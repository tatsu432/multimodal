"""DashboardService — owns all shared components and exposes streaming generators.

Shared object lifecycle mirrors src/run_all.py:
  - Camera source (CameraFrameSource / WebcamFrameSource)
  - VLM client + MemoryWriter + SQLiteWriter (for live QA)
  - QueryPlanner + MemoryRetriever + AnswerGenerator (for LTM)
  - Optional geocode client, location sidecar/server, query log writer

Both streaming generators acquire a per-type lock so concurrent requests
(e.g. two browser tabs) are serialised per pipeline rather than interleaved.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Iterator

import cv2

from src.config import Config, PROJECT_ROOT
from src.frame_source import FrameSource, create_frame_source
from src.geocode_client import GeocodeClient
from src.location import LocationSidecarStore, enrich_location_with_geocode, resolve_location
from src.location_server import LocationServer
from src.ltm_query.answer_generator import AnswerGenerator, format_evidence
from src.ltm_query.evidence import build_evidence_pack
from src.ltm_query.query_planner import QueryPlanner
from src.ltm_query.retrieval import MemoryRetriever, retrieve_with_expansion
from src.memory_db import open_db
from src.memory_writer import MemoryWriter
from src.query_log import QueryLogWriter, build_query_log_record
from src.utils import resize_frame
from src.vlm_client import VLMClient, create_vlm_client

logger = logging.getLogger("memory_log.dashboard.service")


class DashboardService:
    """Initialises and holds all shared components; exposes streaming generators.

    Call ``start()`` after construction to begin camera capture.
    Call ``stop()`` to release resources cleanly.
    """

    def __init__(self, config: Config, no_grounding: bool = False) -> None:
        self._config = config
        self._no_grounding = no_grounding

        # per-pipeline concurrency locks
        self._qa_lock = threading.Lock()
        self._ltm_lock = threading.Lock()

        # --- camera -------------------------------------------------------
        self._source: FrameSource | None = None
        try:
            self._source = create_frame_source(config)
        except Exception as exc:
            logger.warning("Camera source unavailable: %s", exc)

        # --- location sidecar / geocode -----------------------------------
        self._sidecar: LocationSidecarStore | None = None
        self._location_server: LocationServer | None = None
        self._geocode_client: GeocodeClient | None = None

        if config.location_server_enabled:
            self._sidecar = LocationSidecarStore(max_age_sec=config.location_gps_max_age_sec)
            self._location_server = LocationServer(
                store=self._sidecar,
                host=config.location_server_host,
                port=config.location_server_port,
                cert_path=config.location_server_cert,
                key_path=config.location_server_key,
            )

        if config.geocode_enabled:
            try:
                self._geocode_client = GeocodeClient(config)
            except Exception as exc:
                logger.warning("Geocode client unavailable: %s", exc)

        # --- live QA writers ----------------------------------------------
        self._vlm: VLMClient | None = None
        self._writer: MemoryWriter | None = None
        self._db_writer = None  # SQLiteWriter
        self._db_conn: sqlite3.Connection | None = None

        try:
            self._vlm = create_vlm_client(config)
        except Exception as exc:
            logger.warning("VLM client unavailable: %s", exc)

        try:
            self._writer = MemoryWriter(config)
        except Exception as exc:
            logger.warning("MemoryWriter unavailable: %s", exc)

        try:
            indexer = None
            if config.vector_search_enabled:
                try:
                    from src.vector_index import create_memory_indexer
                    indexer = create_memory_indexer(config)
                except Exception as idx_exc:
                    logger.warning("Vector indexer unavailable (write-side): %s", idx_exc)
            self._db_conn = open_db(config.memory_db_path)
            from src.db_writer import SQLiteWriter
            self._db_writer = SQLiteWriter(self._db_conn, PROJECT_ROOT, indexer=indexer)
        except Exception as exc:
            logger.warning("SQLiteWriter unavailable (JSONL-only): %s", exc)

        # --- LTM pipeline -------------------------------------------------
        self._planner: QueryPlanner | None = None
        self._retriever: MemoryRetriever | None = None
        self._answer_gen: AnswerGenerator | None = None
        self._ltm_conn: sqlite3.Connection | None = None
        self._log_writer: QueryLogWriter | None = None
        self._grounder = None  # VisualGrounder | None

        try:
            self._planner = QueryPlanner(config)
        except Exception as exc:
            logger.warning("QueryPlanner unavailable: %s", exc)

        try:
            embedding_client = None
            vector_index = None
            if config.vector_search_enabled:
                try:
                    from src.embeddings import create_embedding_client
                    from src.vector_index import ChromaVectorIndex
                    embedding_client = create_embedding_client(config)
                    if embedding_client is not None:
                        vector_index = ChromaVectorIndex(config.chroma_path, embedding_client.model)
                except Exception as vec_exc:
                    logger.warning("Vector search init failed (LIKE fallback): %s", vec_exc)
            self._ltm_conn = open_db(config.memory_db_path)
            self._retriever = MemoryRetriever(
                self._ltm_conn, config,
                embedding_client=embedding_client,
                vector_index=vector_index,
            )
        except Exception as exc:
            logger.warning("MemoryRetriever unavailable: %s", exc)

        try:
            self._answer_gen = AnswerGenerator(config)
        except Exception as exc:
            logger.warning("AnswerGenerator unavailable: %s", exc)

        if config.query_log_enabled:
            try:
                self._log_writer = QueryLogWriter(config.query_log_db_path)
            except Exception as exc:
                logger.warning("QueryLogWriter unavailable: %s", exc)

        if not no_grounding and config.ltm_use_visual_grounding and self._source is not None:
            try:
                from src.ltm_query.visual_grounding import VisualGrounder
                self._grounder = VisualGrounder(config)
            except Exception as exc:
                logger.warning("VisualGrounder unavailable: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start camera capture (and location sidecar if enabled)."""
        if self._source is not None:
            self._source.start()
        if self._location_server is not None:
            self._location_server.start()

    def stop(self) -> None:
        """Stop camera and release all held resources."""
        if self._source is not None:
            try:
                self._source.stop()
                self._source.release()
            except Exception as exc:
                logger.warning("Error stopping camera source: %s", exc)

        if self._location_server is not None:
            try:
                self._location_server.stop()
            except Exception:
                pass

        if self._geocode_client is not None:
            try:
                self._geocode_client.close()
            except Exception:
                pass

        if self._log_writer is not None:
            try:
                self._log_writer.close()
            except Exception:
                pass

        for conn in (self._db_conn, self._ltm_conn):
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame_jpeg(self, max_width: int = 640, quality: int = 80) -> bytes | None:
        """Return the latest camera frame as JPEG bytes, or ``None`` if unavailable."""
        if self._source is None:
            return None
        ok, frame = self._source.read()
        if not ok or frame is None:
            return None
        frame = resize_frame(frame, max_width=max_width)
        ok2, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok2:
            return None
        return buf.tobytes()

    # ------------------------------------------------------------------
    # Live QA streaming
    # ------------------------------------------------------------------

    def live_qa_stream(self, question: str) -> Iterator[tuple[str, dict]]:
        """Yield ``(event_name, payload)`` tuples for a live-QA question.

        Events: ``token`` (text delta), ``done`` (metadata), ``error``.
        """
        if self._vlm is None:
            yield "error", {"message": "VLM not configured"}
            yield "done", {"latency_s": 0, "frame_count": 0}
            return
        if self._source is None:
            yield "error", {"message": "Camera source not available"}
            yield "done", {"latency_s": 0, "frame_count": 0}
            return

        with self._qa_lock:
            yield from self._live_qa_inner(question)

    def _live_qa_inner(self, question: str) -> Iterator[tuple[str, dict]]:
        num_frames = min(self._config.num_frames_per_query, self._config.frame_buffer_size)
        frames = self._source.get_recent(num_frames)  # type: ignore[union-attr]
        frame_items = (
            self._source.get_recent_items(num_frames)
            if hasattr(self._source, "get_recent_items") else None
        )

        if not frames:
            yield "error", {"message": "No frames available yet — wait a few seconds."}
            yield "done", {"latency_s": 0, "frame_count": 0}
            return

        t_start = time.perf_counter()
        full_answer = ""

        try:
            for token in self._vlm.answer_question_stream(question, frames, frame_items):  # type: ignore[union-attr]
                full_answer += token
                yield "token", {"text": token}
        except Exception as exc:
            logger.error("Live QA VLM error: %s", exc)
            yield "error", {"message": str(exc)}
            yield "done", {"latency_s": round(time.perf_counter() - t_start, 2), "frame_count": len(frames)}
            return

        latency_s = round(time.perf_counter() - t_start, 2)

        # Persist memory (non-fatal)
        location_label = "unknown"
        memory_id = ""
        try:
            location = resolve_location(self._config, self._sidecar, self._config.camera_source_key)
            location = enrich_location_with_geocode(self._config, location, self._geocode_client)
            location_label = location.display_name()

            if self._writer is not None:
                record = self._writer.save_memory(
                    frames=frames,
                    frame_items=frame_items,
                    user_question=question,
                    model_answer=full_answer,
                    location=location,
                    camera_source=self._config.camera_source_key,
                )
                memory_id = getattr(record, "memory_id", "")
                if self._db_writer is not None:
                    self._db_writer.write_active_query_with_event(
                        record, location, frames, frame_items,
                        event_frame_dir=self._config.promoted_event_frame_dir,
                    )
        except Exception as exc:
            logger.error("Failed to persist live-QA memory: %s", exc)

        yield "done", {
            "memory_id": memory_id,
            "location": location_label,
            "latency_s": latency_s,
            "frame_count": len(frames),
        }

    # ------------------------------------------------------------------
    # Long-term memory streaming
    # ------------------------------------------------------------------

    def ltm_stream(self, query: str) -> Iterator[tuple[str, dict]]:
        """Yield ``(event_name, payload)`` tuples for a long-term memory query.

        Events: ``plan``, ``grounding`` (optional), ``retrieval``, ``token``,
        ``done``, ``error``.
        """
        if self._planner is None or self._retriever is None or self._answer_gen is None:
            yield "error", {"message": "LTM pipeline not available (memory DB not set up?)"}
            yield "done", {"latency_s": 0}
            return

        with self._ltm_lock:
            yield from self._ltm_inner(query)

    def _ltm_inner(self, query: str) -> Iterator[tuple[str, dict]]:  # noqa: C901
        t_start = time.perf_counter()
        timings: dict[str, float | None] = {
            "plan_ms": None,
            "grounding_ms": None,
            "retrieval_ms": None,
            "answer_ms": None,
        }
        plan = None
        visual_grounding = None
        results = None
        answer = ""
        error: str | None = None
        expanded = False
        answer_prompt: str | None = None

        try:
            # --- Plan ---------------------------------------------------
            t0 = time.perf_counter()
            plan = self._planner.plan(query)  # type: ignore[union-attr]
            timings["plan_ms"] = (time.perf_counter() - t0) * 1000

            yield "plan", {
                "intent": plan.intent,
                "time_range": (
                    {"start": plan.time_range.start_utc, "end": plan.time_range.end_utc}
                    if plan.time_range else None
                ),
                "location_filter": (
                    {
                        "lat": plan.location_filter.lat,
                        "lon": plan.location_filter.lon,
                        "radius_m": plan.location_filter.radius_m,
                    }
                    if plan.location_filter else None
                ),
                "semantic_query": plan.semantic_query,
                "stores": [s.store for s in plan.stores_to_query],
                "needs_grounding": plan.needs_current_visual_grounding,
                "needs_frames": plan.needs_retrieved_frames,
            }

            # --- Optional visual grounding ------------------------------
            if (
                not self._no_grounding
                and self._config.ltm_use_visual_grounding
                and plan.needs_current_visual_grounding
                and self._grounder is not None
                and self._source is not None
            ):
                t0 = time.perf_counter()
                frames = (
                    self._source.get_recent(4)
                    if hasattr(self._source, "get_recent") else []
                )
                frame_items = (
                    self._source.get_recent_items(4)
                    if hasattr(self._source, "get_recent_items") else None
                )
                if frames:
                    current_loc = resolve_location(
                        self._config, self._sidecar, self._config.camera_source_key
                    )
                    visual_grounding = self._grounder.ground(
                        query, frames, frame_items, current_loc
                    )
                    if visual_grounding:
                        yield "grounding", {
                            "scene": visual_grounding.current_scene_summary,
                            "objects": visual_grounding.visible_objects or [],
                        }
                        # Mutate plan exactly as cli.py does
                        if visual_grounding.semantic_retrieval_query:
                            if plan.semantic_query:
                                plan.semantic_query += " " + visual_grounding.semantic_retrieval_query
                            else:
                                plan.semantic_query = visual_grounding.semantic_retrieval_query
                        if (
                            visual_grounding.suggested_location_radius_m
                            and current_loc.lat is not None
                            and current_loc.lon is not None
                        ):
                            from src.ltm_query.query_planner import LocationFilter
                            plan.location_filter = LocationFilter(
                                lat=current_loc.lat,
                                lon=current_loc.lon,
                                radius_m=visual_grounding.suggested_location_radius_m,
                            )
                timings["grounding_ms"] = (time.perf_counter() - t0) * 1000

            # --- Retrieve -----------------------------------------------
            t0 = time.perf_counter()
            results, expanded = retrieve_with_expansion(plan, self._retriever)  # type: ignore[union-attr]
            timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

            if plan.needs_retrieved_frames and not results.frame_paths and results.promoted_events:
                results.frame_paths = self._retriever._query_frames(results.promoted_events)  # type: ignore[union-attr]

            evidence = build_evidence_pack(query, plan, results, visual_grounding)
            answer_prompt = format_evidence(evidence)

            yield "retrieval", {
                "stores": [
                    {
                        "store": t.store,
                        "method": t.method,
                        "candidate_count": t.candidate_count,
                        "final": t.final_count,
                        "note": t.note,
                    }
                    for t in results.trace
                ],
                "evidence_reasons": evidence.retrieval_reasons,
                "uncertainty_notes": evidence.uncertainty_notes,
                "event_count": len(results.promoted_events),
                "qa_count": len(results.active_queries),
                "expanded": expanded,
            }

            # --- Stream answer ------------------------------------------
            t0 = time.perf_counter()
            for token in self._answer_gen.generate_stream(evidence):  # type: ignore[union-attr]
                answer += token
                yield "token", {"text": token}
            timings["answer_ms"] = (time.perf_counter() - t0) * 1000

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("LTM stream error: %s", exc, exc_info=True)
            yield "error", {"message": error}

        total_ms = (time.perf_counter() - t_start) * 1000
        yield "done", {"latency_s": round(total_ms / 1000, 2)}

        # Log telemetry after stream is complete
        if self._log_writer is not None and plan is not None:
            try:
                record = build_query_log_record(
                    query=query,
                    plan=plan,
                    visual_grounding=visual_grounding,
                    results=results,
                    answer=answer,
                    error=error,
                    timings=timings,
                    total_ms=total_ms,
                    expanded=expanded,
                    no_grounding=self._no_grounding,
                    vlm_provider=self._config.vlm_provider,
                    vlm_model=self._config.vlm_model,
                    planner_raw_response=getattr(self._planner, "last_raw_response", None),
                    answer_prompt=answer_prompt,
                )
                self._log_writer.log(record)
            except Exception as log_exc:
                logger.warning("Failed to write LTM query log: %s", log_exc)
