"""LTM query CLI — REPL for querying long-term visual memories."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

from src.config import Config, PROJECT_ROOT
from src.ltm_query.answer_generator import AnswerGenerator, format_evidence
from src.ltm_query.evidence import VisualGroundingResult, build_evidence_pack
from src.ltm_query.query_planner import QueryPlanner, RetrievalPlan
from src.ltm_query.retrieval import MemoryRetriever
from src.memory_db import open_db
from src.query_log import QueryLogWriter, build_query_log_record

logger = logging.getLogger("memory_log.ltm_query.cli")

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})

_DEICTIC_WORDS = frozenset({
    "this", "here", "that", "these", "those", "current", "now",
    "looking at", "this object", "this room", "this place", "what i see",
})


def _needs_grounding(query: str) -> bool:
    q = query.lower()
    return any(word in q for word in _DEICTIC_WORDS)


def _print_plan(plan: RetrievalPlan) -> None:
    print("\n[Retrieval plan]")
    print(f"  intent: {plan.intent}")
    if plan.time_range:
        print(f"  time_range: {plan.time_range.start_utc[:19]} → {plan.time_range.end_utc[:19]}")
    if plan.location_filter:
        lf = plan.location_filter
        print(f"  location: ({lf.lat:.4f}, {lf.lon:.4f}) ±{lf.radius_m:.0f}m")
    if plan.semantic_query:
        print(f"  semantic_query: {plan.semantic_query}")
    print(f"  visual_grounding: {plan.needs_current_visual_grounding}")
    print(f"  retrieve_frames: {plan.needs_retrieved_frames}")
    stores = ", ".join(s.store for s in plan.stores_to_query)
    print(f"  stores: {stores}")


def _print_evidence_summary(evidence) -> None:
    print("\n[Evidence summary]")
    for reason in evidence.retrieval_reasons:
        print(f"  + {reason}")
    for note in evidence.uncertainty_notes:
        print(f"  ! {note}")
    if evidence.passive_timeline:
        print(f"  passive timeline: {len(evidence.passive_timeline)} segments")
    if evidence.frame_paths:
        print(f"  frames available: {len(evidence.frame_paths)}")


def run_query(
    query: str,
    planner: QueryPlanner,
    retriever: MemoryRetriever,
    answer_gen: AnswerGenerator,
    config: Config,
    grounder=None,
    frame_source=None,
    current_location=None,
    no_grounding: bool = False,
    log_writer: QueryLogWriter | None = None,
) -> None:
    t_start = time.perf_counter()
    timings: dict[str, float | None] = {
        "plan_ms": None,
        "grounding_ms": None,
        "retrieval_ms": None,
        "answer_ms": None,
    }
    plan: RetrievalPlan | None = None
    visual_grounding: VisualGroundingResult | None = None
    results = None
    answer: str | None = None
    error: str | None = None
    expanded = False
    answer_prompt: str | None = None

    try:
        print("\nPlanning...")
        t0 = time.perf_counter()
        plan = planner.plan(query)
        timings["plan_ms"] = (time.perf_counter() - t0) * 1000
        _print_plan(plan)

        if (
            not no_grounding
            and config.ltm_use_visual_grounding
            and plan.needs_current_visual_grounding
            and grounder is not None
            and frame_source is not None
        ):
            print("\nRunning visual grounding on current scene...")
            t0 = time.perf_counter()
            frames = frame_source.get_recent(4) if hasattr(frame_source, "get_recent") else []
            frame_items = frame_source.get_recent_items(4) if hasattr(frame_source, "get_recent_items") else None
            if frames:
                visual_grounding = grounder.ground(query, frames, frame_items, current_location)
                if visual_grounding:
                    print(f"  scene: {visual_grounding.current_scene_summary}")
                    if visual_grounding.semantic_retrieval_query:
                        if plan.semantic_query:
                            plan.semantic_query += " " + visual_grounding.semantic_retrieval_query
                        else:
                            plan.semantic_query = visual_grounding.semantic_retrieval_query
                    if visual_grounding.suggested_location_radius_m and current_location:
                        from src.ltm_query.query_planner import LocationFilter
                        if current_location.lat is not None and current_location.lon is not None:
                            plan.location_filter = LocationFilter(
                                lat=current_location.lat,
                                lon=current_location.lon,
                                radius_m=visual_grounding.suggested_location_radius_m,
                            )
            else:
                print("  no frames available for grounding")
            timings["grounding_ms"] = (time.perf_counter() - t0) * 1000

        print("\nRetrieving...")
        t0 = time.perf_counter()
        results = retriever.retrieve(plan)

        # Sufficiency check: visual_recall with no events → try broader time range
        if (
            plan.intent == "visual_recall"
            and not results.promoted_events
            and not results.active_queries
            and plan.time_range is not None
        ):
            print("  insufficient evidence — expanding time range...")
            from src.ltm_query.query_planner import TimeRange
            from src.ltm_query.retrieval import RetrievalResults
            expanded_plan = RetrievalPlan(
                intent=plan.intent,
                time_range=None,
                location_filter=plan.location_filter,
                semantic_query=plan.semantic_query,
                needs_current_visual_grounding=False,
                needs_retrieved_frames=plan.needs_retrieved_frames,
                stores_to_query=[
                    s for s in plan.stores_to_query
                    if s.store in ("promoted_events", "active_query_memories")
                ],
            )
            expanded_results = retriever.retrieve(expanded_plan)
            results.promoted_events = expanded_results.promoted_events
            results.active_queries = expanded_results.active_queries
            # Merge expansion trace so both attempts are visible in telemetry
            for t in expanded_results.trace:
                t.note = (t.note or "") + " [post-expansion]"
            results.trace.extend(expanded_results.trace)
            if results.promoted_events or results.active_queries:
                print(f"  found {len(results.promoted_events)} events after expansion")
            expanded = True
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        if plan.needs_retrieved_frames and not results.frame_paths and results.promoted_events:
            results.frame_paths = retriever._query_frames(results.promoted_events)

        evidence = build_evidence_pack(query, plan, results, visual_grounding)
        _print_evidence_summary(evidence)
        answer_prompt = format_evidence(evidence)

        print("\nGenerating answer...")
        t0 = time.perf_counter()
        answer = answer_gen.generate(evidence)
        timings["answer_ms"] = (time.perf_counter() - t0) * 1000
        print(f"\nAnswer: {answer}\n")

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise

    finally:
        if log_writer is not None:
            total_ms = (time.perf_counter() - t_start) * 1000
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
                    no_grounding=no_grounding,
                    vlm_provider=config.vlm_provider,
                    vlm_model=config.vlm_model,
                    planner_raw_response=getattr(planner, "last_raw_response", None),
                    answer_prompt=answer_prompt,
                )
                log_writer.log(record)
            except Exception as log_exc:
                logger.warning("Failed to write query log: %s", log_exc)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    from capture.stream_config import configure_decode_logging
    configure_decode_logging()

    parser = argparse.ArgumentParser(
        description="LTM query — ask questions about your long-term visual memories."
    )
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        help="Disable visual grounding (no live camera needed)",
    )
    parser.add_argument(
        "--camera",
        help="Camera preset for visual grounding (optional)",
    )
    parser.add_argument(
        "--url",
        help="Override camera URL for visual grounding",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if args.camera:
        config.camera_preset_override = args.camera
    if args.url:
        config.camera_url_override = args.url

    try:
        conn = open_db(config.memory_db_path)
    except Exception as exc:
        logger.error("Could not open memory DB at %s: %s", config.memory_db_path, exc)
        sys.exit(1)

    query_log_writer: QueryLogWriter | None = None
    if config.query_log_enabled:
        try:
            query_log_writer = QueryLogWriter(config.query_log_db_path)
            logger.info("Query log DB opened: %s", config.query_log_db_path)
        except Exception as exc:
            logger.warning("Could not open query log DB (logging disabled): %s", exc)

    embedding_client = None
    vector_index = None
    if config.vector_search_enabled:
        try:
            from src.embeddings import create_embedding_client
            from src.vector_index import ChromaVectorIndex

            embedding_client = create_embedding_client(config)
            if embedding_client is not None:
                vector_index = ChromaVectorIndex(config.chroma_path, embedding_client.model)
                logger.info(
                    "Vector search enabled: provider=%s model=%s",
                    embedding_client.provider,
                    embedding_client.model,
                )
                if config.embed_auto_backfill:
                    try:
                        from src.embed_index import reconcile_model_index
                        stats = reconcile_model_index(conn, embedding_client, vector_index)
                        total_backfilled = sum(idx for _t, idx in stats.values())
                        if total_backfilled:
                            logger.info(
                                "Auto-backfill: embedded %d missing rows via %s/%s",
                                total_backfilled,
                                embedding_client.provider,
                                embedding_client.model,
                            )
                    except Exception as _backfill_exc:
                        logger.warning(
                            "Auto-backfill skipped (LIKE fallback active): %s", _backfill_exc
                        )
        except Exception as exc:
            logger.warning("Vector search init failed (falling back to LIKE): %s", exc)

    planner = QueryPlanner(config)
    retriever = MemoryRetriever(
        conn, config, embedding_client=embedding_client, vector_index=vector_index
    )
    answer_gen = AnswerGenerator(config)

    grounder = None
    frame_source = None
    current_location = None

    if not args.no_grounding and config.ltm_use_visual_grounding:
        try:
            from src.ltm_query.visual_grounding import VisualGrounder
            from src.frame_source import create_frame_source
            from src.location import resolve_location, LocationSidecarStore

            config_validated = True
            try:
                config.validate()
            except ValueError:
                config_validated = False

            if config_validated:
                grounder = VisualGrounder(config)
                frame_source = create_frame_source(config)
                frame_source.start()
                current_location = resolve_location(config, None, config.camera_source_key)
                print(f"Visual grounding enabled (camera: {config.camera_source_key})")
            else:
                print("Visual grounding disabled (camera config invalid)")
        except Exception as exc:
            logger.warning("Visual grounding init failed: %s", exc)
            print(f"Visual grounding disabled: {exc}")

    print("\nLTM query: ask questions about your past visual memories.")
    print("Examples:")
    print("  - Where was I yesterday?")
    print("  - What did I ask about the camera?")
    print("  - What did I see near this location?")
    print("  - What was here yesterday? (requires visual grounding)")
    print("\nType 'q' to quit.\n")

    try:
        while True:
            sys.stdout.write("> ")
            sys.stdout.flush()
            try:
                query = input().strip()
            except EOFError:
                break

            if not query:
                continue

            if query.lower() in QUIT_COMMANDS:
                break

            try:
                run_query(
                    query=query,
                    planner=planner,
                    retriever=retriever,
                    answer_gen=answer_gen,
                    config=config,
                    grounder=grounder,
                    frame_source=frame_source,
                    current_location=current_location,
                    no_grounding=args.no_grounding,
                    log_writer=query_log_writer,
                )
            except Exception as exc:
                logger.exception("Query error: %s", exc)
                print(f"\nError: {exc}\n")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if frame_source is not None:
            frame_source.stop()
            frame_source.release()
        if query_log_writer is not None:
            try:
                query_log_writer.close()
            except Exception:
                pass
        print("Done.")


if __name__ == "__main__":
    main()
