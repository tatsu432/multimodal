"""Thin wrappers over the real system entry points for eval.

Each driver:
  - Accepts a pre-built Config + isolated DB connection (callers manage isolation).
  - Returns structured result objects (never prints to stdout).
  - Does NOT mock any system behaviour — it calls the real pipeline code.

Config isolation pattern (set these env vars BEFORE Config.from_env()):
    os.environ["MEMORY_DB_PATH"]    = str(run_dir / "memory.sqlite")
    os.environ["CHROMA_PATH"]       = str(run_dir / "chroma")
    os.environ["QUERY_LOG_DB_PATH"] = str(run_dir / "query_logs.sqlite")
    os.environ["FRAME_SOURCE_TYPE"] = "webcam"   # valid; never actually used
    os.environ["GEOCODE_ENABLED"]   = "false"
    config = Config.from_env()
    config.validate()
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import Config
from src.db_writer import SQLiteWriter
from src.ltm_query.answer_generator import AnswerGenerator
from src.ltm_query.evidence import build_evidence_pack
from src.ltm_query.query_planner import QueryPlanner
from src.ltm_query.retrieval import MemoryRetriever, RetrievalResults, retrieve_with_expansion
from src.memory_db import open_db
from src.schema import LocationInfo, MemoryRecord
from src.utils import FrameItem, make_memory_id
from src.vlm_client import create_vlm_client

from evals.manifest import EvalManifest, GoldLocation, MemoryQuestion, SeedMemory
from evals.replay_source import ReplaySource

logger = logging.getLogger("evals.drivers")


# ---- result types ----

@dataclass
class LiveAnswerResult:
    question_id: str
    question: str
    system_answer: str
    frames_used: int
    frame_age_sec: float  # ask_at_sec - latest_frame_media_t
    latency_ms: float


@dataclass
class LtmQueryResult:
    question_id: str
    question: str
    system_answer: str
    plan_intent: str
    retrieval: RetrievalResults
    expanded: bool
    latency_ms: float


# ---- config helpers ----

def build_eval_config(run_dir: Path, extra_env: dict[str, str] | None = None) -> Config:
    """Build a Config that points all DB/storage paths to `run_dir`.

    Sets env vars before calling Config.from_env() so the isolated paths win over .env.
    Call this once per eval run. After calling, further Config.from_env() calls in the
    same process will also see these vars unless explicitly reset.
    """
    import os

    run_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, str] = {
        "MEMORY_DB_PATH": str(run_dir / "memory.sqlite"),
        "CHROMA_PATH": str(run_dir / "chroma"),
        "QUERY_LOG_DB_PATH": str(run_dir / "query_logs.sqlite"),
        "MEMORY_JSONL_PATH": str(run_dir / "memories.jsonl"),
        "OUTPUT_FRAME_DIR": str(run_dir / "frames"),
        "PASSIVE_FRAME_DIR": str(run_dir / "passive_frames"),
        "PROMOTED_EVENT_FRAME_DIR": str(run_dir / "event_frames"),
        "GEOCODE_CACHE_PATH": str(run_dir / "geocode_cache.sqlite"),
        # Harness-safe defaults (override if needed via extra_env)
        "FRAME_SOURCE_TYPE": "webcam",
        "GEOCODE_ENABLED": "false",
        "LOCATION_SERVER_ENABLED": "false",
        "LTM_USE_VISUAL_GROUNDING": "false",
        "EMBED_ON_WRITE": "false",
        "EMBED_AUTO_BACKFILL": "false",
        "LTM_QUERY_LOG_ENABLED": "true",
    }
    if extra_env:
        overrides.update(extra_env)

    for k, v in overrides.items():
        os.environ[k] = v

    config = Config.from_env()
    config.validate()
    return config


# ---- live QA driver ----

def _format_mcq_prompt(question: str, choices: list[str]) -> str:
    """Wrap a question with labeled choices and a letter-first instruction."""
    opts = "\n".join(choices)
    return (
        f"{question}\n\n"
        f"Options:\n{opts}\n\n"
        "Reply with the option letter (A, B, C, or D) on the first line, "
        "then a brief explanation."
    )


def run_live_question(
    question_id: str,
    question: str,
    ask_at_sec: float,
    replay: ReplaySource,
    config: Config,
    num_frames: int | None = None,
    window_sec: float = 30.0,
    choices: list[str] | None = None,
) -> LiveAnswerResult:
    """Answer one live question using frames from `replay` at `ask_at_sec`.

    Args:
        num_frames: override config.num_frames_per_query (useful for sweeping).
        window_sec: lookback window for frame selection.
        choices: if provided (MCQ), the prompt is reformatted to include labeled
                 options and ask the model to lead with a single letter.
    """
    n = num_frames if num_frames is not None else config.num_frames_per_query
    frames, items = replay.frames_at(ask_at_sec, n, window_sec=window_sec)

    # Frame age: how stale is the latest frame relative to the ask time?
    frame_age = 0.0
    if items:
        latest_media_t = items[-1].timestamp - replay.base_epoch
        frame_age = ask_at_sec - latest_media_t

    prompt = _format_mcq_prompt(question, choices) if choices else question

    vlm = create_vlm_client(config)
    t0 = time.monotonic()
    answer = vlm.answer_question(prompt, frames, frame_items=items if items else None)
    latency_ms = (time.monotonic() - t0) * 1000.0

    return LiveAnswerResult(
        question_id=question_id,
        question=question,
        system_answer=answer,
        frames_used=len(frames),
        frame_age_sec=frame_age,
        latency_ms=latency_ms,
    )


# ---- memory seeding ----

def _gold_location_to_info(loc: GoldLocation | None) -> LocationInfo:
    if loc is None:
        return LocationInfo(label="eval", source="eval_harness")
    return LocationInfo(
        label=loc.label or "eval",
        lat=loc.lat,
        lon=loc.lon,
        source="eval_harness",
    )


def _ts_to_utc_iso(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    except (ValueError, TypeError):
        return ts


def _tz_name(ts: str) -> str | None:
    try:
        dt = datetime.fromisoformat(ts)
        offset = dt.utcoffset()
        if offset is None:
            return None
        s = int(offset.total_seconds())
        sign = "+" if s >= 0 else "-"
        h, m = divmod(abs(s) // 60, 60)
        return f"UTC{sign}{h:02d}:{m:02d}"
    except (ValueError, TypeError):
        return None


def seed_memories(
    conn: sqlite3.Connection,
    manifest: EvalManifest,
    project_root: Path,
) -> int:
    """Inject all seed_memories from the manifest into the eval DB.

    Uses the memory's `timestamp` field as the content timestamp (not now()),
    so retrieval time filters work correctly against synthetic past dates.

    Returns the number of records written.
    """
    writer = SQLiteWriter(conn, project_root, indexer=None)
    written = 0

    for idx, mem in enumerate(manifest.seed_memories):
        loc = _gold_location_to_info(mem.location)

        if mem.kind == "active_query":
            # Build a MemoryRecord with the synthetic timestamp
            memory_id = mem.timestamp.replace(":", "-").replace("+", "p").replace(".", "_")
            record = MemoryRecord(
                memory_id=memory_id,
                timestamp=mem.timestamp,
                user_question=mem.user_question or "(eval seed)",
                location=loc,
                model_answer=mem.model_answer,
                frame_paths=mem.frame_paths,
                camera_source=mem.camera_source,
            )
            writer.write_active_query_with_event(
                record=record,
                location=loc,
                frames=[],
                frame_items=None,
            )
            written += 1

        elif mem.kind == "passive":
            obs_id = f"eval_obs_{idx:04d}"
            ts_utc = _ts_to_utc_iso(mem.timestamp)
            tz = _tz_name(mem.timestamp)
            writer.write_passive_observation(
                obs_id=obs_id,
                timestamp_utc=ts_utc,
                timestamp_local=mem.timestamp,
                timezone_name=tz,
                camera_source=mem.camera_source,
                location=loc,
                frame_path=mem.frame_paths[0] if mem.frame_paths else None,
                thumbnail_path=None,
            )
            written += 1

        elif mem.kind == "daily_summary":
            summary_id = f"eval_sum_{idx:04d}"
            writer.write_daily_summary(
                summary_id=summary_id,
                date_local=mem.timestamp[:10],
                timezone_name=_tz_name(mem.timestamp),
                summary_text=mem.summary_text,
                major_places_json=None,
                notable_event_ids_json=None,
                active_query_ids_json=None,
                coverage_start_utc=_ts_to_utc_iso(mem.coverage_start) if mem.coverage_start else None,
                coverage_end_utc=_ts_to_utc_iso(mem.coverage_end) if mem.coverage_end else None,
                raw_model_output=mem.summary_text,
                semantic_search_text=mem.summary_text,
            )
            written += 1

    logger.info("Seeded %d memory records into eval DB", written)
    return written


def replay_ingest_history(
    history_video: Path,
    manifest: EvalManifest,
    conn: sqlite3.Connection,
    config: Config,
    project_root: Path,
    observe_interval_sec: float = 30.0,
    caption_with_vlm: bool = False,
) -> int:
    """Ingest a history video as passive observations with synthetic timestamps.

    This is the replay-ingestion mode for LTM eval. It loops through the video
    at `observe_interval_sec` intervals, writing one passive_observation per step
    with a synthetic timestamp = base_timestamp + media_t.

    Args:
        caption_with_vlm: if True, calls the VLM to generate a model_answer per frame.
                          Adds VLM cost but produces richer semantic text for retrieval.
    Returns:
        Number of passive observations written.
    """
    from evals.replay_source import ReplaySource

    src = ReplaySource(
        history_video,
        sample_interval_sec=observe_interval_sec,
        base_timestamp=manifest.base_timestamp,
    )
    src.load()

    writer = SQLiteWriter(conn, project_root, indexer=None)
    vlm = create_vlm_client(config) if caption_with_vlm else None
    default_loc = _gold_location_to_info(manifest.default_location)
    written = 0

    for entry in src._index:
        t_sec = entry.media_time_sec
        ts_local = datetime.fromtimestamp(entry.synthetic_epoch).astimezone().isoformat(
            timespec="milliseconds"
        )
        ts_utc = _ts_to_utc_iso(ts_local)

        # Optionally caption the frame
        model_answer = ""
        if vlm is not None:
            try:
                model_answer = vlm.answer_question(
                    "Briefly describe what you see in one sentence.",
                    [entry.frame],
                )
            except Exception as exc:
                logger.warning("VLM caption failed at t=%.1fs: %s", t_sec, exc)

        obs_id = f"eval_replay_{int(t_sec):06d}"
        writer.write_passive_observation(
            obs_id=obs_id,
            timestamp_utc=ts_utc,
            timestamp_local=ts_local,
            timezone_name=_tz_name(ts_local),
            camera_source="eval_replay",
            location=default_loc,
            frame_path=None,
            thumbnail_path=None,
        )
        written += 1

    logger.info("Replay-ingested %d passive observations", written)
    return written


# ---- LTM query driver ----

def run_ltm_question(
    question_id: str,
    question: str,
    conn: sqlite3.Connection,
    config: Config,
) -> LtmQueryResult:
    """Run one LTM query through the real 4-stage pipeline.

    Grounding is disabled (no live camera). Returns the answer + full RetrievalResults
    so the caller can compute retrieval metrics.
    """
    planner = QueryPlanner(config)
    retriever = MemoryRetriever(conn, config, embedding_client=None, vector_index=None)
    answer_gen = AnswerGenerator(config)

    t0 = time.monotonic()
    plan = planner.plan(question)
    results, expanded = retrieve_with_expansion(plan, retriever)
    evidence = build_evidence_pack(question, plan, results, visual_grounding=None)
    answer = answer_gen.generate(evidence)
    latency_ms = (time.monotonic() - t0) * 1000.0

    return LtmQueryResult(
        question_id=question_id,
        question=question,
        system_answer=answer,
        plan_intent=plan.intent,
        retrieval=results,
        expanded=expanded,
        latency_ms=latency_ms,
    )
