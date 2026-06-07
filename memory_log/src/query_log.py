"""Standalone SQLite log of long-term-memory query interactions.

This is observability / eval telemetry — deliberately a SEPARATE database file
(``outputs/long_term_query_logs.sqlite``) so it is structurally unreachable by the
memory retriever (``src/ltm_query/retrieval.py``), which only ever opens
``memory.sqlite``. One row is appended per LTM query in ``src/ltm_query/cli.py``.
Mirrors the ``src/geocode_cache.py`` standalone-DB pattern.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.db_writer import _now_utc_iso, _timezone_name, _to_utc_iso
from src.utils import make_memory_id

if TYPE_CHECKING:
    from src.ltm_query.evidence import VisualGroundingResult
    from src.ltm_query.query_planner import RetrievalPlan
    from src.ltm_query.retrieval import RetrievalResults

logger = logging.getLogger("memory_log.query_log")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS long_term_query_logs (
    query_log_id          TEXT PRIMARY KEY,
    timestamp_utc         TEXT NOT NULL,
    timestamp_local       TEXT,
    timezone              TEXT,

    user_query            TEXT NOT NULL,
    intent                TEXT,
    semantic_query        TEXT,

    time_range_start_utc  TEXT,
    time_range_end_utc    TEXT,
    location_lat          REAL,
    location_lon          REAL,
    location_radius_m     REAL,

    used_visual_grounding INTEGER,
    no_grounding_flag     INTEGER,
    expanded              INTEGER,

    plan_json             TEXT,
    visual_grounding_json TEXT,
    retrieved_counts_json TEXT,
    retrieved_ids_json    TEXT,
    frame_paths_json      TEXT,

    answer                TEXT,
    error                 TEXT,

    latency_total_ms      REAL,
    latency_plan_ms       REAL,
    latency_grounding_ms  REAL,
    latency_retrieval_ms  REAL,
    latency_answer_ms     REAL,

    vlm_provider          TEXT,
    vlm_model             TEXT,

    created_at_utc        TEXT NOT NULL,
    extra_json            TEXT
)
"""

CREATE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_ltq_logs_time ON long_term_query_logs(timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_ltq_logs_intent ON long_term_query_logs(intent)",
)


@dataclass
class QueryLogRecord:
    query_log_id: str
    timestamp_utc: str
    timestamp_local: str | None
    timezone: str | None
    user_query: str
    intent: str | None
    semantic_query: str | None
    time_range_start_utc: str | None
    time_range_end_utc: str | None
    location_lat: float | None
    location_lon: float | None
    location_radius_m: float | None
    used_visual_grounding: int
    no_grounding_flag: int
    expanded: int
    plan_json: str | None
    visual_grounding_json: str | None
    retrieved_counts_json: str | None
    retrieved_ids_json: str | None
    frame_paths_json: str | None
    answer: str | None
    error: str | None
    latency_total_ms: float | None
    latency_plan_ms: float | None
    latency_grounding_ms: float | None
    latency_retrieval_ms: float | None
    latency_answer_ms: float | None
    vlm_provider: str | None
    vlm_model: str | None
    created_at_utc: str
    extra_json: str | None = None


class QueryLogWriter:
    """Append-only writer for the standalone query-log database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(CREATE_TABLE_SQL)
        for stmt in CREATE_INDEX_SQL:
            self._conn.execute(stmt)
        self._conn.commit()

    def log(self, record: QueryLogRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO long_term_query_logs (
                    query_log_id, timestamp_utc, timestamp_local, timezone,
                    user_query, intent, semantic_query,
                    time_range_start_utc, time_range_end_utc,
                    location_lat, location_lon, location_radius_m,
                    used_visual_grounding, no_grounding_flag, expanded,
                    plan_json, visual_grounding_json, retrieved_counts_json,
                    retrieved_ids_json, frame_paths_json,
                    answer, error,
                    latency_total_ms, latency_plan_ms, latency_grounding_ms,
                    latency_retrieval_ms, latency_answer_ms,
                    vlm_provider, vlm_model,
                    created_at_utc, extra_json
                ) VALUES (
                    :query_log_id, :timestamp_utc, :timestamp_local, :timezone,
                    :user_query, :intent, :semantic_query,
                    :time_range_start_utc, :time_range_end_utc,
                    :location_lat, :location_lon, :location_radius_m,
                    :used_visual_grounding, :no_grounding_flag, :expanded,
                    :plan_json, :visual_grounding_json, :retrieved_counts_json,
                    :retrieved_ids_json, :frame_paths_json,
                    :answer, :error,
                    :latency_total_ms, :latency_plan_ms, :latency_grounding_ms,
                    :latency_retrieval_ms, :latency_answer_ms,
                    :vlm_provider, :vlm_model,
                    :created_at_utc, :extra_json
                )
                """,
                asdict(record),
            )
            self._conn.commit()
        logger.debug("Logged LTM query %s", record.query_log_id)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _dump(obj: Any) -> str | None:
    """JSON-serialize a dataclass / dict / list, returning None on failure or None input."""
    if obj is None:
        return None
    try:
        data = asdict(obj) if is_dataclass(obj) else obj
        return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        logger.debug("Could not serialize %s: %s", type(obj).__name__, exc)
        return None


def _row_ids(rows: list, key: str) -> list:
    ids: list = []
    for row in rows:
        try:
            value = row[key]
        except (IndexError, KeyError):
            value = None
        if value is not None:
            ids.append(value)
    return ids


def build_query_log_record(
    *,
    query: str,
    plan: RetrievalPlan | None,
    visual_grounding: VisualGroundingResult | None,
    results: RetrievalResults | None,
    answer: str | None,
    error: str | None,
    timings: dict[str, float | None],
    total_ms: float | None,
    expanded: bool,
    no_grounding: bool,
    vlm_provider: str | None,
    vlm_model: str | None,
) -> QueryLogRecord:
    """Assemble a QueryLogRecord from the (possibly partial) pipeline artifacts."""
    memory_id, timestamp_local, _ = make_memory_id()

    time_range = getattr(plan, "time_range", None)
    location_filter = getattr(plan, "location_filter", None)

    counts: dict | None = None
    ids: dict | None = None
    frame_paths: list | None = None
    if results is not None:
        counts = {
            "daily_summaries": len(results.daily_summaries),
            "passive_observations": len(results.passive_rows),
            "promoted_events": len(results.promoted_events),
            "active_query_memories": len(results.active_queries),
            "frames": len(results.frame_paths),
        }
        ids = {
            "daily_summaries": _row_ids(results.daily_summaries, "summary_id"),
            "passive_observations": _row_ids(results.passive_rows, "obs_id"),
            "promoted_events": _row_ids(results.promoted_events, "event_id"),
            "active_query_memories": _row_ids(results.active_queries, "active_query_id"),
        }
        frame_paths = list(results.frame_paths)

    return QueryLogRecord(
        query_log_id=f"qlog_{memory_id}",
        timestamp_utc=_to_utc_iso(timestamp_local),
        timestamp_local=timestamp_local,
        timezone=_timezone_name(timestamp_local),
        user_query=query,
        intent=getattr(plan, "intent", None),
        semantic_query=getattr(plan, "semantic_query", None),
        time_range_start_utc=getattr(time_range, "start_utc", None),
        time_range_end_utc=getattr(time_range, "end_utc", None),
        location_lat=getattr(location_filter, "lat", None),
        location_lon=getattr(location_filter, "lon", None),
        location_radius_m=getattr(location_filter, "radius_m", None),
        used_visual_grounding=1 if visual_grounding is not None else 0,
        no_grounding_flag=1 if no_grounding else 0,
        expanded=1 if expanded else 0,
        plan_json=_dump(plan),
        visual_grounding_json=_dump(visual_grounding),
        retrieved_counts_json=_dump(counts),
        retrieved_ids_json=_dump(ids),
        frame_paths_json=_dump(frame_paths),
        answer=answer,
        error=error,
        latency_total_ms=total_ms,
        latency_plan_ms=timings.get("plan_ms"),
        latency_grounding_ms=timings.get("grounding_ms"),
        latency_retrieval_ms=timings.get("retrieval_ms"),
        latency_answer_ms=timings.get("answer_ms"),
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        created_at_utc=_now_utc_iso(),
        extra_json=None,
    )
