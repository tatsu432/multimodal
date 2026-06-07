"""Deterministic retrieval from the SQLite memory DB based on a RetrievalPlan."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field

from src.config import Config
from src.ltm_query.query_planner import LocationFilter, RetrievalPlan, StoreQuery, TimeRange

logger = logging.getLogger("memory_log.ltm_query.retrieval")

# ~1 degree latitude ≈ 111 km
_KM_PER_DEG_LAT = 111.0


def _radius_to_lat_deg(radius_m: float) -> float:
    return radius_m / 1000.0 / _KM_PER_DEG_LAT


def _radius_to_lon_deg(radius_m: float, lat: float) -> float:
    km_per_deg_lon = _KM_PER_DEG_LAT * math.cos(math.radians(lat))
    if km_per_deg_lon < 0.001:
        return 180.0
    return radius_m / 1000.0 / km_per_deg_lon


def _like_clauses(column: str, keywords: list[str]) -> tuple[str, list[str]]:
    """Build OR-joined LIKE clauses for keyword search."""
    if not keywords:
        return "1=1", []
    parts = " OR ".join(f"{column} LIKE ?" for _ in keywords)
    params = [f"%{kw}%" for kw in keywords]
    return f"({parts})", params


def _time_range_clause(tr: TimeRange | None) -> tuple[str, list[str]]:
    if tr is None:
        return "1=1", []
    return "timestamp_utc BETWEEN ? AND ?", [tr.start_utc, tr.end_utc]


def _location_clause(lf: LocationFilter | None, ts_col: str = "timestamp_utc") -> tuple[str, list]:
    if lf is None:
        return "1=1", []
    lat_d = _radius_to_lat_deg(lf.radius_m)
    lon_d = _radius_to_lon_deg(lf.radius_m, lf.lat)
    sql = (
        "latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
    )
    params = [lf.lat - lat_d, lf.lat + lat_d, lf.lon - lon_d, lf.lon + lon_d]
    return sql, params


@dataclass
class RetrievalResults:
    daily_summaries: list[sqlite3.Row] = field(default_factory=list)
    passive_rows: list[sqlite3.Row] = field(default_factory=list)
    promoted_events: list[sqlite3.Row] = field(default_factory=list)
    active_queries: list[sqlite3.Row] = field(default_factory=list)
    frame_paths: list[str] = field(default_factory=list)


class MemoryRetriever:
    def __init__(self, conn: sqlite3.Connection, config: Config) -> None:
        self._conn = conn
        self._config = config

    def retrieve(self, plan: RetrievalPlan) -> RetrievalResults:
        results = RetrievalResults()
        for sq in plan.stores_to_query:
            self._execute_store_query(sq, plan, results)
        return results

    def _execute_store_query(
        self,
        sq: StoreQuery,
        plan: RetrievalPlan,
        results: RetrievalResults,
    ) -> None:
        store = sq.store
        if store == "daily_summaries":
            results.daily_summaries = self._query_daily_summaries(sq, plan)
        elif store == "passive_observations":
            results.passive_rows = self._query_passive_observations(sq, plan)
        elif store == "promoted_events":
            results.promoted_events = self._query_promoted_events(sq, plan)
        elif store == "active_query_memories":
            results.active_queries = self._query_active_queries(sq, plan)
        elif store == "frames":
            results.frame_paths = self._query_frames(results.promoted_events)
        else:
            logger.warning("Unknown store in plan: %r", store)

    def _query_daily_summaries(self, sq: StoreQuery, plan: RetrievalPlan) -> list[sqlite3.Row]:
        conditions: list[str] = []
        params: list = []

        if plan.time_range:
            conditions.append("date_local BETWEEN substr(?, 1, 10) AND substr(?, 1, 10)")
            params.extend([plan.time_range.start_utc, plan.time_range.end_utc])

        if plan.semantic_query:
            keywords = plan.semantic_query.split()[:5]
            kw_sql, kw_params = _like_clauses("summary_text", keywords)
            conditions.append(kw_sql)
            params.extend(kw_params)

        where = " AND ".join(conditions) if conditions else "1=1"
        limit = sq.top_k or 3
        sql = f"SELECT * FROM daily_summaries WHERE {where} ORDER BY date_local DESC LIMIT ?"
        rows = self._conn.execute(sql, params + [limit]).fetchall()
        logger.debug("daily_summaries: %d rows", len(rows))
        return rows

    def _query_passive_observations(self, sq: StoreQuery, plan: RetrievalPlan) -> list[sqlite3.Row]:
        conditions: list[str] = []
        params: list = []

        tr_sql, tr_params = _time_range_clause(plan.time_range)
        conditions.append(tr_sql)
        params.extend(tr_params)

        loc_sql, loc_params = _location_clause(plan.location_filter)
        conditions.append(loc_sql)
        params.extend(loc_params)

        where = " AND ".join(conditions)
        limit = sq.max_records or self._config.ltm_max_passive_rows
        sql = f"SELECT * FROM passive_observations WHERE {where} ORDER BY timestamp_utc ASC LIMIT ?"
        rows = self._conn.execute(sql, params + [limit]).fetchall()
        logger.debug("passive_observations: %d rows", len(rows))
        return rows

    def _query_promoted_events(self, sq: StoreQuery, plan: RetrievalPlan) -> list[sqlite3.Row]:
        conditions: list[str] = []
        params: list = []

        tr_sql, tr_params = _time_range_clause(
            TimeRange(start_utc=plan.time_range.start_utc, end_utc=plan.time_range.end_utc)
            if plan.time_range else None
        )
        if plan.time_range:
            conditions.append(tr_sql.replace("timestamp_utc", "start_ts_utc"))
            params.extend(tr_params)

        loc_sql, loc_params = _location_clause(plan.location_filter)
        conditions.append(loc_sql)
        params.extend(loc_params)

        if plan.semantic_query:
            keywords = plan.semantic_query.split()[:6]
            for col in ("semantic_search_text", "scene_summary"):
                kw_sql, kw_params = _like_clauses(col, keywords)
                conditions.append(kw_sql)
                params.extend(kw_params)

        where = " AND ".join(conditions) if conditions else "1=1"
        limit = sq.top_k or self._config.ltm_promoted_event_top_k
        sql = f"SELECT * FROM promoted_events WHERE {where} ORDER BY start_ts_utc DESC LIMIT ?"
        rows = self._conn.execute(sql, params + [limit]).fetchall()
        logger.debug("promoted_events: %d rows", len(rows))
        return rows

    def _query_active_queries(self, sq: StoreQuery, plan: RetrievalPlan) -> list[sqlite3.Row]:
        conditions: list[str] = []
        params: list = []

        tr_sql, tr_params = _time_range_clause(plan.time_range)
        conditions.append(tr_sql)
        params.extend(tr_params)

        loc_sql, loc_params = _location_clause(plan.location_filter)
        conditions.append(loc_sql)
        params.extend(loc_params)

        if plan.semantic_query:
            keywords = plan.semantic_query.split()[:6]
            search_cols = ["user_question", "model_answer", "semantic_search_text"]
            all_kw_conditions: list[str] = []
            for col in search_cols:
                kw_sql, kw_params = _like_clauses(col, keywords)
                all_kw_conditions.append(kw_sql)
                params.extend(kw_params)
            conditions.append("(" + " OR ".join(all_kw_conditions) + ")")

        where = " AND ".join(conditions) if conditions else "1=1"
        limit = sq.top_k or self._config.ltm_active_query_top_k
        sql = f"SELECT * FROM active_query_memories WHERE {where} ORDER BY timestamp_utc DESC LIMIT ?"
        rows = self._conn.execute(sql, params + [limit]).fetchall()
        logger.debug("active_query_memories: %d rows", len(rows))
        return rows

    def _query_frames(self, promoted_events: list[sqlite3.Row]) -> list[str]:
        if not promoted_events:
            return []
        final_k = self._config.ltm_final_event_k
        top_events = promoted_events[:final_k]
        event_ids = [row["event_id"] for row in top_events]

        placeholders = ",".join("?" * len(event_ids))
        sql = f"SELECT frame_path FROM frames WHERE promoted_event_id IN ({placeholders}) ORDER BY frame_index ASC LIMIT 24"
        rows = self._conn.execute(sql, event_ids).fetchall()
        paths = [row["frame_path"] for row in rows if row["frame_path"]]
        logger.debug("frames: %d paths from top %d events", len(paths), len(top_events))
        return paths
