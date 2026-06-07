"""Deterministic retrieval from the SQLite memory DB based on a RetrievalPlan."""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.config import Config
from src.ltm_query.query_planner import LocationFilter, RetrievalPlan, StoreQuery, TimeRange

if TYPE_CHECKING:
    from src.embeddings import EmbeddingClient
    from src.vector_index import ChromaVectorIndex

logger = logging.getLogger("memory_log.ltm_query.retrieval")


@dataclass
class StoreTrace:
    """Per-store retrieval audit record — captured inside RetrievalResults.trace."""

    store: str
    method: str  # "vector" | "like" | "metadata"
    candidate_count: int | None  # ChromaDB candidates before SQL filter; None when LIKE/metadata
    sql: str
    params: list
    final_count: int
    note: str | None = None  # set when vector candidates were dropped by SQL filter

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
    trace: list[StoreTrace] = field(default_factory=list)


class MemoryRetriever:
    def __init__(
        self,
        conn: sqlite3.Connection,
        config: Config,
        embedding_client: "EmbeddingClient | None" = None,
        vector_index: "ChromaVectorIndex | None" = None,
    ) -> None:
        self._conn = conn
        self._config = config
        self._embedding_client = embedding_client
        self._vector_index = vector_index
        # per-retrieve() call cache for the query embedding
        self._query_vec_cache: dict[str, list[float] | None] = {}

    def retrieve(self, plan: RetrievalPlan) -> RetrievalResults:
        self._query_vec_cache = {}  # reset per call
        results = RetrievalResults()
        for sq in plan.stores_to_query:
            self._execute_store_query(sq, plan, results)
        return results

    def _append_trace(
        self,
        results: RetrievalResults,
        store: str,
        method: str,
        sql: str,
        params: list,
        final_count: int,
        candidate_count: int | None = None,
    ) -> None:
        """Record one StoreTrace and emit an INFO log line for the live file trace."""
        note: str | None = None
        if method == "vector" and candidate_count is not None and candidate_count > 0 and final_count == 0:
            note = f"{candidate_count} vector candidates excluded by time_range/location filter"

        results.trace.append(StoreTrace(
            store=store,
            method=method,
            candidate_count=candidate_count,
            sql=sql,
            params=params,
            final_count=final_count,
            note=note,
        ))

        # Channel 2: readable live trace in the app-log file (off-terminal)
        if candidate_count is not None:
            cand_str = f"candidates={candidate_count} → rows={final_count}"
        else:
            cand_str = f"rows={final_count}"
        msg = f"%s method=%s %s"
        if note:
            logger.info(msg + " NOTE: %s", store, method, cand_str, note)
        else:
            logger.info(msg, store, method, cand_str)

    def _get_query_vec(self, query: str) -> list[float] | None:
        """Embed *query* and memoize within the current retrieve() call."""
        if query not in self._query_vec_cache:
            try:
                vecs = self._embedding_client.embed([query])  # type: ignore[union-attr]
                self._query_vec_cache[query] = vecs[0] if vecs else None
            except Exception as exc:
                logger.warning("Could not embed query for vector search: %s", exc)
                self._query_vec_cache[query] = None
        return self._query_vec_cache[query]

    def _semantic_candidate_ids(
        self,
        owner_table: str,
        plan: RetrievalPlan,
        limit: int,
    ) -> list[str] | None:
        """Return ranked owner_ids from ChromaDB, or None to fall back to LIKE.

        Returns None when:
        - vector search is disabled in config
        - embedding client or vector index not available
        - no semantic_query in the plan
        - embedding fails (already logged as warning)
        On success returns a (possibly empty) list of ids ranked by similarity.
        """
        if (
            not self._config.vector_search_enabled
            or self._embedding_client is None
            or self._vector_index is None
            or not plan.semantic_query
        ):
            return None

        # Check before embedding the query — avoids a wasted round-trip when the
        # collection for the current model has never been populated (e.g. after
        # switching EMBEDDING_PROVIDER without re-indexing).
        if self._vector_index.count(owner_table) == 0:
            logger.warning(
                "Vector collection for '%s' is empty — model not yet indexed? "
                "Run `uv run python -m src.embed_index` or set EMBED_AUTO_BACKFILL=true. "
                "Falling back to LIKE keyword search.",
                owner_table,
            )
            return None

        query_vec = self._get_query_vec(plan.semantic_query)
        if query_vec is None:
            return None

        try:
            where = self._vector_index.build_where(
                plan.time_range, plan.location_filter, owner_table
            )
            pairs = self._vector_index.search(
                owner_table, query_vec, top_k=limit, where=where
            )
        except Exception as exc:
            logger.warning("Vector search failed for %s: %s", owner_table, exc)
            return None

        return [oid for oid, _dist in pairs]

    def _execute_store_query(
        self,
        sq: StoreQuery,
        plan: RetrievalPlan,
        results: RetrievalResults,
    ) -> None:
        store = sq.store
        if store == "daily_summaries":
            results.daily_summaries = self._query_daily_summaries(sq, plan, results)
        elif store == "passive_observations":
            results.passive_rows = self._query_passive_observations(sq, plan, results)
        elif store == "promoted_events":
            results.promoted_events = self._query_promoted_events(sq, plan, results)
        elif store == "active_query_memories":
            results.active_queries = self._query_active_queries(sq, plan, results)
        elif store == "frames":
            results.frame_paths = self._query_frames(results.promoted_events)
        else:
            logger.warning("Unknown store in plan: %r", store)

    def _query_daily_summaries(
        self, sq: StoreQuery, plan: RetrievalPlan, results: RetrievalResults
    ) -> list[sqlite3.Row]:
        limit = sq.top_k or 3
        candidate_ids = self._semantic_candidate_ids("daily_summaries", plan, limit * 4)

        conditions: list[str] = []
        params: list = []

        if plan.time_range:
            conditions.append("date_local BETWEEN substr(?, 1, 10) AND substr(?, 1, 10)")
            params.extend([plan.time_range.start_utc, plan.time_range.end_utc])

        if candidate_ids is not None:
            if not candidate_ids:
                sql = "SELECT * FROM daily_summaries WHERE 1=0"
                self._append_trace(results, "daily_summaries", "vector", sql, [], 0, candidate_count=0)
                return []
            placeholders = ",".join("?" * len(candidate_ids))
            conditions.append(f"summary_id IN ({placeholders})")
            params.extend(candidate_ids)
            where = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT * FROM daily_summaries WHERE {where}"
            rows = self._conn.execute(sql, params).fetchall()
            id_order = {oid: i for i, oid in enumerate(candidate_ids)}
            rows = sorted(rows, key=lambda r: id_order.get(r["summary_id"], len(candidate_ids)))
            rows = rows[:limit]
            self._append_trace(results, "daily_summaries", "vector", sql, list(params), len(rows), candidate_count=len(candidate_ids))
        else:
            if plan.semantic_query:
                keywords = plan.semantic_query.split()[:5]
                kw_sql, kw_params = _like_clauses("summary_text", keywords)
                conditions.append(kw_sql)
                params.extend(kw_params)
            where = " AND ".join(conditions) if conditions else "1=1"
            method = "like" if plan.semantic_query else "metadata"
            sql = f"SELECT * FROM daily_summaries WHERE {where} ORDER BY date_local DESC LIMIT ?"
            rows = self._conn.execute(sql, params + [limit]).fetchall()
            self._append_trace(results, "daily_summaries", method, sql, list(params) + [limit], len(rows))

        return rows

    def _query_passive_observations(
        self, sq: StoreQuery, plan: RetrievalPlan, results: RetrievalResults
    ) -> list[sqlite3.Row]:
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
        self._append_trace(results, "passive_observations", "metadata", sql, list(params) + [limit], len(rows))
        return rows

    def _query_promoted_events(
        self, sq: StoreQuery, plan: RetrievalPlan, results: RetrievalResults
    ) -> list[sqlite3.Row]:
        limit = sq.top_k or self._config.ltm_promoted_event_top_k
        candidate_ids = self._semantic_candidate_ids("promoted_events", plan, limit * 4)

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

        if candidate_ids is not None:
            if not candidate_ids:
                sql = "SELECT * FROM promoted_events WHERE 1=0"
                self._append_trace(results, "promoted_events", "vector", sql, [], 0, candidate_count=0)
                return []
            placeholders = ",".join("?" * len(candidate_ids))
            conditions.append(f"event_id IN ({placeholders})")
            params.extend(candidate_ids)
            where = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT * FROM promoted_events WHERE {where}"
            rows = self._conn.execute(sql, params).fetchall()
            id_order = {oid: i for i, oid in enumerate(candidate_ids)}
            rows = sorted(rows, key=lambda r: id_order.get(r["event_id"], len(candidate_ids)))
            rows = rows[:limit]
            self._append_trace(results, "promoted_events", "vector", sql, list(params), len(rows), candidate_count=len(candidate_ids))
        else:
            if plan.semantic_query:
                keywords = plan.semantic_query.split()[:6]
                for col in ("semantic_search_text", "scene_summary"):
                    kw_sql, kw_params = _like_clauses(col, keywords)
                    conditions.append(kw_sql)
                    params.extend(kw_params)
            where = " AND ".join(conditions) if conditions else "1=1"
            method = "like" if plan.semantic_query else "metadata"
            sql = f"SELECT * FROM promoted_events WHERE {where} ORDER BY start_ts_utc DESC LIMIT ?"
            rows = self._conn.execute(sql, params + [limit]).fetchall()
            self._append_trace(results, "promoted_events", method, sql, list(params) + [limit], len(rows))

        return rows

    def _query_active_queries(
        self, sq: StoreQuery, plan: RetrievalPlan, results: RetrievalResults
    ) -> list[sqlite3.Row]:
        limit = sq.top_k or self._config.ltm_active_query_top_k
        candidate_ids = self._semantic_candidate_ids("active_query_memories", plan, limit * 4)

        conditions: list[str] = []
        params: list = []

        tr_sql, tr_params = _time_range_clause(plan.time_range)
        conditions.append(tr_sql)
        params.extend(tr_params)

        loc_sql, loc_params = _location_clause(plan.location_filter)
        conditions.append(loc_sql)
        params.extend(loc_params)

        if candidate_ids is not None:
            if not candidate_ids:
                sql = "SELECT * FROM active_query_memories WHERE 1=0"
                self._append_trace(results, "active_query_memories", "vector", sql, [], 0, candidate_count=0)
                return []
            placeholders = ",".join("?" * len(candidate_ids))
            conditions.append(f"active_query_id IN ({placeholders})")
            params.extend(candidate_ids)
            where = " AND ".join(conditions) if conditions else "1=1"
            sql = f"SELECT * FROM active_query_memories WHERE {where}"
            rows = self._conn.execute(sql, params).fetchall()
            id_order = {oid: i for i, oid in enumerate(candidate_ids)}
            rows = sorted(rows, key=lambda r: id_order.get(r["active_query_id"], len(candidate_ids)))
            rows = rows[:limit]
            self._append_trace(results, "active_query_memories", "vector", sql, list(params), len(rows), candidate_count=len(candidate_ids))
        else:
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
            method = "like" if plan.semantic_query else "metadata"
            sql = f"SELECT * FROM active_query_memories WHERE {where} ORDER BY timestamp_utc DESC LIMIT ?"
            rows = self._conn.execute(sql, params + [limit]).fetchall()
            self._append_trace(results, "active_query_memories", method, sql, list(params) + [limit], len(rows))

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


def retrieve_with_expansion(
    plan: RetrievalPlan,
    retriever: MemoryRetriever,
) -> tuple[RetrievalResults, bool]:
    """Run retrieval and optionally expand time range for ``visual_recall`` queries.

    When intent is ``visual_recall``, no events are found, and a ``time_range`` was
    set, this re-runs retrieval without the time constraint over ``promoted_events``
    and ``active_query_memories`` only.  The expansion trace entries are tagged with
    ``[post-expansion]`` so telemetry can distinguish original vs. expanded results.

    Returns ``(results, was_expanded)``.
    """
    results = retriever.retrieve(plan)
    expanded = False

    if (
        plan.intent == "visual_recall"
        and not results.promoted_events
        and not results.active_queries
        and plan.time_range is not None
    ):
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
        for t in expanded_results.trace:
            t.note = (t.note or "") + " [post-expansion]"
        results.trace.extend(expanded_results.trace)
        expanded = True
        logger.info(
            "Time-range expansion: found %d events, %d queries",
            len(results.promoted_events),
            len(results.active_queries),
        )

    return results, expanded
