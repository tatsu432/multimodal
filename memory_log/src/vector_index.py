"""ChromaDB vector index wrapper and write-time memory indexer.

Architecture:
  - ChromaVectorIndex: thin PersistentClient wrapper.  Collections are namespaced as
    ``<owner_table>__<model_slug>`` so switching embedding models never collides; a reindex
    just builds new collections.
  - MemoryIndexer: write-time hook called from SQLiteWriter after a successful commit.
    All errors are non-fatal (logged as warnings).

SQLite remains the source of truth; Chroma is the ANN + metadata-filter engine.
Metadata per document:
  - promoted_events / active_query_memories: ``ts_epoch`` (float), ``lat``, ``lon``
  - daily_summaries: ``ts_epoch`` only (table has no lat/lon columns)
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.config import Config
    from src.embeddings import EmbeddingClient
    from src.ltm_query.query_planner import LocationFilter, TimeRange

logger = logging.getLogger("memory_log.vector_index")

# ~1 degree latitude ≈ 111 km  (mirrors retrieval.py constant)
_KM_PER_DEG_LAT = 111.0

# Stores that carry lat/lon metadata
_SPATIAL_STORES = frozenset({"promoted_events", "active_query_memories"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts_to_epoch(ts: str) -> float:
    """Convert an ISO-8601 timestamp string to a Unix epoch float."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _clean_slug(name: str) -> str:
    """Sanitise a model name for use as a ChromaDB collection name component.

    ChromaDB collection names: 3-63 chars, alphanumeric + hyphens, no leading/trailing
    hyphens, no consecutive periods, must start/end with alphanumeric.
    """
    slug = re.sub(r"[^a-zA-Z0-9\-]", "-", name)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    if not slug or not slug[0].isalpha():
        slug = "m" + slug
    return slug[:48]  # leave headroom for "owner_table__" prefix


def _radius_to_lat_deg(radius_m: float) -> float:
    return radius_m / 1000.0 / _KM_PER_DEG_LAT


def _radius_to_lon_deg(radius_m: float, lat: float) -> float:
    km_per_deg = _KM_PER_DEG_LAT * math.cos(math.radians(lat))
    if km_per_deg < 0.001:
        return 180.0
    return radius_m / 1000.0 / km_per_deg


# ---------------------------------------------------------------------------
# ChromaVectorIndex
# ---------------------------------------------------------------------------

class ChromaVectorIndex:
    """Thin wrapper around a ChromaDB PersistentClient.

    All public methods are safe to call even before Chroma is installed — they raise
    ``ImportError`` with a helpful message if ``chromadb`` is missing.
    """

    def __init__(self, chroma_path: Path, model_slug: str) -> None:
        self._path = chroma_path
        self._slug = _clean_slug(model_slug)
        self._client: Any | None = None
        self._collections: dict[str, Any] = {}

    # --- internal -------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError(
                    "chromadb is not installed. "
                    "Run `uv sync` inside memory_log/ to add it."
                ) from exc
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._path))
        return self._client

    def _get_collection(self, owner_table: str) -> Any:
        if owner_table not in self._collections:
            client = self._get_client()
            name = f"{owner_table}__{self._slug}"
            if len(name) > 63:
                name = name[:63].rstrip("-")
            self._collections[owner_table] = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[owner_table]

    # --- public API -----------------------------------------------------

    def upsert(
        self,
        owner_table: str,
        owner_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Store or update a single embedding with its metadata."""
        col = self._get_collection(owner_table)
        col.upsert(ids=[owner_id], embeddings=[vector], metadatas=[metadata])

    def search(
        self,
        owner_table: str,
        query_vector: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *top_k* (owner_id, distance) pairs ranked by cosine similarity."""
        col = self._get_collection(owner_table)
        count = col.count()
        if count == 0:
            return []
        n_results = min(top_k, count)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        try:
            result = col.query(**kwargs)
        except Exception as exc:
            logger.warning("Chroma search error for %s: %s", owner_table, exc)
            return []
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return list(zip(ids, distances))

    def build_where(
        self,
        time_range: "TimeRange | None",
        location_filter: "LocationFilter | None",
        owner_table: str,
    ) -> dict | None:
        """Build a Chroma ``where`` filter for time range and (if supported) spatial bbox.

        ``daily_summaries`` has no lat/lon — spatial filter is skipped for that store.
        Returns ``None`` when no conditions apply (query without a where clause).
        """
        conditions: list[dict] = []

        if time_range:
            try:
                start_epoch = _ts_to_epoch(time_range.start_utc)
                end_epoch = _ts_to_epoch(time_range.end_utc)
                conditions.append({"ts_epoch": {"$gte": start_epoch}})
                conditions.append({"ts_epoch": {"$lte": end_epoch}})
            except (ValueError, KeyError) as exc:
                logger.debug("Could not parse time range for Chroma where: %s", exc)

        if location_filter and owner_table in _SPATIAL_STORES:
            lat_d = _radius_to_lat_deg(location_filter.radius_m)
            lon_d = _radius_to_lon_deg(location_filter.radius_m, location_filter.lat)
            conditions.append({"lat": {"$gte": float(location_filter.lat - lat_d)}})
            conditions.append({"lat": {"$lte": float(location_filter.lat + lat_d)}})
            conditions.append({"lon": {"$gte": float(location_filter.lon - lon_d)}})
            conditions.append({"lon": {"$lte": float(location_filter.lon + lon_d)}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# MemoryIndexer
# ---------------------------------------------------------------------------

class MemoryIndexer:
    """Write-time hook: embed ``semantic_search_text`` and upsert into ChromaDB.

    All operations are non-fatal — exceptions are caught and logged as warnings so
    embedding failures never block the main write path.
    """

    def __init__(
        self,
        embedding_client: "EmbeddingClient",
        vector_index: ChromaVectorIndex,
    ) -> None:
        self._embedding_client = embedding_client
        self._vector_index = vector_index

    # expose for embed_index CLI
    @property
    def embedding_client(self) -> "EmbeddingClient":
        return self._embedding_client

    @property
    def vector_index(self) -> ChromaVectorIndex:
        return self._vector_index

    def index(
        self,
        owner_table: str,
        owner_id: str,
        text: str,
        timestamp_utc: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> str | None:
        """Embed *text* and upsert into the collection for *owner_table*.

        Returns *owner_id* on success, ``None`` on failure (non-fatal).
        """
        if not text or not text.strip():
            return None
        try:
            vecs = self._embedding_client.embed([text])
            if not vecs:
                return None
            ts_epoch = _ts_to_epoch(timestamp_utc) if timestamp_utc else 0.0
            metadata: dict[str, Any] = {"ts_epoch": ts_epoch}
            if lat is not None:
                metadata["lat"] = float(lat)
            if lon is not None:
                metadata["lon"] = float(lon)
            self._vector_index.upsert(owner_table, owner_id, vecs[0], metadata)
            return owner_id
        except Exception as exc:
            logger.warning(
                "Could not index %s/%s: %s", owner_table, owner_id, exc
            )
            return None

    def index_pair(
        self,
        event_id: str,
        active_query_id: str,
        text: str,
        timestamp_utc: str | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> None:
        """Embed once, upsert to both ``promoted_events`` and ``active_query_memories``.

        A single embedding call covers both collections — the text is shared between
        the promoted event and its linked active query memory.
        """
        if not text or not text.strip():
            return
        try:
            vecs = self._embedding_client.embed([text])
            if not vecs:
                return
            vec = vecs[0]
            ts_epoch = _ts_to_epoch(timestamp_utc) if timestamp_utc else 0.0
            metadata: dict[str, Any] = {"ts_epoch": ts_epoch}
            if lat is not None:
                metadata["lat"] = float(lat)
            if lon is not None:
                metadata["lon"] = float(lon)
            self._vector_index.upsert("promoted_events", event_id, vec, metadata)
            self._vector_index.upsert("active_query_memories", active_query_id, vec, metadata)
        except Exception as exc:
            logger.warning(
                "Could not index pair %s/%s: %s", event_id, active_query_id, exc
            )


def create_memory_indexer(config: "Config") -> MemoryIndexer | None:
    """Build a MemoryIndexer from config, or return None when embeddings are off/unavailable."""
    if not config.vector_search_enabled or not config.embed_on_write:
        return None
    try:
        from src.embeddings import create_embedding_client

        emb_client = create_embedding_client(config)
        if emb_client is None:
            return None
        vector_index = ChromaVectorIndex(config.chroma_path, emb_client.model)
        return MemoryIndexer(emb_client, vector_index)
    except Exception as exc:
        logger.warning("Could not create MemoryIndexer: %s", exc)
        return None
