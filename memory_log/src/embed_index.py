"""Backfill / reindex CLI — embed existing SQLite rows into ChromaDB.

Usage:
    cd memory_log
    uv run python -m src.embed_index                    # index only un-indexed rows
    uv run python -m src.embed_index --force            # re-embed all rows
    uv run python -m src.embed_index --store promoted_events  # one store only

Incremental mode (default) skips rows already present in the CURRENT model's ChromaDB
collection (checked via ``vector_index.existing_ids``), not by ``text_embedding_id``.
This means switching ``EMBEDDING_PROVIDER`` or ``EMBEDDING_MODEL`` and re-running will
automatically populate the new model's collection without touching the old one.
Switching back is instant — the previous collection still holds its docs.

``text_embedding_id`` is still written after each upsert as a best-effort breadcrumb,
but it is NOT used to decide which rows to (re-)embed.

This is idempotent: re-running without --force never re-embeds rows that are already in
the collection.  After a model switch, run this (or let the LTM-query REPL auto-backfill)
to populate the new model-namespaced collection.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from src.config import PROJECT_ROOT, Config
from src.embeddings import create_embedding_client
from src.memory_db import open_db
from src.vector_index import ChromaVectorIndex

if TYPE_CHECKING:
    from src.embeddings import EmbeddingClient

logger = logging.getLogger("memory_log.embed_index")

_BATCH_SIZE = 32

_STORE_CONFIG = {
    "promoted_events": {
        "pk_col": "event_id",
        "ts_col": "start_ts_utc",
        "has_location": True,
    },
    "active_query_memories": {
        "pk_col": "active_query_id",
        "ts_col": "timestamp_utc",
        "has_location": True,
    },
    "daily_summaries": {
        "pk_col": "summary_id",
        "ts_col": "coverage_start_utc",
        "has_location": False,
    },
}

_ALL_STORES = list(_STORE_CONFIG.keys())


def _ts_to_epoch(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def _embed_batch(
    conn: sqlite3.Connection,
    emb_client: "EmbeddingClient",
    vector_index: ChromaVectorIndex,
    store: str,
    rows: list,
) -> int:
    """Embed and upsert *rows* for *store*. Returns count of successfully indexed rows."""
    cfg = _STORE_CONFIG[store]
    pk_col = cfg["pk_col"]
    ts_col = cfg["ts_col"]
    has_location = cfg["has_location"]
    indexed = 0

    for i in range(0, len(rows), _BATCH_SIZE):
        batch = rows[i : i + _BATCH_SIZE]
        texts = [r["semantic_search_text"] for r in batch]
        try:
            vecs = emb_client.embed(texts)
        except Exception as exc:
            logger.warning("Batch embed failed (rows %d–%d): %s", i, i + len(batch), exc)
            continue

        for row, vec in zip(batch, vecs):
            pk = row[pk_col]
            ts = _ts_to_epoch(row[ts_col])
            metadata = {"ts_epoch": ts}
            if has_location:
                lat = row["latitude"]
                lon = row["longitude"]
                if lat is not None:
                    metadata["lat"] = float(lat)
                if lon is not None:
                    metadata["lon"] = float(lon)
            try:
                vector_index.upsert(store, pk, vec, metadata)
                # Best-effort breadcrumb — the Chroma collection is the source of truth
                conn.execute(
                    f"UPDATE {store} SET text_embedding_id = ? WHERE {pk_col} = ?",
                    (pk, pk),
                )
                indexed += 1
            except Exception as exc:
                logger.warning("Could not upsert %s/%s: %s", store, pk, exc)

        conn.commit()
        progress = min(i + _BATCH_SIZE, len(rows))
        print(f"  {store}: {progress}/{len(rows)} processed, {indexed} indexed so far")

    return indexed


def reconcile_model_index(
    conn: sqlite3.Connection,
    emb_client: "EmbeddingClient",
    vector_index: ChromaVectorIndex,
    stores: list[str] | None = None,
    force: bool = False,
) -> dict[str, tuple[int, int]]:
    """Embed SQLite rows that are missing from the current model's ChromaDB collection.

    For each store in *stores* (default: all three), fetches the set of doc ids already
    present in the Chroma collection for this model, then embeds only the missing rows.
    With ``force=True``, re-embeds all eligible rows (useful after a full model switch or
    data correction).

    Returns a dict mapping store name → (total_eligible, indexed_count).

    This function is idempotent and safe to call on every startup: when the current model's
    collection is already fully populated it performs 3 cheap ``get`` calls and returns.
    """
    stores = stores or _ALL_STORES
    results: dict[str, tuple[int, int]] = {}

    for store in stores:
        cfg = _STORE_CONFIG[store]
        pk_col = cfg["pk_col"]

        # Fetch all rows with embeddable text
        rows = conn.execute(
            f"SELECT * FROM {store} "
            "WHERE semantic_search_text IS NOT NULL AND semantic_search_text != ''"
        ).fetchall()
        total = len(rows)

        if not force:
            existing = vector_index.existing_ids(store)
            rows = [r for r in rows if r[pk_col] not in existing]

        if not rows:
            print(f"  {store}: 0/{total} to embed (already up to date)")
            results[store] = (total, 0)
            continue

        action = "force re-embed" if force else f"{len(rows)} missing"
        print(f"\nIndexing {store} ({action} of {total} eligible rows)…")
        indexed = _embed_batch(conn, emb_client, vector_index, store, rows)
        print(f"  {store}: {indexed}/{len(rows)} rows indexed")
        results[store] = (total, indexed)

    return results


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Backfill ChromaDB vector index from existing SQLite memory rows. "
            "Skips rows already in the current model's collection unless --force is given."
        )
    )
    parser.add_argument(
        "--store",
        choices=_ALL_STORES,
        default=None,
        help="Limit to one store (default: all three).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all rows, even those already in the collection.",
    )
    args = parser.parse_args()

    config = Config.from_env()

    if not config.vector_search_enabled:
        print("VECTOR_SEARCH_ENABLED=false — nothing to do.")
        sys.exit(0)

    emb_client = create_embedding_client(config)
    if emb_client is None:
        print(
            "Could not create embedding client. "
            "Check EMBEDDING_PROVIDER / OPENAI_API_KEY / Ollama availability.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Embedding provider: {emb_client.provider}  model: {emb_client.model}  "
        f"(ChromaDB at {config.chroma_path})"
    )

    vector_index = ChromaVectorIndex(config.chroma_path, emb_client.model)

    try:
        conn = open_db(config.memory_db_path)
    except Exception as exc:
        print(f"Could not open memory DB: {exc}", file=sys.stderr)
        sys.exit(1)

    stores_to_run = [args.store] if args.store else None
    results = reconcile_model_index(conn, emb_client, vector_index, stores=stores_to_run, force=args.force)

    grand_total = sum(t for t, _i in results.values())
    grand_indexed = sum(i for _t, i in results.values())
    stores_run = list(results.keys())

    print(f"\nDone. {grand_indexed}/{grand_total} rows indexed across {len(stores_run)} store(s).")
    print(
        f"Verify: sqlite3 {config.memory_db_path} "
        '"SELECT count(*) FROM promoted_events WHERE text_embedding_id IS NOT NULL;"'
    )


if __name__ == "__main__":
    main()
