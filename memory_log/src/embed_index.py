"""Backfill / reindex CLI — embed existing SQLite rows into ChromaDB.

Usage:
    cd memory_log
    uv run python -m src.embed_index                    # index only un-indexed rows
    uv run python -m src.embed_index --force            # re-embed all rows
    uv run python -m src.embed_index --store promoted_events  # one store only

This is idempotent: without --force it skips rows where ``text_embedding_id`` is already
set. After a successful embed, each row's ``text_embedding_id`` is updated with the ChromaDB
doc id (which equals the row primary key).

Run this after switching embedding models to rebuild the model-namespaced collections.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.config import PROJECT_ROOT, Config
from src.embeddings import create_embedding_client
from src.memory_db import open_db
from src.vector_index import ChromaVectorIndex

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


def _index_store(
    conn,
    emb_client,
    vector_index: ChromaVectorIndex,
    store: str,
    force: bool,
) -> tuple[int, int]:
    """Embed and upsert rows for *store*. Returns (total_eligible, indexed_count)."""
    cfg = _STORE_CONFIG[store]
    pk_col = cfg["pk_col"]
    ts_col = cfg["ts_col"]
    has_location = cfg["has_location"]

    where_clause = "semantic_search_text IS NOT NULL AND semantic_search_text != ''"
    if not force:
        where_clause += f" AND text_embedding_id IS NULL"

    rows = conn.execute(
        f"SELECT * FROM {store} WHERE {where_clause}"
    ).fetchall()

    total = len(rows)
    indexed = 0

    for i in range(0, total, _BATCH_SIZE):
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
                conn.execute(
                    f"UPDATE {store} SET text_embedding_id = ? WHERE {pk_col} = ?",
                    (pk, pk),
                )
                indexed += 1
            except Exception as exc:
                logger.warning("Could not upsert %s/%s: %s", store, pk, exc)

        conn.commit()
        progress = min(i + _BATCH_SIZE, total)
        print(f"  {store}: {progress}/{total} processed, {indexed} indexed so far")

    return total, indexed


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Backfill ChromaDB vector index from existing SQLite memory rows. "
            "Skips already-indexed rows unless --force is given."
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
        help="Re-embed all rows, even those already indexed.",
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

    stores_to_run = [args.store] if args.store else _ALL_STORES
    grand_total = 0
    grand_indexed = 0

    for store in stores_to_run:
        print(f"\nIndexing {store} ({'force re-embed' if args.force else 'new rows only'})…")
        total, indexed = _index_store(conn, emb_client, vector_index, store, force=args.force)
        print(f"  {store}: {indexed}/{total} rows indexed")
        grand_total += total
        grand_indexed += indexed

    print(f"\nDone. {grand_indexed}/{grand_total} rows indexed across {len(stores_to_run)} store(s).")
    print(
        f"Verify: sqlite3 {config.memory_db_path} "
        '"SELECT count(*) FROM promoted_events WHERE text_embedding_id IS NOT NULL;"'
    )


if __name__ == "__main__":
    main()
