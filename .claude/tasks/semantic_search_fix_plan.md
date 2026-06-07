Fix: semantic search returns nothing after embedding-model switch

 Context

 Symptom: Semantic (vector) search retrieves nothing for most LTM queries, even when the plan
 correctly selects semantic_search.

 Root cause (confirmed by direct inspection, not inference):
 - memory_log/.env sets EMBEDDING_PROVIDER=openai / EMBEDDING_MODEL=text-embedding-3-small
 (active lines at the bottom, overriding the commented-out ollama defaults).
 - But all 45 existing memories were embedded under the old default nomic-embed-text.
 - ChromaDB collections are model-namespaced as {store}__{model_slug} (vector_index.py:_get_collection).
 Live counts: *__nomic-embed-text = 22/22/1 (populated, 768-dim); *__text-embedding-3-small = 0/0/0 (empty).
 - At query time retrieval.py:_semantic_candidate_ids() → ChromaVectorIndex.search() targets the empty
 openai collection → col.count()==0 → returns []. Each _query_* treats an empty candidate list as
 "definitely no matches" → SELECT ... WHERE 1=0 → zero rows, silently, with no LIKE fallback.

 A plain embed_index (no --force) does not heal this: its incremental gate is
 text_embedding_id IS NULL, but every row already has that column set from the nomic run — so it skips
 everything. Worse, text_embedding_id is already unreliable: the live write path
 (db_writer.py:244-252, 353-360) never sets it — only the offline embed_index.py does.

 User's goal & design decision: They want to freely switch embedding providers (local Ollama for
 production, OpenAI to evaluate quality) and have a model mismatch auto-rebuild so switching "just works."

 My recommendation (agree with the goal, refine the mechanism): Don't "rebuild from scratch on every
 mismatch." Because collections are already model-namespaced, both models' embeddings can coexist. Do an
 incremental per-model backfill: embed only the SQLite rows missing from the current model's collection.
 First switch to a model embeds everything once; switching back is a true zero-row no-op (its collection
 still has its docs). The Chroma collection — not the text_embedding_id column — becomes the source of truth
 for per-model membership. This is strictly better than full-rebuild: cheaper, instant switch-back, enables
 A/B comparison without repeatedly paying to re-embed.

 Changes

 1. src/vector_index.py — expose collection membership

 Add two methods to ChromaVectorIndex (after search()):
 - count(owner_table) -> int: wrap self._get_collection(owner_table).count(), return 0 on exception.
 - existing_ids(owner_table) -> set[str]: set(self._get_collection(owner_table).get(include=[]).get("ids") or []),
 return set() on exception. Verified: col.get(include=[]) returns ids only (chromadb 1.5.9), no vectors fetched.

 2. src/ltm_query/retrieval.py — graceful fallback on empty collection (the silent-failure fix)

 In _semantic_candidate_ids(), before _get_query_vec() (so we skip a wasted embed call), add:
 if self._vector_index.count(owner_table) == 0:
     logger.warning("Vector collection '%s' empty (model not indexed?); falling back to LIKE", owner_table)
     return None
 Returning None triggers the existing LIKE branch in all three _query_* methods (verified: _query_daily_summaries,
 _query_promoted_events, _query_active_queries all gate on if candidate_ids is not None:). This permanently
 prevents a model mismatch from producing silent zero results, independent of backfill.

 3. src/embed_index.py — per-model incremental backfill (replaces broken text_embedding_id IS NULL gate)

 Extract a reusable function reused by both the CLI and startup auto-backfill:
 def reconcile_model_index(conn, emb_client, vector_index, stores=None, force=False) -> dict[str, tuple[int, int]]
 - For each store: existing = vector_index.existing_ids(store) (one get per store).
 - Select rows WHERE semantic_search_text IS NOT NULL AND semantic_search_text != '' (drop the
 text_embedding_id IS NULL predicate). When not force, filter out row[pk_col] in existing in Python.
 - Reuse existing _STORE_CONFIG, _BATCH_SIZE, batch embed loop, and metadata building (ts_epoch from
 start_ts_utc/timestamp_utc/coverage_start_utc; lat/lon from latitude/longitude).
 - Keep writing text_embedding_id after upsert as a best-effort breadcrumb (preserves the verify query);
 do not read it for membership.
 - --force bypasses the membership filter (re-embeds all eligible; upsert is idempotent on id).
 - main() delegates to this function — only behavioral change is the corrected gate (the intended fix).

 4. src/ltm_query/cli.py — auto-backfill on startup (honors "switching just works")

     - --force bypasses the membership filter (re-embeds all eligible; upsert is idempotent on id).
     - main() delegates to this function — only behavioral change is the corrected gate (the intended fix).

     4. src/ltm_query/cli.py — auto-backfill on startup (honors "switching just works")

     Inside the existing if embedding_client is not None: block (after vector_index is built, ~line 263),
     guarded by the new flag:
     if config.embed_auto_backfill:
         try:
             from src.embed_index import reconcile_model_index
             stats = reconcile_model_index(conn, embedding_client, vector_index)
             total = sum(idx for _t, idx in stats.values())
             if total:
                 logger.info("Auto-backfill: embedded %d missing rows via %s/%s",
                             total, embedding_client.provider, embedding_client.model)
         except Exception as exc:
             logger.warning("Auto-backfill skipped (querying continues with LIKE fallback): %s", exc)
     Incremental → cheap no-op once built (3 get + 3 SELECTs). Own try/except so a provider outage never blocks
     the REPL. The INFO log makes the one-time OpenAI cost visible (no silent spend).

     5. src/config.py — new flag

     - Dataclass field embed_auto_backfill: bool (in the vector block after embedding_timeout_sec).
     - from_env: embed_auto_backfill=parse_bool_env(os.getenv("EMBED_AUTO_BACKFILL", "true")).
     - Default true: fulfills the easy-switching goal; incremental design keeps it free once built; the INFO
     log surfaces first-run cost. No validate() change.

     6. Docs

     - memory_log/.env.example: add EMBED_AUTO_BACKFILL=true with a comment explaining auto per-model backfill.
     - CLAUDE.md: update the embed_index section — embed_index is now per-model incremental (heals model
     switches automatically); --force is full rebuild; note auto-backfill on LTM-query startup.

     Critical files

     - src/vector_index.py (count / existing_ids)
     - src/ltm_query/retrieval.py (_semantic_candidate_ids empty-collection → None)
     - src/embed_index.py (extract reconcile_model_index, fix incremental gate)
     - src/ltm_query/cli.py (startup auto-backfill, ~line 263)
     - src/config.py (embed_auto_backfill)
     - memory_log/.env.example, CLAUDE.md

     Out of scope (noted, not blocking)

     - Latent: location_filter excludes Chroma docs lacking lat/lon metadata (a where on a missing key
     drops the doc). Current data has lat/lon populated, so it's not the reported bug, but it will bite mixed
     data once location filters are used. Track as a follow-up (e.g. apply location filtering only at the SQL layer).
     - Latent: daily_summaries with null coverage_start_utc → ts_epoch=0.0, excluded by any time_range.
     Minor; not the current cause.

     Verification

     1. cd memory_log && uv run python -m src.embed_index → should now report it embedded 45 rows into the
     text-embedding-3-small collections (previously skipped them all).
     2. Confirm collections populated:
     sqlite3 outputs/chroma/chroma.sqlite3 "SELECT name, dimension FROM collections;" — the
     *__text-embedding-3-small rows should now show dimension 1536.
     3. uv run python -m src.ltm_query --no-grounding → ask "What did I ask about the camera?" → should return
     non-empty results. Check the log line shows method=vector candidates>0.
     4. Inspect telemetry:
     sqlite3 outputs/long_term_query_logs.sqlite "SELECT retrieval_trace_json FROM long_term_query_logs ORDER BY
     timestamp_utc DESC LIMIT 1;" | jq .
     — candidate_count should be > 0 and final_count > 0 for the semantic stores.
     5. Switch-back test: set EMBEDDING_PROVIDER back to ollama (comment the openai lines), rerun the query —
     should work instantly with 0 rows backfilled (nomic collections already populated), proving coexistence.
     6. Empty-collection fallback test: with a never-indexed model configured and EMBED_AUTO_BACKFILL=false,
     a semantic query should fall back to LIKE (log: "falling back to LIKE") and still return keyword matches
     instead of silent zero.