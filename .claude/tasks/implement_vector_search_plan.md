# Task: Implement vector (semantic) search for LTM retrieval

## Goal
Replace SQLite `LIKE` keyword matching in `memory_log/src/ltm_query/retrieval.py` with real
vector similarity search using **ChromaDB** (PersistentClient, bring-your-own-embeddings) and
a dual-provider embedding abstraction (Ollama primary/default, OpenAI optional). Graceful LIKE
fallback when vector is disabled/unavailable.

## Implementation order

### 1. `.claude/tasks/implement_vector_search_plan.md` — this file ✅

### 2. `memory_log/pyproject.toml` — add `chromadb>=0.5`

### 3. `providers/ollama.py` — add `embeddings(...)` function
- `POST /api/embed` → `{"embeddings": [[...], ...]}`
- Keyword-only args matching `chat()` style

### 4. `memory_log/src/embeddings.py` (new) — provider abstraction
- `EmbeddingClient` ABC: `embed(texts) → list[list[float]]`, `model`, `dim`, `provider`
- `OllamaEmbeddingClient` (default, nomic-embed-text, dim=768)
- `OpenAIEmbeddingClient` (text-embedding-3-small, dim=1536)
- `create_embedding_client(config) → EmbeddingClient | None` (None = graceful disable)

### 5. `memory_log/src/vector_index.py` (new) — ChromaDB wrapper + write-time indexer
- `ChromaVectorIndex(chroma_path, model_slug)` — lazy PersistentClient, per-store collections
  namespaced as `<store>__<model_slug>`, cosine space
- `upsert(owner_table, owner_id, vector, metadata)` and
  `search(owner_table, query_vector, top_k, where=None) → list[(id, distance)]`
- `build_where(time_range, location_filter, owner_table)` — time+location bbox for Chroma where
  (daily_summaries: date-epoch only, no lat/lon)
- `MemoryIndexer(embedding_client, vector_index)` — `index(...)` non-fatal, `index_pair(...)`
- `create_memory_indexer(config) → MemoryIndexer | None`

### 6. `memory_log/src/config.py` — new fields
- `vector_search_enabled`, `embedding_provider`, `embedding_model`,
  `embed_on_write`, `chroma_path`, `embedding_timeout_sec`

### 7. `memory_log/src/db_writer.py` — indexer hook
- `SQLiteWriter.__init__(conn, project_root, indexer=None)` (backward compatible)
- Call `indexer.index_pair(...)` after `write_active_query_with_event` commits
- Call `indexer.index("daily_summaries", ...)` after `write_daily_summary` commits
- Embedding outside the lock/transaction (network I/O)

### 8. `memory_log/src/ltm_query/retrieval.py` — vector ranking
- `MemoryRetriever.__init__(conn, config, embedding_client=None, vector_index=None)`
- `_semantic_candidate_ids(owner_table, plan, limit)` → ranked ids or None
- In each `_query_*` method: use ids if returned, else fall through to existing LIKE
- Memoize query embedding per `retrieve()` call

### 9. `memory_log/src/ltm_query/cli.py` — wire in embedding + vector clients

### 10. `memory_log/src/main.py`, `src/run_all.py`, `src/daily_summary.py` — pass indexer to SQLiteWriter

### 11. `memory_log/src/embed_index.py` (new) — backfill/reindex CLI
- `python -m src.embed_index [--store <name>] [--force]`
- Idempotent; updates `text_embedding_id` column on success

### 12. Docs + config
- `.env.example` — `# --- Vector / semantic search ---` block
- `.gitignore` — `memory_log/outputs/chroma/`
- `STAGES.md`, `memory_log/README.md`, `CLAUDE.md`

## Key integration notes
- PKs: `promoted_events.event_id`, `active_query_memories.active_query_id`, `daily_summaries.summary_id`
- `write_active_query_with_event` returns `(event_id, active_query_id)` — both get indexed
- `daily_summaries` has no lat/lon → skip bbox in Chroma where
- SQLiteWriter construction sites: main.py:285, run_all.py:104, daily_summary.py:237, passive_observer.py:224 (passive untouched)
- `text_embedding_id` stores the ChromaDB doc id (= the row PK)
