# vector_memory — Step 4: Hybrid semantic + structured retrieval

Phase 4 of the wearable multimodal AI assistant. Ask questions about past visual memories using **structured filters**, **keyword scoring**, and **Chroma vector similarity** together.

## What this step does

```text
memories.jsonl
  → build / update Chroma index (local embeddings)
  → user asks a memory question
  → parse time/object/scene/location filters + semantic query text
  → vector search + structured filters + metadata scoring
  → grounded answer with timestamps, evidence, retrieval hints
  → optional LLM summarization (off by default)
```

Step 3 (`memory_search`) rule-based retrieval is **not replaced** — its filter and scoring logic is reused inside the hybrid pipeline.

## Why vector search after keyword/time search

| Approach | Strength | Limitation |
|----------|----------|------------|
| Keyword / metadata (Step 3) | Exact, debuggable matches on `objects`, `scene_type`, time | Misses paraphrases ("food place" vs "restaurant") |
| Semantic / vector (Step 4) | Similar meaning, fuzzy recall | Needs index build; less exact for strict filters |

Hybrid retrieval uses **both**: Chroma proposes semantic candidates; structured filters and keyword scores rerank and explain matches.

## Setup with uv

```bash
cd vector_memory
cp .env.example .env
# Edit .env if memories.jsonl or embedding settings differ
uv sync
```

Point `MEMORY_JSONL_PATH` at Step 2 output (default: `../memory_log/outputs/memories.jsonl`).

First run downloads the default `sentence-transformers/all-MiniLM-L6-v2` weights (~80MB).

## Configuration

| Variable | Description |
|----------|-------------|
| `MEMORY_JSONL_PATH` | Path to Step 2 JSONL (default `../memory_log/outputs/memories.jsonl`) |
| `CHROMA_PERSIST_DIR` | Chroma storage directory (default `outputs/chroma`) |
| `CHROMA_COLLECTION_NAME` | Collection name (default `visual_memories`) |
| `EMBEDDING_PROVIDER` | `sentence_transformers` (default) or `openai` |
| `EMBEDDING_MODEL` | Local model id (default `sentence-transformers/all-MiniLM-L6-v2`) |
| `OPENAI_API_KEY` | Required when `EMBEDDING_PROVIDER=openai` or LLM answerer on |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model (default `text-embedding-3-small`) |
| `DEFAULT_LIMIT` | Max results per query (default `10`) |
| `DEFAULT_SHOULD_STORE_ONLY` | Index/search only `should_store=true` (default `true`) |
| `TIMEZONE` | Timezone for relative time phrases (default `Asia/Tokyo`) |
| `USE_LLM_ANSWERER` | Optional LLM summary (default `false`) |
| `LLM_MODEL` | Model when LLM answerer enabled |
| `REBUILD_INDEX` | `true` clears and rebuilds Chroma collection (default `false`) |

## Build or rebuild the Chroma index

**Incremental (default):** on startup, only new `memory_id` values are embedded and upserted.

**Full rebuild:**

```bash
# In .env: REBUILD_INDEX=true
uv run python -m src.main
# Set REBUILD_INDEX=false afterward for incremental updates
```

Or one-shot:

```bash
REBUILD_INDEX=true uv run python -m src.main
```

## Run

```bash
cd vector_memory
uv run python -m src.main
```

Example session:

```text
Loaded 2 memory records.
Indexed 2 memory records into Chroma collection visual_memories.

Ask a memory question, or type 'q' to quit:
> What memories look similar to a workspace?

Found 2 relevant memories.

1. 2026-06-05 07:47:18 — A person sitting at a desk using a laptop ...
   Objects: person, desk, laptop, ...
   Scene: indoor_workspace
   Privacy: medium
   Image: ../memory_log/outputs/frames/...
   Retrieval: semantic match + scene type match
```

## Example questions to test

| Question | What it exercises |
|----------|-------------------|
| `When did I see something like a restaurant?` | Semantic query + time phrasing |
| `Did I pass by any signs?` | Semantic + keyword `signs` |
| `Where did I see a red object?` | Location/object filters + semantic |
| `Did I see something related to food?` | Semantic paraphrase |
| `What memories look similar to a workspace?` | Scene + semantic |

With only workspace memories in JSONL, restaurant/food/sign questions may correctly return few or no matches — the answerer will say so rather than invent evidence.

## Scoring (transparent)

- Vector: `10 * (1 - cosine_distance)`, clamped 0–10
- Metadata: object +5, scene +4, location +4, keyword in summary +3, visible text +3, etc.
- Final sort: `vector_score + metadata_score`, then timestamp

Debug details log at `INFO` for index; per-hit scores at `DEBUG` (`logging` level).

## Limitations

- English-centric question parser
- No GPS / backend API (location labels often null)
- Chroma index does not auto-update when an existing record’s summary changes (use `REBUILD_INDEX=true`)
- `sentence-transformers` adds PyTorch dependency and first-run download time
- Very small corpora: vector candidate pool may return all records

## Next step

Phase 5 — `location_memory`: richer location metadata and location-aware questions (see repo `STAGES.md`).
