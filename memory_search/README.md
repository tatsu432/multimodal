# memory_search — Step 3: Keyword and time-based memory search

Phase 3 of the wearable multimodal AI assistant. Ask questions about past visual memories stored by Step 2 (`memory_log`).

## What this step does

```text
memory records in memories.jsonl
  → user asks a memory question
  → system parses time/object/scene/keyword filters
  → system scores and retrieves matching records
  → system returns evidence with timestamps and image paths
  → optional LLM summarization (off by default)
```

No vector DB, embeddings, or web UI — just deterministic search over structured JSONL.

## Why keyword/time search before a vector DB

- **Proves the JSONL contract works** — many questions are already answerable from structured fields (`objects`, `scene_type`, `timestamp`, `text_visible`).
- **Fully debuggable** — you can trace exactly why a record matched.
- **No extra services** — no Chroma, embedding API, or index to operate.
- **Fast iteration** — tune rules and scoring without re-indexing.

Semantic search (ChromaDB / embeddings) comes in a later step when paraphrases and fuzzy recall matter.

## Setup with uv

```bash
cd memory_search
cp .env.example .env
# Edit .env if your memories.jsonl lives elsewhere
uv sync
```

Point `MEMORY_JSONL_PATH` at Step 2 output (default: `../memory_log/outputs/memories.jsonl`).

## Configuration

| Variable | Description |
|----------|-------------|
| `MEMORY_JSONL_PATH` | Path to Step 2 JSONL file (default `../memory_log/outputs/memories.jsonl`) |
| `DEFAULT_LIMIT` | Max results per query (default `10`) |
| `DEFAULT_SHOULD_STORE_ONLY` | Only search `should_store=true` records (default `true`) |
| `TIMEZONE` | Timezone for "today", "recently", etc. (default `Asia/Tokyo`) |
| `USE_LLM_ANSWERER` | Optional LLM summarization (default `false`) |
| `LLM_PROVIDER` | `openai` or `ollama` (when LLM answerer enabled) |
| `LLM_MODEL` | Model for LLM answerer (required when LLM answerer enabled) |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` and LLM answerer on |
| `OLLAMA_BASE_URL` | Ollama URL when `LLM_PROVIDER=ollama` |

**Live camera capture** is not in this step — run [`memory_log`](../memory_log/README.md) with `FRAME_SOURCE_TYPE=camera` to record memories from Tapo / phone cameras. This module only searches existing JSONL.

## Run

```bash
cd memory_search
uv run python -m src.main
```

Example session:

```text
Loaded 2 memory records from ../memory_log/outputs/memories.jsonl

Ask a memory question, or type 'q' to quit:
> Did I see a laptop today?

Yes. I found 2 relevant memories.

1. 2026-06-05 07:47:18 — A person sitting at a desk using a laptop ...
   Objects: person, desk, laptop, chair, curtains, cables, box, speaker, clock
   Scene: indoor_workspace
   Privacy: medium
   Text: tapo, AI
   Image: ../memory_log/outputs/frames/2026-06-05T07-47-18.810.jpg
```

## Supported question examples

| Question | What it matches |
|----------|-----------------|
| `What did I see recently?` | Last 30 minutes, recent-bias scoring |
| `What did I see in the last 10 minutes?` | Time window filter |
| `What was on my desk 5 minutes ago?` | Time point ±2 min + object `desk` |
| `Did I see a laptop today?` | Today + object `laptop` |
| `Show me memories involving a person.` | `people_count > 0` |
| `What text did I see?` | Records with non-empty `text_visible` |
| `What did I see at home desk today?` | Location + object + today |
| `Show me high privacy risk memories.` | `privacy_risk=high` filter |

## How retrieval works

1. **Parse** — rule-based extraction of time range, object/scene/location filters, privacy level, keywords.
2. **Filter** — hard filters on `should_store`, time, people, text, privacy, objects, scene, location.
3. **Score** — relevance points for object (+5), scene (+4), location (+4), keyword in summary (+3), etc.
4. **Rank** — score descending, then timestamp descending.
5. **Answer** — template formatter with evidence; optional LLM wraps the same records.

If no records match, the answer says so clearly — nothing is hallucinated.

## Known limitations

- **Rule-based parser** — misses paraphrases ("my computer" won't match `laptop` unless the word appears).
- **No semantic similarity** — "notebook" won't find "laptop" unless both appear in structured fields.
- **Object names must be in JSONL** — search relies on Step 2 VLM extraction quality.
- **Memories loaded at startup** — restart to pick up newly written records.
- **Location search** — only works when `location.label` was set during capture.

## Next step

ChromaDB / embedding-based semantic memory search for paraphrase-tolerant recall.

## Relation to Step 2

[`memory_log`](../memory_log/) writes `outputs/memories.jsonl`. This step reads those records and answers retrospective questions without live capture or VLM calls (unless optional LLM answerer is enabled).
