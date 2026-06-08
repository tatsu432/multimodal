# Project stages

Roadmap for the wearable multimodal AI assistant. This file tracks **what is implemented** and **what is planned next**. For run instructions, see each package README.

---

## Big picture architecture

Eventually you want this:

```text
Wearable / Tapo / phone camera
        ↓
Streaming ingestion (RTSP / WebRTC via MediaMTX)
        ↓
Frame sampler + ring buffer
        ↓
VLM inference (OpenAI or Ollama)
        ↓
Memory layers (passive observation → promoted events → daily summaries)
        ↓
Long-term memory query (retriever + answer generator)
        ↓
Evaluation + service separation + UI
```

**What works today:**

```text
Tapo RTSP / phone WebRTC (MediaMTX) / webcam / video file
        ↓
capture/ (shared stream config + ring buffer)
        ↓
VLM inference (OpenAI or Ollama)
        ↓
[vlm_smoke]        live Q&A only
[memory_log]       Q&A + JSONL + SQLite (active_query_memories + promoted_events + frames)
[passive_observer] periodic background logging → passive_observations table
[run_all]          wearable "on" switch: live Q&A + passive observer in one process
                   (shared camera · location server · lock-guarded DB writer)
        ↓
[daily_summary]    LLM-compressed daily records → daily_summaries table
        ↓
[ltm_query]        query planner → retrieval → evidence pack → grounded answer
                   optional visual grounding for "this/here/current scene" queries
```

---

## Current status

| Package / module | Status | README |
| ---------------- | ------ | ------ |
| `camera_test/` | **Done** — stream validation harness | [camera_test/README.md](camera_test/README.md) |
| `capture/` | **Done** — shared camera ingestion | (root package, see below) |
| `providers/` | **Done** — shared Ollama client | (root package, see below) |
| `vlm_smoke/` | **Done** — live visual QA | [vlm_smoke/README.md](vlm_smoke/README.md) |
| `memory_log/` | **Done** — question-driven JSONL memories + SQLite memory DB | [memory_log/README.md](memory_log/README.md) |
| Passive observation | **Done** — `src/passive_observer.py` periodic background logging | [memory_log/README.md](memory_log/README.md) |
| Promoted events | **Done** — auto-created on active query; SQLite `promoted_events` table | [memory_log/README.md](memory_log/README.md) |
| Active query memories | **Done** — SQLite `active_query_memories` table, linked to events | [memory_log/README.md](memory_log/README.md) |
| Daily summaries | **Done** — `src/daily_summary.py` LLM-compressed daily records | [memory_log/README.md](memory_log/README.md) |
| Long-term memory query | **Done** — `src/ltm_query/` deterministic retrieval + grounded answering | [memory_log/README.md](memory_log/README.md) |
| Unified runner | **Done** — `src/run_all.py` wearable "on" switch (live QA + passive, shared resources) | [memory_log/README.md](memory_log/README.md) |
| Dashboard UI | **Done** — `src/dashboard/` browser UI: live MJPEG frame + streaming Live-QA chat + LTM query panels | [memory_log/README.md](memory_log/README.md) |
| Eval harness | **In Progress** — `memory_log/evals/` end-to-end eval for Live QA + LTM QA | [memory_log/README.md](memory_log/README.md) |
| Eval / API | **Planned** — service separation after eval harness | (this file, Future work) |

---

## Shared infrastructure

Not a separate phase — shared code used by `vlm_smoke` and `memory_log`.

| Module | Role |
| ------ | ---- |
| [`capture/`](capture/) | Camera presets (`tapo-rtsp`, `tapo-webrtc`, `phone-webrtc`), RTSP tuning, threaded ring buffer ([`camera_frame_source.py`](capture/camera_frame_source.py), [`stream_config.py`](capture/stream_config.py)) |
| [`providers/ollama.py`](providers/ollama.py) | Shared local Ollama HTTP client |
| Root [`pyproject.toml`](pyproject.toml) | Publishes `capture` + `providers` as the `multimodal` package |

`camera_test/stream_config.py` re-exports `capture.stream_config`. MediaMTX configs and TLS certs live under [`camera_test/`](camera_test/) and are shared across packages.

---

# Phase 0 — `camera_test`: validate camera streams

**Status: implemented**

Goal: confirm live video ingestion works before using `vlm_smoke` or `memory_log`.

### Supported sources

| `CAMERA_SOURCE` | Camera | Path |
| --------------- | ------ | ---- |
| `tapo-rtsp` | Tapo IP camera | RTSP direct → OpenCV |
| `tapo-webrtc` | Tapo IP camera | RTSP → MediaMTX → WHEP / RTSP relay |
| `phone-webrtc` | Smartphone | WebRTC publish → MediaMTX → RTSP relay |

### Suggested workflow

```text
camera-preview  → confirm source works
camera-sample   → confirm frames save correctly
camera-vlm      → test VLM on live video
```

For production-style flows (config, logging, memory), use `vlm_smoke/` and `memory_log/`.

### Folder layout

```text
camera_test/
├── preview_stream.py, frame_sample.py, live_vlm_qa.py
├── stream_config.py          # re-exports capture/
├── whep_client.py, whep_worker.py, whep_probe.py
├── mediamtx-*.example.yml    # copy to local gitignored yml
└── phone_location.html       # GPS sidecar page for memory_log
```

MediaMTX setup, phone TLS (mkcert), and troubleshooting: see [camera_test/README.md](camera_test/README.md).

### Success criteria

- `uv run camera-preview` shows a live window (or headless read succeeds)
- `uv run camera-sample` saves JPEGs under `sampled_frames/`
- `uv run camera-vlm` answers text questions in a REPL

---

# Phase 1 — `vlm_smoke`: live visual QA

**Status: implemented**

Goal: stable, reproducible live visual question-answering over a frame stream.

### What it does

- Background thread samples frames into a ring buffer (`FRAME_BUFFER_SIZE`, `CAPTURE_SAMPLE_INTERVAL_SEC`)
- Terminal REPL accepts text questions
- Sends the latest N frames (`NUM_FRAMES_PER_QUERY`) to OpenAI or Ollama vision models
- Logs capture health, VLM latency, and optionally saves queried frames under `outputs/sampled_frames/`

### Frame sources

Set `FRAME_SOURCE_TYPE` in `.env`:

| Type | Use |
| ---- | --- |
| `camera` | Tapo RTSP, Tapo/MediaMTX relay, or phone WebRTC — same presets as `camera_test` |
| `webcam` | Local webcam (`WEBCAM_INDEX`) |
| `video` | Looping video file (`VIDEO_PATH`) for smoke tests without a camera |

Uses shared [`capture/`](capture/) for camera presets. Direct Tapo RTSP (`tapo-rtsp`) is the lowest-latency option on LAN.

### Folder layout

```text
vlm_smoke/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   ├── main.py          # REPL entrypoint
│   ├── config.py
│   ├── frame_source.py  # camera / webcam / video
│   ├── vlm_client.py
│   └── utils.py
└── outputs/
    └── sampled_frames/
```

Full setup (MediaMTX, phone publish, Ollama): [vlm_smoke/README.md](vlm_smoke/README.md).

### Success criteria

```bash
cd vlm_smoke
uv run python -m src.main
```

Example questions:

```text
What do you see?
Is there a person?
What object is closest to the camera?
What text is visible?
```

### Out of scope (Phase 1)

- Memory logging or episodic memory
- Vector DB / semantic search
- FastAPI backend or Streamlit UI
- Evaluation harness
- Efficient VLM research

`camera_test/live_vlm_qa.py` is the lightweight predecessor; `vlm_smoke` supersedes it for ongoing work.

---

# Phase 2 — `memory_log`: question-driven visual memory

**Status: implemented**

Goal: persist visual memories as JSONL when you ask questions about the live view.

### What it does

```text
frame stream → background frame buffer
user asks question
  → VLM answers your question (recent frames)
  → append memory to outputs/memories.jsonl (query, answer, frames, location)
```

**Important:** memories are written **only when you ask a question**. There is no timer-based logging. Each question uses **one VLM call** — no second structured-analysis pass.

### Memory record (current format)

```json
{
  "memory_id": "2026-06-06T16-57-09.525",
  "timestamp": "2026-06-06T16:57:09.525+09:00",
  "user_question": "What do you think about his facial expression? how does it change?",
  "model_answer": "He starts with an exaggerated open-mouth expression...",
  "frame_paths": [
    "outputs/frames/2026-06-06T16-57-09.525_f01.jpg",
    "outputs/frames/2026-06-06T16-57-09.525_f02.jpg"
  ],
  "frame_timestamps": [
    "2026-06-06T16:57:00.434+09:00",
    "2026-06-06T16:57:01.634+09:00"
  ],
  "location": {
    "label": "phone location (Yoyogi, Shibuya, Tokyo, Japan)",
    "lat": 35.68499131058371,
    "lon": 139.6963945929086,
    "source": "phone_gps",
    "full_address": "代々木二丁目, 代々木, 渋谷区, 東京都, 151-0053, 日本",
    "city": "渋谷区",
    "prefecture": "東京都",
    "postal_code": "151-0053",
    "country": "日本",
    "geocode_provider": "nominatim",
    "geocoded_at": "2026-06-06T18:48:08.342+09:00"
  },
  "camera_source": "phone-webrtc"
}
```

When `NUM_FRAMES_PER_QUERY=1`, a single `{memory_id}.jpg` is saved. With N > 1, frames are `{memory_id}_f01.jpg`, `_f02.jpg`, etc.

**Legacy records:** older JSONL lines may still contain `summary`, `objects`, `scene_type`, `privacy_risk` from an earlier two-call format. The current writer stores `model_answer` and `frame_paths` instead.

### Location metadata (built in)

Video streams do not carry GPS. Location is resolved without a VLM call:

| Source | How |
| ------ | --- |
| Tapo (`tapo-rtsp`, `tapo-webrtc`) | `TAPO_LOCATION_LABEL` and optional lat/lon in config |
| Phone (`phone-webrtc`) | HTTPS GPS sidecar (`phone_location.html` + `LOCATION_SERVER_*`), else `PHONE_LOCATION_*` fallback |
| Webcam / video | Global `LOCATION_*` env vars |

Optional reverse geocoding (Nominatim + SQLite cache) fills `full_address`, `city`, `prefecture`, etc. at write time. See [memory_log/README.md](memory_log/README.md).

### Folder layout

```text
memory_log/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   ├── main.py
│   ├── memory_writer.py
│   ├── schema.py
│   ├── vlm_client.py
│   ├── frame_source.py
│   ├── location.py
│   ├── location_server.py
│   ├── geocode_client.py
│   └── geocode_cache.py
└── outputs/
    ├── frames/
    ├── memories.jsonl
    └── geocode_cache.sqlite
```

### Success criteria

```bash
cd memory_log
uv run python -m src.main
```

After asking a few questions:

```text
- saved frame images under outputs/frames/
- valid JSONL lines in outputs/memories.jsonl
- location populated per source (config, phone_gps, or geocoded)
- no crash when stream temporarily fails (reconnect handled by capture/)
```

Inspect:

```bash
tail -n 1 outputs/memories.jsonl | jq .
```

### Relation to Phase 1

`vlm_smoke` is interactive QA only. `memory_log` adds persistent JSONL memories **when you ask**, using the same threaded capture model and VLM providers.

---

# Memory layers + LTM query (implemented)

All four memory layers and the long-term query system are now implemented inside `memory_log/`.

### 1. Passive observation memory — `src/passive_observer.py`

Background logging every `PASSIVE_OBSERVATION_INTERVAL_SEC` (default 30s). No VLM. Writes to `passive_observations` SQLite table: timestamp, location, optional frame path + thumbnail, optional pHash.

```bash
cd memory_log && uv run python -m src.passive_observer
```

### 2. Promoted event memory — `src/db_writer.py`

Auto-created whenever the user asks a question (active query path). `source_type='active_query'`, `promotion_reason='user_asked_question'`. The `scene_summary` and `semantic_search_text` are derived from `model_answer` (one-call constraint preserved; marked with `extra_json={"summary_from":"model_answer_fallback"}`).

### 3. Daily summary — `src/daily_summary.py`

LLM-compressed daily records. Input: passive observation timeline + promoted events + active queries for the day. Output: one `daily_summaries` row with structured JSON.

```bash
cd memory_log && uv run python -m src.daily_summary --date 2026-06-06
```

### 4. Long-term memory query — `src/ltm_query/`

Deterministic pipeline:

```text
user query
  → QueryPlanner (LLM → structured RetrievalPlan JSON)
  → optional VisualGrounder (VLM on current frames if "this/here" detected)
  → MemoryRetriever (SQL queries on SQLite: time/location + LIKE keyword search)
  → build_evidence_pack (aggregates passive timeline, events, Q&A, frames)
  → one expansion step if visual_recall intent has no events
  → AnswerGenerator (text-only LLM with evidence context)
```

```bash
cd memory_log && uv run python -m src.ltm_query
cd memory_log && uv run python -m src.ltm_query --no-grounding
```

Each query is logged to `outputs/long_term_query_logs.sqlite` (a **separate** file, never
touched by retrieval): plan JSON + raw LLM response, per-store retrieval trace
(method/candidates/SQL/final-count/drop note), exact answer-generator prompt, answer,
per-stage latency, error. `extra_json.stores_selected_but_empty` flags stores where vector
search found candidates that were eliminated by SQL time/location filters.
Disable with `LTM_QUERY_LOG_ENABLED=false` in `.env`.

Example queries:
- "Where was I yesterday?"
- "What did I ask about the camera?"
- "What did I see near this location?"
- "What was here yesterday?" (requires visual grounding + live camera)

### Current limitations

- `scene_summary` for promoted events uses `model_answer` as a weak proxy (no dedicated VLM call).
- Visual grounding requires a live camera connection for "this/here" queries.
- Passive observation pHash requires `imagehash` (already in dependencies).
- Image embeddings (`image_embedding_id` columns) are reserved but not implemented (v1 scope: text only).

### 6. Vector / semantic search — `src/embeddings.py`, `src/vector_index.py`, `src/embed_index.py`

**Status: implemented**

Real semantic similarity search via **ChromaDB** with a dual embedding provider:
- **Ollama** (default, no API key): `nomic-embed-text`, local `/api/embed`
- **OpenAI** (optional): `text-embedding-3-small`

Collections are **model-namespaced** (`<store>__<model_slug>`) inside `outputs/chroma/`, so
switching providers never collides — a reindex just builds new collections.
SQLite remains the source of truth; Chroma is the ANN + metadata-filter engine.

```bash
# Pull the default embedding model (Ollama path)
ollama pull nomic-embed-text

# Backfill existing rows into ChromaDB
cd memory_log && uv run python -m src.embed_index          # new rows only
uv run python -m src.embed_index --force                   # re-embed all
uv run python -m src.embed_index --store promoted_events   # one store
```

Graceful fallback: `VECTOR_SEARCH_ENABLED=false` (or Ollama down) → queries revert to
SQLite `LIKE` keyword search with no error.

### 7. Later

After core memory layers exist:

- **Evaluation suite** (mostly done) — latency, retrieval quality, answer correctness, hallucination rate
- **Service separation** — API workers, ingestion workers, query service
- **memory refinement** - Promoted Event memory is only based on active query memory, so we should first include some rule-based logic 
- **Refine eval** Use a small set of public benchmarks
- **goal alignment** We have to make sure whether we should set some goals first for each session
- **instance related memory** We might have to deal with the memory about the instance, such as the identity of the person, because the current system cannot distinguish the actual instance of the object, such as you, from the other person Currently, the system is locally hosted, so we cannot use the camera outside of the local network. It would be better to separate services and expose the endpoint correctly
- **video streaming improvement** For smartphones, the video streaming has a bit of lag, and we have to figure out how to ensure truly real-time, stable streaming
- **edge device support** Start using the NVIDIA edge device 
- **efficient inference** Use a vLLM or a sort of more efficient vision-language model serving, but this should require GPUs-
- **Reserach on long term memory or efficient infernece** Research on efficient inference or long-term memory-related work
- **Voide** Enable voice streaming query
---

## Recommended order (current)

```text
0. camera_test      — validate streams
1. vlm_smoke        — live visual QA
2. memory_log       — question-driven JSONL + SQLite active query memories
3. passive_observer — background location/frame logging
4. ltm_query        — long-term memory query CLI
5. run_all          — combined wearable entry point (QA + passive in one process)
6. (TBD)           — eval, services, and more
```
