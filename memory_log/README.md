# memory_log — Step 2: Question-driven visual memory

Phase 2 of the wearable multimodal AI assistant. Ask questions about the live view; each question triggers a **text answer** and a **JSONL memory record** tied to that moment.

## What this step does

```text
frame stream → background frame buffer
user asks question
  → VLM answers your question (recent frames)
  → append memory to outputs/memories.jsonl (query, answer, frames, location)
```

Nothing is written to JSONL on a timer. If you do not ask a question, no new memories are created.

Each question uses **one VLM call** only. There is no second structured-analysis call after the answer.

## Why JSONL + SQLite + ChromaDB

- **Easy to debug** — `tail -f`, `jq`, any text editor for JSONL; standard SQLite tools for the memory DB.
- **Local-first embeddings** — Ollama `nomic-embed-text` for semantic search with no API key required; OpenAI optional.
- **Stable contract** — each line is self-contained, including `user_question` and `model_answer` for later search.
- **Fail-safe progress** — lines are flushed after each write; ChromaDB indexing is non-fatal (LIKE fallback if unavailable).

## Setup with uv

```bash
cd memory_log
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and frame source
uv sync
```

**Migrating from the old timer-based config:** remove `FRAME_SAMPLE_INTERVAL_SECONDS` from `.env` and add `FRAME_BUFFER_SIZE`, `CAPTURE_SAMPLE_INTERVAL_SEC`, and `NUM_FRAMES_PER_QUERY` (see `.env.example`).

**Migrating from the old two-call memory format:** older JSONL lines with `summary` / `objects` / `privacy_risk` still load in `memory_search` and `vector_memory`. New lines store `model_answer` and `frame_paths` instead.

## Configuration


| Variable                                         | Description                                                                                 |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `FRAME_SOURCE_TYPE`                              | `camera`, `webcam`, or `video`                                                              |
| `CAMERA_SOURCE`                                  | When `camera`: `tapo-rtsp`, `tapo-webrtc`, `phone-webrtc`                                   |
| `RTSP_URL`, `PHONE_STREAM_URL`, `RTSP_`*         | Same as `[camera_test](../camera_test/README.md)`                                           |
| `WEBCAM_INDEX`                                   | Webcam device index (default `0`)                                                           |
| `VIDEO_PATH`                                     | Required when `FRAME_SOURCE_TYPE=video`                                                     |
| `VLM_PROVIDER`                                   | `openai` or `ollama`                                                                        |
| `VLM_MODEL`                                      | Vision model (e.g. `gpt-5.5`, `llava`)                                                      |
| `OPENAI_API_KEY`                                 | Required when `VLM_PROVIDER=openai`                                                         |
| `OLLAMA_BASE_URL`                                | Ollama URL when `VLM_PROVIDER=ollama`                                                       |
| `FRAME_BUFFER_SIZE`                              | Ring buffer size for recent frames (default `8`)                                            |
| `CAPTURE_SAMPLE_INTERVAL_SEC`                    | How often the background thread adds frames (default `1.0`) — **not** memory write interval |
| `NUM_FRAMES_PER_QUERY`                           | Frames sent to Q&A and saved per question (default `1`)                                     |
| `OUTPUT_FRAME_DIR`                               | Saved frame images (default `outputs/frames`)                                               |
| `MEMORY_JSONL_PATH`                              | JSONL file (default `outputs/memories.jsonl`)                                               |
| `SAVE_FRAMES`                                    | Save JPEGs for each query frame (default `true`)                                            |
| `MAX_RUNTIME_SECONDS`                            | Optional; unset = run until Ctrl+C                                                          |
| `LOCATION_LABEL`, `LOCATION_LAT`, `LOCATION_LON` | Global location fallback                                                                    |
| `TAPO_LOCATION_`*                                | Fixed location for Tapo cameras                                                             |
| `PHONE_LOCATION_*`                               | Fallback when phone GPS sidecar is off or stale                                             |
| `LOCATION_SERVER_*`                              | Optional HTTPS sidecar for phone GPS (see below)                                            |
| `GEOCODE_*`, `NOMINATIM_BASE_URL`                | Reverse geocode lat/lon to address at write time (see below)                                |
| `VECTOR_SEARCH_ENABLED`                          | `true`/`false` — use ChromaDB semantic search (default `true`, falls back to LIKE)          |
| `EMBEDDING_PROVIDER`                             | `ollama` (default, no API key) or `openai`                                                  |
| `EMBEDDING_MODEL`                                | default: `nomic-embed-text` (ollama) / `text-embedding-3-small` (openai)                    |
| `EMBED_ON_WRITE`                                 | Embed new memories at write time (default `true`)                                           |
| `CHROMA_PATH`                                    | ChromaDB directory (default `outputs/chroma`)                                               |
| `EMBEDDING_TIMEOUT_SEC`                          | Embedding HTTP timeout in seconds (default `30`)                                            |


## Location metadata

Video streams do not carry GPS. Location is resolved without a VLM call:


| Source                                | How location is set                                                                                               |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Tapo** (`tapo-rtsp`, `tapo-webrtc`) | Configure `TAPO_LOCATION_LABEL` and optional `TAPO_LOCATION_LAT` / `TAPO_LOCATION_LON` (fixed camera position)    |
| **iPhone** (`phone-webrtc`)           | Fresh GPS from the location sidecar (`location.source=phone_gps`), else `PHONE_LOCATION_`* or global `LOCATION_*` |
| **Webcam / video**                    | Global `LOCATION_`* env vars                                                                                      |


### Phone GPS sidecar

When using `CAMERA_SOURCE=phone-webrtc`, enable the built-in HTTPS server:

```env
LOCATION_SERVER_ENABLED=true
LOCATION_SERVER_PORT=8765
LOCATION_SERVER_CERT=../camera_test/mediamtx-certs/server.crt
LOCATION_SERVER_KEY=../camera_test/mediamtx-certs/server.key
```

On the phone (same Wi‑Fi, while publishing to MediaMTX), open:

```text
https://YOUR_MAC_IP:8765/
such as https://192.168.11.51:8765/ 
```

Keep that page open so `memory_log` receives live lat/lon. See [`camera_test/README.md`](../camera_test/README.md#phone-gps-sidecar-for-memory_log) for cert setup.

### Reverse geocoding (address from lat/lon)

When a memory has **coordinates but no manual label**, `memory_log` can resolve a **full street address** and structured place fields at write time (not at search time). Results are cached in SQLite so repeated questions in the same area do not hit the network again.

```env
GEOCODE_ENABLED=true
GEOCODE_PROVIDER=nominatim
NOMINATIM_BASE_URL=https://nominatim.openstreetmap.org
GEOCODE_CACHE_PATH=outputs/geocode_cache.sqlite
GEOCODE_SKIP_IF_LABEL_SET=false
```

**When geocoding runs**

- Any source with lat/lon and no cached `full_address` → Nominatim reverse lookup → fills `full_address`, `city`, `prefecture`, `postal_code`, `country` (manual `*_LOCATION_LABEL` is kept as `label`)
- Set `GEOCODE_SKIP_IF_LABEL_SET=true` only if you already store a full address and want to avoid API calls
- Cache key rounds coordinates to ~11 m; walking in one neighborhood reuses one cached address

**Privacy:** full addresses make JSONL more sensitive (home/work identifiable). Disable with `GEOCODE_ENABLED=false`, or keep manual labels only. The public Nominatim service has a **1 request/second** limit — the client throttles and caches; for heavy use, self-host Nominatim.

**Search:** `memory_search` and `vector_memory` match location filters against `label`, `full_address`, `city`, `prefecture`, `postal_code`, and `country`.

## Camera sources (Tapo RTSP, MediaMTX, phone WebRTC)

Same presets as `[vlm_smoke](../vlm_smoke/README.md#camera-sources-tapo-rtsp-mediamtx-phone-webrtc)` and `[camera_test](../camera_test/README.md)`. Set `FRAME_SOURCE_TYPE=camera` in `memory_log/.env`.

MediaMTX YAML and certs: `[camera_test/](../camera_test/)` (`mediamtx-tapo.yml`, `mediamtx-phone.yml`).

### Tapo RTSP (recommended)

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=tapo-rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
RTSP_FLUSH_GRABS=8
TAPO_LOCATION_LABEL=home office
VLM_PROVIDER=openai
VLM_MODEL=gpt-5.5
OPENAI_API_KEY=sk-...
```

Test the same URL in VLC first. Use `stream2` for VLM (lower bandwidth).

### Tapo via MediaMTX

```bash
cd camera_test && mediamtx mediamtx-tapo.yml
```

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=tapo-webrtc
WEBRTC_URL=http://localhost:8889/tapo/whep
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
TAPO_LOCATION_LABEL=living room
```

Python reads `rtsp://127.0.0.1:8554/tapo` automatically (RTSP relay, not WHEP).

### iPhone via MediaMTX WebRTC

Requires HTTPS publish from the phone — see [camera_test README § Smartphone](../camera_test/README.md#3-smartphone-with-webrtc) and [Publish page settings](../camera_test/README.md#publish-page-settings-before-you-tap-publish).

```bash
cd camera_test && mediamtx mediamtx-phone.yml
# Phone: https://YOUR_MAC_IP:8889/phone/publish  (ipconfig getifaddr en0)
```

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=phone-webrtc
PHONE_STREAM_URL=rtsp://127.0.0.1:8554/phone
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
LOCATION_SERVER_ENABLED=true
LOCATION_SERVER_CERT=../camera_test/mediamtx-certs/server.crt
LOCATION_SERVER_KEY=../camera_test/mediamtx-certs/server.key
PHONE_LOCATION_LABEL=walking
```

### Ollama (local VLM, no API key)

```bash
ollama pull llava
```

```env
VLM_PROVIDER=ollama
VLM_MODEL=llava
OLLAMA_BASE_URL=http://localhost:11434
```

## Dashboard UI (recommended)

A browser-based dashboard with a live MJPEG camera frame, a streaming Live-QA chat
panel, and a Long-term Memory search panel (planner → retrieval trace → streaming answer):

```bash
cd memory_log
uv run python -m src.dashboard
# → open http://127.0.0.1:8800/ in your browser
```

Options:
```
--host 0.0.0.0   expose on the LAN (default: 127.0.0.1)
--port 9000      use a different port (default: 8800)
--no-grounding   disable live visual grounding for LTM queries (no camera needed for LTM)
```

Override host/port in `.env`:
```
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8800
```

## Run (CLI / terminal REPL)

```bash
cd memory_log
uv run python -m src.main
# CLI: --camera phone-webrtc --url rtsp://127.0.0.1:8554/phone
```

Wait a few seconds for the capture buffer to fill, then type a question. Type `q` to quit.

### Webcam

```bash
FRAME_SOURCE_TYPE=webcam
WEBCAM_INDEX=0
```

### Video file

```bash
FRAME_SOURCE_TYPE=video
VIDEO_PATH=/path/to/clip.mp4
```

## Example session

```text
Ask a question about the current view, or type 'q' to quit:
> What do you see?

Assistant: thinking...

Assistant: A desk with a laptop and cables near a window.

[2026-06-04T23:12:30+09:00] frames=1 location=home office source=config latency=2.10s
```

## Example memory record

```json
{
  "memory_id": "2026-06-04T23-12-30.123",
  "timestamp": "2026-06-04T23:12:30.123+09:00",
  "user_question": "What do you see?",
  "model_answer": "A desk with a laptop and cables near a window.",
  "frame_paths": ["outputs/frames/2026-06-04T23-12-30.123.jpg"],
  "frame_timestamps": ["2026-06-04T23:12:28.500+09:00"],
  "location": {
    "label": "home office",
    "lat": 35.6812,
    "lon": 139.7671,
    "source": "config",
    "full_address": "1-2-3 Example St, 渋谷区, 東京都 150-0001, Japan",
    "city": "渋谷区",
    "prefecture": "東京都",
    "postal_code": "150-0001",
    "country": "Japan",
    "geocode_provider": "nominatim",
    "geocoded_at": "2026-06-06T16:57:09.525+09:00"
  },
  "camera_source": "tapo-rtsp"
}
```

When `NUM_FRAMES_PER_QUERY=3`, frames are saved as `{memory_id}_f01.jpg`, `_f02.jpg`, `_f03.jpg`.

## Inspect JSONL

```bash
tail -n 1 outputs/memories.jsonl | jq .
jq -r '.model_answer' outputs/memories.jsonl
```

## Shutdown summary

```text
Run summary:
- questions_asked
- memories_written
- vlm_failures
- average_vlm_latency_seconds
```

`average_vlm_latency_seconds` is the mean Q&A latency per question (single VLM call).

## Vector / semantic search

LTM queries use **ChromaDB** + **Ollama embeddings** (or OpenAI) to rank results by cosine
similarity instead of keyword matching. Existing memories must be backfilled:

```bash
# One-time setup (Ollama path — no API key needed)
ollama pull nomic-embed-text

# Backfill existing rows
cd memory_log
uv run python -m src.embed_index           # index new rows only
uv run python -m src.embed_index --force   # re-embed all rows

# Check results
sqlite3 outputs/memory.sqlite "SELECT count(*) FROM promoted_events WHERE text_embedding_id IS NOT NULL;"
```

New memories are embedded automatically at write time when `EMBED_ON_WRITE=true`.
If Ollama is unavailable or `VECTOR_SEARCH_ENABLED=false`, LTM queries silently fall back
to the SQLite `LIKE` keyword path.

To switch to OpenAI embeddings:
```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```
Then re-run `uv run python -m src.embed_index` — new model-namespaced collections
(`<store>__text-embedding-3-small`) are created without overwriting the Ollama collections.

## LTM query telemetry

Each LTM query run appends one row to `outputs/long_term_query_logs.sqlite` when
`LTM_QUERY_LOG_ENABLED=true` (default). The row captures full per-stage I/O for debugging:

| Column | What it contains |
|---|---|
| `plan_json` | Parsed `RetrievalPlan` — intent, time range, location, semantic query, stores |
| `planner_raw_response` | Raw LLM output before JSON parsing (helps debug malformed plans) |
| `retrieval_trace_json` | JSON array — one object per queried store: `store`, `method` (`vector`/`like`/`metadata`), `candidate_count` (ChromaDB hits before SQL filter), `sql`, `params`, `final_count`, and a `note` when candidates were dropped by time/location filters |
| `answer_prompt` | Exact evidence text sent to the answer generator |
| `answer` | Final answer returned to the user |
| `error` | Exception message if the pipeline failed |
| `extra_json` | Summary for quick SQL queries: `vector_used`, `stores_selected_but_empty` (stores where vector found candidates but SQL filters eliminated them all), `expanded` |
| `latency_*_ms` | Per-stage latency: `plan`, `grounding`, `retrieval`, `answer`, `total` |

The `retrieval_trace_json` + `extra_json.stores_selected_but_empty` are designed specifically to
expose the silent "candidates→0 rows" failure where vector search selects memories but the
time-range or location filter in the SQL step eliminates them all.

**Inspect a run:**

```bash
# Trace for the most recent query
sqlite3 outputs/long_term_query_logs.sqlite \
  "SELECT retrieval_trace_json FROM long_term_query_logs ORDER BY timestamp_utc DESC LIMIT 1;" \
  | jq .

# Find queries where vector candidates were dropped by SQL filters
sqlite3 outputs/long_term_query_logs.sqlite \
  "SELECT user_query, extra_json FROM long_term_query_logs WHERE extra_json LIKE '%stores_selected_but_empty%' ORDER BY timestamp_utc DESC;"

# Latency summary
sqlite3 outputs/long_term_query_logs.sqlite \
  "SELECT user_query, intent, round(latency_total_ms) as total_ms, round(latency_retrieval_ms) as retr_ms FROM long_term_query_logs ORDER BY timestamp_utc DESC LIMIT 10;"
```

The app-log file (`RTSP_FFMPEG_LOG`) also records a readable per-stage INFO trace for each
query, including the `method + candidates → rows` line for every store.

## Known limitations

- **OpenAI + Ollama** — set `VLM_PROVIDER` / `VLM_MODEL` (Ollama needs a vision model, e.g. `llava`).
- **Tapo has no GPS** — use config labels/coordinates for fixed cameras.
- **Phone GPS** — requires HTTPS location sidecar page open on the phone.
- **Geocoded addresses** — optional; full street addresses increase log sensitivity.
- **Legacy JSONL** — older records with `summary`/`objects` remain readable by search tools.
- **Image embeddings** — `image_embedding_id` columns are reserved but not yet populated (text only in v1).
- **Switching embedding models** requires a reindex (`embed_index --force`) — collections are model-namespaced.

## Evaluation harness

An end-to-end evaluation harness lives in `evals/`. It replays pre-recorded video files
deterministically, seeds or replays past memories into an **isolated** eval DB (production
`outputs/memory.sqlite` is never touched), and scores answers with deterministic matching
+ an optional LLM judge rubric.

### Quick start

```bash
cd memory_log

# 1. Generate toy dataset (5-min synthetic video + manifest)
uv run python -m evals.make_sample

# 2. Live VQA eval (replay video → ask questions at specific timestamps)
uv run python -m evals.run_live --manifest evals/datasets/toy/desk_001.json

# 3. LTM eval — seed mode (inject structured past memories → ask recall questions)
uv run python -m evals.run_ltm --manifest evals/datasets/toy/desk_001.json --memory-mode seed

# 4. LTM eval — replay mode (ingest history video as passive observations → ask)
uv run python -m evals.run_ltm --manifest evals/datasets/toy/desk_001.json --memory-mode replay

# Options for both runners
--model gpt-4o-mini       # override VLM_MODEL
--limit 5                 # evaluate only first N questions
--no-judge                # skip LLM judge (exact-match only)
--run-id my_run           # custom run ID
```

Results are written to `evals/outputs/eval_runs.sqlite` (one row per question) and a JSON
summary file. Console output shows a dashboard-style breakdown.

### Eval manifest format

Manifests are JSON files describing one eval scenario. Key fields:

```json
{
  "video_id": "desk_001",
  "video_path": "desk_001.mp4",
  "base_timestamp": "2026-01-15T09:00:00+09:00",
  "live_questions": [
    {
      "id": "q1", "ask_at_sec": 15.0, "question": "What is on the desk?",
      "gold_answer": "red bottle", "answer_type": "short_text",
      "gold_evidence_window": [10.0, 24.0]
    }
  ],
  "memory_mode": "seed",
  "seed_memories": [
    {
      "kind": "active_query", "timestamp": "2026-01-15T09:00:15+09:00",
      "user_question": "...", "model_answer": "There is a red bottle on the desk."
    }
  ],
  "memory_questions": [
    {
      "id": "m1", "question": "Where did I put the red bottle?",
      "gold_answer": "on the shelf", "gold_evidence_windows": [[75.0, 120.0]]
    }
  ]
}
```

### Public benchmark adapters

Convert external benchmark datasets into the manifest format. Use `--limit N` (videos, not questions)
to keep early experiments cheap — start with 3–5, scale up once the pipeline looks right.

#### StreamingBench (Live QA)

900 videos · 4 500 MCQ questions · timestamps per question.
Annotations stream from HuggingFace automatically (~500 KB, instant).
Videos are ~203 GB total — download only what you need (see below).

```bash
# Step 1 — stream annotations and generate N manifests (no video download needed yet)
uv run python -m evals.adapters.streaming_bench \
    --limit 5 \
    --out-dir evals/datasets/streaming_bench/

# Choose a specific task category (default: all 4 MCQ categories)
uv run python -m evals.adapters.streaming_bench \
    --configs Real_Time_Visual_Understanding \
    --limit 5 \
    --out-dir evals/datasets/streaming_bench/

# Step 2 — run Live QA eval (MCQ → deterministic scoring, no LLM judge needed)
uv run python -m evals.run_live \
    --manifest evals/datasets/streaming_bench/<video_id>.json \
    --no-judge \
    --limit 10        # limit questions within a single manifest
```

**Manifests without videos** work for testing the metadata pipeline; the runner will error
at frame-fetch time since the video path is a placeholder. To add real videos:

```bash
# Download individual sample videos from the StreamingBench GitHub:
#   https://github.com/THUNLP-MT/StreamingBench  (Google Drive download script)
# Place MP4s as sample_1.mp4, sample_2.mp4, … in a local directory, then:

uv run python -m evals.adapters.streaming_bench \
    --limit 5 \
    --video-dir /path/to/streaming_bench_videos/ \
    --out-dir evals/datasets/streaming_bench/
```

Available `--configs` (MCQ categories):
- `Real_Time_Visual_Understanding` — 2 500 questions
- `Sequential_Question_Answering` — 250 questions
- `Contextual_Understanding` — 500 questions
- `Omni_Source_Understanding` — 1 000 questions

#### EgoSchema (LTM)

Egocentric long-form MCQ (wearable-like). Videos require Ego4D access.

```bash
# Step 1 — download QA annotations (tiny JSON, no video needed)
uv run python -m evals.adapters.egoschema \
    --download-qa \
    --qa-json evals/datasets/egoschema_raw/questions.json

# Step 2 — generate manifests (placeholder video paths if no video-dir)
uv run python -m evals.adapters.egoschema \
    --qa-json evals/datasets/egoschema_raw/questions.json \
    --limit 5 \
    --out-dir evals/datasets/egoschema/

# With local Ego4D clips:
uv run python -m evals.adapters.egoschema \
    --qa-json evals/datasets/egoschema_raw/questions.json \
    --video-dir /path/to/ego4d_clips/ \
    --limit 5 \
    --out-dir evals/datasets/egoschema/

# Step 3 — LTM eval (replay mode: ingest clip as passive observations, then ask)
uv run python -m evals.run_ltm \
    --manifest evals/datasets/egoschema/<uid>.json \
    --memory-mode replay \
    --observe-interval 30
```

Add new benchmarks by subclassing `evals.adapters.base.BenchmarkAdapter` and implementing
`to_manifests()`. The runners are benchmark-agnostic — switching benchmarks = swapping the manifest.

### Metrics

| Task | Metric |
|------|--------|
| Live QA | `answer_accuracy`, `hallucination_rate`, `unanswerable_accuracy`, `mean_frame_age_sec`, `p50/p95_latency_ms` |
| LTM retrieval | `Recall@1/3/5`, `MRR`, `mean_temporal_distance_sec`, `evidence_iou` |
| LTM final answer | `answer_accuracy`, `judge_avg_score` (0-2 rubric), `hallucination flags` |

---

## Relation to Step 1

`vlm_smoke` is interactive QA only. `memory_log` adds persistent JSONL memories **when you ask**, using the same threaded capture model as `vlm_smoke`.