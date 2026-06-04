# memory_log — Step 2: Visual memory as JSONL

Phase 2 of the wearable multimodal AI assistant. It turns a live or recorded frame stream into **long-term visual memory records** stored as **JSONL** plus optional JPEG snapshots.

## What this step does

```text
frame stream
  → sample one frame every N seconds
  → VLM returns structured JSON (scene summary, objects, privacy, etc.)
  → save frame image (optional)
  → append one JSON object per line to outputs/memories.jsonl
```

This is intentionally boring infrastructure: reliable capture, parsing, and append-only storage before anything fancier.

## Why JSONL before a vector DB

- **Easy to debug** — `tail -f`, `jq`, any text editor.
- **No extra services** — no Chroma, embeddings API, or search index to operate yet.
- **Stable contract** — each line is a self-contained memory record you can later index, filter, or embed.
- **Fail-safe progress** — lines are flushed after each write so a crash does not lose prior memories.

Step 3 can add keyword/time-based search on this file; Step 4+ can add embeddings and semantic retrieval.

## Setup with uv

```bash
cd memory_log
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY and your frame source settings
uv sync
```

## Configuration

Copy `.env.example` to `.env`. Key variables:

| Variable | Description |
|----------|-------------|
| `FRAME_SOURCE_TYPE` | `rtmp`, `webcam`, or `video` |
| `RTMP_URL` | RTMP stream URL (GoPro relay, etc.) |
| `WEBCAM_INDEX` | Webcam device index (default `0`) |
| `VIDEO_PATH` | Path to video file when `FRAME_SOURCE_TYPE=video` |
| `VLM_PROVIDER` | `openai` (only provider in this phase) |
| `VLM_MODEL` | OpenAI vision-capable model (e.g. `gpt-5.5`) |
| `OPENAI_API_KEY` | Required API key |
| `FRAME_SAMPLE_INTERVAL_SECONDS` | Seconds between memory samples (default `3`) |
| `OUTPUT_FRAME_DIR` | Directory for saved frames (default `outputs/frames`) |
| `MEMORY_JSONL_PATH` | JSONL output file (default `outputs/memories.jsonl`) |
| `LOCATION_LABEL` | Optional manual location label (no GPS yet) |
| `SAVE_FRAMES` | `true` / `false` — write JPEG per memory |
| `MAX_RUNTIME_SECONDS` | Optional cap; unset = run until Ctrl+C |

## Run

```bash
cd memory_log
uv run python -m src.main
```

### RTMP (GoPro relay)

```bash
# .env
FRAME_SOURCE_TYPE=rtmp
RTMP_URL=rtmp://localhost:1935/live/gopro
FRAME_SAMPLE_INTERVAL_SECONDS=3
```

Ensure your RTMP relay is publishing, then start `memory_log`.

### Webcam

```bash
FRAME_SOURCE_TYPE=webcam
WEBCAM_INDEX=0
```

### Video file

```bash
FRAME_SOURCE_TYPE=video
VIDEO_PATH=/path/to/clip.mp4
MAX_RUNTIME_SECONDS=120   # optional — stop after 2 minutes
```

The loop stops automatically when the video ends.

## Example log line

```text
[2026-06-04T23:12:30+09:00] stored=true privacy=low scene=indoor_workspace objects=desk,laptop latency=2.43s
```

## Example memory record

One line in `outputs/memories.jsonl`:

```json
{
  "memory_id": "2026-06-04T23-12-30.123",
  "timestamp": "2026-06-04T23:12:30.123+09:00",
  "image_path": "outputs/frames/2026-06-04T23-12-30.123.jpg",
  "summary": "A desk with a laptop, cable, and GoPro accessories.",
  "objects": ["desk", "laptop", "cable", "camera"],
  "scene_type": "indoor_workspace",
  "people_count": 0,
  "text_visible": [],
  "location": {
    "label": null,
    "lat": null,
    "lon": null,
    "source": "manual_or_not_available"
  },
  "should_store": true,
  "memory_reason": "Contains meaningful workspace context.",
  "privacy_risk": "low"
}
```

## Inspect JSONL

```bash
# Pretty-print last record
tail -n 1 outputs/memories.jsonl | jq .

# Count memories
wc -l outputs/memories.jsonl

# Filter high-privacy lines
jq 'select(.privacy_risk == "high")' outputs/memories.jsonl
```

## Shutdown summary

On exit (Ctrl+C, max runtime, or end of video), the app prints counters:

```text
Run summary:
- frames_read
- frames_sampled
- memories_written
- vlm_failures
- json_parse_failures
- average_vlm_latency_seconds
```

## Known limitations

- **OpenAI only** — same as Step 1 (`vlm_smoke`).
- **No retrieval** — no search API, embeddings, or Chroma yet.
- **No GPS** — location is a manual label or placeholders only.
- **Synchronous sampling** — main loop reads frames and calls the VLM on interval; long VLM latency delays the next sample.
- **Records with `should_store: false`** are still written so you can audit VLM decisions; check the log `stored=false` lines.
- **Invalid VLM JSON** skips that sample; the process keeps running.

## Next step

Keyword and time-based memory search over `memories.jsonl` (no vector DB required initially).

## Relation to Step 1

`vlm_smoke` provides interactive visual QA. `memory_log` reuses the same frame-source types (`RTMP`, `webcam`, `video`) and OpenAI VLM patterns, but runs an automatic sampling loop instead of a REPL.
