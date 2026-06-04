# memory_log — Step 2: Question-driven visual memory

Phase 2 of the wearable multimodal AI assistant. Ask questions about the live view; each question triggers a **text answer** and optionally a **JSONL memory record** tied to that moment.

## What this step does

```text
frame stream → background frame buffer
user asks question
  → VLM answers your question (recent frames)
  → VLM returns structured JSON (latest frame)
  → append memory to outputs/memories.jsonl (if should_store)
```

Nothing is written to JSONL on a timer. If you do not ask a question, no new memories are created.

## Why JSONL before a vector DB

- **Easy to debug** — `tail -f`, `jq`, any text editor.
- **No extra services** — no Chroma, embeddings API, or search index yet.
- **Stable contract** — each line is self-contained, including `user_question` for later search.
- **Fail-safe progress** — lines are flushed after each write.

## Setup with uv

```bash
cd memory_log
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and frame source
uv sync
```

**Migrating from the old timer-based config:** remove `FRAME_SAMPLE_INTERVAL_SECONDS` from `.env` and add `FRAME_BUFFER_SIZE`, `CAPTURE_SAMPLE_INTERVAL_SEC`, and `NUM_FRAMES_PER_QUERY` (see `.env.example`).

## Configuration

| Variable | Description |
|----------|-------------|
| `FRAME_SOURCE_TYPE` | `rtmp`, `webcam`, or `video` |
| `RTMP_URL` | RTMP stream URL |
| `WEBCAM_INDEX` | Webcam device index (default `0`) |
| `VIDEO_PATH` | Required when `FRAME_SOURCE_TYPE=video` |
| `VLM_PROVIDER` | `openai` |
| `VLM_MODEL` | Vision-capable model |
| `OPENAI_API_KEY` | Required |
| `FRAME_BUFFER_SIZE` | Ring buffer size for recent frames (default `8`) |
| `CAPTURE_SAMPLE_INTERVAL_SEC` | How often the background thread adds frames (default `1.0`) — **not** memory write interval |
| `NUM_FRAMES_PER_QUERY` | Frames sent to Q&A per question (default `1`) |
| `OUTPUT_FRAME_DIR` | Saved frame images (default `outputs/frames`) |
| `MEMORY_JSONL_PATH` | JSONL file (default `outputs/memories.jsonl`) |
| `LOCATION_LABEL` | Optional manual location label |
| `SAVE_FRAMES` | Save JPEG when a memory is stored |
| `MAX_RUNTIME_SECONDS` | Optional; unset = run until Ctrl+C |

## Run

```bash
cd memory_log
uv run python -m src.main
```

Wait a few seconds for the capture buffer to fill, then type a question. Type `q` to quit.

### RTMP

```bash
FRAME_SOURCE_TYPE=rtmp
RTMP_URL=rtmp://localhost:1935/live/gopro
```

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

[2026-06-04T23:12:30+09:00] stored=true privacy=low scene=indoor_workspace objects=desk,laptop latency=4.21s
```

## Example memory record

```json
{
  "memory_id": "2026-06-04T23-12-30.123",
  "timestamp": "2026-06-04T23:12:30.123+09:00",
  "image_path": "outputs/frames/2026-06-04T23-12-30.123.jpg",
  "user_question": "What do you see?",
  "summary": "A desk with a laptop, cable, and accessories.",
  "objects": ["desk", "laptop", "cable"],
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
  "memory_reason": "User asked about the current view.",
  "privacy_risk": "low"
}
```

## Inspect JSONL

```bash
tail -n 1 outputs/memories.jsonl | jq .
jq -r '.user_question' outputs/memories.jsonl
```

## Shutdown summary

```text
Run summary:
- questions_asked
- memories_written
- vlm_failures
- json_parse_failures
- average_vlm_latency_seconds
```

`average_vlm_latency_seconds` is the mean **per-question** total (Q&A + memory analysis).

## Known limitations

- **Two API calls per question** — answer + structured memory (cost/latency).
- **OpenAI only** — same provider support as Step 1.
- **No retrieval** — no search over memories yet.
- **`should_store: false`** — Q&A still prints; JSONL and frame save are skipped.
- **No GPS** — location label only.

## Next step

Keyword and time-based memory search over `memories.jsonl`.

## Relation to Step 1

`vlm_smoke` is interactive QA only. `memory_log` adds persistent JSONL memories **when you ask**, using the same threaded capture model as `vlm_smoke`.
