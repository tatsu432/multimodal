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
| `FRAME_SOURCE_TYPE` | `camera`, `webcam`, or `video` |
| `CAMERA_SOURCE` | When `camera`: `tapo-rtsp`, `tapo-webrtc`, `phone-webrtc` |
| `RTSP_URL`, `PHONE_STREAM_URL`, `RTSP_*` | Same as [`camera_test`](../camera_test/README.md) |
| `WEBCAM_INDEX` | Webcam device index (default `0`) |
| `VIDEO_PATH` | Required when `FRAME_SOURCE_TYPE=video` |
| `VLM_PROVIDER` | `openai` or `ollama` |
| `VLM_MODEL` | Vision model (e.g. `gpt-5.5`, `llava`) |
| `OPENAI_API_KEY` | Required when `VLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` | Ollama URL when `VLM_PROVIDER=ollama` |
| `FRAME_BUFFER_SIZE` | Ring buffer size for recent frames (default `8`) |
| `CAPTURE_SAMPLE_INTERVAL_SEC` | How often the background thread adds frames (default `1.0`) — **not** memory write interval |
| `NUM_FRAMES_PER_QUERY` | Frames sent to Q&A per question (default `1`) |
| `OUTPUT_FRAME_DIR` | Saved frame images (default `outputs/frames`) |
| `MEMORY_JSONL_PATH` | JSONL file (default `outputs/memories.jsonl`) |
| `LOCATION_LABEL` | Optional manual location label |
| `SAVE_FRAMES` | Save JPEG when a memory is stored |
| `MAX_RUNTIME_SECONDS` | Optional; unset = run until Ctrl+C |

## Camera sources (Tapo RTSP, MediaMTX, phone WebRTC)

Same presets as [`vlm_smoke`](../vlm_smoke/README.md#camera-sources-tapo-rtsp-mediamtx-phone-webrtc) and [`camera_test`](../camera_test/README.md). Set `FRAME_SOURCE_TYPE=camera` in `memory_log/.env`.

MediaMTX YAML and certs: [`camera_test/`](../camera_test/) (`mediamtx-tapo.yml`, `mediamtx-phone.yml`).

### Tapo RTSP (recommended)

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=tapo-rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
RTSP_FLUSH_GRABS=8
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

## Run

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
- **OpenAI + Ollama** — set `VLM_PROVIDER` / `VLM_MODEL` (Ollama needs a vision model, e.g. `llava`).
- **No retrieval** — no search over memories yet.
- **`should_store: false`** — Q&A still prints; JSONL and frame save are skipped.
- **No GPS** — location label only.

## Next step

Keyword and time-based memory search: see [`../memory_search/README.md`](../memory_search/README.md).

## Relation to Step 1

`vlm_smoke` is interactive QA only. `memory_log` adds persistent JSONL memories **when you ask**, using the same threaded capture model as `vlm_smoke`.
