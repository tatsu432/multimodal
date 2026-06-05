# camera_test

Lightweight scripts for testing live camera streams before using the full pipeline in `vlm_smoke` and `memory_log`.

All scripts support **RTMP** (e.g. GoPro relay) and **RTSP** (e.g. Tapo IP camera).

## Files

| File | Purpose |
|------|---------|
| `stream_config.py` | Shared stream helpers: resolve protocol/URL from env or CLI, open `cv2.VideoCapture` |
| `preview_rtmp.py` | Live preview only — no saving, no VLM |
| `frame_sample.py` | Live preview + save a JPEG every 2 seconds to `sampled_frames/` |
| `live_vlm_qa.py` | Live preview in background + ask a VLM questions about recent frames in a REPL |

## Prerequisites

From the repo root:

```bash
uv sync
```

Scripts load environment variables from `.env` in the current directory or a parent directory (typically the repo root `.env`).

Run every script from inside `camera_test`:

```bash
cd camera_test
uv run --project .. python <script>.py
```

`--project ..` uses the root `multimodal` package (OpenCV, dotenv, and for `live_vlm_qa.py` the shared `providers` module).

## Configuration

Set these in the repo root `.env` (or `camera_test/.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_PROTOCOL` | `rtmp` | `rtmp` or `rtsp` |
| `RTMP_URL` | `rtmp://localhost:1935/live/gopro` | Used when protocol is `rtmp` |
| `RTSP_URL` | `rtsp://localhost:8554/live/gopro` | Used when protocol is `rtsp` |

CLI flags override env:

- `--protocol rtmp|rtsp`
- `--url <full stream URL>`

### GoPro (RTMP relay)

```env
STREAM_PROTOCOL=rtmp
RTMP_URL=rtmp://localhost:1935/live/gopro
```

### Tapo camera (RTSP)

Create a **Camera Account** in the Tapo app first: **Device Settings → Advanced Settings → Camera Account**. This is separate from your Tapo app login.

```env
STREAM_PROTOCOL=rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
```

- `stream1` — higher quality
- `stream2` — lower bandwidth (good default for sampling and VLM)
- If the password contains special characters (`@`, `!`, `#`, …), URL-encode them (e.g. `!` → `%21`)

### VLM settings (`live_vlm_qa.py` only)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_PROVIDER` | `openai` | `openai` or `ollama` (local, no API key) |
| `VLM_MODEL` | `gpt-5.5` | OpenAI model name, or Ollama model (e.g. `llava`) |
| `OPENAI_API_KEY` | — | Required when `VLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

## Usage

### 1. Preview the stream

Confirm the stream opens and frames display correctly.

```bash
uv run --project .. python preview_rtmp.py
```

Tapo example:

```bash
uv run --project .. python preview_rtmp.py \
  --protocol rtsp \
  --url 'rtsp://camera_user:camera_pass@192.168.1.50:554/stream2'
```

Press **`q`** in the preview window to quit.

### 2. Sample frames to disk

Shows a live preview and saves one JPEG every 2 seconds under `camera_test/sampled_frames/`.

```bash
uv run --project .. python frame_sample.py
```

With explicit RTSP URL:

```bash
uv run --project .. python frame_sample.py \
  --protocol rtsp \
  --url 'rtsp://camera_user:camera_pass@192.168.1.50:554/stream2'
```

Press **`q`** to stop. Output files: `frame_000000.jpg`, `frame_000001.jpg`, …

### 3. Ask a VLM about the live view

Captures recent frames in the background (1 per second) and answers text questions in a REPL.

**OpenAI:**

```env
VLM_PROVIDER=openai
VLM_MODEL=gpt-5.5
OPENAI_API_KEY=your-key-here
```

```bash
uv run --project .. python live_vlm_qa.py --protocol rtsp
```

**Local Ollama (no API key):**

```bash
ollama pull llava
```

```env
VLM_PROVIDER=ollama
VLM_MODEL=llava
```

```bash
uv run --project .. python live_vlm_qa.py --protocol rtsp
```

Example questions:

- What objects are visible?
- Is there a person in front of the camera?
- What changed in the last few seconds?

Type **`q`**, **`quit`**, or **`exit`** to stop.

## Troubleshooting

**Stream won't open**

1. Test the same URL in VLC (*Media → Open Network Stream*).
2. Confirm camera and computer are on the same LAN.
3. For Tapo, verify the Camera Account credentials (not your Tapo app password).
4. Try `stream2` instead of `stream1`.

**RTSP works in VLC but not in Python**

Force TCP transport for FFmpeg/OpenCV:

```bash
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
uv run --project .. python preview_rtmp.py --protocol rtsp --url 'rtsp://...'
```

**`Failed to read frame` loops**

The camera may have dropped the connection. Stop with `q` and restart. For RTSP, TCP transport often helps.

## Suggested workflow

```text
preview_rtmp.py   → confirm stream works
frame_sample.py   → confirm frames save correctly
live_vlm_qa.py    → test VLM on live video
```

For production-style flows (config, logging, memory), use the packages in the repo:

- `vlm_smoke/` — stable live visual QA
- `memory_log/` — structured visual memory to JSONL
