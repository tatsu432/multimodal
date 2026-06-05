# vlm_smoke — Phase 1: Live Visual QA

Phase 1 of the wearable multimodal AI assistant project. This module stabilizes the live visual question-answering loop: capture frames from RTMP, webcam, or a video file, then ask a vision-language model (VLM) questions in the terminal.

## What this phase does

- Continuously samples frames into a small ring buffer (background thread)
- Accepts text questions in a terminal REPL
- Sends the latest frame(s) to OpenAI’s VLM (Responses API)
- Logs capture health, frame counts, VLM latency, and saved frame paths
- Optionally saves frames sent with each query under `outputs/sampled_frames/`

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.12+
- OpenAI API key with access to your chosen vision model
- For RTMP: GoPro (or other camera) streaming to a local RTMP relay (e.g. `rtmp://localhost:1935/live/gopro`)
- OpenCV with FFmpeg support (included via `opencv-python`) for RTMP

## Setup

```bash
cd vlm_smoke
uv sync
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and other options as needed)
```

## Configuration (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `FRAME_SOURCE_TYPE` | `rtmp`, `webcam`, or `video` | `rtmp` |
| `RTMP_URL` | RTMP stream URL | `rtmp://localhost:1935/live/gopro` |
| `WEBCAM_INDEX` | Webcam device index | `0` |
| `VIDEO_PATH` | Path to video file (required for `video`) | — |
| `VLM_PROVIDER` | VLM backend (`openai` only in Phase 1) | `openai` |
| `VLM_MODEL` | OpenAI model name | `gpt-5.5` |
| `OPENAI_API_KEY` | API key (required) | — |
| `FRAME_SAMPLE_DIR` | Where to save queried frames | `outputs/sampled_frames` |
| `NUM_FRAMES_PER_QUERY` | Frames sent per question | `1` |
| `SAVE_QUERIED_FRAMES` | Save frames on each query (`true`/`false`) | `true` |
| `FRAME_BUFFER_SIZE` | Ring buffer capacity | `8` |
| `RTMP_SAMPLE_INTERVAL_SEC` | Seconds between buffered samples | `1.0` |

See [.env.example](.env.example) for a full template.

## How to run

From the `vlm_smoke` directory:

```bash
uv run python -m src.main
```

### RTMP (GoPro live stream)

```bash
# .env
FRAME_SOURCE_TYPE=rtmp
RTMP_URL=rtmp://localhost:1935/live/gopro
```

Start your RTMP relay and GoPro stream first, then run the app. Wait a few seconds for the frame buffer to fill before asking questions.

### Webcam

```bash
# .env
FRAME_SOURCE_TYPE=webcam
WEBCAM_INDEX=0
```

### Video file

```bash
# .env
FRAME_SOURCE_TYPE=video
VIDEO_PATH=/path/to/your/video.mp4
```

The video loops from the start when it reaches the end, so you can smoke-test without a live camera.

## Example questions

```
What do you see?
Is there a person?
What object is closest to the camera?
What text is visible?
```

Type `q`, `quit`, or `exit` to stop.

## Current limitations

- **Single VLM provider**: Only OpenAI is implemented (`VLM_PROVIDER=openai`).
- **Blocking REPL**: The main thread waits for keyboard input; capture runs in a background thread.
- **No preview window**: Headless-friendly; no `cv2.imshow` in this module.
- **RTMP startup delay**: You may need to wait a few seconds after launch before frames appear in the buffer.
- **RTMP reconnects**: Brief read stalls are retried before reconnecting; sustained failures still reconnect. If your relay stops when the last viewer disconnects, avoid running a second RTMP client (e.g. `camera_test/preview_stream.py`) at the same time.
- **API cost and latency**: Each question triggers a VLM API call; latency is logged but not optimized.
- **Coarse sampling**: Frames are sampled at `RTMP_SAMPLE_INTERVAL_SEC` (default 1 Hz), not every video frame.

## Out of scope for Phase 1

Do not expect these in `vlm_smoke` yet:

- Memory logging or episodic memory
- ChromaDB / vector search
- FastAPI backend
- Streamlit UI
- Location metadata
- Evaluation harness
- Efficient VLM research or model comparison

## Related scripts

Lightweight stream testing lives in `camera_test/` (`preview_stream.py`, `frame_sample.py`, `live_vlm_qa.py`). `vlm_smoke` supersedes `live_vlm_qa.py` for ongoing Phase 1 work.

## Project layout

```text
vlm_smoke/
├── README.md
├── pyproject.toml
├── .env.example
├── src/
│   ├── main.py          # REPL entrypoint
│   ├── config.py        # Environment configuration
│   ├── frame_source.py  # RTMP / webcam / video capture
│   ├── vlm_client.py    # OpenAI VLM wrapper
│   └── utils.py         # Buffer, encoding, frame saving
└── outputs/
    └── sampled_frames/  # Saved query frames (gitignored)
```
