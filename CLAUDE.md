# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Wearable multimodal AI assistant that ingests live camera streams, runs VLM inference on frames, and persists visual memories as JSONL. The project uses **uv** as the package manager with per-sub-package virtual environments.

## Running the packages

Each sub-package has its own `pyproject.toml` and `uv.lock`. Always `cd` into the package before running:

```bash
# Phase 0 — validate camera streams
cd camera_test
uv run camera-preview      # live window
uv run camera-sample       # save JPEGs to sampled_frames/
uv run camera-vlm          # terminal VLM Q&A
uv run camera-whep-probe   # diagnose WebRTC/WHEP

# Phase 1 — live visual QA only
cd vlm_smoke
uv run python -m src.main

# Phase 2 — question-driven JSONL memory
cd memory_log
uv run python -m src.main
```

Each package reads config from its local `.env` (copy from `.env.example` and set `OPENAI_API_KEY`).

## Architecture

```
camera / webcam / video file
        ↓
capture/   (shared ring-buffer + stream config)
        ↓
vlm_smoke/src/    OR    memory_log/src/
        ↓                       ↓
 terminal Q&A         terminal Q&A + JSONL memory record
```

**Shared packages** (root `pyproject.toml`, installed as `multimodal`):
- `capture/` — threaded `CameraFrameSource` with ring buffer, RTSP/WHEP source resolution, stale-stream reconnect
- `providers/ollama.py` — Ollama HTTP client

**Sub-packages** depend on `multimodal` via `tool.uv.sources` editable install. Any change to `capture/` or `providers/` is picked up immediately without reinstalling.

**`memory_log` data flow (one VLM call per question):**
1. Background thread (`CameraFrameSource`) samples frames at `CAPTURE_SAMPLE_INTERVAL_SEC` into a deque of `FrameItem`
2. On user question → `VLMClient.answer_question()` → print answer
3. `resolve_location()` picks location from config or live phone GPS sidecar
4. `GeocodeClient` reverse-geocodes lat/lon via Nominatim (SQLite cache at `outputs/geocode_cache.sqlite`)
5. `MemoryWriter.save_memory()` appends to `outputs/memories.jsonl` and saves frame JPEGs

## Key environment variables

| Variable | Where | Notes |
|---|---|---|
| `FRAME_SOURCE_TYPE` | both | `camera`, `webcam`, or `video` |
| `CAMERA_SOURCE` | both | `tapo-rtsp`, `tapo-webrtc`, `phone-webrtc` |
| `VLM_PROVIDER` / `VLM_MODEL` | both | `openai` (default `gpt-5.5`) or `ollama` (e.g. `llava`) |
| `OPENAI_API_KEY` | both | required when `VLM_PROVIDER=openai` |
| `RTSP_URL` | both | Tapo camera RTSP endpoint |
| `PHONE_STREAM_URL` | both | MediaMTX RTSP relay for phone |
| `NUM_FRAMES_PER_QUERY` | both | frames bundled per VLM call (default `1`) |
| `LOCATION_SERVER_ENABLED` | memory_log | start HTTPS GPS sidecar for phone |
| `GEOCODE_ENABLED` | memory_log | Nominatim reverse geocode (default `true`) |
| `RTSP_FFMPEG_LOG` | both | path for FFmpeg + app logs (keeps REPL clean) |

## Camera sources

`resolve_source()` (`capture/stream_config.py`) maps `CAMERA_SOURCE` to `(preset, source_type, url)`:
- `tapo-rtsp` → RTSP direct via OpenCV/FFmpeg
- `tapo-webrtc` → RTSP relay from MediaMTX (default) or raw WHEP
- `phone-webrtc` → MediaMTX RTSP relay (`rtsp://127.0.0.1:8554/phone`) or WHEP

MediaMTX config files (`mediamtx-tapo.yml`, `mediamtx-phone.yml`) and TLS certs live in `camera_test/`. Copy from the `.example.yml` files once — they are shared by all packages.

## Inspecting memory output

```bash
tail -n 1 memory_log/outputs/memories.jsonl | jq .
```

Legacy records may contain `summary`/`objects`/`privacy_risk` (old two-call format). Current format uses `model_answer` and `frame_paths`.
