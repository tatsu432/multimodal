# vlm_smoke — Phase 1: Live Visual QA

Phase 1 of the wearable multimodal AI assistant project. This module stabilizes the live visual question-answering loop: capture frames from **Tapo / phone cameras**, RTMP, webcam, or a video file, then ask a vision-language model (VLM) questions in the terminal.

## What this phase does

- Continuously samples frames into a small ring buffer (background thread)
- Accepts text questions in a terminal REPL
- Sends the latest frame(s) to OpenAI or local Ollama vision models
- Logs capture health, frame counts, VLM latency, and saved frame paths
- Optionally saves frames sent with each query under `outputs/sampled_frames/`

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.12+
- OpenAI API key **or** [Ollama](https://ollama.com) with a vision model (`ollama pull llava`)
- For **camera** sources: same `.env` presets as [`camera_test/`](../camera_test/README.md) (`tapo-rtsp`, `phone-webrtc`, etc.)
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
| `FRAME_SOURCE_TYPE` | `camera`, `rtmp`, `webcam`, or `video` | `camera` |
| `CAMERA_SOURCE` | When `camera`: `tapo-rtsp`, `tapo-webrtc`, `phone-webrtc` | `tapo-rtsp` |
| `RTSP_URL` | Tapo RTSP URL (when `CAMERA_SOURCE=tapo-rtsp`) | see `.env.example` |
| `PHONE_STREAM_URL` | MediaMTX RTSP relay (when `CAMERA_SOURCE=phone-webrtc`) | `rtsp://127.0.0.1:8554/phone` |
| `RTSP_TRANSPORT`, `RTSP_LOW_LATENCY`, `RTSP_FLUSH_GRABS` | RTSP tuning (same as `camera_test`) | — |
| `RTMP_URL` | RTMP stream URL | `rtmp://localhost:1935/live/gopro` |
| `WEBCAM_INDEX` | Webcam device index | `0` |
| `VIDEO_PATH` | Path to video file (required for `video`) | — |
| `VLM_PROVIDER` | `openai` or `ollama` | `openai` |
| `VLM_MODEL` | Model name (e.g. `gpt-5.5`, `llava`) | `gpt-5.5` |
| `OPENAI_API_KEY` | API key (required when `VLM_PROVIDER=openai`) | — |
| `OLLAMA_BASE_URL` | Ollama HTTP URL (when `VLM_PROVIDER=ollama`) | `http://localhost:11434` |
| `FRAME_SAMPLE_DIR` | Where to save queried frames | `outputs/sampled_frames` |
| `NUM_FRAMES_PER_QUERY` | Frames sent per question | `1` |
| `SAVE_QUERIED_FRAMES` | Save frames on each query (`true`/`false`) | `true` |
| `FRAME_BUFFER_SIZE` | Ring buffer capacity | `8` |
| `RTMP_SAMPLE_INTERVAL_SEC` | Seconds between buffered samples | `1.0` |

See [.env.example](.env.example) for a full template.

## Camera sources (Tapo RTSP, MediaMTX, phone WebRTC)

`vlm_smoke` uses the shared [`capture/`](../capture/) module (same presets as [`camera_test`](../camera_test/README.md)). Set `FRAME_SOURCE_TYPE=camera` and pick a `CAMERA_SOURCE` preset.

MediaMTX configs and TLS certs live under [`camera_test/`](../camera_test/) — copy the example YAML files there once, then use them for both `camera_test` and `vlm_smoke`.

### MediaMTX one-time setup

MediaMTX does not read `.example.yml` files directly. From the repo:

```bash
cd camera_test
cp mediamtx-tapo.example.yml mediamtx-tapo.yml    # Tapo RTSP → WebRTC + RTSP relay
cp mediamtx-phone.example.yml mediamtx-phone.yml  # phone WebRTC publish → RTSP relay
# Edit mediamtx-tapo.yml (your Tapo RTSP URL) and mediamtx-phone.yml (LAN IP, certs)
```

`mediamtx-tapo.yml` and `mediamtx-phone.yml` are **gitignored** (camera URLs and cert paths stay local).

```bash
mediamtx mediamtx-tapo.yml    # Tapo relay — use when CAMERA_SOURCE=tapo-webrtc
mediamtx mediamtx-phone.yml   # phone publish — use when CAMERA_SOURCE=phone-webrtc
```

Run only the config you need for your chosen preset.

### 1. Tapo with RTSP (recommended for VLM)

Direct RTSP from the camera — **lowest latency**, no MediaMTX required.

1. In the Tapo app: **Device Settings → Advanced Settings → Camera Account** (username/password for RTSP, not your Tapo login).
2. Note the camera IP: **Device Settings → Network**.
3. Test in VLC: *Media → Open Network Stream…*  
   `rtsp://camera_user:camera_pass@192.168.1.50:554/stream2`
4. Configure `vlm_smoke/.env`:

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=tapo-rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
RTSP_FLUSH_GRABS=8
```

- `stream2` — lower bandwidth (good for VLM); `stream1` — higher quality
- Tapo + OpenCV usually works best with **`RTSP_TRANSPORT=tcp`**

```bash
cd vlm_smoke
uv run python -m src.main
```

FFmpeg warnings go to `rtsp_decode.log` (see `RTSP_FFMPEG_LOG`); the terminal stays for questions and answers.

### 2. Tapo via MediaMTX (optional)

Tapo speaks **RTSP only**. Use this preset if you already run MediaMTX for a browser player or multiple clients. Python reads the **local RTSP relay** (not WHEP) — same as `camera_test`.

1. Confirm direct RTSP works in VLC (see §1).
2. Set your Tapo RTSP URL in `camera_test/mediamtx-tapo.yml` under `paths.tapo.source`.
3. Start MediaMTX:

```bash
cd camera_test
mediamtx mediamtx-tapo.yml
```

4. Verify in a browser: `http://localhost:8889/tapo/`
5. Configure `vlm_smoke/.env`:

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=tapo-webrtc
WEBRTC_URL=http://localhost:8889/tapo/whep
# Python uses RTSP relay by default (WEBRTC_PREVIEW_VIA_RTSP=true in capture/)
# No WHEP in Python — reads:
#   rtsp://127.0.0.1:8554/tapo
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
```

For lowest latency on the same Wi‑Fi, **Tapo RTSP direct** (§1) is usually faster than MediaMTX relay. WebRTC in the browser is still available at `http://localhost:8889/tapo/`.

### 3. iPhone / smartphone via MediaMTX WebRTC

Phones do not expose RTSP. The phone **publishes WebRTC** to MediaMTX; Python reads **`rtsp://127.0.0.1:8554/phone`**. The stream exists **only while the phone is publishing**.

#### TLS (one-time)

Phone browsers require **HTTPS** for camera access on LAN. Use [mkcert](https://github.com/FiloSottile/mkcert):

```bash
ipconfig getifaddr en0          # your Mac Wi‑Fi IP, e.g. 192.168.11.51
brew install mkcert && mkcert -install
cd camera_test
mkdir -p mediamtx-certs
mkcert -key-file mediamtx-certs/server.key -cert-file mediamtx-certs/server.crt \
  localhost 127.0.0.1 YOUR_MAC_IP
```

Install the mkcert root CA on your phone (iOS: Settings → General → About → Certificate Trust Settings).  
Uncomment your LAN IP in `mediamtx-phone.yml` under `webrtcAdditionalHosts`.

Full phone publish settings (codec, bitrate, resolution): see [camera_test README § Publish page settings](../camera_test/README.md#publish-page-settings-before-you-tap-publish).

#### Run

1. Start MediaMTX:

```bash
cd camera_test
mediamtx mediamtx-phone.yml
```

2. On the phone (same Wi‑Fi), open **https** (not http):

```text
https://YOUR_MAC_IP:8889/phone/publish
```

Example: `https://192.168.11.51:8889/phone/publish` if `ipconfig getifaddr en0` prints `192.168.11.51`.

3. Tap **Publish**, allow camera. Verify on Mac: `https://localhost:8889/phone/`

4. Configure `vlm_smoke/.env`:

```env
FRAME_SOURCE_TYPE=camera
CAMERA_SOURCE=phone-webrtc
PHONE_STREAM_URL=rtsp://127.0.0.1:8554/phone
RTSP_TRANSPORT=tcp
RTSP_LOW_LATENCY=true
RTSP_FLUSH_GRABS=8
```

```bash
cd vlm_smoke
uv run python -m src.main
```

Keep the phone publish tab in the foreground; background mobile browsers may pause video.

### Quick reference

| Goal | `CAMERA_SOURCE` | MediaMTX | Python reads |
|------|-----------------|----------|--------------|
| Tapo, lowest latency | `tapo-rtsp` | not required | `RTSP_URL` (camera direct) |
| Tapo + browser WebRTC | `tapo-webrtc` | `mediamtx-tapo.yml` | `rtsp://127.0.0.1:8554/tapo` |
| iPhone as camera | `phone-webrtc` | `mediamtx-phone.yml` + phone publish | `rtsp://127.0.0.1:8554/phone` |

CLI overrides:

```bash
uv run python -m src.main --camera tapo-rtsp --url 'rtsp://user:pass@192.168.1.50:554/stream2'
uv run python -m src.main --camera phone-webrtc --url rtsp://127.0.0.1:8554/phone
```

More troubleshooting (VLC, ICE, WHEP): [`camera_test/README.md`](../camera_test/README.md).

## How to run

From the `vlm_smoke` directory:

```bash
uv run python -m src.main
```

CLI overrides (when `FRAME_SOURCE_TYPE=camera`):

```bash
uv run python -m src.main --camera phone-webrtc --url rtsp://127.0.0.1:8554/phone
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

## VLM provider (OpenAI vs Ollama)

**OpenAI**

```env
VLM_PROVIDER=openai
VLM_MODEL=gpt-5.5
OPENAI_API_KEY=sk-...
```

**Local Ollama** — requires a **vision** model (`ollama pull llava`; text-only models like `qwen3` cannot see frames):

```bash
ollama pull llava
ollama list
```

```env
VLM_PROVIDER=ollama
VLM_MODEL=llava
OLLAMA_BASE_URL=http://localhost:11434
```

Tune `NUM_FRAMES_PER_QUERY` (default `1`) and `FRAME_BUFFER_SIZE` for multi-frame questions.

## Current limitations

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
