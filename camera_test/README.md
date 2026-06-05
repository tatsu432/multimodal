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

## Reference: streaming protocols & tools

Notes on how camera streams fit together — useful when choosing a source, debugging connectivity, or planning beyond this prototype.

### Core mental model

A camera does not automatically support every streaming protocol. A camera produces video frames, and then some device/software exposes those frames through a specific interface or protocol.

```text
Camera sensor
  → image processing
  → video encoder, e.g. H.264/H.265/MJPEG
  → output interface/protocol, e.g. USB, RTSP, RTMP, WebRTC, HTTP
  → receiver
  → decoded frames
  → VLM / downstream processing
```

The key question is not “which protocol is best in theory?” but:

```text
What output can this actual camera expose easily,
and how can I convert that stream into frames for Python/VLM inference?
```

### IP camera

An IP camera is a camera connected to a network, usually via Wi-Fi or Ethernet, that has its own IP address.

Example local IPs:

```text
192.168.1.20
192.168.0.50
10.0.0.12
```

An IP camera is basically:

```text
camera + encoder + small network server
```

Many IP cameras expose streams such as:

```text
rtsp://192.168.1.50:554/live
```

Unlike a USB webcam, which sends frames through the OS camera driver, an IP camera sends video over the network.

```text
USB webcam:
camera → USB → OS camera driver → OpenCV

IP camera:
camera → network → RTSP/HTTP/etc. → OpenCV/FFmpeg/VLC
```

### RTSP

RTSP stands for Real-Time Streaming Protocol.

It is commonly used by IP cameras and security cameras. RTSP is mostly a control protocol: the client asks the camera/server to describe, set up, and play a stream. The actual media is often sent using RTP over UDP or TCP.

Typical use:

```text
OpenCV/VLC/FFmpeg client
  → connects to RTSP server
  → receives video stream
  → decodes frames
```

Example:

```text
rtsp://192.168.1.50:554/live
```

RTSP is usually good for local prototypes where a server or script needs to read frames from a camera. This is what `preview_rtmp.py`, `frame_sample.py`, and `live_vlm_qa.py` use when `STREAM_PROTOCOL=rtsp`.

### RTMP

RTMP stands for Real-Time Messaging Protocol.

It is commonly used for livestream ingestion, for example:

```text
OBS → RTMP → YouTube/Twitch/media server
```

RTMP is usually push-based: an encoder pushes video to a streaming server.

Example:

```text
rtmp://server/live/stream_key
```

RTMP is useful for livestream infrastructure (e.g. GoPro relay to a local media server). It is usually less natural than RTSP or WebRTC for an interactive wearable VLM prototype.

### WebRTC

WebRTC is designed for real-time audio/video communication, such as browser video calls.

It is usually used for:

```text
browser camera
phone camera
real-time video chat
low-latency streaming
interactive apps
```

WebRTC can be peer-to-peer, but production systems often use servers such as SFUs, media relays, or TURN servers.

```text
Simple idea:
device/browser ↔ device/browser

Production idea:
device/browser → media server/SFU/TURN → client/server
```

WebRTC is more complex than RTSP because it involves signaling, NAT traversal, encryption, and real-time network adaptation. It is useful when low latency and browser/mobile integration matter.

### Protocol vs codec

Protocol and codec are different.

```text
Protocol = how video is transported
Codec = how video is compressed
```

Examples:

```text
Protocols:
RTSP, RTMP, WebRTC, HTTP, HLS

Codecs:
H.264, H.265, MJPEG, VP8, VP9, AV1
```

A camera may support RTSP but stream H.265, so the receiver must also support decoding H.265.

### RTSP URL and RTMP URL

An RTSP or RTMP URL is an address that tells the client where to connect and what stream to request.

Example RTSP URL:

```text
rtsp://192.168.1.50:554/live
```

Breakdown:

```text
rtsp://        protocol
192.168.1.50  camera/server IP
554           port
/live         stream path
```

Example RTMP URL:

```text
rtmp://example.com/live/stream_key
```

Modern browsers usually cannot directly play `rtsp://` or `rtmp://` URLs. These URLs usually work in tools like VLC, FFmpeg, OpenCV, or media servers, not directly in Chrome/Safari.

For browsers, the stream often needs to be converted to WebRTC, HLS, or another browser-supported format.

### VLC

VLC is a media player and media utility. It can play local files and network streams.

In this project context, VLC is useful for quickly checking whether a camera stream works.

Example use:

```text
VLC → Open Network Stream → paste RTSP URL
```

VLC is not the protocol. VLC is an application that can speak many protocols and decode many codecs.

```text
RTSP = protocol
VLC = app/tool that can read RTSP
```

### OBS

OBS, or Open Broadcaster Software, is a desktop app for recording and livestreaming.

It can:

```text
capture camera/screen
mix audio
encode video
record locally
stream to RTMP/SRT/etc.
```

Typical use:

```text
camera/screen → OBS → RTMP server / YouTube / Twitch
```

OBS is useful for simulating a video source or pushing a livestream to a media server.

### FFmpeg

FFmpeg is a command-line media tool and library suite.

It can:

```text
read streams
decode/encode video
convert formats
extract frames
record streams
resize videos
push streams to servers
debug media pipelines
```

Examples:

```bash
ffmpeg -i rtsp://192.168.1.50:554/live output.mp4
```

Extract one frame per second:

```bash
ffmpeg -i rtsp://192.168.1.50:554/live -vf fps=1 frame_%04d.jpg
```

For ML/VLM projects, FFmpeg is useful because models usually need image frames, not raw streaming protocols. OpenCV uses FFmpeg/GStreamer under the hood for RTSP/RTMP capture.

### OpenCV as a client

When using OpenCV with an RTSP URL:

```python
import cv2

cap = cv2.VideoCapture("rtsp://192.168.1.50:554/live")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # frame is a decoded image as a NumPy array
```

Python/OpenCV acts as an RTSP client.

The RTSP server may be:

```text
IP camera
media server
another computer running FFmpeg/GStreamer/MediaMTX
```

The flow is:

```text
Python code
  → OpenCV API
  → FFmpeg/GStreamer backend
  → RTSP/RTP stream
  → decoded frame as NumPy array
```

For a USB webcam, OpenCV is not an RTSP client:

```python
cap = cv2.VideoCapture(0)
```

That means:

```text
USB camera → OS camera driver → OpenCV → frame
```

### NAT issue

NAT stands for Network Address Translation.

Devices inside a home/company network usually have private IP addresses:

```text
Mac:    192.168.1.10
Phone:  192.168.1.11
Camera: 192.168.1.12
```

The outside internet cannot directly access those private IPs.

This is why a stream like:

```text
rtsp://192.168.1.12:554/live
```

may work on the same Wi-Fi but fail from outside the network.

Common solutions:

```text
same LAN/Wi-Fi
port forwarding
VPN/Tailscale
cloud relay
TURN server for WebRTC
media server with public IP
```

For local prototypes, NAT is usually not a big issue. For real wearable/cloud systems, NAT becomes important.

### Can any camera use any protocol?

No.

A camera can use a protocol only if:

1. the camera firmware supports that protocol, or
2. intermediate software converts the camera output into that protocol.

Example native RTSP:

```text
IP camera → RTSP stream → OpenCV/VLC/FFmpeg
```

Example converted RTSP:

```text
USB webcam → Mac → FFmpeg/MediaMTX → RTSP stream
```

In the second case, the camera itself does not support RTSP. The Mac is converting the camera frames into an RTSP stream.

### Practical development path

For a first wearable/VLM prototype, prioritize practical frame access over deep protocol knowledge.

Recommended order:

```text
1. Confirm camera stream with VLC.
2. Read/record stream with FFmpeg.
3. Read frames with Python/OpenCV.
4. Sample frames every N seconds.
5. Add timestamps and metadata.
6. Send sampled frames to VLM.
7. Store responses and visual memory.
8. Later, use WebRTC if browser/mobile low-latency interaction is needed.
```

Steps 1–6 map directly to the scripts in this folder (`preview_rtmp.py` → `frame_sample.py` → `live_vlm_qa.py`).

Initial prototype stack:

```text
Camera / GoPro / IP camera / webcam
  → RTSP or USB capture
  → OpenCV or FFmpeg
  → sampled frames
  → VLM inference
  → timestamped memory store
  → later retrieval/QA
```

More product-like stack:

```text
phone/wearable camera
  → WebRTC
  → server/media pipeline
  → VLM inference
  → memory system
  → real-time user interaction
```

### What to learn deeply vs shallowly

For this project, it is not necessary to implement RTSP, RTMP, or WebRTC from scratch.

Important to know:

```text
what protocol the camera exposes
how to test the stream
how to decode it into frames
how to handle latency and dropped frames
how to deal with NAT/firewall issues
how to convert between protocols if needed
```

Not necessary at first:

```text
packet-level RTSP/RTP details
full WebRTC internals
codec implementation details
custom media server implementation
```

The main engineering goal is:

```text
camera stream → reliable frames → VLM → memory system
```
