# camera_test

Lightweight scripts for testing live camera streams before using the full pipeline in `vlm_smoke` and `memory_log`.

All scripts support **RTMP** (e.g. GoPro relay), **RTSP** (e.g. Tapo IP camera), **WebRTC** (WHEP, e.g. MediaMTX), **webcam**, and **local video files**.

## Files

| File | Purpose |
|------|---------|
| `stream_config.py` | Shared helpers: resolve source type/URL from env or CLI, open `cv2.VideoCapture` |
| `preview_stream.py` | Live preview only — no saving, no VLM |
| `frame_sample.py` | Live preview + save a JPEG every 2 seconds to `sampled_frames/` |
| `live_vlm_qa.py` | Background capture + ask a VLM questions about recent frames in a REPL |

## Setup

```bash
cd camera_test
cp .env.example .env
# Edit .env — stream URL, API key for live_vlm_qa, etc.
uv sync
```

Scripts load `.env` from the current directory or a parent directory.

## How to run

From the `camera_test` directory:

```bash
uv run camera-preview
uv run camera-sample
uv run camera-vlm
```

Or run modules directly:

```bash
uv run python preview_stream.py
uv run python frame_sample.py
uv run python live_vlm_qa.py
```

## Configuration

Set these in `camera_test/.env` (see [.env.example](.env.example)):

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_SOURCE_TYPE` | `rtmp` | `rtmp`, `rtsp`, `webrtc`, `webcam`, or `video` |
| `RTMP_URL` | `rtmp://localhost:1935/live/gopro` | Used when source type is `rtmp` |
| `RTSP_URL` | `rtsp://localhost:8554/live/gopro` | Used when source type is `rtsp` |
| `WEBRTC_URL` | `http://localhost:8889/live/whep` | WHEP endpoint when source type is `webrtc` |
| `WEBRTC_ICE_SERVERS` | `stun:stun.l.google.com:19302` | Optional comma-separated STUN/TURN URLs |
| `WEBRTC_OPEN_TIMEOUT_SEC` | `15` | Seconds to wait for the first WebRTC frame |
| `WEBCAM_INDEX` | `0` | Webcam device index |
| `VIDEO_PATH` | — | Required when `FRAME_SOURCE_TYPE=video` |

CLI flags override env:

- `--source-type rtmp|rtsp|webrtc|webcam|video`
- `--url <stream URL, WHEP URL, or video path>`
- `--protocol rtmp|rtsp` — deprecated alias for `--source-type`

### GoPro (RTMP relay)

```env
FRAME_SOURCE_TYPE=rtmp
RTMP_URL=rtmp://localhost:1935/live/gopro
```

### Tapo camera (RTSP)

Create a **Camera Account** in the Tapo app first: **Device Settings → Advanced Settings → Camera Account**. This is separate from your Tapo app login.

```env
FRAME_SOURCE_TYPE=rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
RTSP_TRANSPORT=tcp
```

- `stream1` — higher quality
- `stream2` — lower bandwidth (good default for sampling and VLM)
- If the password contains special characters (`@`, `!`, `#`, …), URL-encode them (e.g. `!` → `%21`)
- **RTSP direct uses `RTSP_URL`**, not `WEBRTC_URL` — WebRTC/WHEP is only for the MediaMTX relay path
- OpenCV reads Tapo RTSP best with **`RTSP_TRANSPORT=tcp`** (MediaMTX can use UDP; OpenCV/FFmpeg often cannot)

### Tapo → WebRTC via MediaMTX

Tapo cameras speak **RTSP**, not WebRTC. To use `FRAME_SOURCE_TYPE=webrtc`, run [MediaMTX](https://github.com/bluenviron/mediamtx) as a relay: it pulls RTSP from the camera and exposes WebRTC/WHEP on the same path name.

1. Confirm RTSP works in VLC first (see above).
2. Create `mediamtx.yml` (start MediaMTX with an explicit path: `mediamtx /path/to/mediamtx.yml`). Check the startup log for `configuration loaded from ...` — if MediaMTX loads a different file than the one you edited, you will see `path 'tapo' is not configured`.
3. Open the browser player at `http://localhost:8889/tapo/` to verify.
4. Point `camera_test` at the WHEP endpoint:

```yaml
# mediamtx.yml — Tapo RTSP in, WebRTC/WHEP out
paths:
  tapo:
    source: rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
    sourceOnDemand: no
    rtspTransport: udp
```

```env
FRAME_SOURCE_TYPE=webrtc
WEBRTC_URL=http://localhost:8889/tapo/whep
```

```bash
uv run camera-preview --source-type webrtc
```

For lowest latency on the same LAN, prefer `rtspTransport: udp` and `sourceOnDemand: no` (see [Latency](#latency-rtsp-vs-webrtc) below). If the stream drops or stutters on Wi‑Fi, switch to `rtspTransport: tcp`.

### WebRTC (WHEP)

WebRTC is consumed via **WHEP** (WebRTC-HTTP Egress Protocol). Point `WEBRTC_URL` at your media server's WHEP endpoint:

```env
FRAME_SOURCE_TYPE=webrtc
WEBRTC_URL=http://localhost:8889/live/whep
```

MediaMTX can also ingest RTMP and expose WebRTC/WHEP on the same path. Publish to path `live`, then:

```bash
uv run camera-preview --source-type webrtc
```

If the browser or Python client is on a different network, configure STUN/TURN via `WEBRTC_ICE_SERVERS`.

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
uv run camera-preview
```

Tapo example:

```bash
uv run camera-preview \
  --source-type rtsp \
  --url 'rtsp://camera_user:camera_pass@192.168.1.50:554/stream2'
```

Webcam:

```bash
uv run camera-preview --source-type webcam
```

Press **`q`** in the preview window to quit.

### 2. Sample frames to disk

Shows a live preview and saves one JPEG every 2 seconds under `camera_test/sampled_frames/`.

```bash
uv run camera-sample
```

With explicit RTSP URL:

```bash
uv run camera-sample \
  --source-type rtsp \
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
uv run camera-vlm --source-type rtsp
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
uv run camera-vlm --source-type rtsp
```

Example questions:

- What objects are visible?
- Is there a person in front of the camera?
- What changed in the last few seconds?

Type **`q`**, **`quit`**, or **`exit`** to stop.

## Latency (RTSP vs WebRTC)

A half-second to one second behind “real time” is **normal** when video goes through RTSP → MediaMTX → WebRTC. For local Python on the same Wi‑Fi, **RTSP direct** is usually the fastest option. Use WebRTC when you want a browser player, multiple clients, or a path toward remote viewing.

| Path | Typical delay (same LAN) | Best for |
|------|--------------------------|----------|
| Tapo → RTSP → `camera_test` | ~200–500 ms | Local preview, VLM, frame sampling |
| Tapo → RTSP → MediaMTX → WebRTC → `camera_test` | ~500 ms–1.5 s (tunable) | Browser preview, relay, WHEP clients |

Even with `rtspTransport: udp` and `sourceOnDemand: no`, the **WebRTC path usually still has more delay than RTSP direct** on the same LAN. Tuning makes the WebRTC chain feel “super fast” compared to its untuned self — it does not make WebRTC faster than skipping MediaMTX altogether.

### What “RTSP → WebRTC” means (MediaMTX as an adapter)

MediaMTX is a **media relay** (think protocol adapter or router for video). It does **not** replace the camera’s video; it **repackages** the same H.264 stream for a different delivery protocol.

```text
Tapo camera
  │  H.264 video inside RTSP (camera’s native output)
  ▼
MediaMTX  ── pulls RTSP, unwraps packets, forwards as WebRTC/RTP
  │  same video, different wire format + session rules
  ▼
Browser or aiortc (WHEP)  ── WebRTC client
```

Important details:

- **Codec usually stays the same** (H.264). MediaMTX typically **forwards** without re-encoding when the codec is compatible. It is not “converting MP4 to something else” — it is changing **how packets are transported and negotiated**.
- **RTSP and WebRTC are transport/session protocols**, not speed ratings. “Slow” vs “fast” depends on **how many hops**, **TCP vs UDP**, **on-demand connects**, and **buffer sizes** — not the label on the protocol.
- MediaMTX sits in the **middle**: one leg is RTSP (camera → server), the other is WebRTC (server → client). You always pay for **both legs** plus whatever buffering each leg adds.

So yes — **adapter/relay** is the right mental model.

### RTSP is not inherently “slow”

It is easy to hear “RTSP is old” or “WebRTC is real-time” and assume RTSP direct must be worse. For a Tapo on the same Wi‑Fi, **RTSP is often the lowest-latency way to get frames into Python**:

| Myth | Reality |
|------|---------|
| “RTSP is slow” | RTSP is a **pull** protocol designed for IP cameras on LANs. Latency is often **200–500 ms** end-to-end when you connect directly. |
| “WebRTC is always faster” | WebRTC is optimized for **browsers, NAT traversal, and smooth calls** — it intentionally keeps a **jitter buffer** (~200–400 ms). That helps stability, not minimum delay. |
| “MediaMTX makes it faster” | MediaMTX **adds a hop**. It enables WebRTC/browser access; it does not remove the camera’s RTSP leg. |

What actually made your stream feel slow before tuning was mostly the **RTSP leg into MediaMTX** (`sourceOnDemand: yes`, TCP retransmits) and the **WebRTC jitter buffer** — not RTSP as a protocol being useless.

### Two legs, two different bottlenecks

Split the path by **who talks which protocol**:

```text
Leg 1 — Tapo → MediaMTX (RTSP)
  Camera encodes H.264 → RTSP/RTP packets → MediaMTX ingests

Leg 2 — MediaMTX → aiortc (WebRTC)
  MediaMTX repackages → WebRTC/RTP + ICE/DTLS → aiortc decodes → OpenCV
```

| Leg | What you tuned | What it fixed |
|-----|----------------|---------------|
| **Leg 1 (RTSP into MediaMTX)** | `rtspTransport: udp`, `sourceOnDemand: no` | Removed TCP retry stalls and “cold start” RTSP connects. This is where most of your speed-up came from. |
| **Leg 2 (WebRTC to aiortc)** | (defaults) | WebRTC still applies jitter buffering and pacing. `camera_test` only keeps the **latest** decoded frame — it does not add a deep queue, but it cannot remove WebRTC’s upstream buffer. |

**There is no special “gain from MediaMTX to aiortc” that beats RTSP direct** — that segment is an extra protocol conversion with its own buffering rules. The win from tuning is making **Leg 1** and session startup snappy so **Leg 2** receives fresh packets immediately.

**RTSP direct** removes Leg 2 entirely:

```text
Tapo → RTSP → OpenCV/ffmpeg in camera_test   (one protocol, one client)
```

That is why `FRAME_SOURCE_TYPE=rtsp` is still the fastest option for local VLM work, even after a well-tuned WebRTC relay.

### Why the WebRTC chain can still feel slower

Each hop adds a small buffer on purpose:

```text
Tapo (H.264 encode)
  → RTSP transport          ← Leg 1 (tune with udp + sourceOnDemand: no)
  → MediaMTX (demux / forward)
  → WebRTC (jitter buffer)  ← Leg 2 (fixed cost for smooth playback)
  → aiortc decode → OpenCV display
```

WebRTC trades a bit of latency for smooth playback — it holds a short jitter buffer so uneven packet arrival does not cause stutter. That alone is often **200–400 ms**. `camera_test` does not add a deep queue: `WebRTCCapture` keeps only the latest frame, similar to OpenCV `BUFFERSIZE=1` for RTSP.

### Why `rtspTransport: udp` is faster (on a good LAN)

RTSP can run over **UDP** or **TCP**:

| Transport | Behavior | Latency | Reliability |
|-----------|----------|---------|-------------|
| **UDP** | Sends packets without waiting for acknowledgements | Lower — newer frames arrive sooner | Lost packets are skipped; fine for live preview on stable Wi‑Fi |
| **TCP** | Retransmits lost packets | Higher — the stream can pause while waiting for retries | Better on noisy Wi‑Fi or marginal networks |

On the same LAN with a Tapo camera, **UDP** often feels much snappier because MediaMTX receives the newest packets immediately instead of waiting for TCP to catch up. If you see tearing, frozen frames, or frequent reconnects, switch back to `rtspTransport: tcp`.

### Why `sourceOnDemand: no` is faster

`sourceOnDemand` controls when MediaMTX connects to the camera's RTSP URL:

| Setting | What MediaMTX does | Latency impact |
|---------|-------------------|----------------|
| `sourceOnDemand: yes` | Connects to Tapo only when a viewer opens the stream; disconnects when idle | **Slower** — each new viewer triggers RTSP handshake, authentication, and waiting for the next keyframe (GOP). Easy to see 0.5–2 s stalls when opening the browser or restarting preview. |
| `sourceOnDemand: no` | Keeps pulling RTSP from Tapo continuously | **Faster** — the stream stays “warm”. WebRTC/WHEP clients get frames immediately because MediaMTX already has live video buffered. |

The tradeoff: `sourceOnDemand: no` uses bandwidth even when nobody is watching (Tapo → MediaMTX pull runs 24/7). For a dev machine on the same network, that is usually acceptable.

### Recommended MediaMTX settings for low-latency Tapo WebRTC

```yaml
paths:
  tapo:
    source: rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
    sourceOnDemand: no      # keep RTSP pull alive
    rtspTransport: udp      # lower latency on stable LAN; use tcp if unstable
```

Also use **`stream2`** (substream) unless you need full resolution — lower bitrate often means less encode delay.

### When RTSP direct is enough

For `camera-preview`, `camera-sample`, and `camera-vlm` on the same machine, skip MediaMTX entirely:

```env
FRAME_SOURCE_TYPE=rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
```

Keep the MediaMTX + WebRTC path when you need `http://localhost:8889/tapo/` in a browser or plan to add remote clients later.

## Troubleshooting

**Stream won't open**

1. Test the same URL in VLC (*Media → Open Network Stream*).
2. Confirm camera and computer are on the same LAN.
3. For Tapo, verify the Camera Account credentials (not your Tapo app password).
4. Try `stream2` instead of `stream1`.

**`path 'tapo' is not configured` (MediaMTX)**

MediaMTX is not loading the YAML file you edited. Check the startup log for `configuration loaded from ...`, ensure `tapo:` is nested under `paths:` with correct indentation, and start with an explicit path: `mediamtx /path/to/mediamtx.yml`.

**`no stream is available on path 'live'` (WebRTC / WHEP)**

`WEBRTC_URL` points at path **`live`**, but MediaMTX has no stream there. The path name in the URL must match the key under `paths:` in `mediamtx.yml`:

| `mediamtx.yml` | Browser test | `WEBRTC_URL` |
|----------------|--------------|--------------|
| `tapo:` | `http://localhost:8889/tapo/` | `http://localhost:8889/tapo/whep` |
| `live:` | `http://localhost:8889/live/` | `http://localhost:8889/live/whep` |

For Tapo you likely configured **`tapo`**, not **`live`**. The default `.env.example` used `live` for GoPro/RTMP examples — update `.env`:

```env
FRAME_SOURCE_TYPE=webrtc
WEBRTC_URL=http://localhost:8889/tapo/whep
```

If the browser URL plays video but WHEP still fails, MediaMTX may not be pulling RTSP yet — set `sourceOnDemand: no` or open the browser player once to start the pull.

**WebRTC preview is laggy (~0.5–1 s)**

Expected for RTSP → WebRTC. Try `sourceOnDemand: no` and `rtspTransport: udp` in `mediamtx.yml`, or use RTSP direct for local Python (see [Latency](#latency-rtsp-vs-webrtc)).

**`backend is generally available but can't be used to capture by name` (RTSP)**

This OpenCV warning is **misleading**. FFmpeg failed to open the RTSP stream (connection, auth, or transport) — not because the backend is wrong.

Common fixes for Tapo:

1. Use **RTSP direct** settings, not WebRTC:

```env
FRAME_SOURCE_TYPE=rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
RTSP_TRANSPORT=tcp
```

2. Confirm the same URL plays in VLC first.
3. URL-encode special characters in the camera password.
4. `camera_test` tries `RTSP_TRANSPORT` first, then the other transport automatically. To force one mode:

```bash
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
uv run camera-preview --source-type rtsp --url 'rtsp://...'
```

Use `rtsp_transport;tcp` — **not** `rtsp_flags;tcp` (that invalid option causes the same error).

**`Failed to read frame` loops**

The camera may have dropped the connection. Stop with `q` and restart. For RTSP direct, TCP transport often helps. For MediaMTX → Tapo, try `rtspTransport: tcp` if UDP is unstable.

## Suggested workflow

```text
preview_stream.py → confirm source works
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

RTSP is usually good for local prototypes where a server or script needs to read frames from a camera. This is what `preview_stream.py`, `frame_sample.py`, and `live_vlm_qa.py` use when `FRAME_SOURCE_TYPE=rtsp`. For WebRTC, set `FRAME_SOURCE_TYPE=webrtc` and a WHEP URL in `WEBRTC_URL`.

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

Steps 1–6 map directly to the scripts in this folder (`preview_stream.py` → `frame_sample.py` → `live_vlm_qa.py`).

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
