# camera_test

Lightweight scripts for testing live camera streams before using the full pipeline in `vlm_smoke` and `memory_log`.

Supported setups:


| `CAMERA_SOURCE` | Camera         | Path                             |
| --------------- | -------------- | -------------------------------- |
| `tapo-rtsp`     | Tapo IP camera | RTSP direct → OpenCV             |
| `tapo-webrtc`   | Tapo IP camera | RTSP → MediaMTX → WHEP           |
| `phone-webrtc`  | Smartphone     | WebRTC publish → MediaMTX → WHEP |


## Files


| File                         | Purpose                                                                |
| ---------------------------- | ---------------------------------------------------------------------- |
| `stream_config.py`           | Shared helpers: resolve camera preset from env or CLI, open capture    |
| `preview_stream.py`          | Live preview only — no saving, no VLM                                  |
| `frame_sample.py`            | Live preview + save a JPEG every 2 seconds to `sampled_frames/`        |
| `live_vlm_qa.py`             | Background capture + ask a VLM questions about recent frames in a REPL |
| `whep_client.py`             | Core WHEP/ICE logic (aiortc)                                           |
| `whep_worker.py`             | Subprocess worker — streams frames to parent (no OpenCV in child)      |
| `whep_probe.py`              | Diagnose WHEP OPTIONS/POST/ICE (`camera-whep-probe`)                   |
| `mediamtx-tapo.example.yml`  | Template MediaMTX config for Tapo → WebRTC/WHEP (copy to `mediamtx-tapo.yml`)   |
| `mediamtx-phone.example.yml` | Template MediaMTX config for phone → WebRTC/WHEP (copy to `mediamtx-phone.yml`) |
| `mediamtx-tapo.yml`          | Your local Tapo MediaMTX config (gitignored — create from example)              |
| `mediamtx-phone.yml`         | Your local phone MediaMTX config (gitignored — create from example)             |


## Setup

```bash
cd camera_test
cp .env.example .env
# Edit .env — camera source, stream URL, API key for live_vlm_qa, etc.
uv sync
```

Scripts load `.env` from the current directory or a parent directory.

### MediaMTX config (Tapo WebRTC and phone WebRTC)

MediaMTX does not read the `.example.yml` files directly. Copy each template once, edit your local copy (RTSP URL, LAN IP, certs), then launch MediaMTX with that file:

```bash
cd camera_test
cp mediamtx-tapo.example.yml mediamtx-tapo.yml    # Tapo RTSP → WebRTC/WHEP
cp mediamtx-phone.example.yml mediamtx-phone.yml  # phone WebRTC publish → WHEP/RTSP
# Edit mediamtx-tapo.yml (camera RTSP URL) and/or mediamtx-phone.yml (LAN IP, certs)
```

`mediamtx-tapo.yml` and `mediamtx-phone.yml` are **gitignored** so camera URLs and TLS paths stay local.

```bash
mediamtx mediamtx-tapo.yml    # Tapo relay
mediamtx mediamtx-phone.yml   # phone publish
```

Use only the config you need — Tapo WebRTC requires `mediamtx-tapo.yml`; phone-as-camera requires `mediamtx-phone.yml` (often with mkcert TLS; see [§3 Smartphone with WebRTC](#3-smartphone-with-webrtc)).

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


| Variable                  | Default                        | Description                                                                      |
| ------------------------- | ------------------------------ | -------------------------------------------------------------------------------- |
| `CAMERA_SOURCE`           | `tapo-rtsp`                    | `tapo-rtsp`, `tapo-webrtc`, or `phone-webrtc`                                    |
| `RTSP_URL`                | Tapo placeholder               | Used when `CAMERA_SOURCE=tapo-rtsp`                                              |
| `RTSP_TRANSPORT`          | `tcp`                          | `tcp` or `udp` for OpenCV/FFmpeg RTSP                                            |
| `WEBRTC_URL`              | path-specific default          | WHEP endpoint for WebRTC presets                                                 |
| `WEBRTC_ICE_SERVERS`      | `stun:stun.l.google.com:19302` | Optional comma-separated STUN/TURN URLs                                          |
| `WEBRTC_OPEN_TIMEOUT_SEC` | `30`                           | Seconds to wait for WHEP connect + first frame                                   |
| `WEBRTC_IPC`              | `subprocess`                   | `subprocess` (default) runs aiortc in a child process; `inprocess` for debugging |


CLI flags override env:

- `--camera tapo-rtsp|tapo-webrtc|phone-webrtc`
- `--url <RTSP URL or WHEP URL>`

### 1. Tapo with RTSP

Before using `camera_test`, confirm RTSP works on your LAN:

1. **Find the camera IP** in the Tapo app: **Device Settings → Network / Networking** (wording varies by model).
2. **Create a Camera Account** (separate from your Tapo app login): **Device Settings → Advanced Settings → Camera Account**. Use that username and password with the IP from step 1 to build the RTSP URL:
  ```text
   rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
  ```
3. **Test in VLC** on your Mac: *Media → Open Network Stream…*, paste the URL, and confirm video plays.

Then set `camera_test/.env`:

```env
CAMERA_SOURCE=tapo-rtsp
RTSP_URL=rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
RTSP_TRANSPORT=tcp
```

- `stream1` — higher quality
- `stream2` — lower bandwidth (good default for sampling and VLM)
- OpenCV reads Tapo RTSP best with `**RTSP_TRANSPORT=tcp**`

```bash
uv run camera-preview --camera tapo-rtsp
```

### 2. Tapo with WebRTC

Tapo cameras speak **RTSP**, not WebRTC. To use `CAMERA_SOURCE=tapo-webrtc`, run [MediaMTX](https://github.com/bluenviron/mediamtx) as a relay: it pulls RTSP from the camera and exposes WebRTC/WHEP.

1. Confirm RTSP works in VLC first (see above).
2. Copy and edit the Tapo config if you have not already (see [MediaMTX config](#mediamtx-config-tapo-webrtc-and-phone-webrtc)): set your camera RTSP URL under `paths.tapo.source` in `mediamtx-tapo.yml`.
3. Start MediaMTX:

```bash
cd camera_test
mediamtx mediamtx-tapo.yml
```

4. Open the browser player at `http://localhost:8889/tapo/` to verify.
5. Point `camera_test` at the WHEP endpoint:

```env
CAMERA_SOURCE=tapo-webrtc
WEBRTC_URL=http://localhost:8889/tapo/whep
```

```bash
uv run camera-whep-probe --url http://localhost:8889/tapo/whep
uv run camera-preview --camera tapo-webrtc
```

Example template: [`mediamtx-tapo.example.yml`](mediamtx-tapo.example.yml) → your local [`mediamtx-tapo.yml`](mediamtx-tapo.yml). For lowest latency on the same LAN, prefer `rtspTransport: udp` and `sourceOnDemand: no`. If the stream drops on Wi‑Fi, switch to `rtspTransport: tcp`.

For local Python on the same Wi‑Fi, **RTSP direct** (`tapo-rtsp`) is usually faster than RTSP → MediaMTX → WebRTC.

### 3. Smartphone with WebRTC

Smartphones do not expose RTSP. Use MediaMTX to accept a **WebRTC publish** from the phone. Python reads the stream via MediaMTX's **RTSP relay** (recommended) or WHEP (experimental with TLS).

Unlike Tapo, the stream exists **only while the phone is actively publishing**.

**Important:** the phone publish page needs **HTTPS**. Browsers only allow camera/microphone access on secure origins (`https://`, or `http://localhost` on the same device). Opening `http://192.168.x.x/...` from your phone is neither — MediaMTX shows *"can't access webcams or microphones. Make sure that WebRTC encryption is enabled"* until you enable `webrtcEncryption` and use `https://` URLs.

#### Phone WebRTC TLS (one-time setup)

Easiest on a dev Mac: [mkcert](https://github.com/FiloSottile/mkcert) (locally trusted certs).

First, find your Mac's **LAN IP** on Wi‑Fi (use this in the cert, `mediamtx-phone.yml`, and the phone publish URL — not a placeholder like `192.168.1.100`):

```bash
ipconfig getifaddr en0
```

If that prints nothing (Ethernet instead of Wi‑Fi), try `ipconfig getifaddr en1`. The phone must use this address on the **same Wi‑Fi** as your Mac.

```bash
brew install mkcert
mkcert -install
cd camera_test
mkdir -p mediamtx-certs
# Replace YOUR_MAC_IP with the address from ipconfig above (e.g. 192.168.11.51)
mkcert -key-file mediamtx-certs/server.key -cert-file mediamtx-certs/server.crt \
  localhost 127.0.0.1 YOUR_MAC_IP
```

Install the mkcert root CA on your phone too (mkcert prints how; on iOS: Settings → General → About → Certificate Trust Settings).

Copy [`mediamtx-phone.example.yml`](mediamtx-phone.example.yml) to `mediamtx-phone.yml` if you have not already (see [MediaMTX config](#mediamtx-config-tapo-webrtc-and-phone-webrtc)). That file points at `mediamtx-certs/` and sets `webrtcEncryption: yes`. Add your LAN IP under `webrtcAdditionalHosts`.

#### Run

1. Start MediaMTX:

```bash
cd camera_test
mediamtx mediamtx-phone.yml
```

2. Confirm your Mac's **LAN IP** (same value as in the cert and `webrtcAdditionalHosts`):

```bash
ipconfig getifaddr en0
```

Use that address in the phone URL below (not `localhost`, not a README example IP). The phone must be on the **same Wi‑Fi**.
3. On the phone, open (**https**, not http):

```text
https://YOUR_MAC_IP:8889/phone/publish
```

Example: if `ipconfig getifaddr en0` prints `192.168.11.51`, open `https://192.168.11.51:8889/phone/publish`.

#### Publish page settings (before you tap Publish)

MediaMTX shows a built-in form on that page. The phone's browser **encodes** the stream with these values; MediaMTX **relays** it (no server-side transcode). Pick settings for your goal — `camera-preview`, `camera-sample`, and `camera-vlm` only need clear video at modest resolution.

| Field | What it does | How to choose |
| ----- | ------------ | ------------- |
| **Video device** | Camera, `screen` (screen share), or `none` | Use the back camera for room/scene capture; `none` if you only need audio. |
| **Video codec** | How video is compressed (VP8, VP9, AV1, H264, H265 — list depends on phone/browser) | **H264** or **VP8** for best compatibility with Python RTSP relay and Mac preview. AV1/VP9/H265 can look sharper but use more phone CPU/battery and may not play everywhere. |
| **Video bitrate (kbps)** | Target encoder bitrate in kilobits per second | **1500–2500** at 720p is enough for VLM and preview. Lower (800–1200) if Wi‑Fi is weak or the phone gets hot; raise only if the picture looks blocky. |
| **Video framerate (ideal)** | Requested FPS (`ideal` — the phone may pick the nearest supported rate) | **15–24** for `camera-vlm` / sampling (those tools do not need 30 fps). **24–30** for smoother `camera-preview`. Higher fps costs bandwidth and battery. |
| **Video width / height (ideal)** | Requested resolution (`ideal` — the phone may snap to a nearby mode, e.g. 1280×720) | **1280×720** is a good default for scene understanding. Use **960×540** to reduce lag and Wi‑Fi load; use **1920×1080** only if you need fine detail. |
| **Audio device** | Microphone or `none` | **`none`** if you only care about video (simplest). Enable the mic if you will speak to the VLM or want ambient sound. |
| **Audio codec** | Audio compression (usually **Opus**; may also offer G722, PCMU, PCMA) | **Opus** — best quality at low bitrate. Others are legacy/telephony. |
| **Audio bitrate (kbps)** | Target audio bitrate | **32–64** for speech; **64–128** for room ambience. Irrelevant if audio device is `none`. |
| **Optimize for voice** | When checked: echo cancellation, noise suppression, auto gain (speech-optimized mic processing) | **On** if you talk to `camera-vlm` or want clear speech. **Off** for raw/ambient audio (music, room tone); can sound worse for voice. |

**Suggested presets**

| Goal | Video | Audio |
| ---- | ----- | ----- |
| **Preview / frame sampling** (`camera-preview`, `camera-sample`) | H264 or VP8, **2000** kbps, **24** fps, **1280×720** | device **`none`** |
| **VLM scene Q&A** (`camera-vlm`) | H264 or VP8, **1500–2000** kbps, **15–20** fps, **1280×720** (720p is plenty; the VLM samples ~1 fps) | Opus **64** kbps, voice **on** only if you ask questions by speech |
| **Low Wi‑Fi / reduce heat** | Same codec, **1000** kbps, **15** fps, **960×540** | **`none`** |

The form saves choices in the page URL (query parameters), so you can bookmark a tuned link. After adjusting, tap **Publish** and allow camera (and microphone, if enabled).

(OBS/WHIP without this form: `https://YOUR_MAC_IP:8889/phone/whip` — set codec/bitrate in OBS instead.)

4. Verify on your Mac: `https://localhost:8889/phone/`
5. Point `camera_test` at the RTSP relay (path name must match `mediamtx-phone.yml`):

```env
CAMERA_SOURCE=phone-webrtc
PHONE_STREAM_URL=rtsp://127.0.0.1:8554/phone
```

```bash
uv run camera-preview --camera phone-webrtc
```

Optional WHEP probe (often fails with `webrtcEncryption` + aiortc — see Troubleshooting):

```bash
uv run camera-whep-probe --url https://localhost:8889/phone/whep
```

#### Phone GPS sidecar for memory_log

When using [`memory_log`](../memory_log/) with `CAMERA_SOURCE=phone-webrtc`, you can attach live GPS without a VLM call:

1. Generate TLS certs (same mkcert step as above) if you have not already.
2. In `memory_log/.env`:

```env
LOCATION_SERVER_ENABLED=true
LOCATION_SERVER_PORT=8765
LOCATION_SERVER_CERT=../camera_test/mediamtx-certs/server.crt
LOCATION_SERVER_KEY=../camera_test/mediamtx-certs/server.key
PHONE_LOCATION_LABEL=walking
```

3. Start `memory_log` (`uv run python -m src.main` from `memory_log/`).
4. On the phone, while publishing at `https://YOUR_MAC_IP:8889/phone/publish`, also open:

```text
https://YOUR_MAC_IP:8765/
```

The page at [`phone_location.html`](phone_location.html) uses the browser Geolocation API and POSTs lat/lon to memory_log. Keep it open in a background tab. Each saved memory then gets `location.source=phone_gps` when a fix is fresh (default max age 120s).

### VLM settings (`camera-vlm` only)

Switch between cloud OpenAI and local Ollama in `.env`:

**OpenAI (API key required)**

```env
VLM_PROVIDER=openai
VLM_MODEL=gpt-5.5
OPENAI_API_KEY=sk-...
```

**Local Ollama (no API key)** — requires a **vision** model (text-only models like `qwen3` or `deepseek-r1` cannot see camera frames):

```bash
# Install Ollama: https://ollama.com — then pull a vision model:
ollama pull llava
# alternatives: ollama pull llama3.2-vision   ollama pull qwen2.5vl:7b
ollama list   # confirm the model name
```

```env
VLM_PROVIDER=ollama
VLM_MODEL=llava
OLLAMA_BASE_URL=http://localhost:11434
```

| Variable          | Default                  | Description                                       |
| ----------------- | ------------------------ | ------------------------------------------------- |
| `VLM_PROVIDER`    | `openai`                 | `openai` or `ollama` (local, no API key)          |
| `VLM_MODEL`       | `gpt-5.5`                | OpenAI model name, or Ollama model (e.g. `llava`) |
| `OPENAI_API_KEY`  | —                        | Required when `VLM_PROVIDER=openai`               |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL                                 |

**Common error:** `model 'llava' not found` — the name in `VLM_MODEL` must match an installed Ollama model exactly (`ollama list`). Run `ollama pull llava` first.


## Usage

### Preview the stream

```bash
uv run camera-preview
```

Tapo RTSP with explicit URL:

```bash
uv run camera-preview \
  --camera tapo-rtsp \
  --url 'rtsp://camera_user:camera_pass@192.168.1.50:554/stream2'
```

Press `**q**` in the preview window to quit.

### Sample frames to disk

Shows a live preview and saves one JPEG every 2 seconds under `camera_test/sampled_frames/`.

```bash
uv run camera-sample
```

Press `**q**` to stop. Output files: `frame_000000.jpg`, `frame_000001.jpg`, …

### Ask a VLM about the live view

Captures recent frames in the background (1 per second) and answers text questions in a REPL.

```bash
uv run camera-vlm --camera tapo-rtsp
```

Type `**q**`, `**quit**`, or `**exit**` to stop.

## Latency (RTSP vs WebRTC)

A half-second to one second behind “real time” is **normal** when video goes through RTSP → MediaMTX → WebRTC. For local Python on the same Wi‑Fi, **Tapo RTSP direct** is usually the fastest option. Use WebRTC when you want a browser player, a phone as the camera, or multiple clients.


| Path                                                          | Typical delay (same LAN) | Best for                                            |
| ------------------------------------------------------------- | ------------------------ | --------------------------------------------------- |
| Tapo → RTSP → `camera_test`                                   | ~200–500 ms              | Local preview, VLM, frame sampling                  |
| Tapo → RTSP → MediaMTX → WebRTC → `camera_test`               | ~500 ms–1.5 s (tunable)  | Browser preview, relay, WHEP clients                |
| Phone → MediaMTX → WebRTC → `camera_test`                     | ~300 ms–1 s              | Wearable / phone-as-camera prototypes               |
| Phone → MediaMTX → **RTSP** → `camera_test` (current default) | ~0.5–2 s                 | Reliable Python preview; not true end-to-end WebRTC |


**Why phone preview can feel slow or freeze:** Python reads MediaMTX over **RTSP**, not WebRTC (WHEP is unreliable with `webrtcEncryption` + aiortc). That adds a transcode/relay hop. OpenCV/FFmpeg also **buffers** frames — the picture can lag, then look frozen when the buffer stalls. `read_frame()` flushes stale frames; tune `RTSP_FLUSH_GRABS` and use `RTSP_TRANSPORT=udp` for local `127.0.0.1:8554`.

### Recommended MediaMTX settings for low-latency Tapo WebRTC

```yaml
paths:
  tapo:
    source: rtsp://camera_user:camera_pass@192.168.1.50:554/stream2
    sourceOnDemand: no
    rtspTransport: udp
```

Use `**stream2**` unless you need full resolution.

## Troubleshooting

`**Class AVFFrameReceiver is implemented in both ...` (macOS)**

OpenCV and aiortc each ship FFmpeg libraries. When both load in one process, macOS may print a harmless duplicate-class warning. WebRTC runs in a **child process** by default (`WEBRTC_IPC=subprocess`) to avoid crashes. If you see real crashes, avoid `WEBRTC_IPC=inprocess` or use `tapo-rtsp` for Python preview.

**Stream won't open**

1. Test the same URL in VLC (*Media → Open Network Stream*).
2. Confirm camera/phone and computer are on the same LAN.
3. For Tapo, verify the Camera Account credentials (not your Tapo app password).
4. Try `stream2` instead of `stream1`.

`**path 'tapo' is not configured` (MediaMTX)**

MediaMTX is not loading the YAML file you edited. Copy the example to `mediamtx-tapo.yml` or `mediamtx-phone.yml`, edit that file, and start with an explicit path: `mediamtx mediamtx-tapo.yml` (from `camera_test/`).

`**no stream is available on path` (WebRTC / WHEP)**


| Setup        | Config file           | Browser test                    | Python URL                                 |
| ------------ | --------------------- | ------------------------------- | ------------------------------------------ |
| Tapo WebRTC  | `mediamtx-tapo.yml`   | `http://localhost:8889/tapo/`   | `http://localhost:8889/tapo/whep`          |
| Phone WebRTC | `mediamtx-phone.yml`  | `https://localhost:8889/phone/` | `rtsp://127.0.0.1:8554/phone` (RTSP relay) |


**Tapo WebRTC:** ensure MediaMTX is pulling RTSP (`sourceOnDemand: no` or open the browser player once).

**Phone WebRTC:** path `**phone`** has no stream until the phone is publishing at `https://YOUR_MAC_IP:8889/phone/publish` (get `YOUR_MAC_IP` with `ipconfig getifaddr en0` on the Mac).

**Phone publish page: "can't access webcams or microphones"**

You opened `http://` from the phone, or `webrtcEncryption` is off / certs are missing. Browsers block camera access on insecure origins. Enable TLS in `mediamtx-phone.yml`, generate certs (see [Phone WebRTC TLS](#phone-webrtc-tls-one-time-setup)), restart MediaMTX, and use **`https://`** on the phone.

**Browser HTTPS works, but Python WHEP fails with `CERTIFICATE_VERIFY_FAILED`**

Chrome trusts mkcert via the macOS keychain; Python's `httpx` uses its own CA bundle and does not. `whep_client` auto-adds the mkcert root CA when `mkcert` is installed. If it still fails:

1. Confirm mkcert is installed: `mkcert -install`
2. Or set `WEBRTC_CA_FILE=$(mkcert -CAROOT)/rootCA.pem` in `.env`
3. Dev-only escape hatch: `WEBRTC_SSL_VERIFY=false`

**WHEP probe: `ice=completed`, `connection=connecting` (DTLS timeout)**

WHEP signaling succeeded but the WebRTC media path never finished DTLS. The browser player works because Chrome's WebRTC stack differs from Python aiortc. With phone + `webrtcEncryption`, aiortc WHEP is unreliable today.

**Fix:** use MediaMTX's RTSP relay instead of WHEP:

```env
CAMERA_SOURCE=phone-webrtc
PHONE_STREAM_URL=rtsp://127.0.0.1:8554/phone
```

Test: `ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/phone` (while the phone is publishing).

**Browser WebRTC works, but `camera-preview --camera tapo-webrtc` fails**

```bash
uv run camera-whep-probe --url http://localhost:8889/tapo/whep
```

Checklist:

1. `WEBRTC_URL` path matches the `tapo:` path in `mediamtx-tapo.yml`.
2. In `mediamtx-tapo.yml`, set `webrtcAdditionalHosts: [127.0.0.1]` (and your Mac LAN IP if needed).
3. Increase timeout: `WEBRTC_OPEN_TIMEOUT_SEC=30`.

**Probe stuck at `ice=checking`**

Add to `mediamtx-tapo.yml` or `mediamtx-phone.yml`:

- `webrtcAdditionalHosts: [127.0.0.1]` (+ your Mac LAN IP)
- `webrtcLocalTCPAddress: :8190`

`**backend is generally available but can't be used to capture by name` (RTSP)**

FFmpeg failed to open the RTSP stream — not an OpenCV backend issue. Use `CAMERA_SOURCE=tapo-rtsp`, `RTSP_TRANSPORT=tcp`, confirm the URL in VLC, and URL-encode special characters in the password.

`**Failed to read frame` loops**

The camera may have dropped the connection. Stop with `q` and restart. For RTSP direct, TCP transport often helps.

**Phone preview laggy or frozen (RTSP relay)**

Python is not receiving WebRTC directly — it reads `rtsp://127.0.0.1:8554/phone`. Try:

1. `PHONE_STREAM_URL=rtsp://127.0.0.1:8554/phone` (not WHEP)
2. `RTSP_TRANSPORT=udp` for local MediaMTX (TCP adds latency)
3. `RTSP_FLUSH_GRABS=12` if the image still freezes (drops buffered stale frames)
4. `RTSP_STALE_SEC=3` — auto-reconnect when pixels stop changing (fixes duplicate JPEG saves)
5. Keep the phone publishing tab in the foreground; background mobile browsers may pause video
6. For sampling without a preview window: `uv run camera-sample --no-preview` (macOS preview can block reads)
7. Compare latency in the browser: `https://localhost:8889/phone/` — if that's smooth but Python isn't, it's the RTSP relay path

**H264 errors in terminal (`corrupted macroblock`, `Missing reference picture`)**

Harmless FFmpeg warnings when joining mid-stream or after a brief phone/WebRTC glitch. They are **not VLM errors**. By default they go to `**camera_test/rtsp_decode.log`** (not the terminal). Set `RTSP_FFMPEG_LOG=off` to print them inline, or `RTSP_FFMPEG_LOG=/path/to/file.log` to customize. If the VLM keeps seeing stale scenes, the capture thread reconnects RTSP automatically; refresh the phone publish page if it keeps happening.

## Suggested workflow

```text
preview_stream.py → confirm source works
frame_sample.py   → confirm frames save correctly
live_vlm_qa.py    → test VLM on live video
```

For production-style flows (config, logging, memory), use the packages in the repo:

- `vlm_smoke/` — stable live visual QA
- `memory_log/` — structured visual memory to JSONL

