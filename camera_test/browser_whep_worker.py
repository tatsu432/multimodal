"""Capture WebRTC via MediaMTX's built-in browser player (Chromium + Playwright).

aiortc often fails DTLS with MediaMTX on macOS while Chromium succeeds. This worker
loads the same page as http://localhost:8889/<path>/ and pipes JPEG frames on stdout
using the same binary protocol as whep_worker.py.

This path is slow (canvas + JPEG + base64 per frame). For low-latency Python preview
with MediaMTX running, prefer WEBRTC_PREVIEW_VIA_RTSP=true (RTSP relay) instead.
"""

from __future__ import annotations

import argparse
import base64
import os
import struct
import sys
import time
from urllib.parse import urlparse

import cv2
import numpy as np

_FRAME_MAGIC = b"WFRM"
_ERROR_MAGIC = b"WERR"
_HEADER = struct.Struct(">4sIII")

_INIT_CAPTURE_JS = """
() => {
  window.__cameraCapture = (quality, scale) => {
    const video = document.querySelector('#video');
    if (!video || video.readyState < 2 || !video.videoWidth) {
      return null;
    }
    if (!window.__captureCanvas) {
      window.__captureCanvas = document.createElement('canvas');
      window.__captureCtx = window.__captureCanvas.getContext('2d', { alpha: false });
    }
    const w = Math.max(2, Math.round(video.videoWidth * scale));
    const h = Math.max(2, Math.round(video.videoHeight * scale));
    window.__captureCanvas.width = w;
    window.__captureCanvas.height = h;
    window.__captureCtx.drawImage(video, 0, 0, w, h);
    return window.__captureCanvas.toDataURL('image/jpeg', quality);
  };
}
"""


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name, "").strip()
    if not val:
        return default
    return float(val)


def _write_frame(frame: np.ndarray) -> None:
    payload = frame.tobytes()
    header = _HEADER.pack(_FRAME_MAGIC, frame.shape[0], frame.shape[1], len(payload))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _write_error(message: str) -> None:
    data = message.encode("utf-8")
    header = _HEADER.pack(_ERROR_MAGIC, 0, 0, len(data))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def whep_url_to_player_url(whep_url: str) -> str:
    parsed = urlparse(whep_url)
    path = parsed.path
    if path.endswith("/whep"):
        path = path[: -len("/whep")] + "/"
    elif not path.endswith("/"):
        path += "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def run(player_url: str, timeout_sec: float) -> int:
    from playwright.sync_api import sync_playwright

    jpeg_quality = _env_float("BROWSER_JPEG_QUALITY", 0.65)
    capture_scale = _env_float("BROWSER_CAPTURE_SCALE", 0.667)
    min_interval = _env_float("BROWSER_MIN_FRAME_INTERVAL_SEC", 0.033)

    deadline = time.time() + timeout_sec
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(player_url, wait_until="domcontentloaded", timeout=int(timeout_sec * 1000))
            page.evaluate(_INIT_CAPTURE_JS)

            opened = False
            last_send = 0.0
            while True:
                if not opened and time.time() > deadline:
                    _write_error(
                        f"Browser WebRTC player did not deliver video within "
                        f"{timeout_sec:.0f}s ({player_url})"
                    )
                    return 1

                data_url = page.evaluate(
                    "([quality, scale]) => window.__cameraCapture(quality, scale)",
                    [jpeg_quality, capture_scale],
                )
                if data_url:
                    now = time.time()
                    if not opened or now - last_send >= min_interval:
                        raw = base64.b64decode(data_url.split(",", 1)[1])
                        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            _write_frame(frame)
                            opened = True
                            last_send = now
                time.sleep(0.002 if opened else 0.05)
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MediaMTX browser-player capture worker (stdout frame protocol)"
    )
    parser.add_argument("--url", required=True, help="WHEP URL (converted to player page)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Open timeout (seconds)")
    args = parser.parse_args()

    player_url = whep_url_to_player_url(args.url)
    try:
        code = run(player_url, args.timeout)
    except Exception as exc:
        _write_error(str(exc))
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
