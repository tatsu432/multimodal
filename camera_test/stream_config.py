"""Shared frame source configuration for camera_test scripts."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

_CAMERA_TEST_DIR = Path(__file__).resolve().parent
DEFAULT_RTSP_FFMPEG_LOG = _CAMERA_TEST_DIR / "rtsp_decode.log"

_ffmpeg_stderr_installed = False

VALID_CAMERA_SOURCES = frozenset({"tapo-rtsp", "tapo-webrtc", "phone-webrtc"})

DEFAULT_RTSP_URL = "rtsp://camera_user:camera_pass@192.168.1.50:554/stream2"
DEFAULT_TAPO_WEBRTC_URL = "http://localhost:8889/tapo/whep"
# Phone publishes WebRTC to MediaMTX; Python reads the RTSP relay (WHEP + TLS is unreliable in aiortc).
DEFAULT_PHONE_RTSP_URL = "rtsp://127.0.0.1:8554/phone"
DEFAULT_PHONE_WHEP_URL = "https://localhost:8889/phone/whep"

CAMERA_LABELS = {
    "tapo-rtsp": "Tapo (RTSP direct)",
    "tapo-webrtc": "Tapo (WebRTC via MediaMTX)",
    "phone-webrtc": "Smartphone (WebRTC via MediaMTX)",
}


DEFAULT_MEDIAMTX_RTSP_BASE = "rtsp://127.0.0.1:8554"


def whep_url_to_rtsp_relay(whep_url: str) -> str:
    """Map WHEP URL to MediaMTX RTSP relay on the same path (e.g. .../tapo/whep → .../tapo)."""
    base = os.getenv("MEDIAMTX_RTSP_BASE", DEFAULT_MEDIAMTX_RTSP_BASE).strip().rstrip("/")
    path_name = _whep_path_name(whep_url)
    return f"{base}/{path_name}"


def _webrtc_preview_via_rtsp() -> bool:
    """Use MediaMTX RTSP relay for Python preview instead of WHEP/browser capture."""
    return _env_bool("WEBRTC_PREVIEW_VIA_RTSP", True)


def resolve_source(
    camera: str | None = None,
    url: str | None = None,
) -> tuple[str, str, str]:
    """
    Resolve camera preset and capture target.

    Returns (camera_source, source_type, target) where source_type is
    ``rtsp`` or ``webrtc`` and target is the stream URL string.
    """
    selected = (camera or os.getenv("CAMERA_SOURCE", "tapo-rtsp")).strip().lower()

    if selected not in VALID_CAMERA_SOURCES:
        raise ValueError(
            f"CAMERA_SOURCE must be one of {sorted(VALID_CAMERA_SOURCES)}, "
            f"got {selected!r}"
        )

    if selected == "tapo-rtsp":
        stream_url = (url or os.getenv("RTSP_URL", DEFAULT_RTSP_URL)).strip()
        if not stream_url:
            raise ValueError("RTSP_URL is required when CAMERA_SOURCE=tapo-rtsp")
        return selected, "rtsp", stream_url

    if selected == "tapo-webrtc":
        webrtc_url = (
            url or os.getenv("WEBRTC_URL", DEFAULT_TAPO_WEBRTC_URL)
        ).strip()
        if not webrtc_url:
            raise ValueError("WEBRTC_URL is required when CAMERA_SOURCE=tapo-webrtc")
        if _webrtc_preview_via_rtsp():
            return selected, "rtsp", whep_url_to_rtsp_relay(webrtc_url)
        return selected, "webrtc", webrtc_url

    stream_url = (url or os.getenv("PHONE_STREAM_URL", DEFAULT_PHONE_RTSP_URL)).strip()
    if not stream_url:
        raise ValueError(
            "PHONE_STREAM_URL (or --url) is required when CAMERA_SOURCE=phone-webrtc"
        )
    if stream_url.startswith("rtsp://"):
        return selected, "rtsp", stream_url
    if stream_url.startswith(("http://", "https://")):
        return selected, "webrtc", stream_url
    raise ValueError(
        f"phone-webrtc stream URL must be rtsp:// or http(s):// WHEP, got {stream_url!r}"
    )


def source_description(camera: str, source_type: str, target: str) -> str:
    label = CAMERA_LABELS.get(camera, camera)
    if camera == "tapo-webrtc" and source_type == "rtsp":
        return f"{label} — RTSP relay {target} (Python preview; browser uses WebRTC)"
    if source_type == "webrtc":
        return f"{label} — WHEP ({target})"
    return f"{label} — {target}"


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes"}


def _is_local_rtsp(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def configure_decode_logging() -> None:
    """
    Route FFmpeg/libav decode warnings to a log file instead of the terminal.

    Call after ``load_dotenv()`` so ``RTSP_FFMPEG_LOG`` from ``.env`` applies.
    """
    install_ffmpeg_stderr_log()


def install_ffmpeg_stderr_log() -> Path | None:
    """
    Redirect OS stderr (fd 2) to a log file.

    FFmpeg/libav (inside OpenCV) writes directly to file descriptor 2, bypassing
    Python's ``sys.stderr`` wrapper. ``dup2`` catches those C-level messages.
    """
    global _ffmpeg_stderr_installed
    if _ffmpeg_stderr_installed:
        return _resolve_ffmpeg_log_path()

    log_path = _resolve_ffmpeg_log_path()
    if log_path is None:
        _ffmpeg_stderr_installed = True
        return None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    terminal_stderr_fd = os.dup(2)
    log_fd = os.open(
        str(log_path),
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    header = (
        f"\n--- RTSP/FFmpeg decode log started "
        f"{datetime.now(timezone.utc).isoformat()} ---\n"
    )
    os.write(log_fd, header.encode("utf-8"))
    os.dup2(log_fd, 2)
    os.close(log_fd)

    # Python tracebacks / library errors still go to the real terminal.
    sys.stderr = os.fdopen(terminal_stderr_fd, "w", buffering=1, closefd=True)

    _ffmpeg_stderr_installed = True
    print(f"[rtsp] FFmpeg decode warnings → {log_path}", file=sys.stdout)
    return log_path


def _resolve_ffmpeg_log_path() -> Path | None:
    raw = os.getenv("RTSP_FFMPEG_LOG", "rtsp_decode.log").strip()
    if raw.lower() in {"0", "false", "no", "off", "none", "disable", "disabled"}:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = _CAMERA_TEST_DIR / path
    return path


def _build_rtsp_ffmpeg_options(url: str, transport: str) -> str:
    """FFmpeg options for OpenCV VideoCapture (semicolon pairs, pipe-separated)."""
    parts = [f"rtsp_transport;{transport}"]
    if _env_bool("RTSP_LOW_LATENCY", _is_local_rtsp(url)):
        parts.extend(
            [
                "fflags;nobuffer+discardcorrupt",
                "flags;low_delay",
                "max_delay;0",
            ]
        )
    return "|".join(parts)


def frame_signature(frame) -> bytes:
    """Compact fingerprint for duplicate / frozen-frame detection."""
    small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
    return small.tobytes()


class StaleStreamDetector:
    """
    Detect when RTSP decode is returning the same pixels repeatedly.

    FFmpeg often keeps returning ``ok=True`` with an old frame after H264
    errors or when the phone/browser publisher pauses.
    """

    def __init__(self, stale_sec: float | None = None) -> None:
        if stale_sec is None:
            stale_sec = float(os.getenv("RTSP_STALE_SEC", "3"))
        self.stale_sec = stale_sec
        self._last_sig: bytes | None = None
        self._unchanged_since = time.time()

    def check(self, frame) -> str | None:
        """
        Return ``duplicate`` if same as previous frame, ``stale`` if unchanged
        for longer than ``stale_sec`` (caller should reconnect).
        """
        sig = frame_signature(frame)
        now = time.time()
        if sig == self._last_sig:
            if now - self._unchanged_since >= self.stale_sec:
                return "stale"
            return "duplicate"
        self._last_sig = sig
        self._unchanged_since = now
        return None

    def reset(self) -> None:
        self._last_sig = None
        self._unchanged_since = time.time()


def release_source(capture) -> None:
    if capture is not None:
        capture.release()


def read_frame(capture, source_type: str):
    """
    Read one display frame from any capture handle returned by ``open_source``.

    For RTSP, drops buffered frames so preview stays near live (OpenCV often
    queues seconds of stale video even when CAP_PROP_BUFFERSIZE is 1).
    """
    if source_type == "webrtc":
        return capture.read()

    flush_grabs = int(os.getenv("RTSP_FLUSH_GRABS", "8"))
    grabbed = False
    for _ in range(max(flush_grabs, 1)):
        if not capture.grab():
            break
        grabbed = True
    if not grabbed:
        return capture.read()
    return capture.retrieve()


def _open_rtsp_stream(url: str) -> cv2.VideoCapture:
    """
    Open an RTSP stream via FFmpeg.

    OpenCV prints a misleading warning when this fails:
    "backend is generally available but can't be used to capture by name"
    That usually means FFmpeg could not connect or negotiate RTSP — not that the backend is wrong.
    """
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    local = _is_local_rtsp(url)
    preferred = os.getenv("RTSP_TRANSPORT", "udp" if local else "tcp").strip().lower()
    transports: list[str] = []
    if preferred in {"tcp", "udp"}:
        transports.append(preferred)
        if _env_bool("RTSP_TRY_ALT_TRANSPORT", not local):
            alt = "udp" if preferred == "tcp" else "tcp"
            if alt not in transports:
                transports.append(alt)
    else:
        transports = ["udp", "tcp"] if local else ["tcp", "udp"]

    last_cap = cv2.VideoCapture()
    for transport in transports:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _build_rtsp_ffmpeg_options(
            url, transport
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        last_cap = cap
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()

    return last_cap


def open_stream(url: str) -> cv2.VideoCapture:
    if not url.startswith("rtsp://"):
        raise ValueError(f"Expected RTSP URL, got {url!r}")
    configure_decode_logging()
    return _open_rtsp_stream(url)


def _whep_path_name(whep_url: str) -> str:
    """Extract MediaMTX path from a WHEP URL like http://host:8889/tapo/whep."""
    from urllib.parse import urlparse

    parsed = urlparse(whep_url.rstrip("/"))
    path = parsed.path.removesuffix("/whep").strip("/")
    return path or "?"


def describe_open_failure(
    camera: str,
    source_type: str,
    target: str,
    label: str,
    *,
    capture: object | None = None,
) -> str:
    lines = [f"Could not open {label}."]

    if source_type == "webrtc":
        path_name = _whep_path_name(target)
        lines.extend(
            [
                "",
                "Common cause: WEBRTC_URL path does not match mediamtx.yml, or no stream on that path.",
                f"  You requested path: {path_name!r}",
                f"  WEBRTC_URL={target}",
            ]
        )
        if camera == "tapo-webrtc":
            lines.extend(
                [
                    "",
                    "For Tapo via MediaMTX (RTSP pull), mediamtx.yml usually defines tapo:",
                    "  paths:",
                    "    tapo:",
                    "      source: rtsp://camera_user:pass@192.168.1.50:554/stream2",
                    "  WEBRTC_URL=http://localhost:8889/tapo/whep",
                    "",
                    "Verify RTSP in VLC first, then browser: http://localhost:8889/tapo/",
                ]
            )
        elif camera == "phone-webrtc":
            lines.extend(
                [
                    "",
                    "For smartphone via MediaMTX, mediamtx.yml usually defines phone:",
                    "  paths:",
                    "    phone:",
                    "      # no source — waits for WebRTC publisher",
                    "  Publish from phone (use Mac LAN IP, HTTPS required):",
                    "    https://YOUR_MAC_IP:8889/phone/publish",
                    "    or WHIP: https://YOUR_MAC_IP:8889/phone/whip",
                    "  Python default (RTSP relay from MediaMTX):",
                    f"    PHONE_STREAM_URL={DEFAULT_PHONE_RTSP_URL}",
                    "",
                    "Phone browsers block camera on http:// LAN URLs — enable",
                    "webrtcEncryption and TLS certs (see README Phone WebRTC TLS).",
                    "",
                    "The stream exists only while the phone is actively publishing.",
                    "Verify in a browser: https://localhost:8889/phone/",
                    "Or test RTSP: ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/phone",
                ]
            )
        lines.extend(
            [
                "",
                "If the browser plays but Python WHEP fails, run:",
                "  uv run camera-whep-probe --url <your WHEP URL>",
                "For tapo-webrtc Python preview, prefer the RTSP relay (default):",
                "  WEBRTC_PREVIEW_VIA_RTSP=true  → rtsp://127.0.0.1:8554/<path>",
                "On macOS, aiortc often cannot complete DTLS with MediaMTX — use",
                "WEBRTC_PREVIEW_VIA_RTSP=true or WEBRTC_IPC=browser (slower).",
                "  uv sync --extra browser-webrtc && uv run playwright install chromium",
                "Other fixes: mediamtx webrtcAdditionalHosts, WEBRTC_OPEN_TIMEOUT_SEC=30.",
            ]
        )
        last_error = getattr(capture, "last_error", None)
        if last_error:
            lines.extend(["", f"Server/client error: {last_error}"])
    elif source_type == "rtsp":
        lines.extend(
            [
                "",
                "OpenCV often prints this misleading warning when RTSP fails:",
                "  'backend is generally available but can't be used to capture by name'",
                "FFmpeg could not open the stream (wrong URL, auth, or transport) —",
                "the backend itself is fine.",
                "",
                "Checklist:",
                "  1. Test the same URL in VLC (Media → Open Network Stream).",
                "  2. Set RTSP_URL in camera_test/.env (CAMERA_SOURCE=tapo-rtsp).",
                f"     RTSP_URL={target}",
                "  3. Tapo + OpenCV usually needs TCP:",
                "     RTSP_TRANSPORT=tcp",
                "  4. URL-encode special characters in the camera password (! → %21).",
                "  5. Use stream2 if stream1 fails:",
                "     .../554/stream2",
            ]
        )
        if target == DEFAULT_RTSP_URL:
            lines.append(
                "  6. You are still on the default placeholder URL — set your Tapo RTSP URL."
            )
        if camera == "tapo-webrtc":
            lines.extend(
                [
                    "",
                    "Python preview uses the MediaMTX RTSP relay (WEBRTC_PREVIEW_VIA_RTSP=true).",
                    "Ensure MediaMTX is running and pulling the Tapo RTSP source.",
                    "For lower latency on localhost, try RTSP_TRANSPORT=udp.",
                    "Browser WebRTC player: same path at http://localhost:8889/<path>/",
                ]
            )
    else:
        lines.append("Confirm the source is reachable and .env / CLI flags are correct.")

    return "\n".join(lines)


def open_source(source_type: str, target: str):
    """Open a capture handle for any supported source type."""
    if source_type == "webrtc":
        from webrtc_capture import WebRTCCapture

        timeout = float(os.getenv("WEBRTC_OPEN_TIMEOUT_SEC", "30"))
        return WebRTCCapture(
            target,
            ice_servers_env=os.getenv("WEBRTC_ICE_SERVERS"),
            open_timeout_sec=timeout,
        )

    return open_stream(target)


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--camera",
        choices=sorted(VALID_CAMERA_SOURCES),
        help="Camera setup (default: CAMERA_SOURCE env or tapo-rtsp)",
    )
    parser.add_argument(
        "--url",
        help="Override RTSP_URL (tapo-rtsp), WEBRTC_URL (tapo-webrtc), or PHONE_STREAM_URL (phone-webrtc)",
    )
