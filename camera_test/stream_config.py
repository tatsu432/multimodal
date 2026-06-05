"""Shared frame source configuration for camera_test scripts."""

from __future__ import annotations

import argparse
import os

import cv2

DEFAULT_RTMP_URL = "rtmp://localhost:1935/live/gopro"
DEFAULT_RTSP_URL = "rtsp://localhost:8554/live/gopro"
DEFAULT_WEBRTC_URL = "http://localhost:8889/live/whep"
VALID_SOURCE_TYPES = frozenset({"rtmp", "rtsp", "webrtc", "webcam", "video"})


def resolve_source(
    source_type: str | None = None,
    protocol: str | None = None,
    url: str | None = None,
) -> tuple[str, str | int]:
    """
    Resolve frame source type and capture target.

    Returns (source_type, target) where target is a stream URL/path string
    or a webcam device index.
    """
    selected = (
        source_type
        or os.getenv("FRAME_SOURCE_TYPE")
        or protocol
        or os.getenv("STREAM_PROTOCOL", "rtmp")
    ).strip().lower()

    if selected not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"FRAME_SOURCE_TYPE must be one of {sorted(VALID_SOURCE_TYPES)}, "
            f"got {selected!r}"
        )

    if selected == "webcam":
        return "webcam", int(os.getenv("WEBCAM_INDEX", "0"))

    if selected == "video":
        video_path = (url or os.getenv("VIDEO_PATH", "")).strip()
        if not video_path:
            raise ValueError(
                "VIDEO_PATH is required when FRAME_SOURCE_TYPE=video"
            )
        return "video", video_path

    if selected == "webrtc":
        webrtc_url = (url or os.getenv("WEBRTC_URL", DEFAULT_WEBRTC_URL)).strip()
        if not webrtc_url:
            raise ValueError(
                "WEBRTC_URL is required when FRAME_SOURCE_TYPE=webrtc"
            )
        return "webrtc", webrtc_url

    if url:
        stream_url = url.strip()
    elif selected == "rtsp":
        stream_url = os.getenv("RTSP_URL", DEFAULT_RTSP_URL)
    else:
        stream_url = os.getenv("RTMP_URL", DEFAULT_RTMP_URL)

    return selected, stream_url.strip()


def source_description(source_type: str, target: str | int) -> str:
    if source_type == "webcam":
        return f"webcam (index {target})"
    if source_type == "webrtc":
        return f"WebRTC WHEP ({target})"
    return f"{source_type.upper()} ({target})"


def _open_rtsp_stream(url: str) -> cv2.VideoCapture:
    """
    Open an RTSP stream via FFmpeg.

    OpenCV prints a misleading warning when this fails:
    "backend is generally available but can't be used to capture by name"
    That usually means     FFmpeg could not connect or negotiate RTSP — not that the backend is wrong.
    """
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    preferred = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
    transports: list[str] = []
    if preferred in {"tcp", "udp"}:
        transports.append(preferred)
        if os.getenv("RTSP_TRY_ALT_TRANSPORT", "true").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            alt = "udp" if preferred == "tcp" else "tcp"
            if alt not in transports:
                transports.append(alt)
    else:
        transports = ["tcp", "udp"]

    last_cap = cv2.VideoCapture()
    for transport in transports:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        last_cap = cap
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()

    return last_cap


def open_stream(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.startswith("rtsp://"):
        return _open_rtsp_stream(source)

    if isinstance(source, str) and source.startswith(
        ("rtmp://", "http://", "https://")
    ):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def describe_open_failure(
    source_type: str,
    target: str | int,
    label: str,
) -> str:
    lines = [f"Could not open {label}."]

    if source_type == "rtsp":
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
                "  2. Set RTSP_URL in camera_test/.env (not WEBRTC_URL).",
                f"     RTSP_URL={target}",
                "  3. Tapo + OpenCV usually needs TCP:",
                "     RTSP_TRANSPORT=tcp",
                "  4. URL-encode special characters in the camera password (! → %21).",
                "  5. Use stream2 if stream1 fails:",
                "     .../554/stream2",
            ]
        )
        if target in {DEFAULT_RTSP_URL, DEFAULT_RTMP_URL}:
            lines.append(
                "  6. You are still on the default placeholder URL — set your Tapo RTSP URL."
            )
    else:
        lines.append("Confirm the source is reachable and .env / CLI flags are correct.")

    return "\n".join(lines)


def open_source(source_type: str, target: str | int):
    """Open a capture handle for any supported source type."""
    if source_type == "webrtc":
        from webrtc_capture import WebRTCCapture, parse_ice_servers

        timeout = float(os.getenv("WEBRTC_OPEN_TIMEOUT_SEC", "15"))
        ice_servers = parse_ice_servers(os.getenv("WEBRTC_ICE_SERVERS"))
        return WebRTCCapture(
            str(target),
            ice_servers=ice_servers,
            open_timeout_sec=timeout,
        )

    return open_stream(target)


def add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-type",
        choices=sorted(VALID_SOURCE_TYPES),
        help="Frame source type (default: FRAME_SOURCE_TYPE env or rtmp)",
    )
    parser.add_argument(
        "--protocol",
        choices=["rtmp", "rtsp"],
        help="Deprecated alias for --source-type rtmp|rtsp",
    )
    parser.add_argument(
        "--url",
        help=(
            "Full stream URL, video file path (--source-type=video), "
            "or WHEP endpoint (--source-type=webrtc)"
        ),
    )


# Backward-compatible aliases used by older docs/scripts.
resolve_stream = resolve_source
add_stream_args = add_source_args
