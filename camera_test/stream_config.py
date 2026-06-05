"""Shared RTMP/RTSP stream configuration for camera_test scripts."""

from __future__ import annotations

import argparse
import os

import cv2

DEFAULT_RTMP_URL = "rtmp://localhost:1935/live/gopro"
DEFAULT_RTSP_URL = "rtsp://localhost:8554/live/gopro"
VALID_PROTOCOLS = frozenset({"rtmp", "rtsp"})


def resolve_stream(
    protocol: str | None = None,
    url: str | None = None,
) -> tuple[str, str]:
    selected = (protocol or os.getenv("STREAM_PROTOCOL", "rtmp")).strip().lower()
    if selected not in VALID_PROTOCOLS:
        raise ValueError(
            f"STREAM_PROTOCOL must be one of {sorted(VALID_PROTOCOLS)}, got {selected!r}"
        )

    if url:
        stream_url = url.strip()
    elif selected == "rtsp":
        stream_url = os.getenv("RTSP_URL", DEFAULT_RTSP_URL)
    else:
        stream_url = os.getenv("RTMP_URL", DEFAULT_RTMP_URL)

    return selected, stream_url


def open_stream(url: str) -> cv2.VideoCapture:
    if url.startswith(("rtmp://", "rtsp://", "http://", "https://")):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(url)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def add_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol",
        choices=sorted(VALID_PROTOCOLS),
        help="Stream protocol (default: STREAM_PROTOCOL env or rtmp)",
    )
    parser.add_argument(
        "--url",
        help="Full stream URL (overrides protocol-specific env URL)",
    )
