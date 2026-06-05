"""Shared frame source configuration for camera_test scripts."""

from __future__ import annotations

import argparse
import os

import cv2

DEFAULT_RTMP_URL = "rtmp://localhost:1935/live/gopro"
DEFAULT_RTSP_URL = "rtsp://localhost:8554/live/gopro"
VALID_SOURCE_TYPES = frozenset({"rtmp", "rtsp", "webcam", "video"})


def resolve_source(
    source_type: str | None = None,
    protocol: str | None = None,
    url: str | None = None,
) -> tuple[str, str | int]:
    """
    Resolve frame source type and OpenCV capture target.

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
    return f"{source_type.upper()} ({target})"


def open_stream(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.startswith(
        ("rtmp://", "rtsp://", "http://", "https://")
    ):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


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
        help="Full stream URL, or video file path when --source-type=video",
    )


# Backward-compatible aliases used by older docs/scripts.
resolve_stream = resolve_source
add_stream_args = add_source_args
