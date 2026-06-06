"""Shared camera capture configuration (Tapo RTSP, phone WebRTC relay, etc.)."""

from capture.camera_frame_source import CameraFrameSource
from capture.stream_config import (
    CAMERA_LABELS,
    VALID_CAMERA_SOURCES,
    StaleStreamDetector,
    add_source_args,
    configure_app_logging,
    configure_decode_logging,
    describe_open_failure,
    open_source,
    read_frame,
    release_source,
    resolve_source,
    source_description,
)

__all__ = [
    "CAMERA_LABELS",
    "CameraFrameSource",
    "StaleStreamDetector",
    "VALID_CAMERA_SOURCES",
    "add_source_args",
    "configure_app_logging",
    "configure_decode_logging",
    "describe_open_failure",
    "open_source",
    "read_frame",
    "release_source",
    "resolve_source",
    "source_description",
]
