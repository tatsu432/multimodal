"""SSE and MJPEG framing helpers — pure functions, no external dependencies."""

from __future__ import annotations

import json


def sse_event(event: str, payload: dict) -> bytes:
    """Encode a single SSE event with a named event type and JSON data.

    Format::

        event: <name>\\n
        data: <json>\\n
        \\n
    """
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


def mjpeg_part(jpeg_bytes: bytes) -> bytes:
    """Encode one MJPEG multipart frame part."""
    header = (
        "--frame\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(jpeg_bytes)}\r\n"
        "\r\n"
    ).encode("utf-8")
    return header + jpeg_bytes + b"\r\n"


MJPEG_CONTENT_TYPE = "multipart/x-mixed-replace; boundary=frame"
