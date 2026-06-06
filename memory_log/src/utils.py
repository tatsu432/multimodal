import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_optional_float_env(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


@dataclass
class FrameItem:
    timestamp: float
    frame: np.ndarray


class LiveFrameBuffer:
    def __init__(self, max_frames: int = 8):
        self.frames: Deque[FrameItem] = deque(maxlen=max_frames)
        self.lock = threading.Lock()

    def add(self, frame: np.ndarray) -> None:
        frame_copy = frame.copy()
        with self.lock:
            self.frames.append(FrameItem(timestamp=time.time(), frame=frame_copy))

    def get_latest_items(self) -> list[FrameItem]:
        with self.lock:
            return list(self.frames)

    def get_recent_frames(self, n: int) -> list[np.ndarray]:
        items = self.get_latest_items()
        if not items:
            return []
        selected = items[-n:]
        return [item.frame for item in selected]

    def get_recent_items(self, n: int) -> list[FrameItem]:
        items = self.get_latest_items()
        if not items:
            return []
        return items[-n:]

    def latest_frame(self) -> np.ndarray | None:
        items = self.get_latest_items()
        if not items:
            return None
        return items[-1].frame.copy()

    def __len__(self) -> int:
        with self.lock:
            return len(self.frames)


def make_memory_id(now: datetime | None = None) -> tuple[str, str, datetime]:
    """
    Returns (memory_id, iso_timestamp, aware_datetime).
    memory_id uses filesystem-safe form: 2026-06-04T23-12-30.123
    timestamp uses ISO8601 with offset: 2026-06-04T23:12:30.123+09:00
    """
    if now is None:
        now = datetime.now().astimezone()

    millis = int(now.microsecond / 1000)
    memory_id = now.strftime("%Y-%m-%dT%H-%M-%S") + f".{millis:03d}"
    timestamp = now.isoformat(timespec="milliseconds")
    return memory_id, timestamp, now


def frame_capture_timestamp_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).astimezone().isoformat(
        timespec="milliseconds"
    )


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resize_frame(frame: np.ndarray, max_width: int = 768) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def encode_frame_as_base64_jpeg(
    frame: np.ndarray,
    max_width: int = 768,
    quality: int = 85,
) -> str:
    frame = resize_frame(frame, max_width=max_width)

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    return base64.b64encode(buffer).decode("utf-8")


def save_frame_image(
    frame: np.ndarray,
    directory: Path,
    memory_id: str,
    max_width: int = 1280,
    *,
    suffix: str = "",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{memory_id}{suffix}.jpg"
    ok = cv2.imwrite(str(path), resize_frame(frame, max_width=max_width))
    if not ok:
        raise RuntimeError(f"Failed to save frame to {path}")
    return path
