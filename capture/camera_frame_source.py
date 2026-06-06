"""Threaded frame source backed by capture.stream_config (Tapo RTSP, phone relay, etc.)."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from capture.stream_config import (
    StaleStreamDetector,
    describe_open_failure,
    frame_signature,
    open_source,
    read_frame,
    release_source,
    source_description,
)

logger = logging.getLogger("capture.camera_frame_source")

OPEN_RETRY_SLEEP_SEC = 2.0
READ_FAILURE_SLEEP_SEC = 0.2
MAX_READ_FAILURES = 5


@dataclass
class FrameItem:
    timestamp: float
    frame: np.ndarray


class _FrameRingBuffer:
    def __init__(self, max_frames: int) -> None:
        self.frames: Deque[FrameItem] = deque(maxlen=max_frames)
        self.lock = threading.Lock()

    def add(self, frame: np.ndarray) -> None:
        frame_copy = frame.copy()
        with self.lock:
            self.frames.append(FrameItem(timestamp=time.time(), frame=frame_copy))

    def get_recent_frames(self, n: int) -> list[np.ndarray]:
        with self.lock:
            items = list(self.frames)
        if not items:
            return []
        return [item.frame for item in items[-n:]]

    def get_recent_items(self, n: int) -> list[FrameItem]:
        with self.lock:
            items = list(self.frames)
        if not items:
            return []
        return items[-n:]

    def latest_frame(self) -> np.ndarray | None:
        with self.lock:
            if not self.frames:
                return None
            return self.frames[-1].frame.copy()


class CameraFrameSource:
    """
    Background sampler for Tapo / phone camera presets resolved via ``resolve_source``.

    Implements the same interface as vlm_smoke / memory_log ``FrameSource`` implementations.
    """

    def __init__(
        self,
        camera: str,
        source_type: str,
        target: str,
        buffer_size: int,
        sample_interval_sec: float,
    ) -> None:
        self.camera = camera
        self.source_type = source_type
        self.target = target
        self.sample_interval_sec = sample_interval_sec
        self.label = source_description(camera, source_type, target)
        self.buffer = _FrameRingBuffer(max_frames=buffer_size)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.camera}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started capture thread for %s", self.label)

    def stop(self) -> None:
        self._stop_event.set()

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.buffer.latest_frame()
        if frame is None:
            return False, None
        return True, frame

    def get_recent(self, n: int) -> list[np.ndarray]:
        return self.buffer.get_recent_frames(n)

    def get_recent_items(self, n: int) -> list[FrameItem]:
        return self.buffer.get_recent_items(n)

    def release(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Released %s", self.label)

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            cap = open_source(self.source_type, self.target)
            if not cap.isOpened():
                logger.warning(
                    "Could not open %s — retrying in %.0fs",
                    self.label,
                    OPEN_RETRY_SLEEP_SEC,
                )
                print(describe_open_failure(self.camera, self.source_type, self.target, self.label))
                release_source(cap)
                time.sleep(OPEN_RETRY_SLEEP_SEC)
                continue

            logger.info("Opened %s", self.label)
            stale_detector = StaleStreamDetector()
            last_sample_time = 0.0
            last_added_sig: bytes | None = None
            read_failures = 0

            while not self._stop_event.is_set():
                ok, frame = read_frame(cap, self.source_type)
                if not ok:
                    read_failures += 1
                    if read_failures >= MAX_READ_FAILURES:
                        logger.warning("Read failures on %s — reconnecting", self.label)
                        break
                    time.sleep(READ_FAILURE_SLEEP_SEC)
                    continue
                read_failures = 0

                if stale_detector.check(frame) == "stale":
                    logger.warning("Stream frozen on %s — reconnecting", self.label)
                    break

                now = time.time()
                if now - last_sample_time >= self.sample_interval_sec:
                    sig = frame_signature(frame)
                    if sig != last_added_sig:
                        last_sample_time = now
                        last_added_sig = sig
                        self.buffer.add(frame)

            release_source(cap)
