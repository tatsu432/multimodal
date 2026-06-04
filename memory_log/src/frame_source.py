import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from src.config import Config

logger = logging.getLogger("memory_log.capture")

MAX_CONSECUTIVE_READ_FAILURES = 30
READ_RETRY_SLEEP_SEC = 0.01
OPEN_RETRY_SLEEP_SEC = 2.0


def open_video_capture(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str) and source.startswith(
        ("rtmp://", "rtsp://", "http://", "https://")
    ):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


class FrameSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class _OpenCVFrameSource(FrameSource):
    """Shared read/reconnect logic for RTMP, webcam, and video file sources."""

    def __init__(self, source_label: str, *, end_of_stream: bool = False):
        self.source_label = source_label
        self.end_of_stream = end_of_stream
        self._cap: cv2.VideoCapture | None = None
        self._stopped = False
        self._stream_ended = False

    @property
    def stream_ended(self) -> bool:
        return self._stream_ended

    def _ensure_open(self) -> bool:
        if self._stopped:
            return False

        if self._cap is not None and self._cap.isOpened():
            return True

        self._open_capture()
        return self._cap is not None and self._cap.isOpened()

    @abstractmethod
    def _open_capture(self) -> None:
        ...

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._stopped or self._stream_ended:
            return False, None

        consecutive_failures = 0

        while not self._stopped and not self._stream_ended:
            if not self._ensure_open():
                time.sleep(OPEN_RETRY_SLEEP_SEC)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    return False, None
                continue

            ok, frame = self._cap.read()  # type: ignore[union-attr]

            if ok and frame is not None and frame.size > 0:
                return True, frame

            consecutive_failures += 1
            if consecutive_failures < MAX_CONSECUTIVE_READ_FAILURES:
                time.sleep(READ_RETRY_SLEEP_SEC)
                continue

            if self.end_of_stream:
                logger.info("End of %s stream", self.source_label)
                self._stream_ended = True
                self._release_cap()
                return False, None

            logger.warning(
                "Sustained read failures on %s; reconnecting",
                self.source_label,
            )
            self._release_cap()
            consecutive_failures = 0

        return False, None

    def _release_cap(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def release(self) -> None:
        self._stopped = True
        self._release_cap()
        logger.info("Released %s frame source", self.source_label)


class RTMPFrameSource(_OpenCVFrameSource):
    def __init__(self, rtmp_url: str):
        super().__init__("rtmp")
        self.rtmp_url = rtmp_url

    def _open_capture(self) -> None:
        logger.info("Opening RTMP stream: %s", self.rtmp_url)
        self._release_cap()
        self._cap = open_video_capture(self.rtmp_url)
        if not self._cap.isOpened():
            logger.warning(
                "Could not open RTMP stream %s. Retrying in %.0f seconds...",
                self.rtmp_url,
                OPEN_RETRY_SLEEP_SEC,
            )


class WebcamFrameSource(_OpenCVFrameSource):
    def __init__(self, webcam_index: int):
        super().__init__("webcam")
        self.webcam_index = webcam_index

    def _open_capture(self) -> None:
        logger.info("Opening webcam index %d", self.webcam_index)
        self._release_cap()
        self._cap = open_video_capture(self.webcam_index)
        if not self._cap.isOpened():
            logger.warning(
                "Could not open webcam %d. Retrying in %.0f seconds...",
                self.webcam_index,
                OPEN_RETRY_SLEEP_SEC,
            )


class VideoFileFrameSource(_OpenCVFrameSource):
    def __init__(self, video_path: str):
        super().__init__("video", end_of_stream=True)
        self.video_path = Path(video_path)

    def _open_capture(self) -> None:
        logger.info("Opening video file: %s", self.video_path)
        self._release_cap()
        self._cap = open_video_capture(str(self.video_path))
        if not self._cap.isOpened():
            logger.warning(
                "Could not open video file %s. Retrying in %.0f seconds...",
                self.video_path,
                OPEN_RETRY_SLEEP_SEC,
            )


def create_frame_source(config: Config) -> FrameSource:
    if config.frame_source_type == "rtmp":
        return RTMPFrameSource(rtmp_url=config.rtmp_url)

    if config.frame_source_type == "webcam":
        return WebcamFrameSource(webcam_index=config.webcam_index)

    if config.frame_source_type == "video":
        return VideoFileFrameSource(video_path=config.video_path)

    raise ValueError(f"Unsupported frame source type: {config.frame_source_type}")
