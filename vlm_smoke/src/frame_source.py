import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from src.config import Config
from src.utils import FrameItem, LiveFrameBuffer

logger = logging.getLogger("vlm_smoke.capture")

# Tolerate brief read stalls (e.g. while main thread encodes/saves frames for VLM).
MAX_CONSECUTIVE_READ_FAILURES = 30
READ_RETRY_SLEEP_SEC = 0.01
OPEN_RETRY_SLEEP_SEC = 2.0


def open_video_capture(source: str | int) -> cv2.VideoCapture:
    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


class FrameSource(ABC):
    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    @abstractmethod
    def get_recent(self, n: int) -> list[np.ndarray]:
        ...

    @abstractmethod
    def release(self) -> None:
        ...


class _ThreadedFrameSource(FrameSource):
    """Base for sources that sample frames in a background thread."""

    def __init__(
        self,
        buffer_size: int,
        sample_interval_sec: float,
        source_label: str,
    ):
        self.buffer = LiveFrameBuffer(max_frames=buffer_size)
        self.sample_interval_sec = sample_interval_sec
        self.source_label = source_label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cap_lock = threading.Lock()
        self._active_cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{self.source_label}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Started %s capture thread", self.source_label)

    def stop(self) -> None:
        self._stop_event.set()
        self._interrupt_active_capture()

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
        with self._cap_lock:
            if self._active_cap is not None:
                self._active_cap.release()
                self._active_cap = None
        logger.info("Released %s frame source", self.source_label)

    def _interrupt_active_capture(self) -> None:
        """Release the open capture so a blocked cap.read() can return during shutdown."""
        with self._cap_lock:
            if self._active_cap is not None:
                self._active_cap.release()
                self._active_cap = None

    def _register_capture(self, cap: cv2.VideoCapture) -> None:
        with self._cap_lock:
            self._active_cap = cap

    def _unregister_capture(self, cap: cv2.VideoCapture) -> None:
        with self._cap_lock:
            if self._active_cap is cap:
                cap.release()
                self._active_cap = None

    def _run_capture_session(
        self,
        cap: cv2.VideoCapture,
        *,
        end_of_stream: bool = False,
    ) -> bool:
        """
        Read frames until stop, sustained read failure, or end-of-stream.
        Returns True when the caller should immediately reopen (e.g. video loop).
        """
        self._register_capture(cap)
        last_sample_time = 0.0
        consecutive_failures = 0
        reopen = False

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()

                if not ok:
                    consecutive_failures += 1
                    if consecutive_failures < MAX_CONSECUTIVE_READ_FAILURES:
                        time.sleep(READ_RETRY_SLEEP_SEC)
                        continue

                    if end_of_stream:
                        logger.info("End of stream reached")
                        reopen = True
                    else:
                        logger.warning(
                            "Sustained read failures on %s (%d attempts), reconnecting",
                            self.source_label,
                            consecutive_failures,
                        )
                        reopen = True
                    break

                consecutive_failures = 0
                now = time.time()
                if now - last_sample_time >= self.sample_interval_sec:
                    last_sample_time = now
                    self.buffer.add(frame)
                    logger.debug(
                        "Sampled %s frame into buffer (size=%d)",
                        self.source_label,
                        len(self.buffer),
                    )
        finally:
            self._unregister_capture(cap)

        return reopen

    def _capture_loop_with_opener(
        self,
        open_capture: Callable[[], cv2.VideoCapture],
        *,
        end_of_stream: bool = False,
        open_failure_message: str,
    ) -> None:
        while not self._stop_event.is_set():
            cap = open_capture()

            if not cap.isOpened():
                logger.warning(open_failure_message)
                time.sleep(OPEN_RETRY_SLEEP_SEC)
                continue

            logger.info("%s opened", self.source_label)
            reopen = self._run_capture_session(cap, end_of_stream=end_of_stream)

            if not reopen and self._stop_event.is_set():
                break

    @abstractmethod
    def _capture_loop(self) -> None:
        ...


class WebcamFrameSource(_ThreadedFrameSource):
    def __init__(
        self,
        webcam_index: int,
        buffer_size: int,
        sample_interval_sec: float,
    ):
        super().__init__(buffer_size, sample_interval_sec, "webcam")
        self.webcam_index = webcam_index

    def _capture_loop(self) -> None:
        logger.info("Opening webcam index %d", self.webcam_index)

        def open_capture() -> cv2.VideoCapture:
            return open_video_capture(self.webcam_index)

        self._capture_loop_with_opener(
            open_capture,
            open_failure_message=(
                f"Could not open webcam {self.webcam_index}. Retrying in "
                f"{OPEN_RETRY_SLEEP_SEC:.0f} seconds..."
            ),
        )


class VideoFileFrameSource(_ThreadedFrameSource):
    def __init__(
        self,
        video_path: str,
        buffer_size: int,
        sample_interval_sec: float,
    ):
        super().__init__(buffer_size, sample_interval_sec, "video")
        self.video_path = Path(video_path)

    def _capture_loop(self) -> None:
        logger.info("Opening video file: %s", self.video_path)

        def open_capture() -> cv2.VideoCapture:
            return open_video_capture(str(self.video_path))

        self._capture_loop_with_opener(
            open_capture,
            end_of_stream=True,
            open_failure_message=(
                f"Could not open video file {self.video_path}. Retrying in "
                f"{OPEN_RETRY_SLEEP_SEC:.0f} seconds..."
            ),
        )


def create_frame_source(config: Config) -> FrameSource:
    if config.frame_source_type == "camera":
        from capture.camera_frame_source import CameraFrameSource
        from capture.stream_config import resolve_source

        camera, source_type, target = resolve_source(
            config.camera_preset_override or config.camera_source,
            config.camera_url_override,
        )
        return CameraFrameSource(
            camera=camera,
            source_type=source_type,
            target=target,
            buffer_size=config.frame_buffer_size,
            sample_interval_sec=config.capture_sample_interval_sec,
        )

    if config.frame_source_type == "webcam":
        return WebcamFrameSource(
            webcam_index=config.webcam_index,
            buffer_size=config.frame_buffer_size,
            sample_interval_sec=config.capture_sample_interval_sec,
        )

    if config.frame_source_type == "video":
        return VideoFileFrameSource(
            video_path=config.video_path,
            buffer_size=config.frame_buffer_size,
            sample_interval_sec=config.capture_sample_interval_sec,
        )

    raise ValueError(f"Unsupported frame source type: {config.frame_source_type}")
