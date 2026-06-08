"""Deterministic, seekable replay frame source for evaluation.

Unlike VideoFileFrameSource (which stamps frames with wall-clock time and consumes
faster-than-real-time), ReplaySource:
  1. Pre-indexes sampled frames at media-time intervals during load().
  2. Tags each frame with a synthetic epoch = base_epoch + media_pts_sec,
     so timestamps are deterministic and reproducible.
  3. Enforces the no-future-leakage rule: frames_at(t) only returns frames
     with media_time <= t.
  4. Implements the FrameSource duck-type for drop-in use with VisualGrounder.

Usage:
    src = ReplaySource("desk_001.mp4", base_timestamp="2026-01-15T09:00:00+09:00")
    src.load()
    frames, items = src.frames_at(ask_at_sec=15.0, n=4)
    # items[i].timestamp == base_epoch + media_pts  (float epoch seconds)
    # items[i].frame    == BGR np.ndarray
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.utils import FrameItem

if TYPE_CHECKING:
    pass

logger = logging.getLogger("evals.replay_source")

_DEFAULT_MAX_WIDTH = 640  # resize on load to keep memory reasonable


@dataclass
class _IndexEntry:
    media_time_sec: float
    synthetic_epoch: float  # base_epoch + media_time_sec
    frame: np.ndarray       # BGR; may be downscaled


class ReplaySource:
    """Deterministic, seekable replay source backed by a video file.

    Args:
        video_path:         Path to the video file (mp4, avi, etc.)
        sample_interval_sec: Sampling interval in media time (default 1.0 s).
        base_timestamp:     ISO8601 synthetic clock origin for t=0.
                            None → current time (non-reproducible, for smoke tests).
        buffer_size:        Unused; kept for API compatibility with FrameSource.
        max_load_width:     Resize frames to at most this width on load (0 = no resize).
    """

    def __init__(
        self,
        video_path: Path | str,
        sample_interval_sec: float = 1.0,
        base_timestamp: str | None = None,
        buffer_size: int = 8,
        max_load_width: int = _DEFAULT_MAX_WIDTH,
    ) -> None:
        self.video_path = Path(video_path)
        self.sample_interval_sec = sample_interval_sec
        self.max_load_width = max_load_width
        self._buffer_size = buffer_size

        if base_timestamp:
            dt = datetime.fromisoformat(base_timestamp)
            self.base_epoch: float = dt.timestamp()
        else:
            self.base_epoch = time.time()

        self._index: list[_IndexEntry] = []
        self._playhead_sec: float = 0.0
        self._loaded = False

    # ---- lifecycle ----

    def load(self) -> None:
        """Pre-index sampled frames from the video. Call once before use."""
        if self._loaded:
            return

        cap = cv2.VideoCapture(str(self.video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video for replay: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(round(fps * self.sample_interval_sec)))

        logger.info(
            "Indexing %s  fps=%.1f  total=%d  sample_every=%d frames",
            self.video_path.name,
            fps,
            total_frames,
            frame_interval,
        )

        frame_idx = 0
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_interval == 0:
                if self.max_load_width > 0:
                    h, w = frame.shape[:2]
                    if w > self.max_load_width:
                        scale = self.max_load_width / w
                        frame = cv2.resize(
                            frame,
                            (self.max_load_width, int(h * scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                media_t = frame_idx / fps
                self._index.append(
                    _IndexEntry(
                        media_time_sec=media_t,
                        synthetic_epoch=self.base_epoch + media_t,
                        frame=frame.copy(),
                    )
                )
                count += 1

            frame_idx += 1

        cap.release()
        self._loaded = True
        duration = frame_idx / fps
        logger.info(
            "Loaded %d sampled frames  duration=%.1fs  size=%d×%d",
            count,
            duration,
            self._index[0].frame.shape[1] if self._index else 0,
            self._index[0].frame.shape[0] if self._index else 0,
        )

    # ---- primary eval API ----

    def duration_sec(self) -> float:
        self._ensure_loaded()
        return self._index[-1].media_time_sec if self._index else 0.0

    def seek(self, t_sec: float) -> None:
        """Set the logical playhead position (affects the FrameSource duck-type API)."""
        self._playhead_sec = t_sec

    def frames_at(
        self,
        t_sec: float,
        n: int,
        window_sec: float = 30.0,
    ) -> tuple[list[np.ndarray], list[FrameItem]]:
        """Return up to `n` most recent sampled frames with media_time in [t-window, t].

        Enforces the no-future-leakage rule: no frame with media_time > t is returned.
        """
        self._ensure_loaded()
        lo = t_sec - window_sec
        eligible = [
            e for e in self._index if lo <= e.media_time_sec <= t_sec
        ]
        selected = eligible[-n:] if len(eligible) > n else eligible
        frames = [e.frame for e in selected]
        items = [FrameItem(timestamp=e.synthetic_epoch, frame=e.frame) for e in selected]
        return frames, items

    def all_frames_up_to(self, t_sec: float) -> list[_IndexEntry]:
        """All indexed entries with media_time <= t_sec."""
        self._ensure_loaded()
        return [e for e in self._index if e.media_time_sec <= t_sec]

    # ---- FrameSource duck-type (for VisualGrounder compatibility) ----

    def start(self) -> None:
        self._ensure_loaded()

    def stop(self) -> None:
        pass

    def release(self) -> None:
        self._index.clear()
        self._loaded = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        frames, _ = self.frames_at(self._playhead_sec, 1)
        if not frames:
            return False, None
        return True, frames[-1]

    def get_recent(self, n: int) -> list[np.ndarray]:
        frames, _ = self.frames_at(self._playhead_sec, n)
        return frames

    def get_recent_items(self, n: int) -> list[FrameItem]:
        _, items = self.frames_at(self._playhead_sec, n)
        return items

    # ---- private ----

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
