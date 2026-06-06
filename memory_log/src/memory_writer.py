import json
import logging
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, Config
from src.schema import LocationInfo, MemoryRecord
from src.utils import (
    FrameItem,
    frame_capture_timestamp_iso,
    make_memory_id,
    relative_path,
    save_frame_image,
)

logger = logging.getLogger("memory_log.writer")


class MemoryWriter:
    def __init__(self, config: Config):
        self.config = config
        self.output_frame_dir = config.output_frame_dir
        self.memory_jsonl_path = config.memory_jsonl_path
        self._ensure_output_dirs()

    def _ensure_output_dirs(self) -> None:
        self.output_frame_dir.mkdir(parents=True, exist_ok=True)
        self.memory_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.memory_jsonl_path.exists():
            self.memory_jsonl_path.touch()

    def save_memory(
        self,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None,
        user_question: str,
        model_answer: str,
        location: LocationInfo,
        camera_source: str | None,
    ) -> MemoryRecord:
        memory_id, timestamp, _ = make_memory_id()

        frame_paths: list[str] = []
        frame_timestamps: list[str] = []

        for index, frame in enumerate(frames):
            suffix = f"_f{index + 1:02d}" if len(frames) > 1 else ""
            if self.config.save_frames:
                saved = save_frame_image(
                    frame,
                    self.output_frame_dir,
                    memory_id,
                    suffix=suffix,
                )
                frame_paths.append(relative_path(saved, PROJECT_ROOT))
            else:
                frame_paths.append(
                    relative_path(
                        self.output_frame_dir / f"{memory_id}{suffix}.jpg",
                        PROJECT_ROOT,
                    )
                )

            if frame_items and index < len(frame_items):
                frame_timestamps.append(
                    frame_capture_timestamp_iso(frame_items[index].timestamp)
                )

        record = MemoryRecord(
            memory_id=memory_id,
            timestamp=timestamp,
            user_question=user_question,
            model_answer=model_answer,
            frame_paths=frame_paths,
            frame_timestamps=frame_timestamps,
            location=location,
            camera_source=camera_source,
        )

        self._append_jsonl(record)
        return record

    def _append_jsonl(self, record: MemoryRecord) -> None:
        line = json.dumps(
            record.model_dump(exclude_defaults=True),
            ensure_ascii=False,
        )
        with self.memory_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

        logger.info("Appended memory %s to %s", record.memory_id, self.memory_jsonl_path)
