import json
import logging
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, Config
from src.schema import LocationInfo, MemoryRecord
from src.utils import make_memory_id, relative_path, save_frame_image

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

    def save_memory(self, frame: np.ndarray, analysis: dict) -> MemoryRecord:
        memory_id, timestamp, _ = make_memory_id()

        image_path = ""
        if self.config.save_frames:
            saved = save_frame_image(
                frame,
                self.output_frame_dir,
                memory_id,
            )
            image_path = relative_path(saved, PROJECT_ROOT)
        else:
            image_path = relative_path(
                self.output_frame_dir / f"{memory_id}.jpg",
                PROJECT_ROOT,
            )

        location = LocationInfo(
            label=self.config.location_label,
            lat=None,
            lon=None,
            source=(
                "manual"
                if self.config.location_label
                else "manual_or_not_available"
            ),
        )

        record = MemoryRecord(
            memory_id=memory_id,
            timestamp=timestamp,
            image_path=image_path,
            summary=analysis["summary"],
            objects=analysis["objects"],
            scene_type=analysis["scene_type"],
            people_count=analysis["people_count"],
            text_visible=analysis["text_visible"],
            location=location,
            should_store=analysis["should_store"],
            memory_reason=analysis["memory_reason"],
            privacy_risk=analysis["privacy_risk"],
        )

        self._append_jsonl(record)
        return record

    def _append_jsonl(self, record: MemoryRecord) -> None:
        line = json.dumps(
            record.model_dump(),
            ensure_ascii=False,
        )
        with self.memory_jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()

        logger.info("Appended memory %s to %s", record.memory_id, self.memory_jsonl_path)
