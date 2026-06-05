import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from src.config import Config
from src.schema import LoadedMemory, MemoryRecord
from src.utils import parse_record_timestamp

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    memories: list[LoadedMemory]
    total_lines: int
    skipped_json: int
    skipped_validation: int
    skipped_timestamp: int
    duplicate_ids: int

    @property
    def valid_count(self) -> int:
        return len(self.memories)


def build_memories_by_id(memories: list[LoadedMemory]) -> dict[str, LoadedMemory]:
    by_id: dict[str, LoadedMemory] = {}
    for item in memories:
        by_id[item.record.memory_id] = item
    return by_id


def load_memories(config: Config) -> LoadResult:
    path = config.memory_jsonl_path

    if not path.is_file():
        logger.warning("Memory file not found: %s", path)
        return LoadResult([], 0, 0, 0, 0, 0)

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        logger.warning("Memory file is empty: %s", path)
        return LoadResult([], 0, 0, 0, 0, 0)

    memories: list[LoadedMemory] = []
    skipped_json = 0
    skipped_validation = 0
    skipped_timestamp = 0
    duplicate_ids = 0
    total_lines = 0
    seen_ids: set[str] = set()

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        total_lines += 1

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning("Line %d: invalid JSON — %s", line_no, exc)
            skipped_json += 1
            continue

        if not isinstance(data, dict):
            logger.warning("Line %d: expected JSON object, got %s", line_no, type(data))
            skipped_validation += 1
            continue

        try:
            record = MemoryRecord.model_validate(data)
        except ValidationError as exc:
            logger.warning("Line %d: validation failed — %s", line_no, exc)
            skipped_validation += 1
            continue

        if record.memory_id in seen_ids:
            logger.warning(
                "Line %d: duplicate memory_id %r — keeping latest",
                line_no,
                record.memory_id,
            )
            duplicate_ids += 1
            memories = [m for m in memories if m.record.memory_id != record.memory_id]
        else:
            seen_ids.add(record.memory_id)

        parsed_ts = parse_record_timestamp(record.timestamp, config.timezone)
        if parsed_ts is None:
            logger.warning(
                "Line %d: malformed timestamp %r — record loaded but excluded from time filters",
                line_no,
                record.timestamp,
            )
            skipped_timestamp += 1

        memories.append(LoadedMemory(record=record, parsed_timestamp=parsed_ts))

    logger.info(
        "Loaded %d records from %s (%d lines, %d json errors, %d validation errors, "
        "%d bad timestamps, %d duplicate ids)",
        len(memories),
        path,
        total_lines,
        skipped_json,
        skipped_validation,
        skipped_timestamp,
        duplicate_ids,
    )

    return LoadResult(
        memories=memories,
        total_lines=total_lines,
        skipped_json=skipped_json,
        skipped_validation=skipped_validation,
        skipped_timestamp=skipped_timestamp,
        duplicate_ids=duplicate_ids,
    )
