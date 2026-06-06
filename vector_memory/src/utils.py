import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.schema import MemoryRecord

logger = logging.getLogger(__name__)


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_record_timestamp(
    timestamp: str,
    tz: ZoneInfo | None = None,
) -> datetime | None:
    stripped = timestamp.strip()
    if not stripped:
        return None
    try:
        dt = datetime.fromisoformat(stripped)
    except ValueError:
        return None
    if dt.tzinfo is None and tz is not None:
        dt = dt.replace(tzinfo=tz)
    return dt


def resolve_image_path(image_path: str, memory_base_dir: Path) -> Path:
    path = Path(image_path)
    if not image_path:
        return memory_base_dir / "_missing_.jpg"
    if path.is_absolute():
        return path
    return memory_base_dir / path


def resolve_record_image_path(record: MemoryRecord, memory_base_dir: Path) -> Path:
    return resolve_image_path(record.primary_image_path(), memory_base_dir)


def format_timestamp_display(dt: datetime | None, fallback: str = "") -> str:
    if dt is None:
        return fallback
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_objects(objects: list[str]) -> str:
    return ", ".join(objects) if objects else "-"


def memory_to_embedding_text(memory: MemoryRecord) -> str:
    location_label = memory.location.label or "not available"
    parts = [
        f"Question: {memory.user_question}",
        f"Answer: {memory.model_answer or memory.summary or 'not available'}",
        f"Location: {location_label}.",
    ]

    if memory.summary and memory.summary != memory.model_answer:
        parts.append(f"Summary: {memory.summary}")
    if memory.objects:
        parts.append(f"Objects: {', '.join(memory.objects)}.")
    if memory.scene_type:
        parts.append(f"Scene type: {memory.scene_type}.")
    if memory.text_visible:
        parts.append(f"Visible text: {', '.join(memory.text_visible)}.")
    if memory.privacy_risk:
        parts.append(f"Privacy risk: {memory.privacy_risk}.")
    if memory.memory_reason:
        parts.append(f"Reason: {memory.memory_reason}")

    return "\n".join(parts)
