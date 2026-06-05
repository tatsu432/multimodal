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
    if path.is_absolute():
        return path
    return memory_base_dir / path


def format_timestamp_display(dt: datetime | None, fallback: str = "") -> str:
    if dt is None:
        return fallback
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_objects(objects: list[str]) -> str:
    return ", ".join(objects) if objects else "-"


def memory_to_embedding_text(memory: MemoryRecord) -> str:
    objects_str = ", ".join(memory.objects) if memory.objects else "none"
    text_visible_str = (
        ", ".join(memory.text_visible) if memory.text_visible else "none"
    )
    location_label = memory.location.label or "not available"

    return (
        f"Summary: {memory.summary}\n"
        f"Objects: {objects_str}.\n"
        f"Scene type: {memory.scene_type}.\n"
        f"Visible text: {text_visible_str}.\n"
        f"Location: {location_label}.\n"
        f"Privacy risk: {memory.privacy_risk}.\n"
        f"Reason: {memory.memory_reason}"
    )
