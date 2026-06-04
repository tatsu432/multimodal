import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


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
