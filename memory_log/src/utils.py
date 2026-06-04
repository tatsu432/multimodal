import base64
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from src.schema import REQUIRED_ANALYSIS_FIELDS

logger = logging.getLogger(__name__)

VALID_PRIVACY_RISKS = frozenset({"low", "medium", "high"})


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_optional_float_env(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def local_timezone() -> ZoneInfo:
    return datetime.now().astimezone().tzinfo or timezone.utc  # type: ignore[return-value]


def make_memory_id(now: datetime | None = None) -> tuple[str, str, datetime]:
    """
    Returns (memory_id, iso_timestamp, aware_datetime).
    memory_id uses filesystem-safe form: 2026-06-04T23-12-30.123
    timestamp uses ISO8601 with offset: 2026-06-04T23:12:30.123+09:00
    """
    if now is None:
        now = datetime.now().astimezone()

    millis = int(now.microsecond / 1000)
    memory_id = now.strftime("%Y-%m-%dT%H-%M-%S") + f".{millis:03d}"
    timestamp = now.isoformat(timespec="milliseconds")
    return memory_id, timestamp, now


def relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def resize_frame(frame: np.ndarray, max_width: int = 768) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


def encode_frame_as_base64_jpeg(
    frame: np.ndarray,
    max_width: int = 768,
    quality: int = 85,
) -> str:
    frame = resize_frame(frame, max_width=max_width)

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    return base64.b64encode(buffer).decode("utf-8")


def save_frame_image(
    frame: np.ndarray,
    directory: Path,
    memory_id: str,
    max_width: int = 1280,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{memory_id}.jpg"
    ok = cv2.imwrite(str(path), resize_frame(frame, max_width=max_width))
    if not ok:
        raise RuntimeError(f"Failed to save frame to {path}")
    return path


_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)


def strip_markdown_json_fences(text: str) -> str:
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _coerce_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [value.strip()]
    logger.warning("Field %s has unexpected type %s; using []", field_name, type(value))
    return []


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_privacy_risk(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_PRIVACY_RISKS:
            return normalized
    return "medium"


def parse_vlm_memory_analysis(raw_text: str) -> dict | None:
    """
    Parse and normalize VLM JSON for memory analysis.
    Returns None if JSON cannot be parsed at all.
    """
    cleaned = strip_markdown_json_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s", exc)
        return None

    if not isinstance(data, dict):
        logger.warning("VLM response is not a JSON object")
        return None

    summary = str(data.get("summary", "")).strip()
    scene_type = str(data.get("scene_type", "unknown_scene")).strip() or "unknown_scene"
    memory_reason = str(data.get("memory_reason", "")).strip()

    if not summary:
        summary = "Scene could not be summarized."
    if not memory_reason:
        memory_reason = "No reason provided by VLM."

    normalized = {
        "summary": summary,
        "scene_type": scene_type,
        "objects": _coerce_str_list(data.get("objects"), "objects"),
        "people_count": max(0, _coerce_int(data.get("people_count"), 0)),
        "text_visible": _coerce_str_list(data.get("text_visible"), "text_visible"),
        "should_store": _coerce_bool(data.get("should_store"), default=True),
        "memory_reason": memory_reason,
        "privacy_risk": _normalize_privacy_risk(data.get("privacy_risk")),
    }

    missing = REQUIRED_ANALYSIS_FIELDS - set(data.keys())
    if missing:
        logger.debug("VLM JSON missing fields (defaults applied): %s", sorted(missing))

    return normalized
