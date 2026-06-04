from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

PrivacyRisk = Literal["low", "medium", "high"]

REQUIRED_ANALYSIS_FIELDS = frozenset(
    {
        "summary",
        "scene_type",
        "objects",
        "people_count",
        "text_visible",
        "should_store",
        "memory_reason",
        "privacy_risk",
    }
)


class LocationInfo(BaseModel):
    label: str | None = None
    lat: float | None = None
    lon: float | None = None
    source: str = "manual_or_not_available"


class MemoryRecord(BaseModel):
    memory_id: str
    timestamp: str
    image_path: str
    user_question: str
    summary: str
    objects: list[str]
    scene_type: str
    people_count: int
    text_visible: list[str]
    location: LocationInfo
    should_store: bool
    memory_reason: str
    privacy_risk: PrivacyRisk

    @field_validator("privacy_risk", mode="before")
    @classmethod
    def normalize_privacy_risk(cls, value: object) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"low", "medium", "high"}:
                return normalized
        raise ValueError("privacy_risk must be low, medium, or high")


@dataclass
class ParsedMemoryQuery:
    original_question: str
    start_time: datetime | None
    end_time: datetime | None
    keywords: list[str]
    object_filters: list[str]
    scene_type_filters: list[str]
    location_filters: list[str]
    privacy_risk: str | None
    people_only: bool
    text_visible_only: bool
    recent_bias: bool
    limit: int = 10


@dataclass
class LoadedMemory:
    record: MemoryRecord
    parsed_timestamp: datetime | None


@dataclass
class ScoredMemory:
    record: MemoryRecord
    score: float
    parsed_timestamp: datetime | None
    display_image_path: str
