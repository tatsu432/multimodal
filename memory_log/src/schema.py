from typing import Literal

from pydantic import BaseModel, Field, field_validator

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
