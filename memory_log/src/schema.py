from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PrivacyRisk = Literal["low", "medium", "high"]


class LocationInfo(BaseModel):
    label: str | None = None
    lat: float | None = None
    lon: float | None = None
    source: str = "manual_or_not_available"
    full_address: str | None = None
    city: str | None = None
    prefecture: str | None = None
    country: str | None = None
    postal_code: str | None = None
    geocode_provider: str | None = None
    geocoded_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        if not normalized.get("full_address") and normalized.get("address"):
            normalized["full_address"] = normalized["address"]
        if not normalized.get("city") and normalized.get("place_name"):
            normalized["city"] = normalized["place_name"]
        if not normalized.get("prefecture") and normalized.get("admin_area"):
            normalized["prefecture"] = normalized["admin_area"]
        return normalized

    def search_text(self) -> str:
        parts = [
            self.label,
            self.full_address,
            self.city,
            self.prefecture,
            self.country,
            self.postal_code,
        ]
        return " ".join(part for part in parts if part)

    def display_name(self) -> str:
        if self.label:
            return self.label
        if self.full_address:
            return self.full_address
        parts = [self.city, self.prefecture, self.country]
        text = ", ".join(part for part in parts if part)
        if text:
            return text
        if self.lat is not None and self.lon is not None:
            return f"{self.lat:.5f},{self.lon:.5f}"
        return "not available"


class MemoryRecord(BaseModel):
    memory_id: str
    timestamp: str
    user_question: str
    location: LocationInfo
    model_answer: str = ""
    frame_paths: list[str] = Field(default_factory=list)
    frame_timestamps: list[str] = Field(default_factory=list)
    camera_source: str | None = None
    image_path: str = ""
    summary: str = ""
    objects: list[str] = Field(default_factory=list)
    scene_type: str = ""
    people_count: int = 0
    text_visible: list[str] = Field(default_factory=list)
    should_store: bool = True
    memory_reason: str = ""
    privacy_risk: PrivacyRisk = "medium"

    @field_validator("privacy_risk", mode="before")
    @classmethod
    def normalize_privacy_risk(cls, value: object) -> str:
        if value is None:
            return "medium"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"low", "medium", "high"}:
                return normalized
        return "medium"

    @model_validator(mode="after")
    def normalize_legacy(self) -> "MemoryRecord":
        if not self.frame_paths and self.image_path:
            self.frame_paths = [self.image_path]
        if not self.model_answer and self.summary:
            self.model_answer = self.summary
        return self

    def primary_image_path(self) -> str:
        if self.frame_paths:
            return self.frame_paths[0]
        return self.image_path

    def display_text(self) -> str:
        return self.model_answer or self.summary or self.user_question
