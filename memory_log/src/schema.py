from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

PrivacyRisk = Literal["low", "medium", "high"]


class LocationInfo(BaseModel):
    label: str | None = None
    lat: float | None = None
    lon: float | None = None
    source: str = "manual_or_not_available"


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
