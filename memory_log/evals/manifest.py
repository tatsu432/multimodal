"""Pydantic manifest models — common currency for eval datasets and benchmark adapters.

Each manifest describes one eval scenario containing:
  - A video file (current/live stream mock)
  - Live QA questions with ask_at_sec + gold answers
  - LTM QA questions with gold answers + gold evidence windows
  - Past memory records (seed or replay-ingestion mode)

Example:
    manifest = load_manifest(Path("evals/datasets/toy/desk_001.json"))
    for q in manifest.live_questions:
        frames, items = replay.frames_at(q.ask_at_sec, n=4)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GoldLocation(BaseModel):
    """Optional synthetic location attached to a question or memory record."""

    label: str | None = None
    lat: float | None = None
    lon: float | None = None


class LiveQuestion(BaseModel):
    """One live-VQA question asked at a specific video timestamp."""

    id: str
    ask_at_sec: float  # media time at which to seek + ask
    question: str
    gold_answer: str
    acceptable_answers: list[str] = Field(default_factory=list)
    unacceptable_answers: list[str] = Field(default_factory=list)
    answer_type: Literal["short_text", "mcq", "unanswerable"] = "short_text"
    choices: list[str] = Field(default_factory=list)  # MCQ only
    gold_evidence_window: tuple[float, float] | None = None  # media-seconds [start, end]
    location: GoldLocation | None = None


class SeedMemory(BaseModel):
    """A structured past-memory record used in LTM seed mode.

    The harness injects this directly into the eval's isolated SQLite DB via
    SQLiteWriter, using `timestamp` as the content timestamp (not wall-clock).
    """

    kind: Literal["active_query", "passive", "daily_summary"] = "active_query"
    timestamp: str  # ISO8601 with offset — written to content timestamp columns
    location: GoldLocation | None = None
    # active_query / passive fields
    user_question: str = ""
    model_answer: str = ""
    frame_paths: list[str] = Field(default_factory=list)
    # daily_summary fields
    summary_text: str = ""
    coverage_start: str | None = None
    coverage_end: str | None = None
    camera_source: str = "eval_replay"


class MemoryQuestion(BaseModel):
    """One LTM question asked after past memories have been built."""

    id: str
    query_time_sec: float = 0.0  # media time for "current video" grounding; 0 = no grounding
    question: str
    gold_answer: str
    acceptable_answers: list[str] = Field(default_factory=list)
    unacceptable_answers: list[str] = Field(default_factory=list)
    answer_type: Literal["short_text", "mcq", "unanswerable"] = "short_text"
    answerable: bool = True
    choices: list[str] = Field(default_factory=list)  # MCQ only
    gold_evidence_windows: list[tuple[float, float]] = Field(default_factory=list)
    location: GoldLocation | None = None


class EvalManifest(BaseModel):
    """Top-level manifest describing one eval scenario."""

    video_id: str
    # Path to the "current/live" video (relative to manifest dir or absolute)
    video_path: str
    # Synthetic clock origin for media t=0. ISO8601 with offset.
    # None → harness uses current time (non-reproducible timestamps, use for smoke tests).
    base_timestamp: str | None = None
    default_location: GoldLocation = Field(default_factory=GoldLocation)
    description: str = ""

    # ---- Live QA ----
    live_questions: list[LiveQuestion] = Field(default_factory=list)

    # ---- LTM QA ----
    # seed  : inject structured seed_memories directly into the isolated DB
    # replay: ingest history_video_path through the real passive-observer pipeline
    memory_mode: Literal["seed", "replay"] = "seed"
    seed_memories: list[SeedMemory] = Field(default_factory=list)
    history_video_path: str | None = None  # replay mode: video to ingest as past memory
    memory_questions: list[MemoryQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ltm(self) -> "EvalManifest":
        if (
            self.memory_mode == "replay"
            and not self.history_video_path
            and self.memory_questions
        ):
            raise ValueError(
                "memory_mode=replay requires history_video_path when memory_questions are defined"
            )
        return self

    # ---- helpers ----

    def resolve_path(self, raw: str, base_dir: Path) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else base_dir / p

    def video_abs(self, base_dir: Path) -> Path:
        return self.resolve_path(self.video_path, base_dir)

    def history_video_abs(self, base_dir: Path) -> Path | None:
        if not self.history_video_path:
            return None
        return self.resolve_path(self.history_video_path, base_dir)


def load_manifest(path: Path) -> EvalManifest:
    """Load and validate a manifest from a JSON file."""
    data = json.loads(path.read_text())
    return EvalManifest.model_validate(data)


def save_manifest(manifest: EvalManifest, path: Path) -> None:
    """Write a manifest to a JSON file (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))
