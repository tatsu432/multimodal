import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils import parse_bool_env, parse_optional_float_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_FRAME_SOURCE_TYPES = frozenset({"rtmp", "webcam", "video"})
VALID_VLM_PROVIDERS = frozenset({"openai"})


@dataclass
class Config:
    frame_source_type: str
    rtmp_url: str
    webcam_index: int
    video_path: str
    vlm_provider: str
    vlm_model: str
    openai_api_key: str
    frame_buffer_size: int
    capture_sample_interval_sec: float
    num_frames_per_query: int
    output_frame_dir: Path
    memory_jsonl_path: Path
    location_label: str | None
    save_frames: bool
    max_runtime_seconds: float | None

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")

        output_frame_dir = Path(
            os.getenv("OUTPUT_FRAME_DIR", "outputs/frames")
        )
        if not output_frame_dir.is_absolute():
            output_frame_dir = PROJECT_ROOT / output_frame_dir

        memory_jsonl_path = Path(
            os.getenv("MEMORY_JSONL_PATH", "outputs/memories.jsonl")
        )
        if not memory_jsonl_path.is_absolute():
            memory_jsonl_path = PROJECT_ROOT / memory_jsonl_path

        location_raw = os.getenv("LOCATION_LABEL", "").strip()
        location_label = location_raw if location_raw else None

        return cls(
            frame_source_type=os.getenv("FRAME_SOURCE_TYPE", "rtmp")
            .strip()
            .lower(),
            rtmp_url=os.getenv("RTMP_URL", "rtmp://localhost:1935/live/gopro"),
            webcam_index=int(os.getenv("WEBCAM_INDEX", "0")),
            video_path=os.getenv("VIDEO_PATH", "").strip(),
            vlm_provider=os.getenv("VLM_PROVIDER", "openai").strip().lower(),
            vlm_model=os.getenv("VLM_MODEL", "gpt-5.5"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            frame_buffer_size=int(os.getenv("FRAME_BUFFER_SIZE", "8")),
            capture_sample_interval_sec=float(
                os.getenv("CAPTURE_SAMPLE_INTERVAL_SEC", "1.0")
            ),
            num_frames_per_query=int(os.getenv("NUM_FRAMES_PER_QUERY", "1")),
            output_frame_dir=output_frame_dir,
            memory_jsonl_path=memory_jsonl_path,
            location_label=location_label,
            save_frames=parse_bool_env(os.getenv("SAVE_FRAMES", "true")),
            max_runtime_seconds=parse_optional_float_env(
                os.getenv("MAX_RUNTIME_SECONDS", "")
            ),
        )

    def validate(self) -> None:
        if self.frame_source_type not in VALID_FRAME_SOURCE_TYPES:
            raise ValueError(
                f"FRAME_SOURCE_TYPE must be one of {sorted(VALID_FRAME_SOURCE_TYPES)}, "
                f"got {self.frame_source_type!r}"
            )

        if self.frame_source_type == "video" and not self.video_path:
            raise ValueError(
                "VIDEO_PATH is required when FRAME_SOURCE_TYPE=video"
            )

        if self.frame_source_type == "video":
            video = Path(self.video_path)
            if not video.is_file():
                raise ValueError(f"Video file not found: {video}")

        if self.vlm_provider not in VALID_VLM_PROVIDERS:
            raise ValueError(
                f"VLM_PROVIDER must be one of {sorted(VALID_VLM_PROVIDERS)}, "
                f"got {self.vlm_provider!r}"
            )

        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required. Copy .env.example to .env and set your key."
            )

        if self.frame_buffer_size < 1:
            raise ValueError("FRAME_BUFFER_SIZE must be at least 1")

        if self.capture_sample_interval_sec <= 0:
            raise ValueError("CAPTURE_SAMPLE_INTERVAL_SEC must be positive")

        if self.num_frames_per_query < 1:
            raise ValueError("NUM_FRAMES_PER_QUERY must be at least 1")

        if (
            self.max_runtime_seconds is not None
            and self.max_runtime_seconds <= 0
        ):
            raise ValueError("MAX_RUNTIME_SECONDS must be positive when set")
