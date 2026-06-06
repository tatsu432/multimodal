import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils import parse_bool_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_FRAME_SOURCE_TYPES = frozenset({"camera", "rtmp", "webcam", "video"})
VALID_VLM_PROVIDERS = frozenset({"openai", "ollama"})


@dataclass
class Config:
    frame_source_type: str
    camera_source: str
    camera_preset_override: str | None
    camera_url_override: str | None
    rtmp_url: str
    webcam_index: int
    video_path: str
    vlm_provider: str
    vlm_model: str
    openai_api_key: str
    ollama_base_url: str
    frame_sample_dir: Path
    num_frames_per_query: int
    save_queried_frames: bool
    frame_buffer_size: int
    rtmp_sample_interval_sec: float

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")

        frame_source_type = os.getenv("FRAME_SOURCE_TYPE", "rtmp").strip().lower()
        frame_sample_dir = Path(
            os.getenv("FRAME_SAMPLE_DIR", "outputs/sampled_frames")
        )
        if not frame_sample_dir.is_absolute():
            frame_sample_dir = PROJECT_ROOT / frame_sample_dir

        return cls(
            frame_source_type=frame_source_type,
            camera_source=os.getenv("CAMERA_SOURCE", "tapo-rtsp").strip().lower(),
            camera_preset_override=None,
            camera_url_override=None,
            rtmp_url=os.getenv("RTMP_URL", "rtmp://localhost:1935/live/gopro"),
            webcam_index=int(os.getenv("WEBCAM_INDEX", "0")),
            video_path=os.getenv("VIDEO_PATH", "").strip(),
            vlm_provider=os.getenv("VLM_PROVIDER", "openai").strip().lower(),
            vlm_model=os.getenv("VLM_MODEL", "gpt-5.5"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).strip(),
            frame_sample_dir=frame_sample_dir,
            num_frames_per_query=int(os.getenv("NUM_FRAMES_PER_QUERY", "1")),
            save_queried_frames=parse_bool_env(
                os.getenv("SAVE_QUERIED_FRAMES", "true")
            ),
            frame_buffer_size=int(os.getenv("FRAME_BUFFER_SIZE", "8")),
            rtmp_sample_interval_sec=float(
                os.getenv("RTMP_SAMPLE_INTERVAL_SEC", "1.0")
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

        if self.frame_source_type == "camera":
            from capture.stream_config import resolve_source

            resolve_source(
                self.camera_preset_override or self.camera_source,
                self.camera_url_override,
            )

        if self.vlm_provider not in VALID_VLM_PROVIDERS:
            raise ValueError(
                f"VLM_PROVIDER must be one of {sorted(VALID_VLM_PROVIDERS)}, "
                f"got {self.vlm_provider!r}"
            )

        if self.vlm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when VLM_PROVIDER=openai. "
                "Copy .env.example to .env and set your key, "
                "or set VLM_PROVIDER=ollama for a local model."
            )

        if not self.vlm_model:
            raise ValueError("VLM_MODEL is required.")

        if self.num_frames_per_query < 1:
            raise ValueError("NUM_FRAMES_PER_QUERY must be at least 1")

        if self.frame_buffer_size < 1:
            raise ValueError("FRAME_BUFFER_SIZE must be at least 1")

        if self.rtmp_sample_interval_sec <= 0:
            raise ValueError("RTMP_SAMPLE_INTERVAL_SEC must be positive")

    @property
    def vlm_source_key(self) -> str:
        if self.frame_source_type == "camera":
            return self.camera_preset_override or self.camera_source
        return self.frame_source_type
