import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.utils import parse_bool_env, parse_optional_float_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALID_FRAME_SOURCE_TYPES = frozenset({"camera", "webcam", "video"})
VALID_VLM_PROVIDERS = frozenset({"openai", "ollama"})
VALID_GEOCODE_PROVIDERS = frozenset({"nominatim"})


def _optional_label(value: str) -> str | None:
    stripped = value.strip()
    return stripped if stripped else None


def _optional_path(value: str) -> Path | None:
    stripped = value.strip()
    if not stripped:
        return None
    path = Path(stripped)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass
class Config:
    frame_source_type: str
    camera_source: str
    camera_preset_override: str | None
    camera_url_override: str | None
    webcam_index: int
    video_path: str
    vlm_provider: str
    vlm_model: str
    openai_api_key: str
    ollama_base_url: str
    frame_buffer_size: int
    capture_sample_interval_sec: float
    num_frames_per_query: int
    output_frame_dir: Path
    memory_jsonl_path: Path
    location_label: str | None
    location_lat: float | None
    location_lon: float | None
    tapo_location_label: str | None
    tapo_location_lat: float | None
    tapo_location_lon: float | None
    phone_location_label: str | None
    phone_location_lat: float | None
    phone_location_lon: float | None
    location_server_enabled: bool
    location_server_host: str
    location_server_port: int
    location_server_cert: Path | None
    location_server_key: Path | None
    location_gps_max_age_sec: float
    geocode_enabled: bool
    geocode_provider: str
    nominatim_base_url: str
    geocode_cache_path: Path
    geocode_timeout_sec: float
    geocode_skip_if_label_set: bool
    save_frames: bool
    max_runtime_seconds: float | None
    # SQLite memory database
    memory_db_path: Path
    # Passive observer
    passive_observation_interval_sec: float
    passive_save_frames: bool
    passive_frame_dir: Path
    promoted_event_frame_dir: Path
    # LTM query retrieval budgets
    ltm_max_passive_rows: int
    ltm_promoted_event_top_k: int
    ltm_active_query_top_k: int
    ltm_final_event_k: int
    ltm_use_visual_grounding: bool
    # Long-term query logging (separate DB; excluded from retrieval)
    query_log_enabled: bool
    query_log_db_path: Path

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

        geocode_cache_path = Path(
            os.getenv("GEOCODE_CACHE_PATH", "outputs/geocode_cache.sqlite")
        )
        if not geocode_cache_path.is_absolute():
            geocode_cache_path = PROJECT_ROOT / geocode_cache_path

        memory_db_path = Path(
            os.getenv("MEMORY_DB_PATH", "outputs/memory.sqlite")
        )
        if not memory_db_path.is_absolute():
            memory_db_path = PROJECT_ROOT / memory_db_path

        query_log_db_path = Path(
            os.getenv("QUERY_LOG_DB_PATH", "outputs/long_term_query_logs.sqlite")
        )
        if not query_log_db_path.is_absolute():
            query_log_db_path = PROJECT_ROOT / query_log_db_path

        passive_frame_dir = Path(
            os.getenv("PASSIVE_FRAME_DIR", "outputs/passive_frames")
        )
        if not passive_frame_dir.is_absolute():
            passive_frame_dir = PROJECT_ROOT / passive_frame_dir

        promoted_event_frame_dir = Path(
            os.getenv("PROMOTED_EVENT_FRAME_DIR", "outputs/event_frames")
        )
        if not promoted_event_frame_dir.is_absolute():
            promoted_event_frame_dir = PROJECT_ROOT / promoted_event_frame_dir

        return cls(
            frame_source_type=os.getenv("FRAME_SOURCE_TYPE", "camera")
            .strip()
            .lower(),
            camera_source=os.getenv("CAMERA_SOURCE", "tapo-rtsp").strip().lower(),
            camera_preset_override=None,
            camera_url_override=None,
            webcam_index=int(os.getenv("WEBCAM_INDEX", "0")),
            video_path=os.getenv("VIDEO_PATH", "").strip(),
            vlm_provider=os.getenv("VLM_PROVIDER", "openai").strip().lower(),
            vlm_model=os.getenv("VLM_MODEL", "gpt-5.5"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).strip(),
            frame_buffer_size=int(os.getenv("FRAME_BUFFER_SIZE", "8")),
            capture_sample_interval_sec=float(
                os.getenv("CAPTURE_SAMPLE_INTERVAL_SEC", "1.0")
            ),
            num_frames_per_query=int(os.getenv("NUM_FRAMES_PER_QUERY", "1")),
            output_frame_dir=output_frame_dir,
            memory_jsonl_path=memory_jsonl_path,
            location_label=_optional_label(os.getenv("LOCATION_LABEL", "")),
            location_lat=parse_optional_float_env(os.getenv("LOCATION_LAT", "")),
            location_lon=parse_optional_float_env(os.getenv("LOCATION_LON", "")),
            tapo_location_label=_optional_label(os.getenv("TAPO_LOCATION_LABEL", "")),
            tapo_location_lat=parse_optional_float_env(
                os.getenv("TAPO_LOCATION_LAT", "")
            ),
            tapo_location_lon=parse_optional_float_env(
                os.getenv("TAPO_LOCATION_LON", "")
            ),
            phone_location_label=_optional_label(
                os.getenv("PHONE_LOCATION_LABEL", "")
            ),
            phone_location_lat=parse_optional_float_env(
                os.getenv("PHONE_LOCATION_LAT", "")
            ),
            phone_location_lon=parse_optional_float_env(
                os.getenv("PHONE_LOCATION_LON", "")
            ),
            location_server_enabled=parse_bool_env(
                os.getenv("LOCATION_SERVER_ENABLED", "false")
            ),
            location_server_host=os.getenv("LOCATION_SERVER_HOST", "0.0.0.0").strip(),
            location_server_port=int(os.getenv("LOCATION_SERVER_PORT", "8765")),
            location_server_cert=_optional_path(
                os.getenv("LOCATION_SERVER_CERT", "")
            ),
            location_server_key=_optional_path(
                os.getenv("LOCATION_SERVER_KEY", "")
            ),
            location_gps_max_age_sec=float(
                os.getenv("LOCATION_GPS_MAX_AGE_SEC", "120")
            ),
            geocode_enabled=parse_bool_env(os.getenv("GEOCODE_ENABLED", "true")),
            geocode_provider=os.getenv("GEOCODE_PROVIDER", "nominatim")
            .strip()
            .lower(),
            nominatim_base_url=os.getenv(
                "NOMINATIM_BASE_URL", "https://nominatim.openstreetmap.org"
            ).strip(),
            geocode_cache_path=geocode_cache_path,
            geocode_timeout_sec=float(os.getenv("GEOCODE_TIMEOUT_SEC", "5")),
            geocode_skip_if_label_set=parse_bool_env(
                os.getenv("GEOCODE_SKIP_IF_LABEL_SET", "false")
            ),
            save_frames=parse_bool_env(os.getenv("SAVE_FRAMES", "true")),
            max_runtime_seconds=parse_optional_float_env(
                os.getenv("MAX_RUNTIME_SECONDS", "")
            ),
            memory_db_path=memory_db_path,
            passive_observation_interval_sec=float(
                os.getenv("PASSIVE_OBSERVATION_INTERVAL_SEC", "30")
            ),
            passive_save_frames=parse_bool_env(
                os.getenv("PASSIVE_SAVE_FRAMES", "true")
            ),
            passive_frame_dir=passive_frame_dir,
            promoted_event_frame_dir=promoted_event_frame_dir,
            ltm_max_passive_rows=int(os.getenv("LTM_MAX_PASSIVE_ROWS", "1000")),
            ltm_promoted_event_top_k=int(
                os.getenv("LTM_PROMOTED_EVENT_TOP_K", "20")
            ),
            ltm_active_query_top_k=int(
                os.getenv("LTM_ACTIVE_QUERY_TOP_K", "10")
            ),
            ltm_final_event_k=int(os.getenv("LTM_FINAL_EVENT_K", "5")),
            ltm_use_visual_grounding=parse_bool_env(
                os.getenv("LTM_USE_VISUAL_GROUNDING", "true")
            ),
            query_log_enabled=parse_bool_env(
                os.getenv("LTM_QUERY_LOG_ENABLED", "true")
            ),
            query_log_db_path=query_log_db_path,
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

        if self.location_server_enabled:
            if self.location_server_port < 1 or self.location_server_port > 65535:
                raise ValueError("LOCATION_SERVER_PORT must be between 1 and 65535")
            if self.location_gps_max_age_sec <= 0:
                raise ValueError("LOCATION_GPS_MAX_AGE_SEC must be positive")
            if self.location_server_cert and not self.location_server_cert.is_file():
                raise ValueError(
                    f"LOCATION_SERVER_CERT not found: {self.location_server_cert}"
                )
            if self.location_server_key and not self.location_server_key.is_file():
                raise ValueError(
                    f"LOCATION_SERVER_KEY not found: {self.location_server_key}"
                )

        if self.geocode_enabled:
            if self.geocode_provider not in VALID_GEOCODE_PROVIDERS:
                raise ValueError(
                    f"GEOCODE_PROVIDER must be one of "
                    f"{sorted(VALID_GEOCODE_PROVIDERS)}, "
                    f"got {self.geocode_provider!r}"
                )
            if self.geocode_timeout_sec <= 0:
                raise ValueError("GEOCODE_TIMEOUT_SEC must be positive")
            if not self.nominatim_base_url.startswith(("http://", "https://")):
                raise ValueError("NOMINATIM_BASE_URL must be an http(s) URL")

    @property
    def vlm_source_key(self) -> str:
        if self.frame_source_type == "camera":
            return self.camera_preset_override or self.camera_source
        return self.frame_source_type

    @property
    def camera_source_key(self) -> str:
        return self.vlm_source_key
