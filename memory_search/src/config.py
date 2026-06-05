import logging
import os
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.utils import parse_bool_env

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    memory_jsonl_path: Path
    memory_base_dir: Path
    default_limit: int
    default_should_store_only: bool
    timezone: ZoneInfo
    use_llm_answerer: bool
    llm_provider: str
    openai_api_key: str
    llm_model: str
    ollama_base_url: str

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")

        memory_jsonl_path = Path(
            os.getenv("MEMORY_JSONL_PATH", "../memory_log/outputs/memories.jsonl")
        )
        if not memory_jsonl_path.is_absolute():
            memory_jsonl_path = PROJECT_ROOT / memory_jsonl_path

        memory_base_dir = _derive_memory_base_dir(memory_jsonl_path)

        timezone_name = os.getenv("TIMEZONE", "Asia/Tokyo").strip()
        try:
            timezone = ZoneInfo(timezone_name)
        except Exception:
            logger.warning(
                "Invalid TIMEZONE %r; falling back to Asia/Tokyo",
                timezone_name,
            )
            timezone = ZoneInfo("Asia/Tokyo")

        return cls(
            memory_jsonl_path=memory_jsonl_path,
            memory_base_dir=memory_base_dir,
            default_limit=int(os.getenv("DEFAULT_LIMIT", "10")),
            default_should_store_only=parse_bool_env(
                os.getenv("DEFAULT_SHOULD_STORE_ONLY", "true")
            ),
            timezone=timezone,
            use_llm_answerer=parse_bool_env(
                os.getenv("USE_LLM_ANSWERER", "false")
            ),
            llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).strip(),
        )

    def validate(self) -> None:
        if self.default_limit < 1:
            raise ValueError("DEFAULT_LIMIT must be at least 1")

        if not self.memory_jsonl_path.is_file():
            logger.warning(
                "Memory file not found: %s — starting with 0 records",
                self.memory_jsonl_path,
            )

        if self.use_llm_answerer:
            if self.llm_provider not in {"openai", "ollama"}:
                raise ValueError(
                    "LLM_PROVIDER must be 'openai' or 'ollama' when USE_LLM_ANSWERER=true"
                )
            if self.llm_provider == "openai" and not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when USE_LLM_ANSWERER=true "
                    "and LLM_PROVIDER=openai"
                )
            if not self.llm_model:
                raise ValueError(
                    "LLM_MODEL is required when USE_LLM_ANSWERER=true"
                )


def _derive_memory_base_dir(memory_jsonl_path: Path) -> Path:
    """
    image_path in JSONL is relative to the memory_log project root.
    If JSONL lives at .../memory_log/outputs/memories.jsonl, base is .../memory_log.
    """
    if memory_jsonl_path.parent.name == "outputs":
        return memory_jsonl_path.parent.parent
    return memory_jsonl_path.parent
