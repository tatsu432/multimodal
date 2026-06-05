import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from src.utils import parse_bool_env

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EmbeddingProvider = Literal["sentence_transformers", "openai"]


@dataclass
class Config:
    memory_jsonl_path: Path
    memory_base_dir: Path
    chroma_persist_dir: Path
    chroma_collection_name: str
    embedding_provider: EmbeddingProvider
    embedding_model: str
    openai_api_key: str
    openai_embedding_model: str
    default_limit: int
    default_should_store_only: bool
    timezone: ZoneInfo
    use_llm_answerer: bool
    llm_model: str
    rebuild_index: bool

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv(PROJECT_ROOT / ".env")

        memory_jsonl_path = Path(
            os.getenv("MEMORY_JSONL_PATH", "../memory_log/outputs/memories.jsonl")
        )
        if not memory_jsonl_path.is_absolute():
            memory_jsonl_path = PROJECT_ROOT / memory_jsonl_path

        chroma_persist_dir = Path(
            os.getenv("CHROMA_PERSIST_DIR", "outputs/chroma")
        )
        if not chroma_persist_dir.is_absolute():
            chroma_persist_dir = PROJECT_ROOT / chroma_persist_dir

        provider_raw = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
        if provider_raw in {"openai", "openai_embeddings"}:
            embedding_provider: EmbeddingProvider = "openai"
        else:
            embedding_provider = "sentence_transformers"

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
            memory_base_dir=_derive_memory_base_dir(memory_jsonl_path),
            chroma_persist_dir=chroma_persist_dir,
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME", "visual_memories"
            ).strip(),
            embedding_provider=embedding_provider,
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ).strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip(),
            default_limit=int(os.getenv("DEFAULT_LIMIT", "10")),
            default_should_store_only=parse_bool_env(
                os.getenv("DEFAULT_SHOULD_STORE_ONLY", "true")
            ),
            timezone=timezone,
            use_llm_answerer=parse_bool_env(
                os.getenv("USE_LLM_ANSWERER", "false")
            ),
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            rebuild_index=parse_bool_env(os.getenv("REBUILD_INDEX", "false")),
        )

    def validate(self) -> None:
        if self.default_limit < 1:
            raise ValueError("DEFAULT_LIMIT must be at least 1")

        if not self.memory_jsonl_path.is_file():
            logger.warning(
                "Memory file not found: %s — starting with 0 records",
                self.memory_jsonl_path,
            )

        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai"
            )

        if self.use_llm_answerer:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when USE_LLM_ANSWERER=true"
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
