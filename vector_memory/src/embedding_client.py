from __future__ import annotations

import logging
from typing import Protocol

from openai import OpenAI

from src.config import Config, EmbeddingProvider

logger = logging.getLogger(__name__)

BATCH_SIZE = 32


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class SentenceTransformersEmbeddingClient:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for local embeddings. "
                    "Install with: uv sync"
                ) from exc
            logger.info("Loading embedding model %s …", self._model_name)
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load embedding model {self._model_name!r}: {exc}"
                ) from exc
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            vectors = model.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(vectors.tolist())
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class OpenAIEmbeddingClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            all_embeddings.extend(item.embedding for item in ordered)
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def create_embedding_client(config: Config) -> EmbeddingClient:
    provider: EmbeddingProvider = config.embedding_provider
    if provider == "openai":
        return OpenAIEmbeddingClient(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
        )
    return SentenceTransformersEmbeddingClient(model_name=config.embedding_model)
