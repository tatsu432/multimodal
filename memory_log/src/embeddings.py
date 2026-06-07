"""Embedding provider abstraction — mirrors the vlm_client.py pattern.

Two providers are supported:
  - ``ollama`` (default/primary): local, no API key, model ``nomic-embed-text``
  - ``openai``: ``text-embedding-3-small`` via the OpenAI API

``create_embedding_client(config)`` returns ``None`` (and logs a warning) when embeddings
are disabled or misconfigured, so every caller degrades gracefully to the LIKE fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger("memory_log.embeddings")


class EmbeddingClient(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; returns one float vector per input."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier string (used for ChromaDB collection namespacing)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension (may be resolved lazily after first call)."""

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider name: 'ollama' or 'openai'."""


class OllamaEmbeddingClient(EmbeddingClient):
    """Embedding client backed by the local Ollama /api/embed endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_sec: float = 30.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout_sec = timeout_sec
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        # nomic-embed-text is 768-dim; resolved lazily on first embed call
        return self._dim if self._dim is not None else 768

    @property
    def provider(self) -> str:
        return "ollama"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from providers.ollama import embeddings as ollama_embed

        vecs = ollama_embed(
            model=self._model,
            input=texts,
            base_url=self._base_url,
            timeout_sec=self._timeout_sec,
        )
        if vecs and self._dim is None:
            self._dim = len(vecs[0])
        return vecs


class OpenAIEmbeddingClient(EmbeddingClient):
    """Embedding client backed by the OpenAI embeddings API."""

    def __init__(self, model: str, api_key: str) -> None:
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._dim: int | None = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        # text-embedding-3-small is 1536-dim by default
        return self._dim if self._dim is not None else 1536

    @property
    def provider(self) -> str:
        return "openai"

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        vecs = [d.embedding for d in resp.data]
        if vecs and self._dim is None:
            self._dim = len(vecs[0])
        return vecs


def create_embedding_client(config: "Config") -> EmbeddingClient | None:
    """Build an EmbeddingClient from config, or return None on disable/misconfiguration."""
    if not config.vector_search_enabled:
        return None
    provider = config.embedding_provider
    model = config.embedding_model
    try:
        if provider == "ollama":
            return OllamaEmbeddingClient(
                model=model,
                base_url=config.ollama_base_url,
                timeout_sec=config.embedding_timeout_sec,
            )
        if provider == "openai":
            if not config.openai_api_key:
                logger.warning(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is missing; "
                    "embeddings disabled"
                )
                return None
            return OpenAIEmbeddingClient(model=model, api_key=config.openai_api_key)
        logger.warning(
            "Unknown EMBEDDING_PROVIDER %r; embeddings disabled", provider
        )
        return None
    except Exception as exc:
        logger.warning("Could not create embedding client (%s): %s", provider, exc)
        return None
