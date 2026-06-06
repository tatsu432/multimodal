from __future__ import annotations

import logging

import chromadb

from src.config import Config
from src.embedding_client import EmbeddingClient
from src.schema import IndexResult, LoadedMemory, VectorHit
from src.utils import memory_to_embedding_text

logger = logging.getLogger(__name__)

UPSERT_BATCH_SIZE = 64


class VectorStore:
    def __init__(self, config: Config, embedding_client: EmbeddingClient) -> None:
        self._config = config
        self._embedding_client = embedding_client
        config.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(config.chroma_persist_dir))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open Chroma at {config.chroma_persist_dir}: {exc}"
            ) from exc
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self._client.get_or_create_collection(
                name=self._config.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create Chroma collection "
                f"{self._config.chroma_collection_name!r}: {exc}"
            ) from exc

    def _rebuild_collection(self) -> None:
        name = self._config.chroma_collection_name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _existing_ids(self) -> set[str]:
        try:
            result = self._collection.get(include=[])
            ids = result.get("ids") or []
            return set(ids)
        except Exception as exc:
            logger.warning("Could not read existing Chroma IDs: %s", exc)
            return set()

    def index_memories(
        self,
        memories: list[LoadedMemory],
        *,
        rebuild: bool,
    ) -> IndexResult:
        if rebuild:
            logger.info("REBUILD_INDEX=true — clearing collection %s", self._config.chroma_collection_name)
            self._rebuild_collection()
            existing_ids: set[str] = set()
        else:
            existing_ids = self._existing_ids()

        to_index: list[LoadedMemory] = []
        skipped_duplicate = 0

        for item in memories:
            record = item.record
            if self._config.default_should_store_only and not record.should_store:
                continue
            if record.memory_id in existing_ids:
                skipped_duplicate += 1
                continue
            to_index.append(item)

        indexed = 0
        for start in range(0, len(to_index), UPSERT_BATCH_SIZE):
            batch = to_index[start : start + UPSERT_BATCH_SIZE]
            ids = [item.record.memory_id for item in batch]
            documents = [memory_to_embedding_text(item.record) for item in batch]
            metadatas = [_record_metadata(item) for item in batch]

            try:
                embeddings = self._embedding_client.embed_texts(documents)
            except Exception as exc:
                raise RuntimeError(f"Embedding failed during indexing: {exc}") from exc

            try:
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            except Exception as exc:
                raise RuntimeError(f"Chroma upsert failed: {exc}") from exc

            indexed += len(batch)

        return IndexResult(
            loaded=len(memories),
            indexed=indexed,
            skipped_duplicate=skipped_duplicate,
        )

    def query(self, query_embedding: list[float], n_results: int) -> list[VectorHit]:
        if n_results < 1:
            return []

        try:
            count = self._collection.count()
        except Exception:
            count = 0

        if count == 0:
            return []

        n_results = min(n_results, count)

        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["distances"],
            )
        except Exception as exc:
            logger.warning("Chroma query failed: %s", exc)
            return []

        ids_nested = result.get("ids") or [[]]
        distances_nested = result.get("distances") or [[]]

        hits: list[VectorHit] = []
        for memory_id, distance in zip(ids_nested[0], distances_nested[0]):
            hits.append(VectorHit(memory_id=memory_id, distance=float(distance)))
        return hits


def _record_metadata(item: LoadedMemory) -> dict[str, str | int | float | bool]:
    record = item.record
    objects_text = ", ".join(record.objects) if record.objects else ""
    text_visible_text = (
        ", ".join(record.text_visible) if record.text_visible else "none"
    )
    return {
        "memory_id": record.memory_id,
        "timestamp": record.timestamp,
        "image_path": record.primary_image_path(),
        "scene_type": record.scene_type,
        "objects_text": objects_text,
        "text_visible_text": text_visible_text,
        "privacy_risk": record.privacy_risk,
        "people_count": record.people_count,
        "location_label": record.location.label or "",
        "location_full_address": record.location.full_address or "",
        "location_city": record.location.city or "",
        "location_prefecture": record.location.prefecture or "",
        "location_postal_code": record.location.postal_code or "",
        "should_store": record.should_store,
    }
