from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta

from src.config import PROJECT_ROOT, Config
from src.embedding_client import EmbeddingClient
from src.schema import LoadedMemory, ParsedMemoryQuery, ScoredMemory, VectorHit
from src.utils import resolve_record_image_path
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

RECENT_BOOST_WINDOW = timedelta(minutes=30)
VECTOR_CANDIDATE_MULTIPLIER = 5
MIN_VECTOR_CANDIDATES = 50


def hybrid_retrieve(
    memories_by_id: dict[str, LoadedMemory],
    query: ParsedMemoryQuery,
    vector_store: VectorStore,
    embedding_client: EmbeddingClient,
    config: Config,
) -> list[ScoredMemory]:
    if not memories_by_id:
        return []

    n_candidates = max(query.limit * VECTOR_CANDIDATE_MULTIPLIER, MIN_VECTOR_CANDIDATES)

    try:
        query_embedding = embedding_client.embed_query(query.semantic_query)
    except Exception as exc:
        logger.warning("Query embedding failed: %s", exc)
        return []

    hits = vector_store.query(query_embedding, n_results=n_candidates)
    if not hits:
        logger.debug("Vector search returned no hits for %r", query.semantic_query)
        return []

    candidates = _hydrate_hits(hits, memories_by_id)
    filtered = _apply_hard_filters(candidates, query, config)

    if not filtered and _has_time_filter(query) and _has_non_time_filters(query):
        logger.debug(
            "No matches with time filter; retrying without time constraints"
        )
        query_no_time = replace(query, start_time=None, end_time=None)
        filtered = _apply_hard_filters(candidates, query_no_time, config)
        query = query_no_time

    distance_by_id = {hit.memory_id: hit.distance for hit in hits}
    scored = [
        _score_memory(item, query, config, distance_by_id.get(item.record.memory_id))
        for item in filtered
    ]
    scored.sort(
        key=lambda item: (
            item.score,
            item.parsed_timestamp or datetime.min.replace(tzinfo=config.timezone),
        ),
        reverse=True,
    )
    return scored[: query.limit]


def _hydrate_hits(
    hits: list[VectorHit],
    memories_by_id: dict[str, LoadedMemory],
) -> list[LoadedMemory]:
    results: list[LoadedMemory] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.memory_id in seen:
            continue
        seen.add(hit.memory_id)
        item = memories_by_id.get(hit.memory_id)
        if item is None:
            logger.debug(
                "Vector hit %r not found in JSONL — skipping",
                hit.memory_id,
            )
            continue
        results.append(item)
    return results


def _has_time_filter(query: ParsedMemoryQuery) -> bool:
    return query.start_time is not None or query.end_time is not None


def _has_non_time_filters(query: ParsedMemoryQuery) -> bool:
    meaningful_keywords = [k for k in query.keywords if not k.isdigit()]
    return bool(
        meaningful_keywords
        or query.object_filters
        or query.scene_type_filters
        or query.location_filters
        or query.privacy_risk
        or query.people_only
        or query.text_visible_only
    )


def _apply_hard_filters(
    memories: list[LoadedMemory],
    query: ParsedMemoryQuery,
    config: Config,
) -> list[LoadedMemory]:
    results: list[LoadedMemory] = []
    time_filter_active = _has_time_filter(query)

    for item in memories:
        record = item.record

        if config.default_should_store_only and not record.should_store:
            continue

        if time_filter_active:
            if item.parsed_timestamp is None:
                continue
            if query.start_time and item.parsed_timestamp < query.start_time:
                continue
            if query.end_time and item.parsed_timestamp > query.end_time:
                continue

        if query.people_only and record.people_count <= 0:
            continue

        if query.text_visible_only and not record.text_visible:
            continue

        if query.privacy_risk and record.privacy_risk != query.privacy_risk:
            continue

        if query.object_filters and not _matches_object_filters(
            record.objects, query.object_filters
        ):
            continue

        if query.scene_type_filters and not _matches_scene_filters(
            record.scene_type, query.scene_type_filters
        ):
            continue

        if query.location_filters and not _matches_location_filters(
            record.location.label, query.location_filters
        ):
            continue

        results.append(item)

    return results


def _matches_object_filters(objects: list[str], filters: list[str]) -> bool:
    lowered = [obj.lower() for obj in objects]
    return any(
        any(filt in obj for obj in lowered)
        for filt in filters
    )


def _matches_scene_filters(scene_type: str, filters: list[str]) -> bool:
    scene_lower = scene_type.lower()
    return any(filt in scene_lower for filt in filters)


def _matches_location_filters(
    label: str | None,
    filters: list[str],
) -> bool:
    if not label:
        return False
    label_lower = label.lower()
    return any(filt in label_lower for filt in filters)


def _vector_score_from_distance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    return max(0.0, min(10.0, 10.0 * (1.0 - distance)))


def _score_memory(
    item: LoadedMemory,
    query: ParsedMemoryQuery,
    config: Config,
    distance: float | None,
) -> ScoredMemory:
    record = item.record
    vector_score = _vector_score_from_distance(distance)
    metadata_score = 0.0
    hints: list[str] = []

    if vector_score > 0.5:
        hints.append("semantic match")

    summary_lower = record.summary.lower()
    answer_lower = record.model_answer.lower()
    scene_lower = record.scene_type.lower()
    reason_lower = record.memory_reason.lower()
    question_lower = record.user_question.lower()
    objects_lower = [obj.lower() for obj in record.objects]
    text_lower = [text.lower() for text in record.text_visible]
    location_label = (record.location.label or "").lower()

    for obj_filter in query.object_filters:
        if any(obj_filter in obj for obj in objects_lower):
            metadata_score += 5
            hints.append("object match")

    for scene_filter in query.scene_type_filters:
        if scene_filter in scene_lower:
            metadata_score += 4
            hints.append("scene type match")

    for loc_filter in query.location_filters:
        if loc_filter in location_label:
            metadata_score += 4
            hints.append("location match")

    for keyword in query.keywords:
        if keyword in answer_lower:
            metadata_score += 4
            hints.append("keyword in answer")
        if keyword in summary_lower:
            metadata_score += 3
            hints.append("keyword in summary")
        if any(keyword in text for text in text_lower):
            metadata_score += 3
            hints.append("visible text match")
        if any(keyword in obj for obj in objects_lower):
            metadata_score += 2
            hints.append("keyword in objects")
        if keyword in scene_lower:
            metadata_score += 2
        if keyword in reason_lower:
            metadata_score += 1
            hints.append("keyword in reason")
        if keyword in question_lower:
            metadata_score += 1

    if query.recent_bias and item.parsed_timestamp is not None:
        now = datetime.now(config.timezone)
        if now - item.parsed_timestamp <= RECENT_BOOST_WINDOW:
            metadata_score += 2
            hints.append("recent")

    total_score = vector_score + metadata_score
    unique_hints = list(dict.fromkeys(hints))

    image_path = resolve_record_image_path(record, config.memory_base_dir)
    try:
        display_path = str(image_path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(image_path)

    logger.debug(
        "memory_id=%s vector=%.2f metadata=%.2f total=%.2f hints=%s",
        record.memory_id,
        vector_score,
        metadata_score,
        total_score,
        unique_hints,
    )

    return ScoredMemory(
        record=record,
        score=total_score,
        vector_score=vector_score,
        metadata_score=metadata_score,
        parsed_timestamp=item.parsed_timestamp,
        display_image_path=display_path,
        retrieval_hints=unique_hints,
    )
