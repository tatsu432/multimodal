import logging
from datetime import datetime, timedelta

from src.config import PROJECT_ROOT, Config
from src.schema import LoadedMemory, ParsedMemoryQuery, ScoredMemory
from src.utils import resolve_record_image_path

logger = logging.getLogger(__name__)

RECENT_BOOST_WINDOW = timedelta(minutes=30)


def retrieve(
    memories: list[LoadedMemory],
    query: ParsedMemoryQuery,
    config: Config,
) -> list[ScoredMemory]:
    if not memories:
        return []

    candidates = _apply_hard_filters(memories, query, config)

    if not candidates and _has_time_filter(query) and _has_non_time_filters(query):
        logger.debug(
            "No matches with time filter; retrying without time constraints"
        )
        query_no_time = ParsedMemoryQuery(
            original_question=query.original_question,
            start_time=None,
            end_time=None,
            keywords=query.keywords,
            object_filters=query.object_filters,
            scene_type_filters=query.scene_type_filters,
            location_filters=query.location_filters,
            privacy_risk=query.privacy_risk,
            people_only=query.people_only,
            text_visible_only=query.text_visible_only,
            recent_bias=query.recent_bias,
            limit=query.limit,
        )
        candidates = _apply_hard_filters(memories, query_no_time, config)

    scored = [_score_memory(item, query, config) for item in candidates]
    scored.sort(
        key=lambda item: (
            item.score,
            item.parsed_timestamp or datetime.min.replace(tzinfo=config.timezone),
        ),
        reverse=True,
    )
    return scored[: query.limit]


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
            record.location, query.location_filters
        ):
            continue

        results.append(item)

    if not results and not _has_non_time_filters(query) and not time_filter_active:
        return list(memories)

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
    location,
    filters: list[str],
) -> bool:
    searchable = location.search_text().lower()
    if not searchable:
        return False
    return any(filt in searchable for filt in filters)


def _score_memory(
    item: LoadedMemory,
    query: ParsedMemoryQuery,
    config: Config,
) -> ScoredMemory:
    record = item.record
    score = 0.0

    summary_lower = record.summary.lower()
    answer_lower = record.model_answer.lower()
    scene_lower = record.scene_type.lower()
    reason_lower = record.memory_reason.lower()
    question_lower = record.user_question.lower()
    objects_lower = [obj.lower() for obj in record.objects]
    text_lower = [text.lower() for text in record.text_visible]
    location_searchable = record.location.search_text().lower()

    for obj_filter in query.object_filters:
        if any(obj_filter in obj for obj in objects_lower):
            score += 5

    for scene_filter in query.scene_type_filters:
        if scene_filter in scene_lower:
            score += 4

    for loc_filter in query.location_filters:
        if loc_filter in location_searchable:
            score += 4

    for keyword in query.keywords:
        if keyword in answer_lower:
            score += 4
        if keyword in summary_lower:
            score += 3
        if any(keyword in text for text in text_lower):
            score += 3
        if any(keyword in obj for obj in objects_lower):
            score += 2
        if keyword in scene_lower:
            score += 2
        if keyword in reason_lower:
            score += 1
        if keyword in question_lower:
            score += 1

    if query.recent_bias and item.parsed_timestamp is not None:
        now = datetime.now(config.timezone)
        if now - item.parsed_timestamp <= RECENT_BOOST_WINDOW:
            score += 2

    if score == 0.0 and not _has_non_time_filters(query):
        score = 1.0

    image_path = resolve_record_image_path(record, config.memory_base_dir)
    try:
        display_path = str(image_path.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(image_path)

    return ScoredMemory(
        record=record,
        score=score,
        parsed_timestamp=item.parsed_timestamp,
        display_image_path=display_path,
    )
