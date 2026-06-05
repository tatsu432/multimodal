import re
from datetime import datetime, timedelta

from src.config import Config
from src.schema import ParsedMemoryQuery

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "i",
        "me",
        "my",
        "did",
        "do",
        "does",
        "have",
        "has",
        "had",
        "what",
        "when",
        "where",
        "who",
        "how",
        "show",
        "see",
        "saw",
        "seen",
        "involving",
        "about",
        "any",
        "some",
        "all",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "and",
        "or",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "that",
        "this",
        "these",
        "those",
        "with",
        "from",
        "last",
        "recently",
        "today",
        "yesterday",
        "minutes",
        "minute",
        "hours",
        "hour",
        "ago",
        "memories",
        "memory",
        "records",
        "record",
        "risk",
        "privacy",
        "high",
        "medium",
        "low",
        "text",
        "person",
        "people",
        "near",
        "like",
        "something",
        "related",
        "look",
        "looks",
        "similar",
        "pass",
        "by",
        "passby",
    }
)

SEMANTIC_PREFIX_PATTERNS = [
    re.compile(r"^when did i see something like\s+", re.IGNORECASE),
    re.compile(r"^when did i see\s+", re.IGNORECASE),
    re.compile(r"^did i see something related to\s+", re.IGNORECASE),
    re.compile(r"^did i see\s+", re.IGNORECASE),
    re.compile(r"^did i pass by any\s+", re.IGNORECASE),
    re.compile(r"^did i pass by\s+", re.IGNORECASE),
    re.compile(r"^where did i see a\s+", re.IGNORECASE),
    re.compile(r"^where did i see\s+", re.IGNORECASE),
    re.compile(r"^what did i see near\s+", re.IGNORECASE),
    re.compile(r"^what did i see\s+", re.IGNORECASE),
    re.compile(r"^what memories look similar to\s+", re.IGNORECASE),
    re.compile(r"^what memories look like\s+", re.IGNORECASE),
    re.compile(r"^what memories are similar to\s+", re.IGNORECASE),
]

OBJECT_KEYWORDS = frozenset(
    {
        "laptop",
        "desk",
        "person",
        "phone",
        "monitor",
        "camera",
        "chair",
        "cable",
        "curtains",
        "speaker",
        "clock",
        "box",
        "cabinet",
        "cups",
        "boxes",
        "table",
        "computer",
        "gopro",
    }
)

SCENE_KEYWORD_MAP: dict[str, str] = {
    "workspace": "workspace",
    "indoor": "indoor",
    "outdoor": "outdoor",
    "street": "street",
    "station": "station",
    "store": "store",
    "home": "home",
}

PRIVACY_PATTERN = re.compile(
    r"\b(high|medium|low)\s+privacy\s+risk\b", re.IGNORECASE
)
LAST_MINUTES_PATTERN = re.compile(
    r"\blast\s+(\d+)\s+minutes?\b", re.IGNORECASE
)
LAST_HOURS_PATTERN = re.compile(
    r"\blast\s+(\d+)\s+hours?\b", re.IGNORECASE
)
MINUTES_AGO_PATTERN = re.compile(
    r"\b(\d+)\s+minutes?\s+ago\b", re.IGNORECASE
)
HOURS_AGO_PATTERN = re.compile(
    r"\b(\d+)\s+hours?\s+ago\b", re.IGNORECASE
)
LOCATION_PATTERN = re.compile(
    r"\b(?:at|near)\s+([a-z0-9][a-z0-9 _-]*)", re.IGNORECASE
)


def parse_query(question: str, config: Config) -> ParsedMemoryQuery:
    normalized = question.strip()
    lower = normalized.lower()

    now = datetime.now(config.timezone)
    start_time: datetime | None = None
    end_time: datetime | None = None
    recent_bias = False

    if "recently" in lower:
        start_time = now - timedelta(minutes=30)
        end_time = now
        recent_bias = True
    elif match := LAST_MINUTES_PATTERN.search(lower):
        minutes = int(match.group(1))
        start_time = now - timedelta(minutes=minutes)
        end_time = now
    elif match := LAST_HOURS_PATTERN.search(lower):
        hours = int(match.group(1))
        start_time = now - timedelta(hours=hours)
        end_time = now
    elif match := MINUTES_AGO_PATTERN.search(lower):
        minutes = int(match.group(1))
        target = now - timedelta(minutes=minutes)
        start_time = target - timedelta(minutes=2)
        end_time = target + timedelta(minutes=2)
    elif match := HOURS_AGO_PATTERN.search(lower):
        hours = int(match.group(1))
        target = now - timedelta(hours=hours)
        start_time = target - timedelta(minutes=5)
        end_time = target + timedelta(minutes=5)
    elif "yesterday" in lower:
        yesterday = (now - timedelta(days=1)).date()
        start_time = datetime(
            yesterday.year,
            yesterday.month,
            yesterday.day,
            0,
            0,
            0,
            tzinfo=config.timezone,
        )
        end_time = datetime(
            yesterday.year,
            yesterday.month,
            yesterday.day,
            23,
            59,
            59,
            999999,
            tzinfo=config.timezone,
        )
    elif "today" in lower:
        today = now.date()
        start_time = datetime(
            today.year,
            today.month,
            today.day,
            0,
            0,
            0,
            tzinfo=config.timezone,
        )
        end_time = now

    people_only = _detect_people_only(lower)
    text_visible_only = _detect_text_visible_only(lower)

    privacy_risk: str | None = None
    if match := PRIVACY_PATTERN.search(lower):
        privacy_risk = match.group(1).lower()

    object_filters = _extract_object_filters(lower)
    if people_only:
        object_filters = [f for f in object_filters if f not in {"person", "people"}]
    scene_type_filters = _extract_scene_filters(lower)
    location_filters = _extract_location_filters(normalized)

    keywords = _extract_keywords(lower, object_filters, scene_type_filters, location_filters)
    semantic_query = _build_semantic_query(
        normalized,
        lower,
        keywords,
        object_filters,
        scene_type_filters,
        location_filters,
    )

    return ParsedMemoryQuery(
        original_question=normalized,
        semantic_query=semantic_query,
        start_time=start_time,
        end_time=end_time,
        keywords=keywords,
        object_filters=object_filters,
        scene_type_filters=scene_type_filters,
        location_filters=location_filters,
        privacy_risk=privacy_risk,
        people_only=people_only,
        text_visible_only=text_visible_only,
        recent_bias=recent_bias,
        limit=config.default_limit,
    )


def _build_semantic_query(
    normalized: str,
    lower: str,
    keywords: list[str],
    object_filters: list[str],
    scene_type_filters: list[str],
    location_filters: list[str],
) -> str:
    text = normalized
    for pattern in SEMANTIC_PREFIX_PATTERNS:
        text = pattern.sub("", text)

    text = LAST_MINUTES_PATTERN.sub("", text)
    text = LAST_HOURS_PATTERN.sub("", text)
    text = MINUTES_AGO_PATTERN.sub("", text)
    text = HOURS_AGO_PATTERN.sub("", text)
    text = PRIVACY_PATTERN.sub("", text)

    for token in ("recently", "yesterday", "today"):
        text = re.sub(rf"\b{token}\b", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip(" ?.,!")

    if not text:
        if keywords:
            return " ".join(keywords)
        if object_filters:
            return " ".join(object_filters)
        if scene_type_filters:
            return " ".join(scene_type_filters)
        if location_filters:
            return " ".join(location_filters)
        return normalized

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    excluded = (
        set(object_filters)
        | set(scene_type_filters)
        | set(location_filters)
        | STOPWORDS
    )
    semantic_tokens = [t for t in tokens if t not in excluded and len(t) >= 2]

    if semantic_tokens:
        return " ".join(semantic_tokens)

    return text if text else normalized


def _detect_people_only(lower: str) -> bool:
    if "involving a person" in lower or "involving people" in lower:
        return True
    if "with a person" in lower or "with people" in lower:
        return True
    if re.search(r"\b(?:person|people)\b", lower) and "what text" not in lower:
        if any(
            phrase in lower
            for phrase in (
                "involving",
                "with a person",
                "with people",
                "memories involving",
                "see a person",
                "see people",
            )
        ):
            return True
        if re.search(r"\b(?:show|memories|memory)\b.*\b(?:person|people)\b", lower):
            return True
    return False


def _detect_text_visible_only(lower: str) -> bool:
    return any(
        phrase in lower
        for phrase in (
            "what text",
            "text did i see",
            "visible text",
            "text visible",
            "any text",
        )
    )


def _extract_object_filters(lower: str) -> list[str]:
    found: list[str] = []
    for keyword in OBJECT_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lower):
            found.append(keyword)
    return found


def _extract_scene_filters(lower: str) -> list[str]:
    found: list[str] = []
    for trigger, scene_token in SCENE_KEYWORD_MAP.items():
        if re.search(rf"\b{re.escape(trigger)}\b", lower):
            found.append(scene_token)
    return found


def _extract_location_filters(question: str) -> list[str]:
    found: list[str] = []
    for match in LOCATION_PATTERN.finditer(question):
        label = match.group(1).strip().lower()
        if label and label not in STOPWORDS:
            found.append(label)
    return found


def _extract_keywords(
    lower: str,
    object_filters: list[str],
    scene_type_filters: list[str],
    location_filters: list[str],
) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", lower)
    excluded = set(object_filters) | set(scene_type_filters) | set(location_filters) | STOPWORDS
    keywords: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in excluded or len(token) < 2:
            continue
        if token.isdigit():
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords
