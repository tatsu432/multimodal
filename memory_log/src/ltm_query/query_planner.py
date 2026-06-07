"""Query planner — interprets a user query into a structured retrieval plan via LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import Config

logger = logging.getLogger("memory_log.ltm_query.query_planner")

_PLANNER_SYSTEM_PROMPT = """\
You are a memory query planner for a wearable AI assistant.
Given the user's question about their past visual memories, return a JSON retrieval plan.

Current local time: {current_time}
Timezone offset: {timezone_offset}

Available memory stores:
- daily_summaries: high-level day summaries (method: date_lookup or semantic_search)
- passive_observations: raw time/location traces every ~30s, no visual analysis \
(method: time_range or location_radius)
- promoted_events: semantic visual events with scene descriptions \
(method: time_range, semantic_search, or location_radius)
- active_query_memories: past user Q&A interactions with the camera \
(method: time_range or semantic_search)
- frames: frame images linked to events (method: by_event)

Intents:
- whereabouts: "where was I", "which location", "was I near"
- visual_recall: "what did I see", "what was there", "describe the scene"
- interaction_recall: "what did I ask", "what did the model say", "did I ask about X"
- current_scene: "this", "here", "current scene", "what I am looking at now"
- general: anything else

Return ONLY valid JSON — no markdown, no explanation:
{
  "intent": "<intent>",
  "time_range": {"start_utc": "<ISO8601>", "end_utc": "<ISO8601>"} or null,
  "location_filter": {"lat": <float>, "lon": <float>, "radius_m": <float>} or null,
  "semantic_query": "<keywords>" or null,
  "needs_current_visual_grounding": <bool>,
  "needs_retrieved_frames": <bool>,
  "stores_to_query": [
    {"store": "<store>", "method": "<method>", "top_k": <int or null>, "max_records": <int or null>}
  ]
}

Rules:
- Set needs_current_visual_grounding=true if the query contains words like: \
this, here, that, these, current, now, looking at, this object, this room, this place.
- For "yesterday", compute start/end from current local time.
- For whereabouts intent, always include passive_observations and daily_summaries.
- For visual_recall, include promoted_events; set needs_retrieved_frames=true when \
the user asks for visual details.
- For interaction_recall, include active_query_memories.
- Prefer metadata filters (time_range, location_radius) over semantic_search.
"""

_FALLBACK_PLAN_JSON = """{
  "intent": "general",
  "time_range": null,
  "location_filter": null,
  "semantic_query": null,
  "needs_current_visual_grounding": false,
  "needs_retrieved_frames": false,
  "stores_to_query": [
    {"store": "promoted_events", "method": "semantic_search", "top_k": 10, "max_records": null},
    {"store": "active_query_memories", "method": "semantic_search", "top_k": 5, "max_records": null}
  ]
}"""


@dataclass
class TimeRange:
    start_utc: str
    end_utc: str


@dataclass
class LocationFilter:
    lat: float
    lon: float
    radius_m: float


@dataclass
class StoreQuery:
    store: str
    method: str
    top_k: int | None = None
    max_records: int | None = None


@dataclass
class RetrievalPlan:
    intent: str
    time_range: TimeRange | None
    location_filter: LocationFilter | None
    semantic_query: str | None
    needs_current_visual_grounding: bool
    needs_retrieved_frames: bool
    stores_to_query: list[StoreQuery] = field(default_factory=list)


def _parse_plan(raw: str) -> RetrievalPlan:
    data = json.loads(raw)
    tr = data.get("time_range")
    lf = data.get("location_filter")
    stores = [
        StoreQuery(
            store=s["store"],
            method=s["method"],
            top_k=s.get("top_k"),
            max_records=s.get("max_records"),
        )
        for s in data.get("stores_to_query", [])
    ]
    return RetrievalPlan(
        intent=data.get("intent", "general"),
        time_range=TimeRange(**tr) if tr else None,
        location_filter=LocationFilter(**lf) if lf else None,
        semantic_query=data.get("semantic_query"),
        needs_current_visual_grounding=bool(data.get("needs_current_visual_grounding")),
        needs_retrieved_frames=bool(data.get("needs_retrieved_frames")),
        stores_to_query=stores,
    )


def _fallback_plan(query: str) -> RetrievalPlan:
    plan = _parse_plan(_FALLBACK_PLAN_JSON)
    plan.semantic_query = query
    return plan


class QueryPlanner:
    def __init__(self, config: Config) -> None:
        self._config = config

    def plan(self, query: str) -> RetrievalPlan:
        now = datetime.now().astimezone()
        tz_offset = now.strftime("%z")  # e.g. +0900
        current_time = now.strftime("%Y-%m-%dT%H:%M:%S") + tz_offset

        system_prompt = (
            _PLANNER_SYSTEM_PROMPT
            .replace("{current_time}", current_time)
            .replace("{timezone_offset}", tz_offset)
        )

        raw: str | None = None
        try:
            raw = self._call_llm(system_prompt, query)
            plan = _parse_plan(raw)
            logger.debug("Query plan parsed: intent=%s stores=%s", plan.intent, [s.store for s in plan.stores_to_query])
            return plan
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Plan parse error (%s); raw=%r; using fallback", exc, raw)
            return _fallback_plan(query)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        if self._config.vlm_provider == "openai":
            return self._call_openai(system_prompt, user_message)
        return self._call_ollama(system_prompt, user_message)

    def _call_openai(self, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._config.openai_api_key)
        response = client.responses.create(
            model=self._config.vlm_model,
            instructions=system_prompt,
            input=user_message,
        )
        return response.output_text.strip()

    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        from providers.ollama import chat as ollama_chat

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return ollama_chat(
            self._config.vlm_model,
            messages,
            base_url=self._config.ollama_base_url,
        ).strip()
