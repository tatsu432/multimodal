"""Visual grounding — resolve deictic references (this/here/current scene) via VLM."""

from __future__ import annotations

import json
import logging

import numpy as np

from src.config import Config
from src.ltm_query.evidence import VisualGroundingResult
from src.schema import LocationInfo
from src.utils import FrameItem

logger = logging.getLogger("memory_log.ltm_query.visual_grounding")

_GROUNDING_SYSTEM_PROMPT = """\
You are a visual grounding module for a memory query system.
The user asked a question that refers to the current visual scene \
(using words like "this", "here", "current", "now", etc.).

Your job is NOT to answer the question. Your job is to describe the current scene
so that a retrieval system can search past memories for relevant context.

Return ONLY valid JSON — no markdown, no explanation:
{
  "current_scene_summary": "<1-2 sentence description of visible scene>",
  "visible_objects": ["<object1>", "<object2>"],
  "place_type": "<indoor_office|indoor_home|outdoor_street|outdoor_park|vehicle|other>",
  "resolved_references": {
    "here": "<what 'here' refers to>",
    "this": "<what 'this' refers to if present>"
  },
  "semantic_retrieval_query": "<keywords for searching past memories>",
  "suggested_location_radius_m": <float or null>
}
"""

_GROUNDING_FALLBACK = VisualGroundingResult(
    current_scene_summary="Could not interpret current scene.",
    visible_objects=[],
    place_type="other",
    resolved_references={},
    semantic_retrieval_query="",
    suggested_location_radius_m=None,
)


class VisualGrounder:
    def __init__(self, config: Config) -> None:
        self._config = config

    def ground(
        self,
        query: str,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None,
        location: LocationInfo | None,
    ) -> VisualGroundingResult | None:
        if not frames:
            logger.debug("Visual grounding skipped: no frames available")
            return None

        location_note = ""
        if location and location.display_name() != "not available":
            location_note = f"\nCurrent location context: {location.display_name()}"

        user_prompt = f"User query: {query}{location_note}\n\nDescribe the current scene for retrieval."

        raw: str | None = None
        try:
            if self._config.vlm_provider == "openai":
                raw = self._call_openai(user_prompt, frames, frame_items)
            else:
                raw = self._call_ollama(user_prompt, frames, frame_items)

            data = json.loads(raw)
            return VisualGroundingResult(
                current_scene_summary=data.get("current_scene_summary", ""),
                visible_objects=data.get("visible_objects", []),
                place_type=data.get("place_type", "other"),
                resolved_references=data.get("resolved_references", {}),
                semantic_retrieval_query=data.get("semantic_retrieval_query", ""),
                suggested_location_radius_m=data.get("suggested_location_radius_m"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Visual grounding parse error (%s); raw=%r", exc, raw)
            return _GROUNDING_FALLBACK

    def _encode_frames(self, frames: list[np.ndarray], frame_items: list[FrameItem] | None) -> list[tuple[str, str | None]]:
        from src.utils import encode_frame_as_base64_jpeg, frame_capture_timestamp_iso

        encoded: list[tuple[str, str | None]] = []
        for i, frame in enumerate(frames[:4]):
            b64 = encode_frame_as_base64_jpeg(frame, max_width=512, quality=80)
            ts: str | None = None
            if frame_items and i < len(frame_items):
                ts = frame_capture_timestamp_iso(frame_items[i].timestamp)
            encoded.append((b64, ts))
        return encoded

    def _call_openai(
        self,
        user_prompt: str,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._config.openai_api_key)
        encoded = self._encode_frames(frames, frame_items)
        content: list = []
        for b64, ts in encoded:
            label = f"Frame{' at ' + ts if ts else ''}:"
            content.append({"type": "input_text", "text": label})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            })
        content.append({"type": "input_text", "text": user_prompt})

        response = client.responses.create(
            model=self._config.vlm_model,
            instructions=_GROUNDING_SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
        )
        return response.output_text.strip()

    def _call_ollama(
        self,
        user_prompt: str,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None,
    ) -> str:
        from providers.ollama import chat as ollama_chat

        encoded = self._encode_frames(frames, frame_items)
        images = [b64 for b64, _ in encoded]
        messages = [
            {"role": "system", "content": _GROUNDING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt, "images": images},
        ]
        return ollama_chat(
            self._config.vlm_model,
            messages,
            base_url=self._config.ollama_base_url,
        ).strip()
