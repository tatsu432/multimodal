"""Answer generator — synthesizes retrieved evidence into a grounded answer via text LLM."""

from __future__ import annotations

import logging
from typing import Iterator

from src.config import Config
from src.ltm_query.evidence import EvidencePack

logger = logging.getLogger("memory_log.ltm_query.answer_generator")

_ANSWER_SYSTEM_PROMPT = """\
You are answering a user's question about their past visual memories stored by a wearable AI assistant.

Rules:
1. Answer ONLY from the evidence provided. Do not invent details.
2. Always mention timestamps and locations when they are relevant.
3. Explicitly distinguish:
   - "Based on saved memories, ..." — when you have clear evidence
   - "I found evidence that ..." — when evidence is partial
   - "I do not have enough memory to know ..." — when evidence is missing
4. If observation coverage was incomplete (camera off, no data), say so.
5. Be concise but specific. Prefer "around 渋谷区 at 15:30" over vague summaries.
"""


def format_evidence(evidence: EvidencePack) -> str:
    lines: list[str] = [f"User query: {evidence.user_query}"]

    if evidence.time_range_description:
        lines.append(f"Interpreted time range: {evidence.time_range_description}")
    if evidence.location_context:
        lines.append(f"Location context: {evidence.location_context}")

    if evidence.visual_grounding:
        vg = evidence.visual_grounding
        lines.append("\n--- Current scene (visual grounding) ---")
        lines.append(f"Scene: {vg.current_scene_summary}")
        if vg.visible_objects:
            lines.append(f"Visible objects: {', '.join(vg.visible_objects)}")
        if vg.resolved_references:
            for ref, desc in vg.resolved_references.items():
                lines.append(f"Resolved '{ref}': {desc}")

    if evidence.daily_summaries:
        lines.append("\n--- Daily summaries ---")
        for row in evidence.daily_summaries:
            lines.append(f"[{row['date_local']}] {row['summary_text']}")

    if evidence.passive_timeline:
        lines.append("\n--- Location timeline (from passive observations) ---")
        for seg in evidence.passive_timeline:
            lines.append(
                f"- {seg.start_local} – {seg.end_local}: "
                f"{seg.location_label} ({seg.observation_count} observations)"
            )

    if evidence.promoted_events:
        lines.append("\n--- Promoted visual events ---")
        for row in evidence.promoted_events:
            ts = (row["timestamp_local"] or row["start_ts_utc"] or "")[:19]
            loc = row["location_label"] or row["city"] or "unknown location"
            summary = row["scene_summary"] or row["raw_vlm_output"] or ""
            lines.append(f"[{ts}] {loc}: {summary[:300]}")

    if evidence.active_queries:
        lines.append("\n--- Past Q&A interactions ---")
        for row in evidence.active_queries:
            ts = (row["timestamp_local"] or row["timestamp_utc"] or "")[:19]
            loc = row["location_label"] or row["city"] or ""
            loc_str = f" ({loc})" if loc else ""
            lines.append(f"[{ts}]{loc_str} Q: {row['user_question']}")
            if row["model_answer"]:
                lines.append(f"  A: {row['model_answer'][:200]}")

    if evidence.uncertainty_notes:
        lines.append("\n--- Retrieval notes ---")
        for note in evidence.uncertainty_notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


class AnswerGenerator:
    def __init__(self, config: Config) -> None:
        self._config = config

    def generate(self, evidence: EvidencePack) -> str:
        evidence_text = format_evidence(evidence)
        logger.info(
            "Answer generation: evidence_prompt_chars=%d events=%d active_queries=%d",
            len(evidence_text),
            len(evidence.promoted_events),
            len(evidence.active_queries),
        )
        try:
            if self._config.vlm_provider == "openai":
                answer = self._call_openai(evidence_text)
            else:
                answer = self._call_ollama(evidence_text)
            logger.info("Answer generated: answer_chars=%d", len(answer))
            return answer
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return f"I encountered an error generating the answer: {exc}"

    def _call_openai(self, evidence_text: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._config.openai_api_key)
        response = client.responses.create(
            model=self._config.vlm_model,
            instructions=_ANSWER_SYSTEM_PROMPT,
            input=evidence_text,
        )
        return response.output_text.strip()

    def _call_ollama(self, evidence_text: str) -> str:
        from providers.ollama import chat as ollama_chat

        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": evidence_text},
        ]
        return ollama_chat(
            model=self._config.vlm_model,
            messages=messages,
            base_url=self._config.ollama_base_url,
        ).strip()

    def generate_stream(self, evidence: EvidencePack) -> Iterator[str]:
        """Yield answer text tokens one by one."""
        evidence_text = format_evidence(evidence)
        logger.info(
            "Streaming answer: evidence_prompt_chars=%d events=%d active_queries=%d",
            len(evidence_text),
            len(evidence.promoted_events),
            len(evidence.active_queries),
        )
        try:
            if self._config.vlm_provider == "openai":
                yield from self._stream_openai(evidence_text)
            else:
                yield from self._stream_ollama(evidence_text)
        except Exception as exc:
            logger.error("Answer stream failed: %s", exc)
            yield f"\n[Error generating answer: {exc}]"

    def _stream_openai(self, evidence_text: str) -> Iterator[str]:
        from openai import OpenAI

        client = OpenAI(api_key=self._config.openai_api_key)
        with client.responses.stream(
            model=self._config.vlm_model,
            instructions=_ANSWER_SYSTEM_PROMPT,
            input=evidence_text,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta

    def _stream_ollama(self, evidence_text: str) -> Iterator[str]:
        from providers.ollama import chat_stream as ollama_chat_stream

        messages = [
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": evidence_text},
        ]
        yield from ollama_chat_stream(
            model=self._config.vlm_model,
            messages=messages,
            base_url=self._config.ollama_base_url,
        )
