import json
import logging
import re

from openai import OpenAI
from providers.ollama import chat as ollama_chat

from src.config import PROJECT_ROOT, Config
from src.schema import ParsedMemoryQuery, ScoredMemory
from src.utils import format_objects, format_timestamp_display, resolve_image_path

logger = logging.getLogger(__name__)

YES_NO_PATTERN = re.compile(
    r"^\s*(?:did|have|was|were)\b", re.IGNORECASE
)

LLM_SYSTEM_PROMPT = (
    "Answer only using the provided memory records. "
    "If the records do not contain enough evidence, say so. "
    "Include timestamps as evidence."
)


def generate_answer(
    query: ParsedMemoryQuery,
    results: list[ScoredMemory],
    config: Config,
) -> str:
    if not results:
        return "I could not find matching memory records for that question."

    evidence_block = _format_evidence_block(results, config)

    if config.use_llm_answerer:
        llm_text = _llm_summarize(query, results, config)
        return f"{llm_text}\n\n{evidence_block}"

    return _template_answer(query, results, config)


def _template_answer(
    query: ParsedMemoryQuery,
    results: list[ScoredMemory],
    config: Config,
) -> str:
    count = len(results)
    lines: list[str] = []

    if YES_NO_PATTERN.match(query.original_question):
        lead = "Yes" if count > 0 else "No"
        lines.append(
            f"{lead}. I found {count} relevant {'memory' if count == 1 else 'memories'}."
        )
    elif query.recent_bias or (
        query.start_time is None and query.end_time is None
    ):
        lines.append(
            f"Found {count} relevant {'memory' if count == 1 else 'memories'}."
        )
    else:
        lines.append(f"Found {count} relevant {'memory' if count == 1 else 'memories'}.")

    lines.append("")
    lines.append(_format_evidence_block(results, config))
    return "\n".join(lines)


def _format_retrieval_line(hints: list[str]) -> str:
    if not hints:
        return "   Retrieval: semantic match"
    return f"   Retrieval: {' + '.join(hints)}"


def _format_evidence_block(
    results: list[ScoredMemory],
    config: Config,
) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        record = item.record
        ts_display = format_timestamp_display(
            item.parsed_timestamp,
            fallback=record.timestamp[:19] if len(record.timestamp) >= 19 else record.timestamp,
        )

        image_path = resolve_image_path(record.image_path, config.memory_base_dir)
        try:
            display_image = str(image_path.relative_to(PROJECT_ROOT))
        except ValueError:
            display_image = item.display_image_path

        image_suffix = ""
        if not image_path.is_file():
            image_suffix = " (image not found)"

        lines.append(f"{index}. {ts_display} — {record.summary}")
        lines.append(f"   Objects: {format_objects(record.objects)}")
        lines.append(f"   Scene: {record.scene_type}")
        lines.append(f"   Privacy: {record.privacy_risk}")
        if record.text_visible:
            lines.append(f"   Text: {format_objects(record.text_visible)}")
        lines.append(f"   Image: {display_image}{image_suffix}")
        lines.append(_format_retrieval_line(item.retrieval_hints))

    return "\n".join(lines)


def _llm_summarize(
    query: ParsedMemoryQuery,
    results: list[ScoredMemory],
    config: Config,
) -> str:
    records_payload = []
    for item in results:
        record = item.record
        records_payload.append(
            {
                "timestamp": format_timestamp_display(
                    item.parsed_timestamp,
                    fallback=record.timestamp,
                ),
                "summary": record.summary,
                "objects": record.objects,
                "scene_type": record.scene_type,
                "text_visible": record.text_visible,
                "privacy_risk": record.privacy_risk,
                "image_path": item.display_image_path,
            }
        )

    user_content = (
        f"Question: {query.original_question}\n\n"
        f"Memory records:\n{json.dumps(records_payload, ensure_ascii=False, indent=2)}"
    )

    try:
        if config.llm_provider == "ollama":
            text = ollama_chat(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                base_url=config.ollama_base_url,
            )
        else:
            client = OpenAI(api_key=config.openai_api_key)
            response = client.responses.create(
                model=config.llm_model,
                input=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            text = response.output_text.strip()

        if text:
            return text
    except Exception as exc:
        logger.warning("LLM answerer failed: %s — falling back to template intro", exc)

    count = len(results)
    return f"Found {count} relevant {'memory' if count == 1 else 'memories'}."
