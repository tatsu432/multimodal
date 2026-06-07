"""Daily summary generator — compress one day's memories into a structured summary."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from src.config import Config, PROJECT_ROOT
from src.db_writer import SQLiteWriter
from src.ltm_query.evidence import aggregate_passive_observations
from src.memory_db import open_db

logger = logging.getLogger("memory_log.daily_summary")

_SUMMARY_SYSTEM_PROMPT = """\
You are a memory summarizer for a wearable AI assistant.
Given structured data from one day's observations and interactions, \
generate a concise daily summary.

Return ONLY valid JSON — no markdown, no explanation:
{
  "summary_text": "<2-4 sentence natural language summary of the day>",
  "major_places": ["<place1>", "<place2>"],
  "notable_event_ids": ["<event_id1>"],
  "active_query_ids": ["<aq_id1>"],
  "uncertainties": ["<gap or limitation>"]
}

Rules:
- summary_text should be a readable description of where the user was and \
what notable things happened.
- major_places: list of distinct locations visited.
- notable_event_ids: event_ids of the most important promoted events.
- active_query_ids: active_query_id of notable Q&A interactions.
- uncertainties: periods with no data or incomplete coverage.
"""


def _build_day_context(
    date_local: str,
    tz_name: str,
    passive_rows: list,
    promoted_events: list,
    active_queries: list,
) -> str:
    lines: list[str] = [
        f"Date: {date_local}",
        f"Timezone: {tz_name}",
    ]

    passive_segments = aggregate_passive_observations(passive_rows)
    if passive_segments:
        lines.append("\nPassive observation location timeline:")
        for seg in passive_segments:
            lines.append(
                f"- {seg.start_local} – {seg.end_local}: "
                f"{seg.location_label}, {seg.observation_count} observations"
            )
    else:
        lines.append("\nPassive observations: none recorded for this day.")

    if promoted_events:
        lines.append("\nPromoted events:")
        for row in promoted_events:
            ts = (row["timestamp_local"] or row["start_ts_utc"] or "")[:16]
            loc = row["location_label"] or row["city"] or "unknown location"
            summary = row["scene_summary"] or row["raw_vlm_output"] or "(no description)"
            evt_id = row["event_id"]
            lines.append(f"- {ts}, {loc}: {summary[:200]} [id: {evt_id}]")
    else:
        lines.append("\nPromoted events: none recorded for this day.")

    if active_queries:
        lines.append("\nActive query memories:")
        for row in active_queries:
            ts = (row["timestamp_local"] or row["timestamp_utc"] or "")[:16]
            loc = row["location_label"] or row["city"] or ""
            loc_str = f" ({loc})" if loc else ""
            aq_id = row["active_query_id"]
            lines.append(f"- {ts}{loc_str}, user asked: \"{row['user_question']}\" "
                         f"answer: \"{(row['model_answer'] or '')[:150]}\" [id: {aq_id}]")
    else:
        lines.append("\nActive query memories: none recorded for this day.")

    return "\n".join(lines)


class DailySummaryGenerator:
    def __init__(self, config: Config) -> None:
        self._config = config

    def generate(self, date_local: str, conn, db_writer: SQLiteWriter) -> dict:
        # Parse timezone from the system
        local_now = datetime.now().astimezone()
        tz_offset = local_now.strftime("%z")  # e.g. +0900
        tz_name = f"UTC{tz_offset[:3]}:{tz_offset[3:]}"

        # Build date range in UTC (approximate: use local date with offset)
        start_local = f"{date_local}T00:00:00{tz_offset[:3]}:{tz_offset[3:]}"
        end_local = f"{date_local}T23:59:59{tz_offset[:3]}:{tz_offset[3:]}"
        try:
            start_utc = datetime.fromisoformat(start_local).astimezone(timezone.utc).isoformat(timespec="milliseconds")
            end_utc = datetime.fromisoformat(end_local).astimezone(timezone.utc).isoformat(timespec="milliseconds")
        except ValueError:
            logger.error("Invalid date format: %r", date_local)
            raise

        # Query each table for the day
        passive_rows = conn.execute(
            "SELECT * FROM passive_observations WHERE timestamp_utc BETWEEN ? AND ? ORDER BY timestamp_utc",
            (start_utc, end_utc),
        ).fetchall()

        promoted_events = conn.execute(
            "SELECT * FROM promoted_events WHERE start_ts_utc BETWEEN ? AND ? ORDER BY start_ts_utc",
            (start_utc, end_utc),
        ).fetchall()

        active_queries = conn.execute(
            "SELECT * FROM active_query_memories WHERE timestamp_utc BETWEEN ? AND ? ORDER BY timestamp_utc",
            (start_utc, end_utc),
        ).fetchall()

        logger.info(
            "Day %s: %d passive, %d events, %d queries",
            date_local, len(passive_rows), len(promoted_events), len(active_queries),
        )

        day_context = _build_day_context(
            date_local, tz_name, passive_rows, promoted_events, active_queries
        )

        raw_output = self._call_llm(day_context)
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("Could not parse summary JSON; using raw output as summary_text")
            parsed = {"summary_text": raw_output, "major_places": [], "uncertainties": []}

        summary_text = parsed.get("summary_text", raw_output)
        major_places = parsed.get("major_places", [])
        notable_event_ids = parsed.get("notable_event_ids", [])
        active_query_ids = parsed.get("active_query_ids", [])
        uncertainties = parsed.get("uncertainties", [])

        summary_id = f"sum_{date_local}"
        semantic_text = summary_text + " " + " ".join(major_places)

        db_writer.write_daily_summary(
            summary_id=summary_id,
            date_local=date_local,
            timezone_name=tz_name,
            summary_text=summary_text,
            major_places_json=json.dumps(major_places, ensure_ascii=False),
            notable_event_ids_json=json.dumps(notable_event_ids),
            active_query_ids_json=json.dumps(active_query_ids),
            coverage_start_utc=start_utc,
            coverage_end_utc=end_utc,
            raw_model_output=raw_output,
            semantic_search_text=semantic_text,
        )

        result = {
            "summary_id": summary_id,
            "date_local": date_local,
            "summary_text": summary_text,
            "major_places": major_places,
            "notable_event_ids": notable_event_ids,
            "active_query_ids": active_query_ids,
            "uncertainties": uncertainties,
            "stats": {
                "passive_observations": len(passive_rows),
                "promoted_events": len(promoted_events),
                "active_queries": len(active_queries),
            },
        }
        return result

    def _call_llm(self, day_context: str) -> str:
        if self._config.vlm_provider == "openai":
            return self._call_openai(day_context)
        return self._call_ollama(day_context)

    def _call_openai(self, day_context: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._config.openai_api_key)
        response = client.responses.create(
            model=self._config.vlm_model,
            instructions=_SUMMARY_SYSTEM_PROMPT,
            input=day_context,
        )
        return response.output_text.strip()

    def _call_ollama(self, day_context: str) -> str:
        from providers.ollama import chat as ollama_chat

        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": day_context},
        ]
        return ollama_chat(
            self._config.vlm_model,
            messages,
            base_url=self._config.ollama_base_url,
        ).strip()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Generate a daily summary from memory DB records."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Local date to summarize, e.g. 2026-06-06",
    )
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"Error: --date must be YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
        sys.exit(1)

    config = Config.from_env()

    try:
        conn = open_db(config.memory_db_path)
        db_writer = SQLiteWriter(conn, PROJECT_ROOT)
    except Exception as exc:
        logger.error("Could not open memory DB: %s", exc)
        sys.exit(1)

    gen = DailySummaryGenerator(config)
    try:
        result = gen.generate(args.date, conn, db_writer)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.exception("Daily summary generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
