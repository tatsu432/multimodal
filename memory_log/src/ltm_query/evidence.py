"""Evidence pack builder — assembles retrieved rows into a structured context for answer generation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class PassiveTimelineSegment:
    start_local: str
    end_local: str
    location_label: str
    observation_count: int


@dataclass
class VisualGroundingResult:
    current_scene_summary: str
    visible_objects: list[str]
    place_type: str
    resolved_references: dict[str, str]
    semantic_retrieval_query: str
    suggested_location_radius_m: float | None = None


@dataclass
class EvidencePack:
    user_query: str
    time_range_description: str | None
    location_context: str | None
    visual_grounding: VisualGroundingResult | None
    daily_summaries: list[sqlite3.Row]
    passive_timeline: list[PassiveTimelineSegment]
    promoted_events: list[sqlite3.Row]
    active_queries: list[sqlite3.Row]
    frame_paths: list[str]
    retrieval_reasons: list[str]
    uncertainty_notes: list[str] = field(default_factory=list)


def _local_ts_display(ts: str | None) -> str:
    if not ts:
        return "unknown time"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return ts[:19] if ts else "unknown time"


def aggregate_passive_observations(
    rows: list[sqlite3.Row],
) -> list[PassiveTimelineSegment]:
    """Group passive observations into ~1-hour location segments for display."""
    if not rows:
        return []

    segments: list[PassiveTimelineSegment] = []
    seg_start: str | None = None
    seg_end: str | None = None
    seg_location: str = "unknown location"
    seg_count = 0

    _SEGMENT_GAP_MINUTES = 60

    def _flush() -> None:
        if seg_start is not None:
            segments.append(
                PassiveTimelineSegment(
                    start_local=_local_ts_display(seg_start),
                    end_local=_local_ts_display(seg_end),
                    location_label=seg_location,
                    observation_count=seg_count,
                )
            )

    prev_ts: datetime | None = None
    for row in rows:
        ts_str = row["timestamp_local"] or row["timestamp_utc"]
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue

        loc_label = (
            row["location_label"]
            or row["city"]
            or row["full_address"]
            or "unknown location"
        )

        gap_minutes = 0.0
        if prev_ts is not None:
            try:
                gap_minutes = abs((ts - prev_ts).total_seconds()) / 60.0
            except Exception:
                gap_minutes = 0.0

        if seg_start is None:
            seg_start = ts_str
            seg_end = ts_str
            seg_location = loc_label
            seg_count = 1
        elif gap_minutes > _SEGMENT_GAP_MINUTES or loc_label != seg_location:
            _flush()
            seg_start = ts_str
            seg_end = ts_str
            seg_location = loc_label
            seg_count = 1
        else:
            seg_end = ts_str
            seg_count += 1

        prev_ts = ts

    _flush()
    return segments


def build_evidence_pack(
    user_query: str,
    plan,  # RetrievalPlan
    results,  # RetrievalResults
    visual_grounding: VisualGroundingResult | None,
) -> EvidencePack:
    from src.ltm_query.retrieval import RetrievalResults

    reasons: list[str] = []
    uncertainty: list[str] = []

    if results.daily_summaries:
        reasons.append(f"{len(results.daily_summaries)} daily summary records found")
    if results.passive_rows:
        reasons.append(f"{len(results.passive_rows)} passive observation rows retrieved")
    else:
        uncertainty.append("No passive observation data found for the requested period")

    if results.promoted_events:
        reasons.append(f"{len(results.promoted_events)} promoted visual events found")
    else:
        uncertainty.append("No promoted events found — visual detail may be limited")

    if results.active_queries:
        reasons.append(f"{len(results.active_queries)} past Q&A interactions found")

    if visual_grounding:
        reasons.append(f"Visual grounding resolved: {visual_grounding.current_scene_summary[:80]}")

    tr_desc: str | None = None
    if plan.time_range:
        try:
            s = datetime.fromisoformat(plan.time_range.start_utc).astimezone()
            e = datetime.fromisoformat(plan.time_range.end_utc).astimezone()
            tr_desc = f"{s.strftime('%Y-%m-%d %H:%M')} to {e.strftime('%Y-%m-%d %H:%M %Z')}"
        except Exception:
            tr_desc = f"{plan.time_range.start_utc} to {plan.time_range.end_utc}"

    loc_ctx: str | None = None
    if plan.location_filter:
        loc_ctx = f"near ({plan.location_filter.lat:.4f}, {plan.location_filter.lon:.4f}) within {plan.location_filter.radius_m:.0f}m"

    passive_timeline = aggregate_passive_observations(results.passive_rows)

    return EvidencePack(
        user_query=user_query,
        time_range_description=tr_desc,
        location_context=loc_ctx,
        visual_grounding=visual_grounding,
        daily_summaries=results.daily_summaries,
        passive_timeline=passive_timeline,
        promoted_events=results.promoted_events,
        active_queries=results.active_queries,
        frame_paths=results.frame_paths,
        retrieval_reasons=reasons,
        uncertainty_notes=uncertainty,
    )
