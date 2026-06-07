"""Write memory records to the SQLite memory database."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.schema import LocationInfo, MemoryRecord
from src.utils import FrameItem, frame_capture_timestamp_iso, relative_path, save_frame_image

logger = logging.getLogger("memory_log.db_writer")

_MAX_SCENE_SUMMARY_CHARS = 200
_MAX_SEMANTIC_TEXT_CHARS = 300


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _to_utc_iso(iso_local: str) -> str:
    """Convert a local ISO8601 timestamp (with offset) to UTC ISO8601."""
    try:
        dt = datetime.fromisoformat(iso_local)
        return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")
    except (ValueError, TypeError):
        return iso_local


def _timezone_name(iso_local: str) -> str | None:
    try:
        dt = datetime.fromisoformat(iso_local)
        offset = dt.utcoffset()
        if offset is None:
            return None
        total_sec = int(offset.total_seconds())
        sign = "+" if total_sec >= 0 else "-"
        h, m = divmod(abs(total_sec) // 60, 60)
        return f"UTC{sign}{h:02d}:{m:02d}"
    except (ValueError, TypeError):
        return None


def _location_columns(location: LocationInfo) -> dict:
    return {
        "latitude": location.lat,
        "longitude": location.lon,
        "location_label": location.label,
        "full_address": location.full_address,
        "city": location.city,
        "prefecture": location.prefecture,
        "country": location.country,
        "postal_code": location.postal_code,
        "location_source": location.source,
        "geocode_provider": location.geocode_provider,
        "geocoded_at": location.geocoded_at,
    }


def _build_semantic_search_text(
    model_answer: str,
    camera_source: str | None,
    location: LocationInfo,
) -> str:
    parts: list[str] = []
    if model_answer:
        parts.append(model_answer[:_MAX_SEMANTIC_TEXT_CHARS])
    if camera_source:
        parts.append(camera_source)
    loc_text = location.display_name()
    if loc_text and loc_text != "not available":
        parts.append(loc_text)
    return " ".join(parts)


class SQLiteWriter:
    def __init__(self, conn: sqlite3.Connection, project_root: Path) -> None:
        self._conn = conn
        self._project_root = project_root

    def write_active_query_with_event(
        self,
        record: MemoryRecord,
        location: LocationInfo,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None,
        event_frame_dir: Path | None = None,
    ) -> tuple[str, str]:
        """
        Write one promoted_events row, one active_query_memories row, and
        N frames rows — all in a single transaction.

        Returns (event_id, active_query_id).
        """
        event_id = f"evt_{record.memory_id}"
        active_query_id = f"aq_{record.memory_id}"
        now_utc = _now_utc_iso()
        ts_utc = _to_utc_iso(record.timestamp)
        tz_name = _timezone_name(record.timestamp)
        loc = _location_columns(location)
        scene_summary = record.model_answer[:_MAX_SCENE_SUMMARY_CHARS] if record.model_answer else ""
        semantic_text = _build_semantic_search_text(record.model_answer, record.camera_source, location)
        extra = json.dumps({"summary_from": "model_answer_fallback"})

        # Save frames to event_frame_dir if provided (reuse existing frame paths otherwise)
        saved_frame_paths: list[str] = []
        if event_frame_dir is not None and frames:
            for idx, frame in enumerate(frames):
                suffix = f"_f{idx + 1:02d}" if len(frames) > 1 else ""
                try:
                    saved = save_frame_image(frame, event_frame_dir, record.memory_id, suffix=suffix)
                    saved_frame_paths.append(relative_path(saved, self._project_root))
                except (OSError, RuntimeError) as exc:
                    logger.warning("Could not save event frame %d: %s", idx, exc)
        else:
            saved_frame_paths = list(record.frame_paths)

        with self._conn:
            # promoted_events
            self._conn.execute(
                """
                INSERT OR IGNORE INTO promoted_events (
                    event_id, start_ts_utc, end_ts_utc, timestamp_local, timezone,
                    source_type, promotion_reason, camera_source,
                    latitude, longitude, location_label, full_address,
                    city, prefecture, country, postal_code, location_source,
                    scene_summary, semantic_search_text, raw_vlm_output,
                    created_at_utc, extra_json
                ) VALUES (
                    :event_id, :start_ts_utc, :end_ts_utc, :timestamp_local, :timezone,
                    :source_type, :promotion_reason, :camera_source,
                    :latitude, :longitude, :location_label, :full_address,
                    :city, :prefecture, :country, :postal_code, :location_source,
                    :scene_summary, :semantic_search_text, :raw_vlm_output,
                    :created_at_utc, :extra_json
                )
                """,
                {
                    "event_id": event_id,
                    "start_ts_utc": ts_utc,
                    "end_ts_utc": ts_utc,
                    "timestamp_local": record.timestamp,
                    "timezone": tz_name,
                    "source_type": "active_query",
                    "promotion_reason": "user_asked_question",
                    "camera_source": record.camera_source,
                    **loc,
                    "scene_summary": scene_summary,
                    "semantic_search_text": semantic_text,
                    "raw_vlm_output": record.model_answer,
                    "created_at_utc": now_utc,
                    "extra_json": extra,
                },
            )

            # active_query_memories
            self._conn.execute(
                """
                INSERT OR IGNORE INTO active_query_memories (
                    active_query_id, timestamp_utc, timestamp_local, timezone,
                    linked_event_id, user_question, model_answer, camera_source,
                    latitude, longitude, location_label, full_address,
                    city, prefecture, country, postal_code, location_source,
                    semantic_search_text, raw_vlm_output,
                    created_at_utc, extra_json
                ) VALUES (
                    :active_query_id, :timestamp_utc, :timestamp_local, :timezone,
                    :linked_event_id, :user_question, :model_answer, :camera_source,
                    :latitude, :longitude, :location_label, :full_address,
                    :city, :prefecture, :country, :postal_code, :location_source,
                    :semantic_search_text, :raw_vlm_output,
                    :created_at_utc, :extra_json
                )
                """,
                {
                    "active_query_id": active_query_id,
                    "timestamp_utc": ts_utc,
                    "timestamp_local": record.timestamp,
                    "timezone": tz_name,
                    "linked_event_id": event_id,
                    "user_question": record.user_question,
                    "model_answer": record.model_answer,
                    "camera_source": record.camera_source,
                    **loc,
                    "semantic_search_text": semantic_text,
                    "raw_vlm_output": record.model_answer,
                    "created_at_utc": now_utc,
                    "extra_json": None,
                },
            )

            # frames
            for idx, frame_path in enumerate(saved_frame_paths):
                frame_id = f"frame_{record.memory_id}_{idx + 1:02d}"
                frame_ts_local: str | None = None
                frame_ts_utc: str | None = None
                if record.frame_timestamps and idx < len(record.frame_timestamps):
                    frame_ts_local = record.frame_timestamps[idx]
                    frame_ts_utc = _to_utc_iso(frame_ts_local)
                elif frame_items and idx < len(frame_items):
                    frame_ts_local = frame_capture_timestamp_iso(frame_items[idx].timestamp)
                    frame_ts_utc = _to_utc_iso(frame_ts_local)

                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO frames (
                        frame_id, promoted_event_id, active_query_id,
                        timestamp_utc, timestamp_local, frame_index,
                        frame_path, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        frame_id,
                        event_id,
                        active_query_id,
                        frame_ts_utc,
                        frame_ts_local,
                        idx + 1,
                        frame_path,
                        now_utc,
                    ),
                )

        logger.info("Wrote active query %s + event %s to SQLite", active_query_id, event_id)
        return event_id, active_query_id

    def write_passive_observation(
        self,
        obs_id: str,
        timestamp_utc: str,
        timestamp_local: str,
        timezone_name: str | None,
        camera_source: str | None,
        location: LocationInfo,
        frame_path: str | None,
        thumbnail_path: str | None,
        frame_timestamp: str | None = None,
        phash: str | None = None,
    ) -> None:
        now_utc = _now_utc_iso()
        loc = _location_columns(location)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO passive_observations (
                    obs_id, timestamp_utc, timestamp_local, timezone,
                    camera_source,
                    latitude, longitude,
                    location_label, full_address, city, prefecture,
                    country, postal_code, location_source, geocode_provider, geocoded_at,
                    frame_path, thumbnail_path, frame_timestamp,
                    phash,
                    created_at_utc
                ) VALUES (
                    :obs_id, :timestamp_utc, :timestamp_local, :timezone,
                    :camera_source,
                    :latitude, :longitude,
                    :location_label, :full_address, :city, :prefecture,
                    :country, :postal_code, :location_source, :geocode_provider, :geocoded_at,
                    :frame_path, :thumbnail_path, :frame_timestamp,
                    :phash,
                    :created_at_utc
                )
                """,
                {
                    "obs_id": obs_id,
                    "timestamp_utc": timestamp_utc,
                    "timestamp_local": timestamp_local,
                    "timezone": timezone_name,
                    "camera_source": camera_source,
                    **loc,
                    "frame_path": frame_path,
                    "thumbnail_path": thumbnail_path,
                    "frame_timestamp": frame_timestamp,
                    "phash": phash,
                    "created_at_utc": now_utc,
                },
            )
        logger.debug("Wrote passive observation %s", obs_id)

    def write_daily_summary(
        self,
        summary_id: str,
        date_local: str,
        timezone_name: str | None,
        summary_text: str,
        major_places_json: str | None,
        notable_event_ids_json: str | None,
        active_query_ids_json: str | None,
        coverage_start_utc: str | None,
        coverage_end_utc: str | None,
        raw_model_output: str | None,
        semantic_search_text: str | None = None,
    ) -> None:
        now_utc = _now_utc_iso()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO daily_summaries (
                    summary_id, date_local, timezone,
                    summary_text, major_places_json, notable_event_ids_json,
                    active_query_ids_json, coverage_start_utc, coverage_end_utc,
                    semantic_search_text, raw_model_output, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary_id,
                    date_local,
                    timezone_name,
                    summary_text,
                    major_places_json,
                    notable_event_ids_json,
                    active_query_ids_json,
                    coverage_start_utc,
                    coverage_end_utc,
                    semantic_search_text,
                    raw_model_output,
                    now_utc,
                ),
            )
        logger.info("Wrote daily summary %s", summary_id)
