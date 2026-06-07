"""SQLite memory database — schema initialization and connection management."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("memory_log.memory_db")

_DDL = """
CREATE TABLE IF NOT EXISTS passive_observations (
    obs_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    timestamp_local TEXT,
    timezone TEXT,

    camera_source TEXT,

    latitude REAL,
    longitude REAL,
    location_accuracy_m REAL,

    location_label TEXT,
    full_address TEXT,
    city TEXT,
    prefecture TEXT,
    country TEXT,
    postal_code TEXT,
    location_source TEXT,
    geocode_provider TEXT,
    geocoded_at TEXT,

    frame_path TEXT,
    thumbnail_path TEXT,
    frame_timestamp TEXT,

    phash TEXT,
    image_embedding_id TEXT,

    created_at_utc TEXT NOT NULL,
    extra_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_passive_observations_time
ON passive_observations(timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_passive_observations_location
ON passive_observations(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_passive_observations_camera_time
ON passive_observations(camera_source, timestamp_utc);

CREATE TABLE IF NOT EXISTS promoted_events (
    event_id TEXT PRIMARY KEY,

    start_ts_utc TEXT NOT NULL,
    end_ts_utc TEXT,
    timestamp_local TEXT,
    timezone TEXT,

    source_type TEXT NOT NULL,
    promotion_reason TEXT,

    camera_source TEXT,

    latitude REAL,
    longitude REAL,
    location_label TEXT,
    full_address TEXT,
    city TEXT,
    prefecture TEXT,
    country TEXT,
    postal_code TEXT,
    location_source TEXT,

    scene_summary TEXT,
    object_tags_json TEXT,
    action_tags_json TEXT,
    place_tags_json TEXT,

    semantic_search_text TEXT,
    text_embedding_id TEXT,

    raw_vlm_output TEXT,

    created_at_utc TEXT NOT NULL,
    extra_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_promoted_events_time
ON promoted_events(start_ts_utc);

CREATE INDEX IF NOT EXISTS idx_promoted_events_location
ON promoted_events(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_promoted_events_source_time
ON promoted_events(source_type, start_ts_utc);

CREATE TABLE IF NOT EXISTS active_query_memories (
    active_query_id TEXT PRIMARY KEY,

    timestamp_utc TEXT NOT NULL,
    timestamp_local TEXT,
    timezone TEXT,

    linked_event_id TEXT,

    user_question TEXT NOT NULL,
    model_answer TEXT,

    camera_source TEXT,

    latitude REAL,
    longitude REAL,
    location_label TEXT,
    full_address TEXT,
    city TEXT,
    prefecture TEXT,
    country TEXT,
    postal_code TEXT,
    location_source TEXT,

    semantic_search_text TEXT,
    text_embedding_id TEXT,

    raw_vlm_output TEXT,

    created_at_utc TEXT NOT NULL,
    extra_json TEXT,

    FOREIGN KEY(linked_event_id) REFERENCES promoted_events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_active_query_memories_time
ON active_query_memories(timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_active_query_memories_event
ON active_query_memories(linked_event_id);

CREATE INDEX IF NOT EXISTS idx_active_query_memories_location
ON active_query_memories(latitude, longitude);

CREATE TABLE IF NOT EXISTS frames (
    frame_id TEXT PRIMARY KEY,

    passive_obs_id TEXT,
    promoted_event_id TEXT,
    active_query_id TEXT,

    timestamp_utc TEXT,
    timestamp_local TEXT,
    frame_index INTEGER,

    frame_path TEXT NOT NULL,
    thumbnail_path TEXT,

    image_embedding_id TEXT,

    created_at_utc TEXT NOT NULL,
    extra_json TEXT,

    FOREIGN KEY(passive_obs_id) REFERENCES passive_observations(obs_id),
    FOREIGN KEY(promoted_event_id) REFERENCES promoted_events(event_id),
    FOREIGN KEY(active_query_id) REFERENCES active_query_memories(active_query_id)
);

CREATE INDEX IF NOT EXISTS idx_frames_event
ON frames(promoted_event_id);

CREATE INDEX IF NOT EXISTS idx_frames_active_query
ON frames(active_query_id);

CREATE INDEX IF NOT EXISTS idx_frames_passive_obs
ON frames(passive_obs_id);

CREATE TABLE IF NOT EXISTS daily_summaries (
    summary_id TEXT PRIMARY KEY,
    date_local TEXT NOT NULL,
    timezone TEXT,

    summary_text TEXT NOT NULL,

    major_places_json TEXT,
    notable_event_ids_json TEXT,
    active_query_ids_json TEXT,

    coverage_start_utc TEXT,
    coverage_end_utc TEXT,

    semantic_search_text TEXT,
    text_embedding_id TEXT,

    raw_model_output TEXT,

    created_at_utc TEXT NOT NULL,
    extra_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_daily_summaries_date
ON daily_summaries(date_local);
"""


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open the memory SQLite database, creating schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL mode: readers do not block writers; multiple processes can write safely
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    for statement in _DDL.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    conn.commit()
    logger.debug("Memory DB schema initialized")
