# Plan: Long-Term Memory and Query System

## Context

`memory_log` currently writes one JSONL record per user question and has no way to query accumulated memories. The task is to add:
1. A structured SQLite database as the primary memory store
2. Passive observation background logging (no VLM, every 30s)
3. SQLite integration for the existing active-query flow
4. A long-term memory query CLI (`ltm_query`) with deterministic retrieval + LLM answer generation
5. A daily summary generator
6. All inside `memory_log/` — no new top-level package

Existing JSONL behavior must not break. The task spec is in `.claude/tasks/long_term_memory_and_query_implementation.md`.

---

## Architecture Overview

```
camera stream
  ↓
ring buffer (capture/)
  ↓
[passive_observer.py] ──────────────→ passive_observations table
[main.py Q&A]          ─→ JSONL (unchanged)
                        ─→ promoted_events + active_query_memories + frames tables
                                                ↓
                              [daily_summary.py CLI]
                                                ↓
                              [ltm_query CLI]
                                query_planner → retrieval → evidence → answer
```

---

## New Files (all under `memory_log/`)

### Foundation Layer

**`src/memory_db.py`**
- `get_connection(db_path: Path) -> sqlite3.Connection` — WAL mode, thread-safe
- `init_schema(conn)` — creates all 5 tables and 11 indexes (DDL as per task spec)
- `open_db(db_path: Path) -> sqlite3.Connection` — calls both; used by all writers

Schema: `passive_observations`, `promoted_events`, `active_query_memories`, `frames`, `daily_summaries` exactly as specified in the task. All timestamp fields stored as ISO8601 TEXT UTC.

**`src/db_writer.py`**
- `SQLiteWriter` class, initialized with `conn: sqlite3.Connection`
- `write_active_query_with_event(record: MemoryRecord, location: LocationInfo, frames: list[np.ndarray], frame_items: list[FrameItem] | None) -> tuple[str, str]` — returns `(event_id, active_query_id)`
  - Generates IDs: `evt_<memory_id>`, `aq_<memory_id>`, `frame_<memory_id>_<idx>`
  - `source_type = 'active_query'`, `promotion_reason = 'user_asked_question'`
  - `scene_summary` = first 200 chars of `model_answer` (weak summary; marks `extra_json = {"summary_from": "model_answer_fallback"}`)
  - `semantic_search_text` = concat of: model_answer (≤300 chars) + " " + camera_source + " " + location.display_name()
  - Inserts promoted event row, active query row, and frame rows in one transaction
- `write_passive_observation(obs_id, timestamp_utc, timestamp_local, timezone, camera_source, location, frame_path, thumbnail_path, phash) -> None`
  - All fields nullable except obs_id, timestamp_utc, created_at_utc
- `write_daily_summary(summary_id, date_local, timezone, summary_text, major_places_json, notable_event_ids_json, active_query_ids_json, coverage_start_utc, coverage_end_utc, raw_model_output) -> None`

### Passive Observer

**`src/passive_observer.py`** — also serves as `__main__` entry
- `PassiveObserver` class
  - Initialized with config, frame source, location sidecar, geocode client, db writer
  - `run()` — loop every `PASSIVE_OBSERVATION_INTERVAL_SEC` seconds; on tick:
    1. `source.get_recent_items(1)` — grab one representative frame
    2. `resolve_location()` + `enrich_location_with_geocode()`
    3. Optionally save frame with `save_frame_image()` to `passive_frame_dir`
    4. Optionally save thumbnail (128px wide resize)
    5. Optionally compute `phash` via `imagehash.phash()` if `imagehash` installed
    6. `db_writer.write_passive_observation(...)`
  - Handles `KeyboardInterrupt` gracefully, calls `source.stop()` / `source.release()`
- `main()` — mirrors `memory_log/src/main.py` startup (load .env, parse args, create frame source, location server, geocode client) but runs `PassiveObserver.run()` instead of REPL
- Run: `uv run python -m src.passive_observer`

### Daily Summary

**`src/daily_summary.py`** — also serves as `__main__` entry
- `DailySummaryGenerator` class
  - `generate(date_local: str, timezone: str, conn: sqlite3.Connection) -> DailySummaryRow`
  - Queries:
    - passive_observations for that date → aggregate into timeline segments (group by hour, cluster by location)
    - promoted_events for that date
    - active_query_memories for that date
  - Builds structured input text (as per task spec format)
  - Calls LLM (text-only, no vision) with separate summary prompt → parses JSON response
  - Writes to `daily_summaries` table
- `main()` — parses `--date YYYY-MM-DD`, loads config/db, runs generator
- Run: `uv run python -m src.daily_summary --date 2026-06-06`

### LTM Query System

**`src/ltm_query/__init__.py`** — empty, marks package

**`src/ltm_query/config.py`**
- `LTMConfig` — thin wrapper around parent `Config` with the new LTM env vars
- Or just extend `memory_log/src/config.py` Config dataclass with the new fields

**`src/ltm_query/query_planner.py`**
- Dataclasses:
  ```python
  @dataclass
  class TimeRange:
      start_utc: str
      end_utc: str

  @dataclass
  class LocationFilter:
      lat: float; lon: float; radius_m: float

  @dataclass
  class StoreQuery:
      store: str       # daily_summaries|passive_observations|promoted_events|active_query_memories|frames
      method: str      # date_lookup|time_range|semantic_search|location_radius
      top_k: int | None
      max_records: int | None

  @dataclass
  class RetrievalPlan:
      intent: str      # whereabouts|visual_recall|interaction_recall|current_scene|general
      time_range: TimeRange | None
      location_filter: LocationFilter | None
      semantic_query: str | None
      needs_current_visual_grounding: bool
      needs_retrieved_frames: bool
      stores_to_query: list[StoreQuery]
  ```
- `QueryPlanner` class
  - Calls LLM text completion with structured JSON output prompt
  - Current local time and timezone injected into system prompt
  - Returns `RetrievalPlan`; falls back to a safe default plan on parse error

**`src/ltm_query/retrieval.py`**
- `retrieve(plan: RetrievalPlan, conn: sqlite3.Connection, config: Config) -> RetrievalResults`
- Executes per-store queries based on plan:
  - `daily_summaries`: date lookup by time_range or semantic LIKE
  - `passive_observations`: time+location SQL filter → returned as raw rows for aggregation
  - `promoted_events`: time/location filter + LIKE on `semantic_search_text` + `scene_summary`
  - `active_query_memories`: time/location filter + LIKE on `user_question` + `model_answer` + `semantic_search_text`
  - `frames`: by linked event_id of top events
- Respects budget: `ltm_promoted_event_top_k`, `ltm_active_query_top_k`, `ltm_max_passive_rows`
- Returns `RetrievalResults(daily_summaries, passive_rows, promoted_events, active_queries, frame_paths)`
- Pluggable interface: `Retriever` ABC with `keyword_retrieve()` so vector retrieval can be swapped in later

**`src/ltm_query/evidence.py`**
- `PassiveTimelineSegment(start_local, end_local, location_label, observation_count)`
- `EvidencePack` dataclass (as per task spec)
- `aggregate_passive_observations(rows: list) -> list[PassiveTimelineSegment]` — clusters by hour/location
- `build_evidence_pack(query, plan, results, visual_grounding_result) -> EvidencePack`

**`src/ltm_query/visual_grounding.py`**
- `VisualGroundingResult` dataclass:
  ```python
  @dataclass
  class VisualGroundingResult:
      current_scene_summary: str
      visible_objects: list[str]
      place_type: str
      resolved_references: dict[str, str]
      semantic_retrieval_query: str
      suggested_location_radius_m: float | None
  ```
- `VisualGrounder` class — reuses `VLMClient.answer_question()` with a structured JSON output prompt
- `ground(query: str, frames: list[np.ndarray], frame_items: list[FrameItem] | None, location: LocationInfo | None) -> VisualGroundingResult | None`
- Separate system prompt from query planner and answer generator

**`src/ltm_query/answer_generator.py`**
- `AnswerGenerator` class
- `generate(query: str, evidence: EvidencePack) -> str`
- Text-only LLM call (no images)
- Prompt instructs: answer only from evidence, cite timestamps/places, use "I found evidence that..." / "I do not have enough memory..." hedging language

**`src/ltm_query/cli.py`** — also serves as `__main__`
- REPL loop (same pattern as `memory_log/src/main.py`)
- Init: load config, open DB, optionally init frame source (for visual grounding)
- Per question:
  1. `QueryPlanner.plan(query)` → `RetrievalPlan`; print plan
  2. If `needs_current_visual_grounding` and frames available: `VisualGrounder.ground(...)` → enriches plan's `semantic_query` and `location_filter`
  3. `retrieve(plan, conn, config)` → `RetrievalResults`; print evidence summary
  4. Sufficiency check: if `intent == 'visual_recall'` and no promoted_events/frames retrieved → one expansion step: re-query with broader time range
  5. `build_evidence_pack(...)` → `EvidencePack`
  6. Optionally load frame images for retrieved events if `needs_retrieved_frames`
  7. `AnswerGenerator.generate(query, evidence)` → print final answer
- `--no-grounding` flag: disables visual grounding regardless of config
- Run: `uv run python -m src.ltm_query`

---

## Modified Files

### `memory_log/src/config.py`
Add these fields to the `Config` dataclass and `from_env()`:
```python
memory_db_path: Path                       # MEMORY_DB_PATH, default outputs/memory.sqlite
passive_observation_interval_sec: float    # PASSIVE_OBSERVATION_INTERVAL_SEC, default 30.0
passive_save_frames: bool                  # PASSIVE_SAVE_FRAMES, default True
passive_frame_dir: Path                    # PASSIVE_FRAME_DIR, default outputs/passive_frames
promoted_event_frame_dir: Path             # PROMOTED_EVENT_FRAME_DIR, default outputs/event_frames
ltm_max_passive_rows: int                  # LTM_MAX_PASSIVE_ROWS, default 1000
ltm_promoted_event_top_k: int             # LTM_PROMOTED_EVENT_TOP_K, default 20
ltm_active_query_top_k: int               # LTM_ACTIVE_QUERY_TOP_K, default 10
ltm_final_event_k: int                    # LTM_FINAL_EVENT_K, default 5
ltm_use_visual_grounding: bool            # LTM_USE_VISUAL_GROUNDING, default True
```

### `memory_log/src/main.py`
In `_handle_question()`, after `record = writer.save_memory(...)` succeeds, add:
```python
# SQLite write — non-blocking: log error but don't fail the Q&A
try:
    db_writer.write_active_query_with_event(record, location, frames, frame_items)
except Exception as exc:
    logger.error("SQLite write failed (JSONL still saved): %s", exc)
```
`db_writer: SQLiteWriter | None` is passed into `_handle_question` (None if DB init fails at startup). Initialize in `main()` after config validation.

### `memory_log/pyproject.toml`
Add dependency:
```toml
"imagehash>=4.3.1",
```
(`imagehash` → pHash for passive observation dedup; small, stable, no breaking deps)

### `memory_log/.env.example`
Add the new env vars block (with sensible defaults).

---

## Implementation Order

Each step is independently testable before the next:

1. **`memory_db.py`** — DDL only; test: `python -c "from src.memory_db import open_db; open_db(Path('test.sqlite'))"`
2. **`db_writer.py`** — write methods; test: call `write_passive_observation` directly
3. **`config.py`** — add new fields; validate existing tests still pass
4. **`main.py`** — add SQLite write after JSONL; test: ask one question, verify both JSONL and SQLite row appear
5. **`passive_observer.py`** — standalone mode; test: run 2 minutes, count rows in `passive_observations`
6. **`ltm_query/query_planner.py`** + **`retrieval.py`** + **`evidence.py`** — core query pipeline; test with existing DB rows
7. **`ltm_query/visual_grounding.py`** — isolated VLM call; test with static frame
8. **`ltm_query/answer_generator.py`** — test with mock evidence pack
9. **`ltm_query/cli.py`** — end-to-end test with example queries
10. **`daily_summary.py`** — test: `uv run python -m src.daily_summary --date 2026-06-06`
11. **Docs** — update `STAGES.md`, `memory_log/README.md`, `CLAUDE.md`, `.env.example`

---

## Key Design Decisions

**IDs**: `evt_<memory_id>`, `aq_<memory_id>`, `obs_<memory_id>`, `frame_<memory_id>_<idx>`, `sum_<date_local>` — all human-readable, sortable, debuggable.

**Timestamps**: Always store as UTC TEXT (`timestamp_utc`) for SQL sorting. Store local as `timestamp_local` for display. Parse from existing ISO8601 strings using `datetime.fromisoformat()`.

**No second VLM call on active query**: `scene_summary` = truncated `model_answer`; mark `extra_json = {"summary_from": "model_answer_fallback"}`. One-call constraint preserved.

**No embeddings in this phase**: SQLite `LIKE` / keyword search on `semantic_search_text`. Design retrieval behind a `Retriever` ABC so ChromaDB can be swapped in later.

**SQLite WAL mode**: Allows `passive_observer` and `main.py` Q&A to write concurrently without blocking each other.

**Visual grounding optional**: Requires live frame source. `ltm_query CLI` skips grounding if frame source unavailable or `--no-grounding` passed.

**Passive observer does NOT use VLM**: Frame metadata + location only. Cheap enough to run continuously.

**`db_writer` failure is non-fatal in `main.py`**: JSONL write always runs first; SQLite write errors are caught and logged. This satisfies "do not break current memory_log behavior."

**Reuse, don't duplicate**:
- `resolve_location()` + `enrich_location_with_geocode()` → reused by passive observer
- `create_frame_source()` → reused by passive observer and ltm_query CLI
- `save_frame_image()` + `resize_frame()` → reused by db_writer for event frames
- `create_vlm_client()` → reused by visual grounder and answer generator

---

## Verification

### Step 3: Existing flow still works
```bash
cd memory_log && uv run python -m src.main
# Ask a question → check JSONL still appended
tail -n 1 outputs/memories.jsonl | jq .
```

### Step 4: SQLite writes on active query
```bash
uv run python -c "
import sqlite3; c = sqlite3.connect('outputs/memory.sqlite')
print(c.execute('SELECT COUNT(*) FROM active_query_memories').fetchone())
print(c.execute('SELECT COUNT(*) FROM promoted_events').fetchone())
print(c.execute('SELECT COUNT(*) FROM frames').fetchone())
"
```

### Step 5: Passive observer creates rows
```bash
uv run python -m src.passive_observer &
sleep 90
uv run python -c "
import sqlite3; c = sqlite3.connect('outputs/memory.sqlite')
print(c.execute('SELECT COUNT(*) FROM passive_observations').fetchone())
"
```

### Step 9: LTM query end-to-end
```bash
cd memory_log && uv run python -m src.ltm_query
# > Where was I yesterday?
# → prints retrieval plan, evidence summary, grounded answer
# > What did I ask about the camera?
# → finds matching active_query_memories rows
```

### Step 10: Daily summary
```bash
uv run python -m src.daily_summary --date 2026-06-06
# → prints structured JSON summary + row in daily_summaries
```
