We need to implement the next phase of the wearable multimodal AI assistant project.

Please inspect the current repository first before editing. The current project already has:

* `camera_test/`: camera stream validation
* `capture/`: shared camera ingestion and ring buffer
* `providers/`: shared provider clients, including Ollama
* `vlm_smoke/`: live visual QA
* `memory_log/`: question-driven JSONL memory logging

The current `memory_log` package writes memories only when the user asks a question. The current record includes roughly:

* `memory_id`
* `timestamp`
* `user_question`
* `model_answer`
* `frame_paths`
* `frame_timestamps`
* `location`
* `camera_source`

Now we want to implement the next memory layer and long-term memory query system.

Important design principles:

1. Do not break the current `memory_log` behavior.
2. Keep JSONL as an optional raw/debug log, but introduce SQLite as the main structured memory database.
3. Use `uv` and existing project style.
4. Reuse existing camera ingestion, ring buffer, VLM client, location, geocode, and frame-saving logic where possible.
5. Do not implement a complex multi-agent system yet.
6. Build a deterministic, debuggable MVP with a bounded optional re-retrieval/visual-grounding step.
7. Do not store raw image bytes in the database. Store frame paths / thumbnail paths / clip paths.
8. Store long VLM outputs in `TEXT` fields, not fixed-length `VARCHAR`.
9. Use flexible JSON columns as text where needed, but keep retrieval-critical fields as normal SQL columns.
10. Start with SQLite. Do not require Postgres, PostGIS, or cloud infrastructure for this phase.
11. Use separate prompts for context retrieval / query planning and answer generation, even if the same underlying model is used.

High-level target architecture:

```text
camera stream
  ↓
ring buffer
  ↓
passive observation writer
  ↓
promoted event writer
  ↓
active query memory writer
  ↓
daily summary generator
  ↓
long-term memory query
      ↓
      optional visual grounding for “this / here / current scene”
      ↓
      context retriever / query planner
      ↓
      selected DB queries
      ↓
      evidence pack builder
      ↓
      optional visual interpretation on selected frames
      ↓
      answer generator
```

Memory types to implement:

1. PassiveObservationMemory
2. PromotedEventMemory
3. ActiveQueryMemory
4. DailySummaryMemory

Do not implement SegmentMemory as a first-class memory type in this phase. If you need grouping for queries like “Where was I yesterday?”, compute it from passive observations on demand. Segment-like grouping can be a derived/cache layer later.

---

## DATABASE DESIGN

Create a SQLite database under something like:

```text
memory_log/outputs/memory.sqlite
```

Add a small database module, for example:

```text
memory_log/src/memory_db.py
memory_log/src/memory_schema.py
```

or another clean structure consistent with the repo.

Please add initialization code that creates these tables if missing.

Table 1: `passive_observations`

Purpose:
Cheap source-of-truth traces for time/location/camera coverage. These are not rich semantic memories. They are used for questions like:

* “Where was I yesterday?”
* “Where was I around 3 PM?”
* “Was I near this location yesterday?”
* “Which camera source was active?”

Schema suggestion:

```sql
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
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_passive_observations_time
ON passive_observations(timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_passive_observations_location
ON passive_observations(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_passive_observations_camera_time
ON passive_observations(camera_source, timestamp_utc);
```

Behavior:

* Passive observation should be written periodically in the background.
* Default persistent interval: 30 seconds.
* Make it configurable via env var, e.g. `PASSIVE_OBSERVATION_INTERVAL_SEC=30`.
* Store metadata and optionally one representative frame path/thumbnail path.
* Do not run a VLM call every 30 seconds.
* Do not create text embeddings for passive observations in this phase.
* It is okay if passive observations are initially metadata + frame path only.

Table 2: `promoted_events`

Purpose:
Richer semantic visual events. These are created when something is meaningful enough to store as a long-term visual memory.

Initial promotion triggers:

* Active user query happened.
* Optional manual promotion.
* Optional simple time anchor.
* Optional future visual novelty, but do not overcomplicate this now.

For this phase, the most important path is:
When an active query happens, create or link a promoted event using the same frame context, because user attention implies potential importance.

Schema suggestion:

```sql
CREATE TABLE IF NOT EXISTS promoted_events (
    event_id TEXT PRIMARY KEY,

    start_ts_utc TEXT NOT NULL,
    end_ts_utc TEXT,
    timestamp_local TEXT,
    timezone TEXT,

    source_type TEXT NOT NULL, -- active_query, manual, visual_change, time_anchor, import
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
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_promoted_events_time
ON promoted_events(start_ts_utc);

CREATE INDEX IF NOT EXISTS idx_promoted_events_location
ON promoted_events(latitude, longitude);

CREATE INDEX IF NOT EXISTS idx_promoted_events_source_time
ON promoted_events(source_type, start_ts_utc);
```

Important:

* `scene_summary` should be concise.
* `raw_vlm_output` can be long.
* `semantic_search_text` should be deliberately constructed for semantic retrieval. Do not dump raw JSON into the embedding text.
* Good `semantic_search_text` example:

```text
Indoor office desk scene. Visible objects include laptop, notebook, coffee cup, cables, and camera. The user appears to be working at a desk. Place type: office.
```

* Exact timestamp, latitude, longitude, postal code, and camera source should be stored as metadata, not forced into the semantic text.

Table 3: `active_query_memories`

Purpose:
Record the actual user question and model answer. This is different from a promoted event.

It should answer questions like:

* “What did I ask earlier about the camera?”
* “What did the model tell me about the router?”
* “What was my question yesterday near this location?”

Schema suggestion:

```sql
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
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_active_query_memories_time
ON active_query_memories(timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_active_query_memories_event
ON active_query_memories(linked_event_id);

CREATE INDEX IF NOT EXISTS idx_active_query_memories_location
ON active_query_memories(latitude, longitude);
```

Behavior:

* Whenever the existing `memory_log` writes a question-driven JSONL memory, also write an `active_query_memories` row.
* Also create a linked `promoted_events` row with `source_type='active_query'`.
* The active query and promoted event are linked but not identical.
* Active query frames are the frames used to answer the user’s question.
* Promoted event frames can initially use the same frames for convenience, but the schema should not assume they must always be the same.

Table 4: `frames`

Purpose:
Normalize frame paths instead of storing arrays everywhere.

Schema suggestion:

```sql
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
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_frames_event
ON frames(promoted_event_id);

CREATE INDEX IF NOT EXISTS idx_frames_active_query
ON frames(active_query_id);

CREATE INDEX IF NOT EXISTS idx_frames_passive_obs
ON frames(passive_obs_id);
```

Table 5: `daily_summaries`

Purpose:
Compressed high-level summaries for long-horizon recall.

Schema suggestion:

```sql
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
```

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_daily_summaries_date
ON daily_summaries(date_local);
```

Daily summary generation:

* Input should be structured text from:

  * passive observation time/location groups
  * promoted events
  * active query memories
* Do not feed all raw frames from the day into the daily summarizer.
* The role of the daily summary is abstract compression, not discovering new visual details from all videos.
* If some promoted events have frame paths and weak summaries, it is okay to include their summaries and metadata only for now.
* Later visual enrichment can be added.

Suggested daily summary input format:

```text
Date: YYYY-MM-DD
Timezone: Asia/Tokyo

Passive observation location timeline:
- 09:00–10:30: near Home / Shibuya, N observations
- 11:00–18:00: near an office / Nihonbashi, N observations

Promoted events:
- 11:10, Nihonbashi: Office desk with laptop and camera setup.
- 15:20, Nihonbashi: User looked at a whiteboard.

Active query memories:
- 11:11, user asked: “Is my camera streaming?” answer: “Yes, the stream appears active.”
```

Daily summary output should be structured, preferably JSON parsed into:

* `summary_text`
* `major_places`
* `notable_event_ids`
* `active_query_ids`
* `uncertainties`

---

## ACTIVE QUERY INTEGRATION

Modify the current `memory_log` flow so that when the user asks a question:

1. The system still performs the current VLM call over recent frames.
2. The system still appends the existing JSONL memory for backwards compatibility.
3. The system also writes to SQLite:

   * one `promoted_events` row
   * one `active_query_memories` row linked to that event
   * one or more `frames` rows linked to both the active query and promoted event
4. The location fields should be copied from the existing location metadata logic.
5. Use IDs that are stable and human-debuggable, e.g. `evt_<timestamp>`, `aq_<timestamp>`, `frame_<timestamp>_<index>`.

For the promoted event created from an active query:

* `source_type = 'active_query'`
* `promotion_reason = 'user_asked_question'`
* `scene_summary`: initially can be derived from the model answer, but preferably ask the VLM for structured output only if that does not break the existing one-call behavior. If adding a second VLM call is too much right now, use the model answer as a weak summary and mark this in `extra_json`.
* `semantic_search_text`: combine model answer + important visual words + broad place type if available.
* Do not include raw lat/lon/timestamps in the semantic text except when semantically useful.

---

## PASSIVE OBSERVATION WRITER

Add a passive observation mode or background component.

Preferred MVP:

* Add a CLI entrypoint or config flag to `memory_log` so we can run passive logging.
* Every `PASSIVE_OBSERVATION_INTERVAL_SEC` seconds, take one representative frame from the ring buffer and write:

  * timestamp
  * camera source
  * location metadata
  * frame path / thumbnail path if saved
  * optional perceptual hash if easy
* Do not run VLM.
* Do not run embeddings.
* Do not do semantic summaries.
* Keep it cheap and reliable.

Possible command:

```bash
cd memory_log
uv run python -m src.passive_observer
```

or another clean command consistent with the existing project.

Success criteria:

* Running passive observer for a few minutes creates rows in `passive_observations`.
* Frame paths, if enabled, point to real files.
* Location metadata is populated using existing location logic.
* Stream reconnect behavior should remain robust.

---

## PROMOTED EVENT MEMORY

For this phase, promoted events should be created at least from active queries.

Optional if simple:

* Add a manual promotion utility that promotes the latest frame buffer to an event.
* Add a simple time-anchor promotion rule, e.g. every 15–20 minutes while passive observation is active.
* Do not implement complex visual novelty detection unless it is very small and isolated.

Important:

* Promoted event is semantic/visual memory.
* Passive observation is source-of-truth time/location trace.
* Active query memory is user interaction history.
* These are related but distinct.

---

## LONG-TERM MEMORY QUERY SYSTEM

Implement a long-term memory query module.

Suggested files:

```text
memory_log/src/ltm_query/
  __init__.py
  query_planner.py
  retrieval.py
  evidence.py
  answer_generator.py
  visual_grounding.py
  cli.py
```

or use a simpler flat structure if that fits the repo better.

We want this workflow:

```text
user long-memory query
  ↓
optional visual grounding if query refers to current scene
  ↓
query planner / context retriever
  ↓
deterministic retrieval from selected DB tables
  ↓
evidence pack builder
  ↓
optional visual interpretation on selected retrieved frames
  ↓
answer generator
```

Do not implement free-form multi-agent behavior. Use a deterministic pipeline with at most one bounded re-retrieval/expansion step later.

Implement at least a CLI:

```bash
cd memory_log
uv run python -m src.ltm_query.cli
```

or:

```bash
uv run python -m src.ltm_query
```

The CLI should accept a user query and print:

* the retrieval plan
* retrieved evidence summary
* final answer

---

## QUERY PLANNER / CONTEXT RETRIEVER

Use a separate prompt from answer generation.

The query planner should not answer the user directly. It should return structured JSON.

It should decide:

* intent
* time range
* location constraint
* semantic query
* which stores/tables to query
* whether current visual grounding is needed
* whether retrieved frames may be needed
* retrieval budget

Supported memory stores:

* `daily_summaries`
* `passive_observations`
* `promoted_events`
* `active_query_memories`
* `frames`

Example planner output:

```json
{
  "intent": "whereabouts",
  "time_range": {
    "start_local": "2026-06-06T00:00:00+09:00",
    "end_local": "2026-06-07T00:00:00+09:00"
  },
  "location_filter": null,
  "semantic_query": null,
  "needs_current_visual_grounding": false,
  "needs_retrieved_frames": false,
  "stores_to_query": [
    {
      "store": "daily_summaries",
      "method": "date_lookup",
      "top_k": 1
    },
    {
      "store": "passive_observations",
      "method": "time_range",
      "max_records": 1000
    }
  ]
}
```

For queries like:

```text
“What was here yesterday?”
“Have I seen this before?”
“When did I last see this object?”
```

the planner should set:

```json
{
  "needs_current_visual_grounding": true
}
```

because “here” or “this” needs recent-frame interpretation.

---

## VISUAL GROUNDING

Visual grounding is not the same as final answering.

Use VLM visual grounding only when the query depends on current visual context, such as:

* here
* this
* that
* these
* current scene
* what I am looking at
* this object
* this room
* this place

Visual grounding input:

* user query
* latest N frames from ring buffer
* current location metadata if available

Visual grounding output should be structured JSON:

```json
{
  "current_scene_summary": "The user is looking at an office desk with a laptop, cables, and a camera setup.",
  "visible_objects": ["laptop", "cables", "camera", "desk"],
  "place_type": "office",
  "resolved_references": {
    "here": "current physical location and current visible desk scene"
  },
  "semantic_retrieval_query": "office desk laptop cables camera setup",
  "suggested_location_radius_m": 100
}
```

The visual grounder should not answer the long-term memory question. It only creates better retrieval constraints.

---

## RETRIEVAL POLICY

Do not use one global top-k.

Use query-dependent retrieval budgets.

Initial defaults:

* `daily_summaries`: top 1–3 by date or semantic search
* `passive_observations`: retrieve by time/location filters, not vector search; max 500–1000 raw rows, then aggregate
* `promoted_events`: initial top 20, final 5–8
* `active_query_memories`: initial top 10, final 3–5
* `frames`: only when needed; select frames from top 3–5 events, 2–8 frames per event

Prefer metadata filters first:

* time range
* location radius
* camera source

Then semantic search if available.

For now, if no vector DB is implemented:

* use SQLite LIKE / simple keyword matching over `semantic_search_text`, `scene_summary`, `user_question`, and `model_answer`
* design the code so vector retrieval can be plugged in later

If adding ChromaDB is easy and does not destabilize the project:

* add a small optional vector retriever abstraction
* embed only deliberately constructed `semantic_search_text`
* do not embed raw JSON
* do not embed exact lat/lon/timestamps unless they are semantically meaningful
* exact metadata should be used as SQL filters

Important:

* For passive observations, aggregate before sending to the answer generator.
* Example aggregation:

```text
09:00–10:30: near Shibuya, 120 observations
11:00–18:00: near Nihonbashi / an office, 840 observations
19:30–20:00: near Shinjuku, 60 observations
```

---

## EVIDENCE PACK

Create an explicit evidence pack object before answer generation.

The answer generator should receive a clean, compact evidence pack, not raw DB rows.

Evidence pack should include:

* original user query
* interpreted time range
* interpreted location
* visual grounding result if used
* retrieved daily summaries
* aggregated passive observation timeline
* retrieved promoted events
* retrieved active query memories
* selected frame paths if any
* retrieval reasons
* uncertainty notes

Example evidence item:

```json
{
  "memory_type": "promoted_event",
  "memory_id": "evt_2026...",
  "retrieval_reason": "near current location yesterday and semantically related to desk/camera setup",
  "timestamp_local": "2026-06-06T15:20:00+09:00",
  "location_label": "Nihonbashi, Tokyo",
  "scene_summary": "Office desk with laptop, notebook, and camera setup.",
  "frame_paths": ["..."]
}
```

---

## ANSWER GENERATOR

Use a separate prompt from query planning.

The answer generator should:

* answer using only the evidence pack
* mention timestamps and places when useful
* explicitly say when evidence is incomplete
* never invent unseen events, locations, or visual details
* distinguish:

  * “I found evidence that...”
  * “I do not have enough memory to know...”
  * “Based on saved memories only...”

Example final answer for “Where was I yesterday?”:

```text
Based on saved passive observations and summaries, yesterday you were mostly around Nihonbashi / an office from late morning to early evening. I also found a later location cluster around Shinjuku. This is based on saved observations only, so it may miss periods where the camera/location logger was inactive.
```

Example final answer for “What was here yesterday?”:

```text
I grounded “here” as your current desk/office scene near Nihonbashi. Looking at yesterday’s memories near the same location, I found promoted events showing a laptop, notebook, cables, and a camera setup on the desk. I also found passive observations confirming you were near this location during the afternoon. I do not have enough visual evidence to know every object that was present throughout the day.
```

---

## SUFFICIENCY CHECK / BOUNDED RETRIEVAL

For this phase, keep it simple.

Implement either:

1. deterministic sufficiency checks, or
2. a small LLM-based sufficiency checker returning JSON

But do not allow unlimited agent loops.

Allowed:

* one optional expansion step if evidence is clearly insufficient

Examples:

* If query asks “where,” passive observations are enough.
* If query asks “what did I see,” promoted events and/or frames are needed.
* If query asks “this/here,” visual grounding is needed.
* If query asks object-level visual details and only passive observations were retrieved, mark insufficient and retrieve promoted events/frames if available.

---

## DAILY SUMMARY GENERATOR

Add a command to generate a daily summary for a given local date.

Example:

```bash
uv run python -m src.daily_summary --date 2026-06-06
```

or another repo-consistent command.

Input:

* passive observations for that date, aggregated by location/time
* promoted events for that date
* active query memories for that date

Output:

* one row in `daily_summaries`
* JSON/debug print to console

Do not use raw daily frames by default.

---

## CONFIGURATION

Add env vars as needed:

```text
MEMORY_DB_PATH=outputs/memory.sqlite
PASSIVE_OBSERVATION_INTERVAL_SEC=30
PASSIVE_SAVE_FRAMES=true
PASSIVE_FRAME_DIR=outputs/passive_frames
PROMOTED_EVENT_FRAME_DIR=outputs/event_frames
LTM_MAX_PASSIVE_ROWS=1000
LTM_PROMOTED_EVENT_TOP_K=20
LTM_ACTIVE_QUERY_TOP_K=10
LTM_FINAL_EVENT_K=5
LTM_USE_VISUAL_GROUNDING=true
```

Use sensible defaults.

---

## TESTING / VALIDATION

Add lightweight tests or validation scripts if the repo already uses tests. If not, add simple manual commands and document them.

Minimum success criteria:

1. Existing `memory_log` interactive Q&A still works.
2. Asking a question still writes JSONL.
3. Asking a question now also writes:

   * one active query memory
   * one linked promoted event
   * frame rows
4. Passive observer writes passive observation rows every configured interval.
5. Daily summary command creates a daily summary row.
6. Long-term query CLI can answer:

   * “Where was I yesterday?”
   * “What did I ask about the camera?”
   * “What did I see near this location?”
   * “What was here yesterday?” if visual grounding is enabled and frames are available
7. All commands should run through `uv`.

Please update documentation:

* update the project stages / roadmap file
* update `memory_log/README.md`
* explain the new SQLite DB
* explain memory types
* explain long-term query workflow
* explain example commands
* explain current limitations

---

## IMPLEMENTATION STYLE

Please implement this incrementally and cleanly.

Before making large changes:

1. Inspect existing code.
2. Identify current entrypoints and data flow.
3. Reuse existing modules when possible.
4. Avoid duplicated VLM/location/frame-saving logic.
5. Keep functions small and testable.
6. Use type hints and dataclasses/Pydantic if already used in the repo.
7. Do not introduce heavy dependencies unless necessary.
8. Avoid breaking current CLI behavior.
9. Prefer explicit, boring, deterministic code over clever agentic abstractions.

After implementation, show:

* files changed
* how to run the new commands
* any limitations or TODOs
* how the new system maps to:

  * passive observation memory
  * promoted event memory
  * active query memory
  * daily summary
  * long-term memory query answering
