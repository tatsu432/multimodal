# Task: Long-term query logging (`long_term_query_logs`)

## Context

The LTM query pipeline (`memory_log/src/ltm_query/`) —
`QueryPlanner → (VisualGrounder) → MemoryRetriever → build_evidence_pack → AnswerGenerator` —
runs and prints, then discards everything. There is no record of what was asked, how the LLM
planned it, what was retrieved, the answer, or per-stage latency. That makes the retrieval
system impossible to evaluate, debug, or regression-test.

**Goal:** persist one row per LTM query interaction for observability / eval.

**Hard constraint (user):** *do not include it in normal long-term memory retrieval.* The log
is telemetry about the assistant, not a memory the assistant should ever recall.

## Design decision — separate SQLite file (`outputs/long_term_query_logs.sqlite`)

A standalone DB file with its own writer, mirroring `src/geocode_cache.py`. Rationale:

1. **Separation of concerns** — eval telemetry vs. the domain memory stores are different
   bounded contexts.
2. **Structural exclusion, not conventional.** `MemoryRetriever.__init__` (`retrieval.py:67`)
   only ever holds the `memory.sqlite` connection; `_execute_store_query` (`retrieval.py:84-95`)
   is a hardcoded dispatch over exactly five tables. A separate file is physically unreachable
   by retrieval.
3. **Existing precedent** — `geocode_cache.sqlite` already isolates a cross-cutting concern
   this way.
4. **Independent lifecycle** — the log grows per query and can be rotated/deleted/exported
   without touching memory data or its WAL.

## Implementation

### New file `src/query_log.py` (mirrors `src/geocode_cache.py`)
- `CREATE_TABLE_SQL` + indexes for `long_term_query_logs`.
- `QueryLogRecord` dataclass (all columns).
- `QueryLogWriter`: `__init__(path)` (mkdir, connect `check_same_thread=False`, WAL +
  `busy_timeout`, lock, ensure schema, commit), `log(record)` (named INSERT under lock+commit),
  `close()`.
- `build_query_log_record(...)`: assembles a record from pipeline artifacts, tolerating `None`.
  Reuses `make_memory_id()` (`src/utils.py:72`) and `_to_utc_iso`/`_timezone_name`/`_now_utc_iso`
  (`src/db_writer.py`); `dataclasses.asdict` for `plan`/`visual_grounding`; `len(...)` + row-id
  extraction for retrieved counts/ids. Decoupled via `TYPE_CHECKING` (no runtime `ltm_query`
  import).

**Columns:** `query_log_id` (qlog_<memory_id>), `timestamp_utc/local`, `timezone`,
`user_query`, `intent`, `semantic_query`, `time_range_start/end_utc`,
`location_lat/lon/radius_m`, `used_visual_grounding`, `no_grounding_flag`, `expanded`,
`plan_json`, `visual_grounding_json`, `retrieved_counts_json`, `retrieved_ids_json`,
`frame_paths_json`, `answer`, `error`, `latency_total/plan/grounding/retrieval/answer_ms`,
`vlm_provider`, `vlm_model`, `created_at_utc`, `extra_json`. Indexes on `timestamp_utc`, `intent`.

### `src/config.py`
Add `query_log_enabled: bool` + `query_log_db_path: Path` (mirror `memory_db_path` resolution);
env `LTM_QUERY_LOG_ENABLED` (default true) + `QUERY_LOG_DB_PATH`
(default `outputs/long_term_query_logs.sqlite`).

### `src/ltm_query/cli.py` (single hook point)
- Import `time` + `QueryLogWriter, build_query_log_record`.
- `main()`: create the writer (gated on `config.query_log_enabled`, non-fatal), pass
  `log_writer=` into `run_query`, `close()` in the `finally`.
- `run_query`: add `log_writer` param; wrap body in `try/except/finally`; time each stage with
  `time.perf_counter()`; set `expanded=True` in the sufficiency-expansion block; capture `error`
  and re-raise (preserves `main()`'s error printing); in `finally`, build + write the record
  (its own `try/except`, non-fatal). Backward compatible — `run_query`'s only caller is
  `main()` and the new param defaults to `None`.

### Docs + env
- `.env.example`: `LTM_QUERY_LOG_ENABLED=true`, `QUERY_LOG_DB_PATH=outputs/long_term_query_logs.sqlite`.
- `CLAUDE.md` + `STAGES.md`: note per-query logging to the separate, retrieval-excluded DB.

## Verification
- `uv run python -m src.ltm_query --no-grounding`; ask 2 queries; confirm rows + non-null
  `intent`/`plan_json`/`latency_total_ms` in `outputs/long_term_query_logs.sqlite`.
- `grep -rn long_term_query_logs src/ltm_query/` → no matches (exclusion is structural).
- Memory retrieval + answers unchanged.

## Out of scope
Write-only (no reader CLI). No change to `memory.sqlite`, `memory_db.py`, or `retrieval.py`.
