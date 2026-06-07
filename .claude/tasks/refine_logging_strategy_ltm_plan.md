# Task: Refine LTM query logging — full per-run traceability

## Problem

LTM query failures are opaque. When a query returns empty results, the telemetry row in
`long_term_query_logs.sqlite` does not reveal:

1. Whether vector search selected candidates that were subsequently dropped by SQL filters
2. The actual SQL and params issued to each store
3. What evidence the answer generator received as input
4. The planner's raw LLM response (only the parsed plan JSON is stored)

## Implementation order

### Step 1 — `src/ltm_query/retrieval.py`
- Add `@dataclass StoreTrace` with: `store`, `method` ("vector"|"like"|"metadata"),
  `candidate_count: int | None`, `sql: str`, `params: list`, `final_count: int`,
  `note: str | None`
- Add `trace: list[StoreTrace]` to `RetrievalResults`
- In each `_query_*` method, record one `StoreTrace` and append to `results.trace`
- Detect silent vector→0 drop and set warning `note`
- Upgrade `logger.debug` → `logger.info` with method + candidate→final info

### Step 2 — `src/ltm_query/query_planner.py`
- Add `self.last_raw_response: str | None = None` instance var (reset each `plan()` call)
- Store `raw` before parsing so it's available even on fallback
- Upgrade `logger.debug` → `logger.info` for intent/stores

### Step 3 — `src/ltm_query/answer_generator.py`
- Rename `_format_evidence` → `format_evidence` (one internal caller to update)
- Add `logger.info` for prompt + answer lengths

### Step 4 — `src/query_log.py`
- Add 3 new columns: `planner_raw_response TEXT`, `retrieval_trace_json TEXT`, `answer_prompt TEXT`
- Add `_ensure_columns(conn)` for zero-downtime migration of existing DB
- Call `_ensure_columns` in `QueryLogWriter.__init__`
- Update `QueryLogRecord` dataclass, INSERT statement
- Update `build_query_log_record()`: new params, fill `retrieval_trace_json`, fill `extra_json`

### Step 5 — `src/ltm_query/cli.py`
- In `run_query()`: capture `answer_prompt` from `format_evidence(evidence)` and
  `planner_raw` from `planner.last_raw_response`
- Pass both into `build_query_log_record()`
- Expansion path: merge `expanded_results.trace` into `results.trace`

### Step 6 — Docs
- `memory_log/README.md`: document the enriched telemetry columns
- `STAGES.md`: note telemetry now captures full per-stage I/O

## Verification

```bash
cd memory_log
uv run python -m src.ltm_query --no-grounding
# ask a query, then:
sqlite3 outputs/long_term_query_logs.sqlite \
  "SELECT retrieval_trace_json FROM long_term_query_logs ORDER BY timestamp_utc DESC LIMIT 1;" \
  | jq .
# expect: per-store {store, method, candidate_count, sql, final_count, note}

sqlite3 outputs/long_term_query_logs.sqlite \
  "SELECT length(answer_prompt), length(planner_raw_response) FROM long_term_query_logs ORDER BY timestamp_utc DESC LIMIT 1;"
# expect: non-zero lengths
```
