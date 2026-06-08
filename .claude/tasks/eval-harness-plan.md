# Eval Harness — Implementation Plan & Checklist

End-to-end evaluation for **Live Visual QA** and **Long-Term Memory QA**, driven by manifests.
Lives in `memory_log/evals/` (reuses memory_log's `.venv` + `from src...` imports).

## Design principles
- **End-to-end, not per-unit.** Drive the system through its real entry points so the eval
  survives internal memory/DB refactors. Stage diagnostics (retrieval recall, frame age) come
  from telemetry the system already emits.
- **Mock the stream.** Deterministic, seekable replay of an existing video file. Rule:
  at simulated time `t`, only frames/memories with timestamp ≤ `t` are visible (no leakage).
- **Flexible benchmarks.** Generic adapter ABC; ≥1 runnable public adapter per task.

## Decisions
1. Scope = local MVP **+ public adapters** (≥1 per task, switchable).
2. LTM past memory = **direct DB seeding (default)** + **optional replay-ingestion**.
3. Scoring = **LLM-judge rubric + deterministic exact/alias fast path** + retrieval metrics.
4. Ship a **tiny generated sample** (runs out-of-the-box).

## Verified facts (no core refactor needed)
- `SQLiteWriter.write_active_query_with_event` writes content timestamps from
  `record.timestamp` (db_writer.py:114-116,157-159); only `created_at_utc` is `now()`. →
  synthetic timestamps via a `MemoryRecord` with chosen `timestamp`.
- `write_passive_observation` / `write_daily_summary` take explicit timestamps.
- Live QA entry: `create_vlm_client(config).answer_question(question, frames, frame_items)`.
- LTM entry: `QueryPlanner.plan` → `retrieve_with_expansion` → `build_evidence_pack` →
  `AnswerGenerator.generate` (4 pure calls, mirrors ltm_query/cli.py:260-264).
- `FrameSource` contract: `start/stop/read/get_recent/get_recent_items/release`;
  `FrameItem(timestamp, frame)` from `src.utils`.

## Layout
```
memory_log/evals/
  __init__.py  manifest.py  replay_source.py  drivers.py  scorers.py
  run_live.py  run_ltm.py  report.py  make_sample.py
  adapters/  base.py  streaming_bench.py  egoschema.py  ego4d_nlq.py
  datasets/ (manifests + media; media gitignored)
  outputs/  (eval_runs.sqlite + reports; gitignored)
```
Config isolation: runners set `MEMORY_DB_PATH`/`CHROMA_PATH`/`QUERY_LOG_DB_PATH` (+ model/source
env) before `Config.from_env()` so eval never touches production memory.sqlite.

## Build order / checklist
- [x] 1. `manifest.py` (Pydantic models + loader) + `replay_source.py` (seekable, no-leak) +
      `make_sample.py` (toy mp4 + manifest)
- [x] 2. `drivers.py` live-answer + `scorers.py` (deterministic + live metrics) + `run_live.py`
      + `report.py` → Live QA runs E2E on toy clip
- [x] 3. `drivers.py` seed + LTM query + retrieval metrics + `run_ltm.py` → LTM seed mode E2E
- [x] 4. replay-ingest driver → LTM replay mode; LLM-judge scorer
- [x] 5. `adapters/base.py` + `streaming_bench.py` (Live) + `egoschema.py` (LTM); document
      `ego4d_nlq.py`
- [x] 6. Docs: `memory_log/README.md`, `STAGES.md`, `.gitignore`

## Status: COMPLETE — verified E2E on 2026-06-08

## Verification
From `memory_log/`:
```
uv run python -m evals.make_sample
uv run python -m evals.run_live --manifest evals/datasets/toy/desk_001.json --limit 5
uv run python -m evals.run_ltm  --manifest evals/datasets/toy/desk_001.json --memory-mode seed
uv run python -m evals.run_ltm  --manifest evals/datasets/toy/desk_001.json --memory-mode replay
```
Check eval_runs.sqlite rows; confirm production memory.sqlite untouched; spot-check judge rationales.
