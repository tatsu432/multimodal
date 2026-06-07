# Plan: Unified runner — live QA + passive memory in one process (`src/run_all.py`)

## Context

`memory_log` has two long-running entry points that each spin up their own copy of the same
resources:

- `src/main.py` — live Q&A REPL → active memory (promoted_events + active_query_memories + frames) + JSONL
- `src/passive_observer.py` — timer loop → passive_observations every `PASSIVE_OBSERVATION_INTERVAL_SEC` (no VLM)

They **cannot run at the same time as two OS processes** because each calls
`create_frame_source(config)` + `source.start()`, starts a `LocationServer`, and opens the
memory DB. The second process collides on three shared resources:

1. **Camera device / stream** — two `FrameSource` objects open the same webcam/RTSP/phone URL.
2. **Location server port** — both bind `LOCATION_SERVER_PORT` (default `8765`); the second
   `ThreadingHTTPServer((host, port))` raises `OSError: Address already in use`
   (`location_server.py:174`, no `SO_REUSEPORT`).
3. **SQLite writes** — two connections serialize on the WAL writer lock.

For the wearable use case we want a single "power-on" switch that runs **both** concurrently.
The building blocks are already concurrency-safe within one process:

- `_FrameRingBuffer` is fully `threading.Lock`-guarded; two reader threads can call
  `get_recent()` / `get_recent_items()` on the **same** `CameraFrameSource` safely
  (`capture/camera_frame_source.py:37-65`).
- `LocationServer.start()` is idempotent within a process; `LocationSidecarStore` is lock-guarded.
- `GeocodeCache` shares one `sqlite3` connection (`check_same_thread=False`) under a lock.
- `open_db()` **already** opens with `check_same_thread=False` + WAL (`memory_db.py:208-211`),
  so one connection can be shared across threads — provided writes are serialized.

**Decision (confirmed with user):** add a new `src/run_all.py` module — the combined entry
point (the wearable "power-on" switch) — that hoists every shared resource into a single
instance, runs the **passive observer on a background daemon thread**, and runs the
**live-QA REPL on the foreground thread**. `main.py` and `passive_observer.py` remain working
standalone modules.

---

## Architecture

```
                 uv run python -m src.run_all      (the wearable "on" switch)
                              │
        ┌─────────────────────┴──────────────────────┐
   build ONE of each:  frame source · location server+sidecar · geocode client · DB conn+SQLiteWriter
        │                                             │
  foreground thread                            background daemon thread
  run_repl()  (src.main)                       PassiveObserver.run()  (src.passive_observer)
  → VLM answer + JSONL                          → passive_observations every interval
  → promoted_events + active_query             (no VLM)
        │                                             │
        └──────────────► shared SQLiteWriter (threading.Lock) ◄────────┘
                         shared camera ring buffer (already lock-safe)
```

Only `run_all`'s `main()` owns the lifecycle (`source.start()` / `release()`, server stop,
conn close). Both loops are pure consumers of the shared instances.

---

## New file: `src/run_all.py`

Reuses existing code — does **not** re-implement the REPL or the observer.

Imports: `from src.main import run_repl, RunStats, print_run_summary`;
`from src.passive_observer import PassiveObserver`; plus the same shared-resource imports
`main.py` uses (`create_frame_source`, `create_vlm_client`, `MemoryWriter`,
`LocationSidecarStore`, `LocationServer`, `GeocodeClient`, `open_db`, `SQLiteWriter`,
`add_source_args`, `configure_decode_logging`, `ensure_model`, `OllamaError`).

`main()`:
1. `load_dotenv(PROJECT_ROOT / ".env")`, `configure_decode_logging()`.
2. argparse: `add_source_args(parser)` + `--no-passive`.
3. `config = Config.from_env()`, apply `--camera`/`--url`, `config.validate()` (exit 1 on `ValueError`).
4. ollama → `ensure_model(...)` (exit 1 on `OllamaError`).
5. Shared location server + sidecar (gated on `config.location_server_enabled`), `start()`.
6. Shared geocode client (gated on `config.geocode_enabled`).
7. `source = create_frame_source(config)`.
8. Shared DB (non-fatal like `main.py`): `open_db(config.memory_db_path)` → `SQLiteWriter(conn, PROJECT_ROOT)`;
   on failure warn → `db_writer = None` (JSONL-only; passive skipped).
9. `vlm = create_vlm_client(config)`, `writer = MemoryWriter(config)`.
10. `source.start()`.
11. If `not args.no_passive` and `db_writer is not None`: `stop_event = threading.Event()`,
    `observer = PassiveObserver(config, source, db_writer, sidecar, geocode_client, stop_event=stop_event, quiet=True)`,
    daemon `threading.Thread(target=observer.run, name="passive-observer").start()`.
12. `run_repl(config, source, vlm, writer, sidecar, geocode_client, db_writer=db_writer)` in try/except KeyboardInterrupt.
13. `finally` (ordered): `stop_event.set()` → `passive_thread.join(timeout=interval+2)` →
    `source.stop()` → `source.release()` → `location_server.stop()` → `geocode_client.close()` →
    `conn.close()` → `print_run_summary(stats)`.
14. `if __name__ == "__main__": main()`.

Run: `uv run python -m src.run_all`.

---

## Modified files

### `src/passive_observer.py` — embeddable loop (backward compatible)
- `__init__`: add `stop_event: threading.Event | None = None`, `quiet: bool = False`;
  store `self._stop = stop_event or threading.Event()`, `self._quiet = quiet`.
- `run()`: `while True:` → `while not self._stop.is_set():`; `time.sleep(0.5)` → `self._stop.wait(0.5)`.
- Banner + per-tick `print(...)`: when `self._quiet`, route to `logger.info`/`logger.debug`.
- Standalone `main()` unchanged (new kwargs default to today's behavior).

### `src/db_writer.py` — serialize writes on the shared connection
- `import threading`; `SQLiteWriter.__init__` adds `self._lock = threading.Lock()`.
- Wrap `write_active_query_with_event`, `write_passive_observation`, `write_daily_summary` bodies in `with self._lock:`.

### `src/memory_db.py`
- Add `conn.execute("PRAGMA busy_timeout=5000")` in `open_db` after WAL.

### Docs
- `STAGES.md`, `memory_log/README.md`, `CLAUDE.md`: document `uv run python -m src.run_all` + `--no-passive`.

---

## Verification
- Regression: `uv run python -m src.main` and `uv run python -m src.passive_observer` still work standalone.
- Combined: `uv run python -m src.run_all` → ask a question, wait ~60s, Ctrl+C clean shutdown.
- Row counts: `passive_observations` grows on the interval; `promoted_events`/`active_query_memories` grow per question — same run, no lock errors.
- `uv run python -m src.run_all --no-passive` behaves like `src.main`.
