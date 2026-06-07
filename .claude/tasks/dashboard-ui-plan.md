# Dashboard UI Plan

## Goal
Add a browser-based dashboard (`src/dashboard/`) to `memory_log` that replaces the two
terminal REPLs with a visual, conversational interface:

1. **Live QA panel** — live MJPEG camera frame + streaming chat (question → streaming VLM answer).
2. **Long-term Memory panel** — query input with three progressive stages:
   - 📋 Query Planner output (intent, time range, semantic query, stores)
   - 🗂 Retrieval results (per-store method/count trace + evidence/uncertainty notes)
   - 💬 Streaming answer

## Stack decision
**stdlib `http.server` — zero new dependencies.**

Reason: all underlying work is sync/thread-based (OpenCV, OpenAI SDK, SQLite).
FastAPI's async model would need full threadpool bridging for zero throughput gain.
`ThreadingHTTPServer` naturally handles two long-lived connections (MJPEG + SSE) on
parallel daemon threads. Pattern mirrors `src/location_server.py` exactly.

Token streaming via SSE over POST (browser uses `fetch()` + `getReader()`).

## Files created
```
memory_log/src/dashboard/
  __init__.py     empty package marker
  __main__.py     entry point: Config.from_env() → DashboardService + DashboardServer → block
  service.py      DashboardService — shared objects + streaming generators
  server.py       DashboardServer — ThreadingHTTPServer + request routing
  sse.py          SSE/MJPEG framing helpers (pure functions, no deps)
  static/
    index.html    single-page UI (vanilla JS, inline CSS, no CDN)
```

## Files modified (additive only)
| File | Change |
|---|---|
| `providers/ollama.py` | Add `chat_stream()` function (NDJSON line-iterator) |
| `memory_log/src/vlm_client.py` | Add abstract `answer_question_stream()` + both impls |
| `memory_log/src/ltm_query/answer_generator.py` | Add `generate_stream()` |
| `memory_log/src/ltm_query/retrieval.py` | Add `retrieve_with_expansion()` module function |
| `memory_log/src/ltm_query/cli.py` | Replace inline expansion block with `retrieve_with_expansion()` |
| `memory_log/src/config.py` | Add `dashboard_host`, `dashboard_port`, `dashboard_cert`, `dashboard_key` |
| `memory_log/.env.example` | Document the 4 new `DASHBOARD_*` vars |

## Run command
```bash
cd memory_log
uv run python -m src.dashboard
# → http://127.0.0.1:8800/
```

## SSE event protocol
### `/api/qa/stream` (POST `{question}`)
| event | payload |
|---|---|
| `token` | `{text: "..."}` |
| `done` | `{memory_id, location, latency_s, frame_count}` |
| `error` | `{message}` |

### `/api/ltm/stream` (POST `{query}`)
| event | payload |
|---|---|
| `plan` | `{intent, time_range, location_filter, semantic_query, stores[], needs_grounding}` |
| `grounding` | `{scene, objects[]}` (optional) |
| `retrieval` | `{stores[{store,method,candidate_count,final,note}], evidence_reasons[], uncertainty_notes[]}` |
| `token` | `{text: "..."}` |
| `done` | `{latency_s}` |
| `error` | `{message}` |

## Verification steps
1. `uv run python -m src.dashboard` → open http://127.0.0.1:8800/
2. Left panel shows MJPEG live frame within seconds
3. Live QA: ask "What do you see?" → tokens stream; `tail -1 outputs/memories.jsonl | jq .` shows new row
4. LTM: ask "Where was I yesterday?" → Planner fills, Retrieval fills, Answer streams
5. `sqlite3 outputs/long_term_query_logs.sqlite "SELECT user_query,intent,latency_total_ms FROM long_term_query_logs ORDER BY timestamp_utc DESC LIMIT 1;"`
6. Ctrl-C → clean shutdown, no traceback
7. Existing REPLs (`src.main`, `src.ltm_query`) still run unchanged
