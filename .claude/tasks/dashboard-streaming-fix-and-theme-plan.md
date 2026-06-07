# Plan: Fix VLM streaming bug + redesign dashboard colors with theme switcher

## Goal

Two issues to resolve:
1. **Streaming bug** — the final VLM answer never streamed in either Live QA or Long-term Memory because both OpenAI paths called `stream.text_stream` which does not exist in openai 2.41.0.
2. **Color redesign** — remove the purple/violet "purpose-based" palette; replace with CSS variables that support a Blue (default) / Teal / Slate switcher in the header. Preserved: green for connected/ready, orange/yellow for alert, 📹/🖼 image icons.

## Root cause (streaming)

`client.responses.stream(...)` returns a `ResponseStream`. In openai 2.41.0 it has only
`close`, `get_final_response`, `until_done` — **no `.text_stream`**. The correct approach
is to iterate the stream object directly and filter for
`event.type == "response.output_text.delta"`, yielding `event.delta`.

## Changes

### 1. `memory_log/src/vlm_client.py`
Replace in `OpenAIVLMClient.answer_question_stream`:
```python
# before (broken)
for text in stream.text_stream:
    yield text

# after (correct)
for event in stream:
    if event.type == "response.output_text.delta":
        yield event.delta
```

### 2. `memory_log/src/ltm_query/answer_generator.py`
Same fix in `_stream_openai`:
```python
# before (broken)
for text in stream.text_stream:
    yield text

# after (correct)
for event in stream:
    if event.type == "response.output_text.delta":
        yield event.delta
```

### 3. `memory_log/src/dashboard/static/index.html`
- Refactor all hard-coded hex colors into CSS custom properties (`--accent`, `--bg`,
  `--surface`, `--text`, `--user-bg`, etc.).
- Add three theme definitions via `[data-theme="blue"]` / `[data-theme="teal"]` /
  `[data-theme="slate"]` on `<html>`.
- Add `<select id="themeSelect">` in the header; on change update `data-theme` attribute
  and persist to `localStorage`.
- Status colors (`--ok #22c55e`, `--ready #4ade80`, `--alert #f59e0b`, `--error #f87171`)
  are declared once in `:root` and never overridden — they stay consistent across themes.
- Replace two inline `color:#444` strings with `color:var(--text-faint)`.
- Ollama paths and live-dot hardcoded status greens/oranges left unchanged.

## Verification

1. `cd memory_log && uv run python -m src.dashboard`
2. **Live QA** — ask "What do you see?" → tokens stream into the assistant bubble.
3. **LTM** — ask "Where was I yesterday?" → Planner / Retrieval panels fill, then answer streams.
4. **Theme switch** — dropdown Blue → Teal → Slate updates accent + buttons + user bubble instantly;
   green/orange/red statuses unchanged; reload restores last selection.
5. Existing REPLs (`python -m src.main`, `python -m src.ltm_query`) unaffected — only
   `answer_question_stream` and `_stream_openai` changed; non-streaming paths untouched.

## Status

✅ Implemented (all three files changed).
