"""Multi-run comparison report.

Reads eval_runs.sqlite, computes aggregate metrics for each run, and outputs
a standalone HTML file with side-by-side bar charts and a summary table.

Usage (from memory_log/):
    uv run python -m evals.compare <run_id_1> <run_id_2> ... \\
        [--db evals/outputs/eval_runs.sqlite] \\
        [--out evals/outputs/comparison.html] \\
        [--title "My experiment: model comparison"]

Example:
    uv run python -m evals.compare \\
        sb_50_gpt4omini_live_20260608T120000 \\
        sb_50_gpt4o_live_20260608T130000 \\
        sb_50_llava_live_20260608T140000 \\
        --title "StreamingBench: GPT-4o-mini vs GPT-4o vs LLaVA"

To list available run IDs:
    uv run python -m evals.compare --list
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent / "outputs" / "eval_runs.sqlite"

# ---- metric definitions per task ----
_LIVE_METRICS = [
    ("answer_accuracy",       "Answer Accuracy",    "higher"),
    ("judge_avg_score",       "Judge Avg (0-2)",    "higher"),
    ("unanswerable_accuracy", "Unanswerable Acc",   "higher"),
    ("hallucination_rate",    "Hallucination Rate", "lower"),
    ("mean_frame_age_sec",    "Mean Frame Age (s)", "lower"),
    ("p50_latency_ms",        "P50 Latency (ms)",   "lower"),
    ("p95_latency_ms",        "P95 Latency (ms)",   "lower"),
]

_LTM_METRICS = [
    ("answer_accuracy", "Answer Accuracy",      "higher"),
    ("recall_at_1",     "Recall@1",             "higher"),
    ("recall_at_3",     "Recall@3",             "higher"),
    ("recall_at_5",     "Recall@5",             "higher"),
    ("mrr",             "MRR",                  "higher"),
    ("mean_evidence_iou","Evidence IoU",         "higher"),
    ("judge_avg_score", "Judge Avg (0-2)",       "higher"),
    ("p50_latency_ms",  "P50 Latency (ms)",      "lower"),
    ("p95_latency_ms",  "P95 Latency (ms)",      "lower"),
]


def _open_db(db_path: Path) -> sqlite3.Connection:
    from evals.report import open_report_db
    conn = open_report_db(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _list_runs(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT run_id, task, model, manifest_id, n_videos, run_ts FROM eval_runs ORDER BY run_ts DESC"
    ).fetchall()
    if not rows:
        print("No runs in database.")
        return
    print(f"{'RUN_ID':<50} {'TASK':<5} {'MODEL':<20} {'MANIFEST':<30} {'N_VIDEOS'}")
    print("-" * 120)
    for r in rows:
        print(
            f"{r['run_id']:<50} {r['task'] or '':<5} {(r['model'] or ''):<20}"
            f" {(r['manifest_id'] or ''):<30} {r['n_videos'] or ''}"
        )


def _load_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM eval_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    if row is None:
        return None

    run = dict(row)

    # Load summary from DB if present
    if run.get("summary_json"):
        stored = json.loads(run["summary_json"])
        run["_summary"] = stored
    else:
        # Recompute from raw results
        run["_summary"] = _recompute_summary(conn, run_id, run.get("task", "live"))

    return run


def _recompute_summary(conn: sqlite3.Connection, run_id: str, task: str) -> dict:
    """Aggregate metrics directly from eval_results rows."""
    rows = conn.execute(
        "SELECT * FROM eval_results WHERE run_id=? AND task=?", (run_id, task)
    ).fetchall()
    if not rows:
        return {}

    n = len(rows)
    video_ids = set(r["video_id"] for r in rows if r["video_id"])

    if task == "live":
        exact = [r["exact_match"] for r in rows if r["exact_match"] is not None]
        latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
        judge_rows = [r for r in rows if r["judge_score"] is not None]
        return {
            "n_videos": len(video_ids),
            "live": {
                "n": n,
                "answer_accuracy": sum(exact) / len(exact) if exact else None,
                "judge_avg_score": sum(r["judge_score"] for r in judge_rows) / len(judge_rows) if judge_rows else None,
                "p50_latency_ms": latencies[len(latencies) // 2] if latencies else None,
                "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
            },
        }
    else:
        exact = [r["exact_match"] for r in rows if r["exact_match"] is not None]
        r1 = [r["recall_at_1"] for r in rows if r["recall_at_1"] is not None]
        mrr = [r["mrr"] for r in rows if r["mrr"] is not None]
        latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
        judge_rows = [r for r in rows if r["judge_score"] is not None]
        return {
            "n_videos": len(video_ids),
            "ltm": {
                "n": n,
                "answer_accuracy": sum(exact) / len(exact) if exact else None,
                "recall_at_1": sum(r1) / len(r1) if r1 else None,
                "mrr": sum(mrr) / len(mrr) if mrr else None,
                "judge_avg_score": sum(r["judge_score"] for r in judge_rows) / len(judge_rows) if judge_rows else None,
                "p50_latency_ms": latencies[len(latencies) // 2] if latencies else None,
                "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
            },
        }


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 10:
            return f"{v:.3f}"
        return f"{v:.1f}"
    return str(v)


# ---- HTML generation ----

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d26; --border: #2a2d3a;
    --text: #e2e8f0; --muted: #8892a4; --accent: #4f8ef7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: ui-monospace, monospace; padding: 24px; }}
  h1 {{ font-size: 1.25rem; margin-bottom: 4px; color: #fff; }}
  .meta {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 24px; }}
  .section {{ margin-bottom: 40px; }}
  h2 {{ font-size: 0.95rem; color: var(--muted); text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: var(--surface); padding: 8px 12px; text-align: left;
        color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: var(--surface); }}
  .best {{ color: #34d399; font-weight: 600; }}
  .worst {{ color: #f87171; }}
  .charts-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
                  gap: 20px; margin-top: 16px; }}
  .chart-card {{ background: var(--surface); border: 1px solid var(--border);
                 border-radius: 8px; padding: 16px; }}
  .chart-card h3 {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 12px; }}
  canvas {{ max-height: 220px; }}
  .run-chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
  .chip {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 4px; padding: 4px 10px; font-size: 0.78rem; }}
  .chip span {{ display: inline-block; width: 10px; height: 10px;
                border-radius: 2px; margin-right: 6px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Generated {ts} &nbsp;·&nbsp; {n_runs} run(s) &nbsp;·&nbsp; DB: {db_path}</p>

<div class="run-chips">
{run_chips}
</div>

{sections}

<script>
const COLORS = {colors};
const RUNS   = {run_labels};

Chart.defaults.color = '#8892a4';
Chart.defaults.borderColor = '#2a2d3a';

function makeBar(id, label, values, direction) {{
  const ctx = document.getElementById(id);
  if (!ctx) return;
  const finite = values.filter(v => v !== null);
  const best = finite.length ? (direction === 'higher' ? Math.max(...finite) : Math.min(...finite)) : null;
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: RUNS,
      datasets: [{{
        label: label,
        data: values,
        backgroundColor: values.map((v, i) =>
          v === best ? COLORS[i % COLORS.length] + 'ff' : COLORS[i % COLORS.length] + '88'),
        borderColor: values.map((v, i) => COLORS[i % COLORS.length]),
        borderWidth: 1,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: true,
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        label: ctx => ctx.parsed.y !== null ? ctx.parsed.y.toFixed(3) : 'N/A'
      }}}} }},
      scales: {{
        x: {{ ticks: {{ maxRotation: 30 }} }},
        y: {{ beginAtZero: true }}
      }}
    }}
  }});
}}

{chart_calls}
</script>
</body>
</html>
"""

_PALETTE = [
    "#4f8ef7", "#34d399", "#f59e0b", "#f87171",
    "#a78bfa", "#38bdf8", "#fb7185", "#86efac",
]


def _build_html(runs: list[dict], title: str, db_path: str) -> str:
    task = runs[0].get("task", "live")
    metric_defs = _LIVE_METRICS if task == "live" else _LTM_METRICS
    summary_key = "live" if task == "live" else "ltm"

    labels = [r["run_id"] for r in runs]
    short_labels = [r["run_id"][:30] + ("…" if len(r["run_id"]) > 30 else "") for r in runs]

    # Run chips
    chips_html = "\n".join(
        f'<div class="chip"><span style="background:{_PALETTE[i % len(_PALETTE)]}"></span>{r["run_id"]}'
        f'  &nbsp;<span style="color:#8892a4">({r.get("task","?")} · {r.get("model","?")} · '
        f'n={r["_summary"].get("n_videos") or r["_summary"].get(summary_key, {}).get("n","?")})</span></div>'
        for i, r in enumerate(runs)
    )

    # Summary table
    table_rows = ""
    for key, label, direction in metric_defs:
        values = [r["_summary"].get(summary_key, {}).get(key) for r in runs]
        finite = [v for v in values if v is not None]
        if not finite:
            continue
        best = max(finite) if direction == "higher" else min(finite)
        worst = min(finite) if direction == "higher" else max(finite)
        cells = ""
        for v in values:
            if v is None:
                cells += "<td>—</td>"
            elif v == best and len(finite) > 1:
                cells += f'<td class="best">{_fmt(v)}</td>'
            elif v == worst and len(finite) > 1:
                cells += f'<td class="worst">{_fmt(v)}</td>'
            else:
                cells += f"<td>{_fmt(v)}</td>"
        table_rows += f"<tr><td>{label}</td>{cells}</tr>\n"

    header_cells = "".join(f"<th>{lbl[:30]}</th>" for lbl in short_labels)
    table_html = f"""
<div class="section">
  <h2>Metrics summary</h2>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>
"""

    # Charts
    chart_cards = ""
    chart_calls = ""
    for key, label, direction in metric_defs:
        values = [r["_summary"].get(summary_key, {}).get(key) for r in runs]
        if all(v is None for v in values):
            continue
        card_id = f"chart_{key}"
        chart_cards += (
            f'<div class="chart-card"><h3>{label}</h3>'
            f'<canvas id="{card_id}"></canvas></div>\n'
        )
        vals_js = json.dumps([v for v in values])
        chart_calls += f'makeBar("{card_id}", {json.dumps(label)}, {vals_js}, {json.dumps(direction)});\n'

    charts_section = f"""
<div class="section">
  <h2>Charts</h2>
  <div class="charts-grid">
{chart_cards}
  </div>
</div>
"""

    return _HTML_TEMPLATE.format(
        title=title,
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_runs=len(runs),
        db_path=db_path,
        run_chips=chips_html,
        sections=table_html + charts_section,
        colors=json.dumps(_PALETTE),
        run_labels=json.dumps(short_labels),
        chart_calls=chart_calls,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare multiple eval runs")
    parser.add_argument("run_ids", nargs="*", help="Run IDs to compare")
    parser.add_argument("--db", type=Path, default=_DB_PATH, help="Path to eval_runs.sqlite")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "outputs" / "comparison.html",
        help="Output HTML path",
    )
    parser.add_argument("--title", default="Eval Run Comparison", help="Report title")
    parser.add_argument("--list", action="store_true", help="List available run IDs and exit")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"ERROR: database not found: {args.db}", file=sys.stderr)
        print("Run some evals first, then compare.", file=sys.stderr)
        return 1

    conn = _open_db(args.db)

    if args.list:
        _list_runs(conn)
        return 0

    if not args.run_ids:
        print("ERROR: provide run IDs to compare, or use --list to see available runs.", file=sys.stderr)
        return 1

    runs = []
    for run_id in args.run_ids:
        run = _load_run(conn, run_id)
        if run is None:
            print(f"WARNING: run_id not found: {run_id!r}", file=sys.stderr)
            continue
        runs.append(run)

    if not runs:
        print("ERROR: none of the specified run IDs were found.", file=sys.stderr)
        return 1

    # Warn if tasks differ
    tasks = set(r.get("task") for r in runs)
    if len(tasks) > 1:
        print(f"WARNING: mixing tasks {tasks} — charts will show available metrics only.", file=sys.stderr)

    html = _build_html(runs, args.title, str(args.db))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    print(f"Comparison report → {args.out}")
    print(f"Open in browser:   open {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
