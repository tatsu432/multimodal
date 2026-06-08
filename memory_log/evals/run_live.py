"""Live Visual QA evaluation runner.

Replays a video file, asks each live_question from the manifest at the correct
media timestamp, scores the system answer, and prints a summary report.

Usage (from memory_log/):
    uv run python -m evals.run_live \\
        --manifest evals/datasets/toy/desk_001.json \\
        [--model gpt-4o-mini] \\
        [--num-frames 4] \\
        [--window-sec 30] \\
        [--limit 5] \\
        [--no-judge] \\
        [--run-id my_run_001]

Each run is written to evals/outputs/eval_runs.sqlite; a JSON summary is
also written to evals/outputs/<run_id>_live.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evals.run_live")

_EVALS_DIR = Path(__file__).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live VQA eval runner")
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to EvalManifest JSON"
    )
    parser.add_argument("--model", default=None, help="Override VLM_MODEL env var")
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="LLM judge model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames per query (default: from .env)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=30.0,
        help="Lookback window for frame selection",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Replay source sampling interval (s)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max questions to evaluate"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge (only exact-match scoring)",
    )
    parser.add_argument(
        "--run-id", default=None, help="Unique run identifier (default: auto-generated)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=_EVALS_DIR / "outputs", help="Output directory"
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    # ---- imports (after path setup) ----
    from evals.drivers import LiveAnswerResult, build_eval_config, run_live_question
    from evals.manifest import load_manifest
    from evals.replay_source import ReplaySource
    from evals.report import EvalReport, RunMeta
    from evals.scorers import JudgeResult, LiveScore, exact_or_alias_match, llm_judge

    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    run_id = (
        args.run_id
        or f"{manifest.video_id}_live_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    run_dir = args.out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run ID: %s", run_id)
    logger.info(
        "Manifest: %s  (%d live questions)",
        manifest_path.name,
        len(manifest.live_questions),
    )

    # ---- config ----
    extra_env: dict[str, str] = {}
    if args.model:
        extra_env["VLM_MODEL"] = args.model
    config = build_eval_config(run_dir, extra_env)
    logger.info("VLM model: %s", config.vlm_model)

    # ---- replay source ----
    video_path = manifest.video_abs(manifest_dir)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 1

    replay = ReplaySource(
        video_path,
        sample_interval_sec=args.sample_interval,
        base_timestamp=manifest.base_timestamp,
    )
    logger.info("Loading replay source: %s", video_path.name)
    replay.load()
    logger.info(
        "Duration: %.1fs  sampled frames: %d", replay.duration_sec(), len(replay._index)
    )

    # ---- report ----
    report = EvalReport(
        RunMeta(
            run_id=run_id,
            task="live",
            manifest_id=manifest.video_id,
            model=config.vlm_model,
            config_extra={
                "num_frames": args.num_frames or config.num_frames_per_query,
                "window_sec": args.window_sec,
                "sample_interval": args.sample_interval,
                "judge": not args.no_judge,
            },
        )
    )

    questions = manifest.live_questions
    if args.limit:
        questions = questions[: args.limit]

    logger.info("Evaluating %d live question(s)…", len(questions))

    for q in questions:
        logger.info("  t=%.1fs  %s", q.ask_at_sec, q.question[:60])

        # ---- call the real pipeline ----
        result: LiveAnswerResult = run_live_question(
            question_id=q.id,
            question=q.question,
            ask_at_sec=q.ask_at_sec,
            replay=replay,
            config=config,
            num_frames=args.num_frames,
            window_sec=args.window_sec,
            choices=q.choices if q.answer_type == "mcq" else None,
        )
        logger.info("    → %s  (%.0fms)", result.system_answer[:80], result.latency_ms)

        # ---- exact match ----
        em = exact_or_alias_match(
            result.system_answer,
            q.gold_answer,
            q.acceptable_answers,
            q.unacceptable_answers,
            q.answer_type,
        )

        # ---- LLM judge ----
        judge = JudgeResult(score=0, skipped=True)
        if not args.no_judge and config.openai_api_key:
            gold_evidence = ""
            if q.gold_evidence_window:
                s, e = q.gold_evidence_window
                gold_evidence = (
                    f"The event occurred between {s:.1f}s and {e:.1f}s in the video."
                )
            judge = llm_judge(
                question=q.question,
                gold_answer=q.gold_answer,
                gold_evidence=gold_evidence,
                system_answer=result.system_answer,
                retrieved_evidence=f"{result.frames_used} frame(s) at t={q.ask_at_sec:.1f}s",
                openai_api_key=config.openai_api_key,
                model=args.judge_model,
            )

        score = LiveScore(
            exact_match=em,
            judge=judge,
            frame_age_sec=result.frame_age_sec,
            frames_used=result.frames_used,
            latency_ms=result.latency_ms,
        )
        report.add_live(
            score=score,
            question_id=q.id,
            question=q.question,
            system_answer=result.system_answer,
            ask_at_sec=q.ask_at_sec,
            gold_answer=q.gold_answer,
            answer_type=q.answer_type,
        )

    # ---- summary ----
    report.print_summary()
    json_out = run_dir / f"{run_id}_live.json"
    report.save_json(json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
