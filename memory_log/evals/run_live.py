"""Live Visual QA evaluation runner.

Single-manifest mode (one video):
    uv run python -m evals.run_live \\
        --manifest evals/datasets/toy/desk_001.json \\
        [--model gpt-4o-mini] [--limit 5] [--no-judge] [--run-id my_run]

Batch mode (N videos — aggregates metrics across all manifests):
    uv run python -m evals.run_live \\
        --manifest-dir evals/datasets/streaming_bench/ \\
        --n-videos 50 \\
        [--model gpt-4o-mini] [--limit 10] [--no-judge] [--run-id sb_50_gpt4omini]

Results go to evals/outputs/eval_runs.sqlite (one run_id, all questions).
Compare runs with:
    uv run python -m evals.compare <run_id_1> <run_id_2> ...
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


def _collect_manifests(manifest_dir: Path, n_videos: int | None) -> list[Path]:
    paths = sorted(manifest_dir.glob("*.json"))
    if n_videos:
        paths = paths[:n_videos]
    return paths


def _eval_one_manifest(
    manifest_path: Path,
    report,
    config,
    args,
    imports: dict,
) -> int:
    """Evaluate all questions in one manifest. Returns number of questions evaluated."""
    load_manifest = imports["load_manifest"]
    ReplaySource = imports["ReplaySource"]
    run_live_question = imports["run_live_question"]
    exact_or_alias_match = imports["exact_or_alias_match"]
    JudgeResult = imports["JudgeResult"]
    LiveScore = imports["LiveScore"]
    llm_judge = imports["llm_judge"]

    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    if not manifest.live_questions:
        logger.info("  No live_questions in %s — skipped", manifest_path.name)
        return 0

    video_path = manifest.video_abs(manifest_dir)
    if not video_path.exists():
        logger.warning("  Video not found: %s — skipped", video_path)
        return 0

    replay = imports["ReplaySource"](
        video_path,
        sample_interval_sec=args.sample_interval,
        base_timestamp=manifest.base_timestamp,
    )
    replay.load()
    logger.info(
        "  %s  %.1fs  %d frames  %d questions",
        manifest_path.name, replay.duration_sec(), len(replay._index),
        len(manifest.live_questions),
    )

    questions = manifest.live_questions
    if args.limit:
        questions = questions[: args.limit]

    for q in questions:
        result = run_live_question(
            question_id=q.id,
            question=q.question,
            ask_at_sec=q.ask_at_sec,
            replay=replay,
            config=config,
            num_frames=args.num_frames,
            window_sec=args.window_sec,
            choices=q.choices if q.answer_type == "mcq" else None,
        )
        logger.info("    t=%.1fs  → %s  (%.0fms)", q.ask_at_sec, result.system_answer[:70], result.latency_ms)

        em = exact_or_alias_match(
            result.system_answer,
            q.gold_answer,
            q.acceptable_answers,
            q.unacceptable_answers,
            q.answer_type,
        )

        judge = JudgeResult(score=0, skipped=True)
        if not args.no_judge and config.openai_api_key:
            gold_evidence = ""
            if q.gold_evidence_window:
                s, e = q.gold_evidence_window
                gold_evidence = f"The event occurred between {s:.1f}s and {e:.1f}s in the video."
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
            video_id=manifest.video_id,
        )

    replay.release()
    return len(questions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live VQA eval runner")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="Single manifest JSON path")
    src.add_argument("--manifest-dir", type=Path, help="Directory of manifest JSONs (batch mode)")

    parser.add_argument(
        "--n-videos", type=int, default=None,
        help="Max manifests to process in batch mode (default: all in directory)",
    )
    parser.add_argument("--model", default=None, help="Override VLM_MODEL env var")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="LLM judge model")
    parser.add_argument("--num-frames", type=int, default=None, help="Frames per query")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Frame lookback window (s)")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Replay sampling interval (s)")
    parser.add_argument("--limit", type=int, default=None, help="Max questions per manifest")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    parser.add_argument("--out-dir", type=Path, default=_EVALS_DIR / "outputs", help="Output directory")
    args = parser.parse_args(argv)

    # ---- collect manifests ----
    if args.manifest:
        manifest_paths = [args.manifest.resolve()]
        batch_label = manifest_paths[0].stem
    else:
        manifest_paths = _collect_manifests(args.manifest_dir.resolve(), args.n_videos)
        if not manifest_paths:
            print(f"ERROR: no manifests found in {args.manifest_dir}", file=sys.stderr)
            return 1
        batch_label = args.manifest_dir.name
        logger.info("Batch mode: %d manifests from %s", len(manifest_paths), args.manifest_dir.name)

    # ---- imports ----
    from evals.drivers import build_eval_config, run_live_question
    from evals.manifest import load_manifest
    from evals.replay_source import ReplaySource
    from evals.report import EvalReport, RunMeta
    from evals.scorers import JudgeResult, LiveScore, exact_or_alias_match, llm_judge

    imports = dict(
        load_manifest=load_manifest,
        ReplaySource=ReplaySource,
        run_live_question=run_live_question,
        exact_or_alias_match=exact_or_alias_match,
        JudgeResult=JudgeResult,
        LiveScore=LiveScore,
        llm_judge=llm_judge,
    )

    # ---- run config (shared across all manifests) ----
    run_id = args.run_id or f"{batch_label}_live_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    run_dir = args.out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    extra_env: dict[str, str] = {}
    if args.model:
        extra_env["VLM_MODEL"] = args.model
    config = build_eval_config(run_dir, extra_env)

    logger.info("Run ID : %s", run_id)
    logger.info("Model  : %s", config.vlm_model)
    logger.info("Videos : %d", len(manifest_paths))

    report = EvalReport(
        RunMeta(
            run_id=run_id,
            task="live",
            manifest_id=batch_label,
            model=config.vlm_model,
            config_extra={
                "n_videos": len(manifest_paths),
                "num_frames": args.num_frames or config.num_frames_per_query,
                "window_sec": args.window_sec,
                "sample_interval": args.sample_interval,
                "judge": not args.no_judge,
                "limit_per_video": args.limit,
            },
        ),
        db_path=args.out_dir / "eval_runs.sqlite",
    )

    # ---- evaluate all manifests ----
    total_q = 0
    for i, mp in enumerate(manifest_paths, 1):
        logger.info("[%d/%d] %s", i, len(manifest_paths), mp.name)
        total_q += _eval_one_manifest(mp, report, config, args, imports)

    logger.info("Total questions evaluated: %d", total_q)

    # ---- summary ----
    report.print_summary()
    json_out = run_dir / f"{run_id}_live.json"
    report.save_json(json_out)
    print(f"\nJSON report → {json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
