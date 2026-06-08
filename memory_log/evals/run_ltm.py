"""Long-Term Memory QA evaluation runner.

Builds an isolated memory DB (seed or replay-ingest), then evaluates
memory_questions from the manifest through the real LTM query pipeline.

Usage (from memory_log/):
    # Seed mode (structured past records from manifest)
    uv run python -m evals.run_ltm \\
        --manifest evals/datasets/toy/desk_001.json \\
        [--memory-mode seed] \\
        [--model gpt-4o-mini] \\
        [--limit 5] \\
        [--no-judge]

    # Replay mode (ingest history_video as passive observations)
    uv run python -m evals.run_ltm \\
        --manifest evals/datasets/toy/desk_001.json \\
        --memory-mode replay \\
        [--caption-history]    # call VLM to caption each observation frame

Each run writes to evals/outputs/eval_runs.sqlite + a JSON report.
The eval's memory.sqlite is isolated in evals/outputs/runs/<run_id>/ and
never touches the production outputs/memory.sqlite.
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
logger = logging.getLogger("evals.run_ltm")

_EVALS_DIR = Path(__file__).parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LTM QA eval runner")
    parser.add_argument("--manifest", required=True, type=Path, help="Path to EvalManifest JSON")
    parser.add_argument(
        "--memory-mode",
        choices=["seed", "replay", "manifest"],
        default=None,
        help="How to build past memory (default: from manifest's memory_mode field)",
    )
    parser.add_argument("--model", default=None, help="Override VLM_MODEL (used for LTM answer generation)")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="LLM judge model")
    parser.add_argument("--caption-history", action="store_true",
                        help="[replay mode] VLM-caption each passive observation frame")
    parser.add_argument("--observe-interval", type=float, default=30.0,
                        help="[replay mode] Passive observation interval in seconds (default: 30)")
    parser.add_argument("--limit", type=int, default=None, help="Max memory questions to evaluate")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge scoring")
    parser.add_argument("--run-id", default=None, help="Unique run ID (auto-generated if omitted)")
    parser.add_argument("--out-dir", type=Path, default=_EVALS_DIR / "outputs")
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    from evals.drivers import build_eval_config, replay_ingest_history, run_ltm_question, seed_memories
    from evals.manifest import load_manifest
    from evals.report import EvalReport, RunMeta
    from evals.scorers import (
        JudgeResult, LtmScore, ExactMatchResult,
        exact_or_alias_match, llm_judge, retrieval_metrics,
    )
    from src.config import PROJECT_ROOT
    from src.memory_db import open_db

    manifest = load_manifest(manifest_path)
    manifest_dir = manifest_path.parent

    memory_mode = args.memory_mode or manifest.memory_mode
    run_id = args.run_id or (
        f"{manifest.video_id}_ltm_{memory_mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    run_dir = args.out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run ID: %s", run_id)
    logger.info(
        "Manifest: %s  memory_mode=%s  (%d memory questions)",
        manifest_path.name, memory_mode, len(manifest.memory_questions),
    )

    # ---- config ----
    extra_env: dict[str, str] = {}
    if args.model:
        extra_env["VLM_MODEL"] = args.model
    config = build_eval_config(run_dir, extra_env)
    logger.info("VLM model: %s", config.vlm_model)
    logger.info("Isolated DB: %s", config.memory_db_path)

    # ---- open isolated DB ----
    conn = open_db(config.memory_db_path)

    # ---- build past memory ----
    if memory_mode == "seed":
        if not manifest.seed_memories:
            logger.warning("seed mode but manifest has no seed_memories — DB will be empty")
        n = seed_memories(conn, manifest, PROJECT_ROOT)
        logger.info("Seeded %d memory records", n)

    elif memory_mode == "replay":
        # Seed the same structured records as seed mode so both modes share identical
        # active_query_memories + promoted_events. This makes the comparison fair:
        # if seed scores high but replay scores low, the passive write path is the bottleneck.
        if manifest.seed_memories:
            n_seeded = seed_memories(conn, manifest, PROJECT_ROOT)
            logger.info("Seeded %d structured memory records (shared with seed mode)", n_seeded)
        else:
            logger.warning("replay mode: manifest has no seed_memories — active_query table will be empty")

        history_path = manifest.history_video_abs(manifest_dir)
        if history_path is None:
            # Fall back to video_path — useful when the same clip contains the full past timeline.
            logger.warning(
                "manifest has no history_video_path; falling back to video_path=%s",
                manifest.video_path,
            )
            history_path = manifest.video_abs(manifest_dir)
        if not history_path.exists():
            print(f"ERROR: history video not found: {history_path}", file=sys.stderr)
            return 1
        logger.info("Replay-ingesting history video: %s", history_path.name)
        n = replay_ingest_history(
            history_path,
            manifest,
            conn,
            config,
            PROJECT_ROOT,
            observe_interval_sec=args.observe_interval,
            caption_with_vlm=args.caption_history,
        )
        logger.info("Ingested %d passive observations (on top of seeded structured memories)", n)

    else:
        logger.warning("Unknown memory_mode=%r — DB will be empty", memory_mode)

    # ---- report ----
    report = EvalReport(
        RunMeta(
            run_id=run_id,
            task="ltm",
            manifest_id=manifest.video_id,
            model=config.vlm_model,
            config_extra={
                "memory_mode": memory_mode,
                "judge": not args.no_judge,
                "observe_interval": args.observe_interval,
            },
        )
    )

    questions = manifest.memory_questions
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        logger.warning("No memory questions in manifest — nothing to evaluate")
        return 0

    logger.info("Evaluating %d memory question(s)…", len(questions))

    # Determine base_epoch for retrieval metric computation
    base_epoch: float
    if manifest.base_timestamp:
        base_epoch = datetime.fromisoformat(manifest.base_timestamp).timestamp()
    else:
        import time
        base_epoch = time.time()

    for q in questions:
        logger.info("  [%s] %s", q.id, q.question[:70])

        result = run_ltm_question(
            question_id=q.id,
            question=q.question,
            conn=conn,
            config=config,
        )
        logger.info(
            "    intent=%s  retrieved=%d events  → %s  (%.0fms)",
            result.plan_intent,
            len(result.retrieval.promoted_events),
            result.system_answer[:80],
            result.latency_ms,
        )

        # ---- exact match ----
        em = exact_or_alias_match(
            result.system_answer,
            q.gold_answer,
            q.acceptable_answers,
            q.unacceptable_answers,
            q.answer_type,
        )

        # ---- retrieval metrics ----
        ret_score = retrieval_metrics(
            result.retrieval,
            q.gold_evidence_windows,
            base_epoch,
        )
        logger.info(
            "    Recall@1=%s  MRR=%.2f  temporal_dist=%.1fs",
            "✓" if ret_score.recall_at_1 else "✗",
            ret_score.mrr,
            ret_score.min_temporal_distance_sec
            if ret_score.min_temporal_distance_sec != float("inf") else -1,
        )

        # ---- LLM judge ----
        judge = JudgeResult(score=0, skipped=True)
        if not args.no_judge and config.openai_api_key:
            gold_evidence_desc = ""
            if q.gold_evidence_windows:
                windows_str = "; ".join(f"{s:.0f}s–{e:.0f}s" for s, e in q.gold_evidence_windows)
                gold_evidence_desc = f"Relevant video window(s): {windows_str}"

            retrieved_texts = []
            for row in result.retrieval.promoted_events[:3]:
                if row["scene_summary"]:
                    retrieved_texts.append(f"[Event] {row['scene_summary']}")
            for row in result.retrieval.active_queries[:3]:
                if row["model_answer"]:
                    retrieved_texts.append(f"[QA] {row['model_answer']}")

            judge = llm_judge(
                question=q.question,
                gold_answer=q.gold_answer,
                gold_evidence=gold_evidence_desc,
                system_answer=result.system_answer,
                retrieved_evidence="\n".join(retrieved_texts) or "(nothing retrieved)",
                openai_api_key=config.openai_api_key,
                model=args.judge_model,
            )

        score = LtmScore(
            exact_match=em,
            judge=judge,
            retrieval=ret_score,
            latency_ms=result.latency_ms,
            plan_intent=result.plan_intent,
            expanded=result.expanded,
        )
        report.add_ltm(
            score=score,
            question_id=q.id,
            question=q.question,
            system_answer=result.system_answer,
            gold_answer=q.gold_answer,
            gold_windows=q.gold_evidence_windows,
            answer_type=q.answer_type,
            retrieval_trace=result.retrieval.trace,
        )

    # ---- summary ----
    report.print_summary()
    json_out = run_dir / f"{run_id}_ltm.json"
    report.save_json(json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
