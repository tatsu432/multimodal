"""Eval run persistence and reporting.

Appends results to evals/outputs/eval_runs.sqlite (one row per question).
Prints a dashboard-style summary to stdout.
Optionally writes a JSON report file.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from evals.scorers import (
    JudgeResult,
    LtmScore,
    LiveScore,
    RetrievalScore,
    aggregate_live_scores,
    aggregate_ltm_scores,
)

logger = logging.getLogger("evals.report")

_DB_PATH = Path(__file__).parent / "outputs" / "eval_runs.sqlite"

_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id      TEXT PRIMARY KEY,
    run_ts      TEXT NOT NULL,
    task        TEXT NOT NULL,
    manifest_id TEXT,
    model       TEXT,
    config_json TEXT,
    summary_json TEXT,
    n_videos    INTEGER,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    result_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES eval_runs(run_id),
    task        TEXT NOT NULL,
    video_id    TEXT,
    question_id TEXT NOT NULL,
    question    TEXT,
    gold_answer TEXT,
    system_answer TEXT,
    answer_type TEXT,
    exact_match INTEGER,
    is_abstention INTEGER,
    judge_score INTEGER,
    judge_halluc_object  INTEGER,
    judge_halluc_time    INTEGER,
    judge_halluc_location INTEGER,
    judge_should_abstain  INTEGER,
    judge_rationale TEXT,
    -- live-specific
    ask_at_sec         REAL,
    frame_age_sec      REAL,
    frames_used        INTEGER,
    -- ltm-specific
    query_time_sec     REAL,
    plan_intent        TEXT,
    retrieval_expanded INTEGER,
    recall_at_1        INTEGER,
    recall_at_3        INTEGER,
    recall_at_5        INTEGER,
    mrr                REAL,
    min_temporal_dist_sec REAL,
    evidence_iou       REAL,
    -- shared
    latency_ms         REAL,
    retrieval_trace_json TEXT,
    created_at         TEXT NOT NULL
);
"""

_MIGRATE_DDL = [
    "ALTER TABLE eval_runs ADD COLUMN n_videos INTEGER",
    "ALTER TABLE eval_results ADD COLUMN video_id TEXT",
]


def open_report_db(db_path: Path = _DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_DDL)
    # Apply non-destructive migrations for existing DBs (ignore if column already exists)
    for stmt in _MIGRATE_DDL:
        try:
            conn.execute(stmt)
            conn.commit()
        except sqlite3.OperationalError:
            pass
    return conn


@dataclass
class RunMeta:
    run_id: str
    task: str           # "live" | "ltm"
    manifest_id: str
    model: str
    config_extra: dict  # any other config info for the report


class EvalReport:
    """Collects per-item results, persists them, and prints a summary."""

    def __init__(self, meta: RunMeta, db_path: Path = _DB_PATH) -> None:
        self.meta = meta
        self._conn = open_report_db(db_path)
        self._live_scores: list[LiveScore] = []
        self._ltm_scores: list[LtmScore] = []
        self._ltm_question_meta: list[dict] = []  # gold answers for printing
        self._live_question_meta: list[dict] = []
        self._video_ids: set[str] = set()

        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._conn.execute(
            "INSERT OR IGNORE INTO eval_runs"
            " (run_id, run_ts, task, manifest_id, model, config_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                meta.run_id, now, meta.task, meta.manifest_id, meta.model,
                json.dumps(meta.config_extra), now,
            ),
        )
        self._conn.commit()

    def add_live(
        self,
        score: LiveScore,
        question_id: str,
        question: str,
        system_answer: str,
        ask_at_sec: float,
        gold_answer: str,
        answer_type: str,
        video_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        em = score.exact_match
        jd = score.judge
        if video_id:
            self._video_ids.add(video_id)
        self._conn.execute(
            """INSERT INTO eval_results (
                run_id, task, video_id, question_id, question, gold_answer, system_answer,
                answer_type, exact_match, is_abstention,
                judge_score, judge_halluc_object, judge_halluc_time, judge_halluc_location,
                judge_should_abstain, judge_rationale,
                ask_at_sec, frame_age_sec, frames_used, latency_ms, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.meta.run_id, "live", video_id, question_id,
                question, gold_answer, system_answer,
                answer_type, int(em.matched), int(em.is_abstention),
                jd.score if not jd.skipped else None,
                int(jd.hallucinated_object), int(jd.hallucinated_time),
                int(jd.hallucinated_location), int(jd.should_have_abstained),
                jd.rationale,
                ask_at_sec, score.frame_age_sec, score.frames_used, score.latency_ms, now,
            ),
        )
        self._conn.commit()
        self._live_scores.append(score)
        self._live_question_meta.append({"question": question, "gold": gold_answer, "ask_at": ask_at_sec})

    def add_ltm(
        self,
        score: LtmScore,
        question_id: str,
        question: str,
        system_answer: str,
        gold_answer: str,
        gold_windows: list[tuple[float, float]],
        answer_type: str,
        retrieval_trace: list | None = None,
        video_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        em = score.exact_match
        jd = score.judge
        ret = score.retrieval
        if video_id:
            self._video_ids.add(video_id)
        self._conn.execute(
            """INSERT INTO eval_results (
                run_id, task, video_id, question_id, question, gold_answer, system_answer,
                answer_type, exact_match, is_abstention,
                judge_score, judge_halluc_object, judge_halluc_time, judge_halluc_location,
                judge_should_abstain, judge_rationale,
                plan_intent, retrieval_expanded,
                recall_at_1, recall_at_3, recall_at_5, mrr,
                min_temporal_dist_sec, evidence_iou,
                latency_ms, retrieval_trace_json, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.meta.run_id, "ltm", video_id, question_id,
                question, gold_answer, system_answer,
                answer_type, int(em.matched), int(em.is_abstention),
                jd.score if not jd.skipped else None,
                int(jd.hallucinated_object), int(jd.hallucinated_time),
                int(jd.hallucinated_location), int(jd.should_have_abstained),
                jd.rationale,
                score.plan_intent, int(score.expanded),
                int(ret.recall_at_1), int(ret.recall_at_3), int(ret.recall_at_5), ret.mrr,
                ret.min_temporal_distance_sec if ret.min_temporal_distance_sec != float("inf") else None,
                ret.evidence_iou,
                score.latency_ms,
                json.dumps([vars(t) if hasattr(t, "__dict__") else str(t) for t in (retrieval_trace or [])]),
                now,
            ),
        )
        self._conn.commit()
        self._ltm_scores.append(score)
        self._ltm_question_meta.append({"question": question, "gold": gold_answer, "windows": gold_windows})

    def print_summary(self) -> None:
        meta = self.meta
        print(f"\n{'='*60}")
        print(f"Eval run : {meta.run_id}")
        print(f"Task     : {meta.task}")
        print(f"Manifest : {meta.manifest_id}")
        print(f"Model    : {meta.model}")
        print(f"{'='*60}")

        if self._live_scores:
            agg = aggregate_live_scores(self._live_scores)
            print("\nLive QA")
            print(f"  n                    : {agg['n']}")
            print(f"  answer_accuracy      : {agg['answer_accuracy']:.1%}")
            if agg["unanswerable_accuracy"] is not None:
                print(f"  unanswerable_acc     : {agg['unanswerable_accuracy']:.1%}")
            if agg["hallucination_rate"] is not None:
                print(f"  hallucination_rate   : {agg['hallucination_rate']:.1%}")
            if agg["judge_avg_score"] is not None:
                print(f"  judge_avg_score      : {agg['judge_avg_score']:.2f} / 2")
            print(f"  mean_frame_age       : {agg['mean_frame_age_sec']:.2f}s")
            print(f"  p50_latency          : {agg['p50_latency_ms']:.0f}ms")
            print(f"  p95_latency          : {agg['p95_latency_ms']:.0f}ms")

            print("\n  Per-item results:")
            for i, (s, meta_q) in enumerate(zip(self._live_scores, self._live_question_meta)):
                em_icon = "✓" if s.exact_match.matched else "✗"
                judge_str = f"judge={s.judge.score}" if not s.judge.skipped else ""
                print(
                    f"  [{em_icon}] t={meta_q['ask_at']:.0f}s  "
                    f"Q: {meta_q['question'][:50]}  "
                    f"→ {s.exact_match.answer_type}  {judge_str}"
                )

        if self._ltm_scores:
            agg = aggregate_ltm_scores(self._ltm_scores)
            print("\nLTM QA — Memory Retrieval")
            print(f"  n                    : {agg['n']}")
            print(f"  Recall@1             : {agg['recall_at_1']:.1%}")
            print(f"  Recall@3             : {agg['recall_at_3']:.1%}")
            print(f"  Recall@5             : {agg['recall_at_5']:.1%}")
            print(f"  MRR                  : {agg['mrr']:.3f}")
            print(f"  mean_temporal_dist   : {agg['mean_temporal_distance_sec']:.1f}s")
            print(f"  mean_evidence_iou    : {agg['mean_evidence_iou']:.3f}")
            print("\nLTM QA — Final Answer")
            print(f"  answer_accuracy      : {agg['answer_accuracy']:.1%}")
            if agg["judge_avg_score"] is not None:
                print(f"  judge_avg_score      : {agg['judge_avg_score']:.2f} / 2")
            print(f"  p50_latency          : {agg['p50_latency_ms']:.0f}ms")
            print(f"  p95_latency          : {agg['p95_latency_ms']:.0f}ms")

            print("\n  Per-item results:")
            for s, meta_q in zip(self._ltm_scores, self._ltm_question_meta):
                em_icon = "✓" if s.exact_match.matched else "✗"
                r1 = "R@1✓" if s.retrieval.recall_at_1 else "R@1✗"
                judge_str = f"judge={s.judge.score}" if not s.judge.skipped else ""
                print(
                    f"  [{em_icon}] {r1}  intent={s.plan_intent}  "
                    f"Q: {meta_q['question'][:45]}  {judge_str}"
                )

        n_videos = len(self._video_ids)
        if n_videos > 1:
            print(f"\nVideos evaluated : {n_videos}")
        print(f"Results stored → evals/outputs/eval_runs.sqlite  run_id={self.meta.run_id}")

    def finalize(self) -> dict:
        """Write aggregated summary_json + n_videos back to eval_runs row.

        Call once after all questions have been added. Returns the summary dict.
        """
        summary = {
            "n_videos": len(self._video_ids),
            "live": aggregate_live_scores(self._live_scores) if self._live_scores else {},
            "ltm": aggregate_ltm_scores(self._ltm_scores) if self._ltm_scores else {},
        }
        self._conn.execute(
            "UPDATE eval_runs SET summary_json=?, n_videos=? WHERE run_id=?",
            (json.dumps(summary, default=str), len(self._video_ids) or None, self.meta.run_id),
        )
        self._conn.commit()
        return summary

    def save_json(self, out_path: Path) -> None:
        summary = self.finalize()
        data = {
            "run_id": self.meta.run_id,
            "task": self.meta.task,
            "manifest_id": self.meta.manifest_id,
            "model": self.meta.model,
            "config": self.meta.config_extra,
            **summary,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("JSON report → %s", out_path)
