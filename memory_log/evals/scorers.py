"""Scoring functions for Live QA and LTM QA evaluation.

Three scoring modes:
  1. Deterministic fast path: exact/alias match, MCQ match, abstention detection.
  2. LLM judge (rubric 0/1/2) for open-ended answers with hallucination flags.
  3. Retrieval metrics: Recall@K, MRR, evidence_window_IoU, temporal distance.

The LLM judge calls the OpenAI text-completion API (or any model available via
the existing OpenAI client). Results include per-item rationales for spot-checking.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3
    from src.ltm_query.retrieval import RetrievalResults

logger = logging.getLogger("evals.scorers")

# ---- Abstention patterns ----
_ABSTENTION_PATTERNS = [
    r"(cannot|can't|unable to)\s+(determine|see|tell|know|answer)",
    r"(not (visible|shown|in (the|my) (view|frame|field)))",
    r"I (don'?t|do not) (know|have enough|see)",
    r"not (enough|enough information|possible to (determine|tell))",
    r"no (information|way to tell)",
    r"the (image|frame|video) (does not|doesn't) (show|contain|have)",
]
_ABSTENTION_RE = re.compile("|".join(_ABSTENTION_PATTERNS), re.IGNORECASE)


# ---- result types ----

@dataclass
class ExactMatchResult:
    matched: bool
    is_abstention: bool  # True when the system answer signals "I don't know"
    answer_type: str     # "short_text" | "mcq" | "unanswerable"


@dataclass
class JudgeResult:
    score: int  # 0 = wrong/unsupported, 1 = partial, 2 = correct and supported
    hallucinated_object: bool = False
    hallucinated_time: bool = False
    hallucinated_location: bool = False
    should_have_abstained: bool = False
    rationale: str = ""
    skipped: bool = False  # True when judge was not requested or failed


@dataclass
class RetrievalScore:
    recall_at_1: bool
    recall_at_3: bool
    recall_at_5: bool
    mrr: float  # reciprocal rank of first correct retrieval (0 if not found in any)
    min_temporal_distance_sec: float  # to nearest gold window
    evidence_iou: float  # max IoU with any gold window
    gold_found: bool  # at least one gold window hit


@dataclass
class LiveScore:
    exact_match: ExactMatchResult
    judge: JudgeResult
    frame_age_sec: float
    frames_used: int
    latency_ms: float


@dataclass
class LtmScore:
    exact_match: ExactMatchResult
    judge: JudgeResult
    retrieval: RetrievalScore
    latency_ms: float
    plan_intent: str
    expanded: bool = False


# ---- deterministic scoring ----

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip(".,!?"))


def exact_or_alias_match(
    system_answer: str,
    gold_answer: str,
    acceptable: list[str],
    unacceptable: list[str],
    answer_type: str,
) -> ExactMatchResult:
    """Check if the system answer matches gold/acceptable/MCQ expectations."""
    norm_sys = _normalize(system_answer)
    is_abstention = bool(_ABSTENTION_RE.search(system_answer))

    if answer_type == "unanswerable":
        # Correct if the system abstains (or says "I don't know")
        return ExactMatchResult(matched=is_abstention, is_abstention=is_abstention, answer_type=answer_type)

    if answer_type == "mcq":
        # Accept if any acceptable answer appears as a substring
        all_ok = [_normalize(a) for a in [gold_answer] + acceptable]
        matched = any(ok in norm_sys or norm_sys in ok for ok in all_ok)
        return ExactMatchResult(matched=matched, is_abstention=is_abstention, answer_type=answer_type)

    # short_text: exact or substring match against gold + acceptable_answers
    all_ok = [_normalize(a) for a in [gold_answer] + acceptable]
    all_bad = [_normalize(a) for a in unacceptable]

    # Penalise if explicitly wrong
    if any(bad in norm_sys for bad in all_bad):
        return ExactMatchResult(matched=False, is_abstention=is_abstention, answer_type=answer_type)

    matched = any(ok in norm_sys or norm_sys in ok for ok in all_ok)
    return ExactMatchResult(matched=matched, is_abstention=is_abstention, answer_type=answer_type)


# ---- LLM judge ----

_JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator for a wearable AI assistant's answers to visual memory questions.

Given a question, gold answer, gold evidence, system answer, and retrieved evidence:
Score the system answer and detect hallucinations.

Return ONLY valid JSON — no markdown, no explanation:
{
  "score": <0|1|2>,
  "hallucinated_object": <true|false>,
  "hallucinated_time": <true|false>,
  "hallucinated_location": <true|false>,
  "should_have_abstained": <true|false>,
  "rationale": "<one sentence>"
}

Score rubric:
  2 = Correct AND supported by evidence (or correctly abstains when unanswerable).
  1 = Partially correct, or vague but not harmful, or minor factual error.
  0 = Incorrect, contradicted by evidence, or unsupported hallucination.

Hallucination flags:
  hallucinated_object   = mentions an object not present in evidence or video
  hallucinated_time     = states a time/date that contradicts evidence
  hallucinated_location = states a location that contradicts evidence
  should_have_abstained = question is unanswerable from evidence, but system guessed
"""

_JUDGE_USER_TEMPLATE = """\
Question: {question}

Gold answer: {gold_answer}

Gold evidence description: {gold_evidence}

System answer: {system_answer}

Retrieved evidence (what the system had access to):
{retrieved_evidence}
"""


def llm_judge(
    question: str,
    gold_answer: str,
    gold_evidence: str,
    system_answer: str,
    retrieved_evidence: str,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    """Call an LLM to score the system answer using the strict rubric.

    Falls back to skipped=True JudgeResult on any error, so the rest of the
    eval continues even if the judge API is unavailable.
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        user_msg = _JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            gold_evidence=gold_evidence or "(no evidence provided)",
            system_answer=system_answer,
            retrieved_evidence=retrieved_evidence or "(nothing retrieved)",
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
        data = json.loads(raw)
        return JudgeResult(
            score=int(data.get("score", 0)),
            hallucinated_object=bool(data.get("hallucinated_object", False)),
            hallucinated_time=bool(data.get("hallucinated_time", False)),
            hallucinated_location=bool(data.get("hallucinated_location", False)),
            should_have_abstained=bool(data.get("should_have_abstained", False)),
            rationale=str(data.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return JudgeResult(score=0, rationale=f"judge_error: {exc}", skipped=True)


# ---- retrieval metrics ----

def _ts_to_epoch(ts_utc: str) -> float | None:
    """Parse a UTC ISO8601 string to epoch seconds. Returns None on failure."""
    try:
        dt = datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """IoU between two 1-D intervals."""
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def retrieval_metrics(
    results: "RetrievalResults",
    gold_windows_media_sec: list[tuple[float, float]],
    base_epoch: float,
) -> RetrievalScore:
    """Compute retrieval quality metrics.

    Gold evidence windows are given in media-seconds; we convert them to
    absolute epoch ranges using base_epoch (= replay.base_epoch).

    A retrieved memory is considered a "hit" if its timestamp falls inside any gold window.
    For promoted_events, we check the interval [start_ts_utc, end_ts_utc].
    For active_queries and passive_obs, we check the single timestamp.
    """
    if not gold_windows_media_sec:
        return RetrievalScore(
            recall_at_1=False, recall_at_3=False, recall_at_5=False,
            mrr=0.0, min_temporal_distance_sec=float("inf"),
            evidence_iou=0.0, gold_found=False,
        )

    # Convert gold windows to epoch ranges
    gold_epoch_windows = [
        (base_epoch + s, base_epoch + e) for s, e in gold_windows_media_sec
    ]

    def _is_hit(row_start_epoch: float, row_end_epoch: float | None) -> bool:
        row_end = row_end_epoch if row_end_epoch is not None else row_start_epoch
        for gs, ge in gold_epoch_windows:
            # Hit if the row's time interval overlaps the gold window
            if row_start_epoch <= ge and row_end >= gs:
                return True
        return False

    def _best_iou(row_start: float, row_end: float | None) -> float:
        row_end = row_end if row_end is not None else row_start
        if row_start == row_end:
            # Point event: define IoU as 1.0 if the point is inside any gold window.
            return max(
                1.0 if gs <= row_start <= ge else 0.0
                for gs, ge in gold_epoch_windows
            )
        return max((_interval_iou(row_start, row_end, gs, ge) for gs, ge in gold_epoch_windows), default=0.0)

    def _min_temporal_dist(row_start: float, row_end: float | None) -> float:
        row_mid = ((row_end if row_end else row_start) + row_start) / 2
        return min(
            abs(row_mid - (gs + ge) / 2) for gs, ge in gold_epoch_windows
        )

    # Flatten all retrieved rows in rank order (promoted_events first, then active_queries)
    ranked: list[tuple[float, float | None]] = []  # (start_epoch, end_epoch|None)

    for row in results.promoted_events:
        s = _ts_to_epoch(row["start_ts_utc"])
        e = _ts_to_epoch(row["end_ts_utc"])
        if s is not None:
            ranked.append((s, e))

    for row in results.active_queries:
        t = _ts_to_epoch(row["timestamp_utc"])
        if t is not None:
            ranked.append((t, None))

    for row in results.passive_rows:
        t = _ts_to_epoch(row["timestamp_utc"])
        if t is not None:
            ranked.append((t, None))

    # Compute metrics
    hit_rank: int | None = None
    for rank, (rs, re) in enumerate(ranked, start=1):
        if _is_hit(rs, re):
            hit_rank = rank
            break

    recall_at_1 = hit_rank is not None and hit_rank <= 1
    recall_at_3 = hit_rank is not None and hit_rank <= 3
    recall_at_5 = hit_rank is not None and hit_rank <= 5
    mrr = (1.0 / hit_rank) if hit_rank is not None else 0.0

    if ranked:
        min_dist = min(_min_temporal_dist(rs, re) for rs, re in ranked)
        best_iou = max(_best_iou(rs, re) for rs, re in ranked)
    else:
        min_dist = float("inf")
        best_iou = 0.0

    return RetrievalScore(
        recall_at_1=recall_at_1,
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        mrr=mrr,
        min_temporal_distance_sec=min_dist,
        evidence_iou=best_iou,
        gold_found=hit_rank is not None,
    )


# ---- aggregate helpers ----

def aggregate_live_scores(scores: list[LiveScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {}
    exact = [s.exact_match.matched for s in scores]
    unanswerable = [s for s in scores if s.exact_match.answer_type == "unanswerable"]
    judge_scored = [s for s in scores if not s.judge.skipped]
    halluc = [s for s in scores if not s.judge.skipped and (
        s.judge.hallucinated_object or s.judge.hallucinated_time or s.judge.hallucinated_location
    )]
    latencies = [s.latency_ms for s in scores]
    latencies_sorted = sorted(latencies)
    return {
        "n": n,
        "answer_accuracy": sum(exact) / n,
        "unanswerable_accuracy": (
            sum(s.exact_match.matched for s in unanswerable) / len(unanswerable)
            if unanswerable else None
        ),
        "hallucination_rate": len(halluc) / len(judge_scored) if judge_scored else None,
        "judge_avg_score": (
            sum(s.judge.score for s in judge_scored) / len(judge_scored) if judge_scored else None
        ),
        "mean_frame_age_sec": sum(s.frame_age_sec for s in scores) / n,
        "p50_latency_ms": latencies_sorted[n // 2],
        "p95_latency_ms": latencies_sorted[int(n * 0.95)],
    }


def aggregate_ltm_scores(scores: list[LtmScore]) -> dict:
    n = len(scores)
    if n == 0:
        return {}
    exact = [s.exact_match.matched for s in scores]
    ret = [s.retrieval for s in scores]
    judge_scored = [s for s in scores if not s.judge.skipped]
    latencies_sorted = sorted(s.latency_ms for s in scores)
    return {
        "n": n,
        "answer_accuracy": sum(exact) / n,
        "recall_at_1": sum(r.recall_at_1 for r in ret) / n,
        "recall_at_3": sum(r.recall_at_3 for r in ret) / n,
        "recall_at_5": sum(r.recall_at_5 for r in ret) / n,
        "mrr": sum(r.mrr for r in ret) / n,
        "mean_temporal_distance_sec": sum(
            r.min_temporal_distance_sec for r in ret if r.min_temporal_distance_sec != float("inf")
        ) / max(1, sum(r.gold_found for r in ret)),
        "mean_evidence_iou": sum(r.evidence_iou for r in ret) / n,
        "judge_avg_score": (
            sum(s.judge.score for s in judge_scored) / len(judge_scored) if judge_scored else None
        ),
        "p50_latency_ms": latencies_sorted[n // 2],
        "p95_latency_ms": latencies_sorted[int(n * 0.95)],
    }
