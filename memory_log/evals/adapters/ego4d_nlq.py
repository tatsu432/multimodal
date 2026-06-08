"""Ego4D NLQ (Natural Language Queries) adapter — documented stub.

Ego4D Episodic Memory / NLQ task:
  https://ego4d-data.org/docs/benchmarks/episodic-memory/
  Ego4D challenge: https://eval.ai/web/challenges/challenge-page/1626/overview

Status: INTERFACE READY — not implemented (requires Ego4D license + large download).

=== What NLQ is ===
Natural Language Queries over ~800 hours of egocentric video (Ego4D).
Each item: a clip + natural-language query whose answer is a temporal moment
in the clip (start_sec, end_sec). Example:
  query: "Where did I put my phone?"
  answer: [moment 742.3s, 749.1s]  ← NOT a text answer, a time localization

This is the closest public benchmark to our LTM task, but the output modality
is different: Ego4D NLQ expects temporal localization, while our system produces
a free-text answer. The mapping below bridges the gap.

=== Mapping to EvalManifest ===
  video_uid (Ego4D clip)     → history_video_path (memory_mode=replay)
  query text                 → memory_question.question
  temporal response window   → gold_evidence_windows (seconds in clip)
  No canonical "text answer" — we derive gold_answer from the scene
    at the response window (via VLM captioning during adapter conversion)
    or use a templated answer like "at around {start:.0f}s into the clip".

=== Access requirements ===
1. Sign Ego4D license at https://ego4d-data.org/docs/start-here/
2. Download NLQ annotations: ego4d --datasets nlq_annotations --output_dir <dir>
3. Download video clips: ego4d --datasets clips --output_dir <video_dir>
4. Install ego4d SDK: pip install ego4d

=== NLQ annotation format ===
{
  "videos": [
    {
      "video_uid": "...",
      "clips": [
        {
          "clip_uid": "...",
          "clip_start_sec": ..., "clip_end_sec": ...,
          "annotations": [
            {
              "annotation_uid": "...",
              "language_queries": [
                {
                  "query": "Where did I put the phone?",
                  "video_start_sec": 742.3,
                  "video_end_sec": 749.1,
                  "template": "...",
                  "query_type": "..."
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

=== Implementation sketch ===
When implemented, this adapter would:
  1. Parse the NLQ JSON annotations.
  2. For each (clip_uid, query):
     a. Find the clip .mp4.
     b. Build EvalManifest with memory_mode=replay (clip → past memory).
     c. Convert [video_start_sec, video_end_sec] → gold_evidence_windows
        (relative to clip start: gold_windows_media_sec = [start - clip_start, end - clip_start]).
     d. For gold_answer: either use a template ("the item was visible at ~{t:.0f}s")
        or VLM-caption the gold window frames.
     e. query_time_sec=0 (no live grounding, this is a pure recall task).
  3. Save one manifest per query (or group by clip_uid, one manifest per clip
     with multiple memory_questions).

Because gold_evidence_windows are available from annotations, this adapter
would give the BEST retrieval metrics (Recall@K, MRR, IoU) of any benchmark.

=== RTV-Bench / SVBench alternatives ===
For streaming Live QA with temporal grounding, RTV-Bench and SVBench are
alternatives to StreamingBench. Their formats are similar (timestamp + MCQ);
adapters would follow the same pattern as streaming_bench.py.
"""

from __future__ import annotations

from pathlib import Path

from evals.adapters.base import BenchmarkAdapter
from evals.manifest import EvalManifest


class Ego4DNLQAdapter(BenchmarkAdapter):
    """Stub adapter for Ego4D Episodic Memory / NLQ task.

    Not yet implemented — see module docstring for requirements and design.
    Raises NotImplementedError on to_manifests().
    """

    @property
    def name(self) -> str:
        return "ego4d_nlq"

    def download(self, target_dir: Path) -> None:
        raise NotImplementedError(
            "Ego4D download requires signing the data license agreement.\n"
            "See: https://ego4d-data.org/docs/start-here/\n"
            "Then: ego4d --datasets nlq_annotations clips --output_directory <dir>"
        )

    def to_manifests(self) -> list[EvalManifest]:
        raise NotImplementedError(
            "Ego4DNLQAdapter is a documented stub — not yet implemented.\n"
            "See evals/adapters/ego4d_nlq.py for the mapping design.\n"
            "Implement to_manifests() following the sketch in the module docstring."
        )
