"""StreamingBench adapter — Live QA public benchmark.

StreamingBench: https://streamingbench.github.io/
  - 900 videos with 4500 human-curated QA pairs
  - Multi-timestamp MCQ: each question is tied to a specific time offset in the video
  - Task categories: real-time understanding, omni-source, contextual understanding
  - Available on HuggingFace: lmms-lab/StreamingBench

Download steps:
  pip install datasets huggingface_hub
  huggingface-cli login   # or set HF_TOKEN env var
  python -m evals.adapters.streaming_bench --download --raw-dir evals/datasets/streaming_bench_raw/

Then convert to manifests:
  python -m evals.adapters.streaming_bench \\
      --raw-dir evals/datasets/streaming_bench_raw/ \\
      --out-dir evals/datasets/streaming_bench/ \\
      --limit 50

Then run Live QA eval:
  uv run python -m evals.run_live \\
      --manifest evals/datasets/streaming_bench/<video_id>.json \\
      --no-judge   # MCQ → deterministic scoring, judge not needed

Mapping from StreamingBench to EvalManifest:
  video_id            → manifest video_id
  question + options  → LiveQuestion (answer_type=mcq, choices=[...])
  timestamp           → ask_at_sec
  correct_option      → gold_answer (option text) + acceptable_answers
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from evals.adapters.base import BenchmarkAdapter
from evals.manifest import EvalManifest, GoldLocation, LiveQuestion, save_manifest

logger = logging.getLogger("evals.adapters.streaming_bench")

# HuggingFace dataset identifier
_HF_DATASET_ID = "lmms-lab/StreamingBench"
_HF_SUBSET = "default"


def _download(target_dir: Path) -> None:
    """Download StreamingBench from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install 'datasets': pip install datasets huggingface_hub")

    logger.info("Downloading %s to %s …", _HF_DATASET_ID, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(_HF_DATASET_ID, split="test")
    out = target_dir / "streaming_bench.jsonl"
    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps(dict(row)) + "\n")
    logger.info("Saved %d rows → %s", len(ds), out)


class StreamingBenchAdapter(BenchmarkAdapter):
    """Convert StreamingBench QA pairs to LiveQuestion manifests.

    Expected raw_dir layout (after download):
        raw_dir/
          streaming_bench.jsonl   ← one row per QA pair from HuggingFace
          videos/                 ← mp4 files (video_id.mp4)
    """

    @property
    def name(self) -> str:
        return "streaming_bench"

    def download(self, target_dir: Path) -> None:
        _download(target_dir)

    def to_manifests(self) -> list[EvalManifest]:
        jsonl_path = self.raw_dir / "streaming_bench.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"StreamingBench data not found at {jsonl_path}. "
                f"Run with --download first."
            )

        # Group QA pairs by video_id
        by_video: dict[str, list[dict]] = {}
        with jsonl_path.open() as f:
            for line in f:
                row = json.loads(line)
                vid = row.get("video_id") or row.get("video_name") or row.get("id", "unknown")
                by_video.setdefault(str(vid), []).append(row)

        manifests: list[EvalManifest] = []
        video_ids = list(by_video.keys())
        if self.limit:
            video_ids = video_ids[: self.limit]

        for vid_id in video_ids:
            qa_pairs = by_video[vid_id]
            video_path = self.raw_dir / "videos" / f"{vid_id}.mp4"

            live_questions: list[LiveQuestion] = []
            for i, row in enumerate(qa_pairs):
                timestamp = float(row.get("timestamp", row.get("time", row.get("ask_at_sec", 0.0))))
                question_text = row.get("question", row.get("Q", ""))
                options = [
                    row.get("option0") or row.get("A", ""),
                    row.get("option1") or row.get("B", ""),
                    row.get("option2") or row.get("C", ""),
                    row.get("option3") or row.get("D", ""),
                ]
                options = [o for o in options if o]

                # Correct answer: index or letter
                answer_key = row.get("answer", row.get("correct_option", row.get("gt", 0)))
                if isinstance(answer_key, str) and answer_key.upper() in "ABCD":
                    answer_idx = "ABCD".index(answer_key.upper())
                else:
                    try:
                        answer_idx = int(answer_key)
                    except (ValueError, TypeError):
                        answer_idx = 0

                gold_answer = options[answer_idx] if answer_idx < len(options) else ""

                live_questions.append(LiveQuestion(
                    id=f"{vid_id}_q{i}",
                    ask_at_sec=timestamp,
                    question=question_text,
                    gold_answer=gold_answer,
                    acceptable_answers=[gold_answer],
                    answer_type="mcq",
                    choices=options,
                ))

            manifest = EvalManifest(
                video_id=f"sb_{vid_id}",
                video_path=str(video_path) if video_path.exists() else f"videos/{vid_id}.mp4",
                description=f"StreamingBench video {vid_id}",
                live_questions=live_questions,
            )
            manifests.append(manifest)

        logger.info("Converted %d StreamingBench videos to manifests", len(manifests))
        return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamingBench adapter")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory with raw benchmark files")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for manifest JSONs")
    parser.add_argument("--limit", type=int, default=None, help="Max videos to convert")
    parser.add_argument("--download", action="store_true", help="Download dataset from HuggingFace first")
    args = parser.parse_args()

    if args.download:
        _download(args.raw_dir)

    adapter = StreamingBenchAdapter(raw_dir=args.raw_dir, limit=args.limit)
    manifests = adapter.to_manifests()

    out_dir = args.out_dir or (Path("evals") / "datasets" / "streaming_bench")
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in manifests:
        save_manifest(m, out_dir / f"{m.video_id}.json")

    print(f"Wrote {len(manifests)} manifests to {out_dir}")
    print("\nNext:")
    print(f"  uv run python -m evals.run_live --manifest {out_dir}/<video_id>.json --no-judge")


if __name__ == "__main__":
    main()
