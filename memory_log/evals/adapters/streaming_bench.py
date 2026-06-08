"""StreamingBench adapter — Live QA public benchmark.

StreamingBench: https://streamingbench.github.io/
  GitHub: https://github.com/THUNLP-MT/StreamingBench
  Paper:  https://arxiv.org/abs/2411.03628

  - 900 videos, 4500 MCQ questions across 5 task categories
  - Each question is tied to a specific timestamp in the video
  - Videos are NOT on HuggingFace (~203 GB total); annotations are tiny (<1 MB)

HuggingFace dataset (annotations only, instant stream):
  mjuicem/StreamingBench  — 4 MCQ configs we support:
    Real_Time_Visual_Understanding (2500 Q)
    Sequential_Question_Answering  (250 Q)
    Contextual_Understanding       (500 Q)
    Omni_Source_Understanding      (1000 Q)

  Proactive_Output is excluded (open-ended, not MCQ).

Videos:
  Download from the StreamingBench GitHub — they provide a script + Google Drive links.
  Place MP4s in a local directory (e.g. streaming_bench_videos/) named as the sample IDs
  extracted from question_id: "Real-Time Visual Understanding_sample_1_3" → video "sample_1".
  Pass that directory as --video-dir. Without it, manifests still generate with placeholder paths.

Quick start (annotations only, no video download):
  uv run python -m evals.adapters.streaming_bench \\
      --out-dir evals/datasets/streaming_bench/ --limit 20

With local videos:
  uv run python -m evals.adapters.streaming_bench \\
      --video-dir /path/to/streaming_bench_videos/ \\
      --out-dir evals/datasets/streaming_bench/ --limit 20

Then run Live QA eval (deterministic MCQ scoring, no judge needed):
  uv run python -m evals.run_live \\
      --manifest evals/datasets/streaming_bench/<video_id>.json --no-judge
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

from evals.adapters.base import BenchmarkAdapter
from evals.manifest import EvalManifest, LiveQuestion, save_manifest

logger = logging.getLogger("evals.adapters.streaming_bench")

_HF_DATASET_ID = "mjuicem/StreamingBench"

# Task configs that are MCQ-compatible (skip Proactive_Output which is open-ended)
_MCQ_CONFIGS = [
    "Real_Time_Visual_Understanding",
    "Sequential_Question_Answering",
    "Contextual_Understanding",
    "Omni_Source_Understanding",
]

# question_id format: "<Category>_sample_<N>_<Q>" or "<Category>_sample_<N>"
_SAMPLE_RE = re.compile(r"sample_(\d+)", re.IGNORECASE)


def _parse_video_id(question_id: str) -> str:
    """Extract the video sample ID from a question_id string."""
    m = _SAMPLE_RE.search(question_id)
    return f"sample_{m.group(1)}" if m else question_id.split("_")[0]


def _parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS or MM:SS string to seconds."""
    parts = ts.strip().split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return float(parts[0])
    except (ValueError, TypeError):
        return 0.0


def _parse_options(options_str: str) -> list[str]:
    """Parse the options field (stored as a Python list literal string)."""
    try:
        opts = ast.literal_eval(options_str)
        return [str(o) for o in opts]
    except Exception:
        return [options_str]


def _answer_idx_from_letter(answer: str, options: list[str]) -> int:
    """Convert an answer letter (A/B/C/D) to an index into options."""
    letter = answer.strip().upper()
    if letter in "ABCD":
        return "ABCD".index(letter)
    try:
        return int(answer)
    except (ValueError, TypeError):
        return 0


class StreamingBenchAdapter(BenchmarkAdapter):
    """Convert StreamingBench annotations to LiveQuestion manifests.

    Streams annotations from HuggingFace (no full video download required).
    Videos are resolved from video_dir if provided; otherwise placeholder paths are used.
    """

    def __init__(
        self,
        raw_dir: Path | None = None,
        limit: int | None = None,
        video_dir: Path | None = None,
        configs: list[str] | None = None,
    ) -> None:
        # raw_dir unused (annotations come from HF streaming), kept for ABC compat
        super().__init__(raw_dir=raw_dir or Path("."), limit=limit)
        self.video_dir = video_dir
        self.configs = configs or _MCQ_CONFIGS

    @property
    def name(self) -> str:
        return "streaming_bench"

    def download(self, target_dir: Path) -> None:
        logger.info(
            "StreamingBench annotations are streamed directly from HuggingFace — "
            "no separate download step needed.\n"
            "For videos (~203 GB total), see: https://github.com/THUNLP-MT/StreamingBench"
        )

    def to_manifests(self) -> list[EvalManifest]:
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Install 'datasets': pip install datasets huggingface_hub")

        # Group rows by video sample across all configs
        by_video: dict[str, list[dict]] = {}

        for config in self.configs:
            logger.info("Streaming config: %s …", config)
            try:
                ds = load_dataset(_HF_DATASET_ID, config, streaming=True)
                split_name = config  # split name matches config in this dataset
                split = ds[split_name]
            except Exception as exc:
                logger.warning("Could not load config %s: %s", config, exc)
                continue

            for row in split:
                vid_id = _parse_video_id(row.get("question_id", ""))
                key = f"{config}__{vid_id}"
                row["_config"] = config
                row["_vid_id"] = vid_id
                by_video.setdefault(key, []).append(row)

        # Apply limit (per video, not per question)
        video_keys = list(by_video.keys())
        if self.limit:
            video_keys = video_keys[: self.limit]

        manifests: list[EvalManifest] = []
        for key in video_keys:
            rows = by_video[key]
            config = rows[0]["_config"]
            vid_id = rows[0]["_vid_id"]

            # Resolve video path
            if self.video_dir:
                video_path = self.video_dir / f"{vid_id}.mp4"
                video_path_str = str(video_path)
            else:
                video_path_str = f"PLACEHOLDER:{vid_id}.mp4"

            live_questions: list[LiveQuestion] = []
            for i, row in enumerate(rows):
                options = _parse_options(row.get("options", "[]"))
                answer_letter = row.get("answer", "A")
                ans_idx = _answer_idx_from_letter(answer_letter, options)
                gold = options[ans_idx] if ans_idx < len(options) else answer_letter
                ask_at = _parse_timestamp(row.get("time_stamp", "0"))
                q_id = row.get("question_id", f"{key}_q{i}")

                live_questions.append(LiveQuestion(
                    id=q_id,
                    ask_at_sec=ask_at,
                    question=row.get("question", ""),
                    gold_answer=gold,
                    acceptable_answers=[gold],
                    answer_type="mcq",
                    choices=options,
                ))

            manifest = EvalManifest(
                video_id=f"sb_{config[:4].lower()}_{vid_id}",
                video_path=video_path_str,
                description=f"StreamingBench {config} {vid_id}",
                live_questions=live_questions,
            )
            manifests.append(manifest)

        logger.info("Generated %d manifests from StreamingBench", len(manifests))
        return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StreamingBench adapter — stream annotations, generate manifests"
    )
    parser.add_argument(
        "--video-dir", type=Path, default=None,
        help="Local directory containing MP4 files named sample_N.mp4 (optional).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("evals") / "datasets" / "streaming_bench",
        help="Output directory for manifest JSONs.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        choices=_MCQ_CONFIGS,
        help="Task configs to include (default: all 4 MCQ configs).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max videos to convert.")
    # Kept for backwards compat but now a no-op
    parser.add_argument("--raw-dir", type=Path, default=None, help="Unused (annotations streamed from HF).")
    parser.add_argument("--download", action="store_true", help="No-op: annotations are streamed from HF.")
    args = parser.parse_args()

    if args.download:
        print(
            "INFO: StreamingBench annotations are streamed from HuggingFace automatically.\n"
            "      For videos (~203 GB), see: https://github.com/THUNLP-MT/StreamingBench\n"
            "      Pass --video-dir <dir> once you have the MP4 files locally."
        )

    adapter = StreamingBenchAdapter(
        video_dir=args.video_dir,
        limit=args.limit,
        configs=args.configs,
    )

    try:
        manifests = adapter.to_manifests()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if not manifests:
        print("No manifests generated — check the HuggingFace dataset access.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for m in manifests:
        save_manifest(m, args.out_dir / f"{m.video_id}.json")

    print(f"Wrote {len(manifests)} manifests → {args.out_dir}")
    placeholder_count = sum(1 for m in manifests if m.video_path.startswith("PLACEHOLDER:"))
    if placeholder_count:
        print(
            f"\nNote: {placeholder_count} manifests have placeholder video paths.\n"
            f"Download videos from https://github.com/THUNLP-MT/StreamingBench and\n"
            f"pass --video-dir <dir> to resolve real paths."
        )
    print(
        f"\nNext:\n"
        f"  uv run python -m evals.run_live \\\n"
        f"      --manifest {args.out_dir}/<video_id>.json --no-judge"
    )


if __name__ == "__main__":
    main()
