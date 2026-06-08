"""EgoSchema adapter — LTM public benchmark (egocentric, MCQ).

EgoSchema: https://egoschema.github.io/
  - 5000 multiple-choice questions over ~180-second egocentric video clips
  - Questions require holistic temporal understanding of the full clip
  - Conceptual match: egocentric (first-person) POV ≈ wearable memory camera
  - Text QA: available on HuggingFace (lmms-lab/EgoSchema / QA pairs)
  - VIDEO FILES: require Ego4D data access agreement (see below)

=== Accessing EgoSchema videos ===
EgoSchema videos are a subset of Ego4D clips. To get them:
  1. Sign the Ego4D data license at https://ego4d-data.org/docs/start-here/
  2. Use the Ego4D CLI: ego4d --datasets clips --output_directory <dir>
  3. EgoSchema video UIDs are listed in the question JSON: "q_uid" → "video_uid"

If you only have the QA text (no videos), this adapter will generate manifests
with memory_mode=seed using the question text as a seed memory record, but
the recall evaluation will be limited (no real visual history to retrieve from).

=== Using with videos ===
Once you have videos:
  python -m evals.adapters.egoschema \\
      --qa-json evals/datasets/egoschema_raw/questions.json \\
      --video-dir /path/to/ego4d_clips/ \\
      --out-dir evals/datasets/egoschema/ \\
      --limit 20

Then run LTM eval in replay mode:
  uv run python -m evals.run_ltm \\
      --manifest evals/datasets/egoschema/<uid>.json \\
      --memory-mode replay \\
      [--caption-history]     # VLM-caption passive observations for richer retrieval

=== HuggingFace QA download ===
  pip install datasets
  python -m evals.adapters.egoschema --download-qa --qa-json evals/datasets/egoschema_raw/questions.json

Mapping to EvalManifest:
  video_uid + ~180s clip → history_video_path (memory_mode=replay)
  5 options              → memory_question (answer_type=mcq, choices=[...])
  correct_option (0-4)   → gold_answer
  no timestamp           → query_time_sec=0 (no live grounding)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from evals.adapters.base import BenchmarkAdapter
from evals.manifest import EvalManifest, GoldLocation, MemoryQuestion, save_manifest

logger = logging.getLogger("evals.adapters.egoschema")

_HF_DATASET_ID = "lmms-lab/EgoSchema"
_CLIP_DURATION_SEC = 180.0  # EgoSchema clips are always ~180s


def _download_qa(out_path: Path) -> None:
    """Download EgoSchema QA pairs (text only, no videos) from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install 'datasets': pip install datasets huggingface_hub")

    logger.info("Downloading EgoSchema QA from %s …", _HF_DATASET_ID)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(_HF_DATASET_ID, split="test")
    with out_path.open("w") as f:
        json.dump([dict(row) for row in ds], f, indent=2)
    logger.info("Saved %d QA pairs → %s", len(ds), out_path)


def _load_qa_json(qa_json: Path) -> list[dict]:
    with qa_json.open() as f:
        data = json.load(f)
    # Accept both list-of-dicts and dict-keyed formats
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    raise ValueError(f"Unexpected format in {qa_json}")


class EgoSchemaAdapter(BenchmarkAdapter):
    """Convert EgoSchema QA pairs to LTM manifests (memory_mode=replay).

    Args:
        raw_dir:   Not used directly (set to dummy Path if no videos).
        qa_json:   Path to the questions.json (from HF download or manual).
        video_dir: Directory containing <video_uid>.mp4 clips. May be None
                   if only testing text-only QA (memory_mode will fall back to seed).
        limit:     Max questions per manifest (1 question per clip = 1 manifest).
    """

    def __init__(
        self,
        raw_dir: Path,
        qa_json: Path,
        video_dir: Path | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(raw_dir, limit)
        self.qa_json = qa_json
        self.video_dir = video_dir

    @property
    def name(self) -> str:
        return "egoschema"

    def download(self, target_dir: Path) -> None:
        _download_qa(target_dir / "questions.json")
        print(
            "\nNOTE: EgoSchema video files require Ego4D data access.\n"
            "Visit https://ego4d-data.org/docs/start-here/ to sign the agreement,\n"
            "then download clips with: ego4d --datasets clips --output_directory <dir>\n"
        )

    def to_manifests(self) -> list[EvalManifest]:
        rows = _load_qa_json(self.qa_json)
        if self.limit:
            rows = rows[: self.limit]

        manifests: list[EvalManifest] = []

        for row in rows:
            q_uid = str(row.get("q_uid") or row.get("id", "unknown"))
            video_uid = str(row.get("video_uid") or q_uid)
            question_text = row.get("question") or row.get("question_text", "")

            # Options: try various field name conventions
            options = []
            for key in ["option 0", "option0", "A", "option_0"]:
                if key in row:
                    for i in range(5):
                        for fmt in [f"option {i}", f"option{i}", "ABCDE"[i], f"option_{i}"]:
                            if fmt in row and row[fmt]:
                                options.append(str(row[fmt]))
                                break
                    break

            if not options:
                # Fallback: collect any key with 'option' in the name
                option_keys = sorted(k for k in row if "option" in k.lower())
                options = [str(row[k]) for k in option_keys if row.get(k)]

            answer_key = row.get("answer", row.get("correct_option", row.get("ans", 0)))
            try:
                answer_idx = int(answer_key)
            except (ValueError, TypeError):
                if isinstance(answer_key, str) and answer_key.upper() in "ABCDE":
                    answer_idx = "ABCDE".index(answer_key.upper())
                else:
                    answer_idx = 0

            gold_answer = options[answer_idx] if answer_idx < len(options) else ""

            # Video path
            video_path_str = ""
            memory_mode = "seed"
            history_video_path: str | None = None

            if self.video_dir:
                candidate = self.video_dir / f"{video_uid}.mp4"
                if candidate.exists():
                    video_path_str = str(candidate)
                    history_video_path = str(candidate)
                    memory_mode = "replay"
                else:
                    logger.warning("Video not found: %s  (seed mode)", candidate)
                    video_path_str = str(candidate)  # record path even if missing

            seed_memories = []
            if memory_mode == "seed":
                # Without videos, create a text-only seed memory from the question context
                # (useful for testing the QA pipeline structure, not visual retrieval)
                from evals.manifest import SeedMemory
                seed_memories = [
                    SeedMemory(
                        kind="active_query",
                        timestamp="2026-01-01T09:00:00+00:00",
                        user_question="(EgoSchema clip context)",
                        model_answer=f"Egocentric clip {video_uid}: {question_text[:200]}",
                        camera_source="egoschema",
                    )
                ]

            mq = MemoryQuestion(
                id=q_uid,
                query_time_sec=0.0,  # EgoSchema: no specific timestamp
                question=question_text,
                gold_answer=gold_answer,
                acceptable_answers=[gold_answer],
                answer_type="mcq",
                answerable=True,
                choices=options,
                # No gold_evidence_windows — EgoSchema provides answer but not timestamps
            )

            manifest = EvalManifest(
                video_id=f"ego_{q_uid}",
                video_path=video_path_str or f"videos/{video_uid}.mp4",
                description=f"EgoSchema clip {video_uid}",
                memory_mode=memory_mode,
                seed_memories=seed_memories,
                history_video_path=history_video_path,
                memory_questions=[mq],
            )
            manifests.append(manifest)

        logger.info("Converted %d EgoSchema items to manifests (mode=%s)", len(manifests), memory_mode)
        return manifests


def main() -> None:
    parser = argparse.ArgumentParser(description="EgoSchema LTM benchmark adapter")
    parser.add_argument("--qa-json", type=Path, required=False, help="Path to EgoSchema questions.json")
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory with ego4d clip mp4s")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--download-qa", action="store_true",
                        help="Download QA pairs from HuggingFace (no videos)")
    args = parser.parse_args()

    if args.download_qa:
        qa_path = args.qa_json or Path("evals/datasets/egoschema_raw/questions.json")
        _download_qa(qa_path)
        return

    if not args.qa_json or not args.qa_json.exists():
        print("ERROR: --qa-json is required (and must exist) when not using --download-qa")
        return

    adapter = EgoSchemaAdapter(
        raw_dir=args.qa_json.parent,
        qa_json=args.qa_json,
        video_dir=args.video_dir,
        limit=args.limit,
    )
    manifests = adapter.to_manifests()

    out_dir = args.out_dir or Path("evals/datasets/egoschema")
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in manifests:
        save_manifest(m, out_dir / f"{m.video_id}.json")

    print(f"Wrote {len(manifests)} manifests to {out_dir}")
    print("\nNext (requires Ego4D videos):")
    print(f"  uv run python -m evals.run_ltm --manifest {out_dir}/<uid>.json --memory-mode replay")


if __name__ == "__main__":
    main()
