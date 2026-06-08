"""Generate a tiny toy evaluation dataset.

Creates:
  evals/datasets/toy/desk_001.mp4   — synthetic 5-min video with scripted events
  evals/datasets/toy/desk_001.json  — matching EvalManifest

Run:
    cd memory_log
    uv run python -m evals.make_sample

The generated video uses colored shapes + text overlays as "objects" so the whole
pipeline can run end-to-end without real footage.

Scripted timeline (all times in seconds):
  0–9   : Empty desk. White background.
  10–24 : Red bottle appears on left side of desk.
  25–49 : Blue laptop opens on right side.
  50–74 : Both objects visible.
  75–119: Red bottle is "moved" to the shelf (right side, different y-position).
  120–299: Final stable state — laptop open, bottle on shelf.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DATASETS_DIR = Path(__file__).parent / "datasets" / "toy"
_FPS = 5
_WIDTH = 640
_HEIGHT = 360
_DURATION_SEC = 300  # 5 minutes


def _make_frame(t: float) -> np.ndarray:
    """Render a synthetic frame at media time `t` with visible event cues."""
    frame = np.full((_HEIGHT, _WIDTH, 3), 245, dtype=np.uint8)  # near-white bg

    # ---- floor / desk surface ----
    desk_y = int(_HEIGHT * 0.6)
    cv2.rectangle(frame, (0, desk_y), (_WIDTH, _HEIGHT), (210, 200, 190), -1)

    # ---- shelf (upper area) ----
    shelf_y = int(_HEIGHT * 0.25)
    cv2.rectangle(frame, (0, shelf_y), (_WIDTH, shelf_y + 8), (160, 130, 100), -1)

    # ---- red bottle on desk (t=10–74) ----
    if 10 <= t < 75:
        bx, by = 120, desk_y - 80
        # Body
        cv2.rectangle(frame, (bx, by), (bx + 35, by + 80), (30, 40, 200), -1)
        # Cap
        cv2.rectangle(frame, (bx + 8, by - 20), (bx + 27, by), (20, 20, 160), -1)
        cv2.putText(
            frame, "RED BOTTLE", (bx - 10, by - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 40, 200), 1,
        )

    # ---- red bottle on shelf (t=75–299) ----
    if t >= 75:
        bx, by = 200, shelf_y - 55
        cv2.rectangle(frame, (bx, by), (bx + 35, by + 55), (30, 40, 200), -1)
        cv2.rectangle(frame, (bx + 8, by - 14), (bx + 27, by), (20, 20, 160), -1)
        cv2.putText(
            frame, "RED BOTTLE", (bx - 10, by - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 40, 200), 1,
        )

    # ---- laptop on desk (t=25–299) ----
    if t >= 25:
        lx, ly = 400, desk_y - 100
        # Screen
        cv2.rectangle(frame, (lx, ly), (lx + 140, ly + 90), (50, 50, 50), -1)
        cv2.rectangle(frame, (lx + 5, ly + 5), (lx + 135, ly + 85), (80, 160, 80), -1)
        # Base
        cv2.rectangle(frame, (lx - 10, desk_y - 10), (lx + 155, desk_y), (80, 80, 80), -1)
        cv2.putText(
            frame, "LAPTOP", (lx + 30, ly - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1,
        )

    # ---- phase label ----
    if t < 10:
        phase = "Empty desk"
    elif t < 25:
        phase = "Red bottle on desk"
    elif t < 50:
        phase = "Laptop opens, bottle on desk"
    elif t < 75:
        phase = "Both on desk"
    elif t < 120:
        phase = "Bottle moved to shelf"
    else:
        phase = "Laptop open, bottle on shelf"

    cv2.putText(
        frame, phase, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1,
    )
    cv2.putText(
        frame, f"t={t:.1f}s", (10, _HEIGHT - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1,
    )

    return frame


def make_video(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, _FPS, (_WIDTH, _HEIGHT))

    total_frames = _DURATION_SEC * _FPS
    for i in range(total_frames):
        t = i / _FPS
        writer.write(_make_frame(t))

    writer.release()
    logger.info("Video written → %s  (%.0fs @ %dfps)", out_path, _DURATION_SEC, _FPS)


def make_manifest(video_path: Path, out_path: Path) -> None:
    """Write a manifest that exercises both Live QA and LTM QA."""
    manifest = {
        "video_id": "desk_001",
        "video_path": str(video_path.name),  # relative to manifest dir
        "base_timestamp": "2026-01-15T09:00:00+09:00",
        "description": "Toy scripted desk scenario — red bottle & laptop",
        "default_location": {"label": "home desk", "lat": 35.68, "lon": 139.76},

        # ---- Live QA ----
        "live_questions": [
            {
                "id": "lq1",
                "ask_at_sec": 15.0,
                "question": "What object do you see on the desk?",
                "gold_answer": "a red bottle",
                "acceptable_answers": ["red bottle", "a red bottle on the desk", "bottle"],
                "answer_type": "short_text",
                "gold_evidence_window": [10.0, 24.0],
            },
            {
                "id": "lq2",
                "ask_at_sec": 35.0,
                "question": "Is there a laptop visible?",
                "gold_answer": "yes",
                "acceptable_answers": ["yes", "a laptop", "yes there is a laptop"],
                "answer_type": "short_text",
                "gold_evidence_window": [25.0, 50.0],
            },
            {
                "id": "lq3",
                "ask_at_sec": 5.0,
                "question": "Is there an object behind the camera that I cannot see?",
                "gold_answer": "I cannot determine that from the visible scene",
                "acceptable_answers": [
                    "cannot determine", "not visible", "I cannot see", "unable to tell",
                    "I don't know",
                ],
                "answer_type": "unanswerable",
            },
        ],

        # ---- LTM QA ----
        # history_video_path == video_path: the same clip contains the full past timeline.
        # seed mode uses the structured seed_memories; replay mode ingests this video.
        "memory_mode": "seed",
        "history_video_path": str(video_path.name),
        "seed_memories": [
            {
                "kind": "active_query",
                "timestamp": "2026-01-15T09:00:15+09:00",
                "location": {"label": "home desk", "lat": 35.68, "lon": 139.76},
                "user_question": "What is on the desk?",
                "model_answer": "There is a red bottle on the left side of the desk.",
                "camera_source": "eval_replay",
            },
            {
                "kind": "active_query",
                "timestamp": "2026-01-15T09:01:30+09:00",
                "location": {"label": "home desk", "lat": 35.68, "lon": 139.76},
                "user_question": "What changed on the desk?",
                "model_answer": (
                    "The red bottle was moved from the desk to the shelf above. "
                    "A laptop is now open on the right side of the desk."
                ),
                "camera_source": "eval_replay",
            },
            {
                "kind": "passive",
                "timestamp": "2026-01-15T09:02:00+09:00",
                "location": {"label": "home desk", "lat": 35.68, "lon": 139.76},
                "model_answer": "Scene shows a desk with an open laptop and a red bottle on the shelf.",
                "camera_source": "eval_replay",
            },
        ],
        "memory_questions": [
            {
                "id": "mq1",
                "query_time_sec": 0.0,
                "question": "Where did I put the red bottle?",
                "gold_answer": "on the shelf",
                "acceptable_answers": ["shelf", "on the shelf", "on the bookshelf", "the shelf"],
                "unacceptable_answers": ["desk", "floor", "table"],
                "answerable": True,
                "gold_evidence_windows": [[75.0, 120.0]],
            },
            {
                "id": "mq2",
                "query_time_sec": 0.0,
                "question": "Did I open the laptop?",
                "gold_answer": "yes",
                "acceptable_answers": ["yes", "the laptop was opened", "a laptop was open"],
                "unacceptable_answers": ["no"],
                "answerable": True,
                "gold_evidence_windows": [[25.0, 299.0]],
            },
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("Manifest written → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy eval dataset")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DATASETS_DIR,
        help="Output directory (default: evals/datasets/toy/)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=_DURATION_SEC,
        help="Video duration in seconds (default: 300)",
    )
    args = parser.parse_args()

    video_path = args.out_dir / "desk_001.mp4"
    manifest_path = args.out_dir / "desk_001.json"

    make_video(video_path)
    make_manifest(video_path, manifest_path)
    print(f"\nGenerated:\n  video   : {video_path}\n  manifest: {manifest_path}")
    print(
        "\nNext steps:\n"
        "  uv run python -m evals.run_live --manifest evals/datasets/toy/desk_001.json\n"
        "  uv run python -m evals.run_ltm  --manifest evals/datasets/toy/desk_001.json"
    )


if __name__ == "__main__":
    main()
