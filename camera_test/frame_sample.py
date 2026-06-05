import argparse
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

from stream_config import (
    StaleStreamDetector,
    add_source_args,
    describe_open_failure,
    frame_signature,
    open_source,
    read_frame,
    release_source,
    resolve_source,
    source_description,
)

SAVE_DIR = Path("sampled_frames")
INTERVAL_SEC = 2.0


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample frames from Tapo (RTSP/WebRTC) or smartphone (WebRTC)."
    )
    add_source_args(parser)
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview window (avoids macOS window focus stalls)",
    )
    args = parser.parse_args()

    camera, source_type, target = resolve_source(
        camera=args.camera,
        url=args.url,
    )
    label = source_description(camera, source_type, target)
    print(f"Opening {label}")

    SAVE_DIR.mkdir(exist_ok=True)

    frame_id = 0
    window_title = f"Camera sample — {camera}"
    cap = None

    try:
        while True:
            cap = open_source(source_type, target)
            if not cap.isOpened():
                print("Could not open source — retrying in 2s...")
                print(
                    describe_open_failure(
                        camera, source_type, target, label, capture=cap
                    )
                )
                release_source(cap)
                cap = None
                time.sleep(2)
                continue

            stale_detector = StaleStreamDetector()
            last_save_time = 0.0
            last_saved_sig: bytes | None = None
            read_failures = 0

            while True:
                ok, frame = read_frame(cap, source_type)
                if not ok:
                    read_failures += 1
                    if read_failures >= 5:
                        print("Too many read failures — reconnecting RTSP...")
                        break
                    time.sleep(0.2)
                    continue
                read_failures = 0

                stale_state = stale_detector.check(frame)
                if stale_state == "stale":
                    print(
                        "Stream frozen (identical frames) — reconnecting RTSP. "
                        "If the browser also paused, bring the phone publish tab to the foreground."
                    )
                    break

                now = time.time()
                if now - last_save_time >= INTERVAL_SEC:
                    sig = frame_signature(frame)
                    if sig == last_saved_sig:
                        print(
                            "Skipping save — duplicate frame (publisher or RTSP may be stalled)"
                        )
                    else:
                        last_save_time = now
                        last_saved_sig = sig
                        path = SAVE_DIR / f"frame_{frame_id:06d}.jpg"
                        cv2.imwrite(str(path), frame)
                        print(f"Saved {path}")
                        frame_id += 1

                if not args.no_preview:
                    cv2.imshow(window_title, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return
            release_source(cap)
            cap = None
            time.sleep(0.5)
    finally:
        release_source(cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
