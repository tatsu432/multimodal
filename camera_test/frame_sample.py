import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from stream_config import (
    add_source_args,
    describe_open_failure,
    open_source,
    resolve_source,
    source_description,
)

SAVE_DIR = Path("sampled_frames")
INTERVAL_SEC = 2.0


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample frames from RTMP, RTSP, WebRTC, webcam, or video."
    )
    add_source_args(parser)
    args = parser.parse_args()

    source_type, target = resolve_source(
        source_type=args.source_type,
        protocol=args.protocol,
        url=args.url,
    )
    label = source_description(source_type, target)
    print(f"Opening {label}")

    SAVE_DIR.mkdir(exist_ok=True)

    cap = open_source(source_type, target)

    if not cap.isOpened():
        raise RuntimeError(
            describe_open_failure(source_type, target, label, capture=cap)
        )

    import cv2  # after WebRTC/aiortc init to reduce macOS FFmpeg load-order issues

    last_save_time = 0.0
    frame_id = 0
    window_title = f"Camera sample — {source_type}"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if source_type == "video":
                    cap.release()
                    cap = open_source(source_type, target)
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not reopen {label}")
                    continue

                print("Failed to read frame")
                time.sleep(1)
                continue

            now = time.time()

            if now - last_save_time >= INTERVAL_SEC:
                last_save_time = now

                path = SAVE_DIR / f"frame_{frame_id:06d}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Saved {path}")

                frame_id += 1

            cv2.imshow(window_title, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
