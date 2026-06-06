import argparse
import time

import cv2
from dotenv import load_dotenv

from stream_config import (
    StaleStreamDetector,
    add_source_args,
    configure_decode_logging,
    describe_open_failure,
    open_source,
    read_frame,
    release_source,
    resolve_source,
    source_description,
)


def main() -> None:
    load_dotenv()
    configure_decode_logging()

    parser = argparse.ArgumentParser(
        description="Preview Tapo (RTSP/WebRTC) or smartphone (WebRTC) streams."
    )
    add_source_args(parser)
    args = parser.parse_args()

    camera, source_type, target = resolve_source(
        camera=args.camera,
        url=args.url,
    )
    label = source_description(camera, source_type, target)
    print(f"Opening {label}")

    window_title = f"Camera preview — {camera}"
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
            while True:
                ok, frame = read_frame(cap, source_type)
                if not ok:
                    print("Failed to read frame — reconnecting...")
                    release_source(cap)
                    cap = None
                    time.sleep(0.5)
                    break

                if stale_detector.check(frame) == "stale":
                    print("Stream frozen — reconnecting RTSP...")
                    release_source(cap)
                    cap = None
                    time.sleep(0.5)
                    break

                cv2.imshow(window_title, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
    finally:
        release_source(cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
