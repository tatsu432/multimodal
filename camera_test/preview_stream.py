import argparse

import cv2
from dotenv import load_dotenv

from stream_config import (
    add_source_args,
    open_stream,
    resolve_source,
    source_description,
)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Preview a live RTMP/RTSP stream, webcam, or video file."
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

    cap = open_stream(target)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {label}")

    window_title = f"Camera preview — {source_type}"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if source_type == "video":
                    cap.release()
                    cap = open_stream(target)
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not reopen {label}")
                    continue

                print("Failed to read frame")
                break

            cv2.imshow(window_title, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
