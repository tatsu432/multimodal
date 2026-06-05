import argparse
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

from stream_config import add_stream_args, open_stream, resolve_stream

SAVE_DIR = Path("sampled_frames")
INTERVAL_SEC = 2.0


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Sample frames from a live RTMP or RTSP stream."
    )
    add_stream_args(parser)
    args = parser.parse_args()

    protocol, stream_url = resolve_stream(protocol=args.protocol, url=args.url)
    print(f"Opening {protocol.upper()} stream: {stream_url}")

    SAVE_DIR.mkdir(exist_ok=True)

    cap = open_stream(stream_url)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {protocol.upper()} stream: {stream_url}")

    last_save_time = 0.0
    frame_id = 0
    window_title = f"GoPro {protocol.upper()}"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            time.sleep(1)
            continue

        now = time.time()

        if now - last_save_time >= INTERVAL_SEC:
            last_save_time = now

            path = SAVE_DIR / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path}")

            # Later: send this frame to a VLM
            frame_id += 1

        cv2.imshow(window_title, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
