import argparse

import cv2
from dotenv import load_dotenv

from stream_config import add_stream_args, open_stream, resolve_stream


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Preview a live RTMP or RTSP stream.")
    add_stream_args(parser)
    args = parser.parse_args()

    protocol, stream_url = resolve_stream(protocol=args.protocol, url=args.url)
    print(f"Opening {protocol.upper()} stream: {stream_url}")

    cap = open_stream(stream_url)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {protocol.upper()} stream: {stream_url}")

    window_title = f"GoPro {protocol.upper()} Stream"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        cv2.imshow(window_title, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
