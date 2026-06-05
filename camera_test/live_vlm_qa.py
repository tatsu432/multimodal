import argparse
import base64
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from stream_config import add_stream_args, open_stream, resolve_stream


@dataclass
class FrameItem:
    timestamp: float
    frame: np.ndarray


class LiveFrameBuffer:
    def __init__(self, max_frames: int = 8):
        self.frames: Deque[FrameItem] = deque(maxlen=max_frames)
        self.lock = threading.Lock()

    def add(self, frame: np.ndarray) -> None:
        with self.lock:
            self.frames.append(FrameItem(timestamp=time.time(), frame=frame.copy()))

    def get_latest(self) -> List[FrameItem]:
        with self.lock:
            return list(self.frames)


def encode_frame_as_base64_jpeg(
    frame: np.ndarray,
    max_width: int = 768,
    quality: int = 85,
) -> str:
    """
    OpenCV frame is BGR.
    For JPEG encoding, BGR is okay because cv2.imencode expects OpenCV-style image arrays.
    """
    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    return base64.b64encode(buffer).decode("utf-8")


def capture_stream_loop(
    protocol: str,
    stream_url: str,
    buffer: LiveFrameBuffer,
    stop_event: threading.Event,
    sample_interval_sec: float = 1.0,
) -> None:
    """
    Continuously reads the live stream and stores sampled recent frames.
    """
    print(f"[capture] Opening {protocol.upper()} stream: {stream_url}")

    while not stop_event.is_set():
        cap = open_stream(stream_url)

        if not cap.isOpened():
            print("[capture] Could not open stream. Retrying in 2 seconds...")
            time.sleep(2)
            continue

        print("[capture] Stream opened.")
        last_sample_time = 0.0

        while not stop_event.is_set():
            ok, frame = cap.read()

            if not ok:
                print("[capture] Failed to read frame. Reconnecting...")
                break

            now = time.time()

            if now - last_sample_time >= sample_interval_sec:
                last_sample_time = now
                buffer.add(frame)

        cap.release()


def ask_vlm_about_recent_frames(
    client: OpenAI,
    model: str,
    question: str,
    frames: List[FrameItem],
    max_frames_to_send: int = 4,
) -> str:
    """
    Sends recent frames + user's text question to the VLM.
    """
    if not frames:
        return "No frames are available yet. Wait a few seconds and ask again."

    selected_frames = frames[-max_frames_to_send:]

    content = [
        {
            "type": "input_text",
            "text": (
                "You are answering questions about a live GoPro camera stream. "
                "Use only the visual evidence in the provided recent frames. "
                "If the answer is uncertain or not visible, say that clearly. "
                "Be concise but specific.\n\n"
                f"User question: {question}"
            ),
        }
    ]

    for i, item in enumerate(selected_frames):
        age_sec = time.time() - item.timestamp
        b64 = encode_frame_as_base64_jpeg(item.frame)

        content.append(
            {
                "type": "input_text",
                "text": f"Frame {i + 1}: captured about {age_sec:.1f} seconds ago.",
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            }
        )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    return response.output_text


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ask a VLM questions about a live RTMP or RTSP stream."
    )
    add_stream_args(parser)
    args = parser.parse_args()

    protocol, stream_url = resolve_stream(protocol=args.protocol, url=args.url)
    model = os.getenv("VLM_MODEL", "gpt-5.5")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    frame_buffer = LiveFrameBuffer(max_frames=8)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_stream_loop,
        args=(protocol, stream_url, frame_buffer, stop_event),
        kwargs={"sample_interval_sec": 1.0},
        daemon=True,
    )
    capture_thread.start()

    print(f"\nLive VLM QA started ({protocol.upper()}).")
    print("Ask questions like:")
    print("- What objects are visible?")
    print("- Is there a person in front of the camera?")
    print("- What changed in the last few seconds?")
    print("- Is the camera facing a desk, street, or room?")
    print("\nType 'q' or 'quit' to stop.\n")

    try:
        while True:
            question = input("You: ").strip()

            if question.lower() in {"q", "quit", "exit"}:
                break

            frames = frame_buffer.get_latest()

            print("VLM: thinking...")
            answer = ask_vlm_about_recent_frames(
                client=client,
                model=model,
                question=question,
                frames=frames,
                max_frames_to_send=4,
            )

            print(f"\nVLM: {answer}\n")

    finally:
        stop_event.set()
        print("Stopped.")


if __name__ == "__main__":
    main()