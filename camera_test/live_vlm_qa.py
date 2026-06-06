import argparse
import base64
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from providers.ollama import OllamaError, chat as ollama_chat, ensure_model

from stream_config import (
    StaleStreamDetector,
    add_source_args,
    configure_decode_logging,
    frame_signature,
    open_source,
    read_frame,
    release_source,
    resolve_source,
    source_description,
)


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
    camera: str,
    source_type: str,
    target: str,
    buffer: LiveFrameBuffer,
    stop_event: threading.Event,
    sample_interval_sec: float = 1.0,
) -> None:
    """
    Continuously reads frames from the configured source and stores recent samples.
    """
    label = source_description(camera, source_type, target)
    print(f"[capture] Opening {label}")

    while not stop_event.is_set():
        cap = open_source(source_type, target)

        if not cap.isOpened():
            print("[capture] Could not open source. Retrying in 2 seconds...")
            time.sleep(2)
            continue

        print("[capture] Source opened.")
        stale_detector = StaleStreamDetector()
        last_sample_time = 0.0
        last_added_sig: bytes | None = None
        read_failures = 0

        while not stop_event.is_set():
            ok, frame = read_frame(cap, source_type)

            if not ok:
                read_failures += 1
                if read_failures >= 5:
                    print("[capture] Too many read failures. Reconnecting...")
                    break
                time.sleep(0.2)
                continue
            read_failures = 0

            if stale_detector.check(frame) == "stale":
                print("[capture] Stream frozen — reconnecting...")
                break

            now = time.time()

            if now - last_sample_time >= sample_interval_sec:
                sig = frame_signature(frame)
                if sig != last_added_sig:
                    last_sample_time = now
                    last_added_sig = sig
                    buffer.add(frame)

        release_source(cap)


def _build_question_prompt(question: str, frames: List[FrameItem]) -> tuple[str, list[str]]:
    prompt_parts = [
        (
            "You are answering questions about a live camera stream. "
            "Use only the visual evidence in the provided recent frames. "
            "If the answer is uncertain or not visible, say that clearly. "
            "Be concise but specific."
        ),
        f"User question: {question}",
    ]
    images: list[str] = []

    for i, item in enumerate(frames):
        age_sec = time.time() - item.timestamp
        prompt_parts.append(
            f"Frame {i + 1}: captured about {age_sec:.1f} seconds ago."
        )
        images.append(encode_frame_as_base64_jpeg(item.frame))

    return "\n\n".join(prompt_parts), images


def ask_vlm_about_recent_frames(
    *,
    provider: str,
    model: str,
    question: str,
    frames: List[FrameItem],
    max_frames_to_send: int = 4,
    openai_client: OpenAI | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> str:
    """
    Sends recent frames + user's text question to the configured VLM provider.
    """
    if not frames:
        return "No frames are available yet. Wait a few seconds and ask again."

    selected_frames = frames[-max_frames_to_send:]
    prompt, images = _build_question_prompt(question, selected_frames)

    if provider == "openai":
        if openai_client is None:
            raise ValueError("OpenAI client is required when VLM_PROVIDER=openai")

        content: list[dict] = [{"type": "input_text", "text": prompt}]
        for b64 in images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )

        response = openai_client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        return response.output_text

    if provider == "ollama":
        return ollama_chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": images,
                }
            ],
            base_url=ollama_base_url,
        )

    raise ValueError(f"Unsupported VLM provider: {provider!r}")


def read_user_question() -> str:
    """Prompt on stdout — never stderr (fd 2 may be redirected to rtsp_decode.log)."""
    sys.stdout.write("\nYou: ")
    sys.stdout.flush()
    return input().strip()


def main() -> None:
    load_dotenv()
    configure_decode_logging()

    parser = argparse.ArgumentParser(
        description="Ask a VLM questions about Tapo or smartphone camera streams."
    )
    add_source_args(parser)
    args = parser.parse_args()

    camera, source_type, target = resolve_source(
        camera=args.camera,
        url=args.url,
    )
    provider = os.getenv("VLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("VLM_MODEL", "gpt-5.5")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

    if provider not in {"openai", "ollama"}:
        raise ValueError("VLM_PROVIDER must be 'openai' or 'ollama'")

    openai_client: OpenAI | None = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when VLM_PROVIDER=openai. "
                "Set VLM_PROVIDER=ollama to use a local model instead."
            )
        openai_client = OpenAI(api_key=api_key)
    elif provider == "ollama":
        ensure_model(model, base_url=ollama_base_url)

    frame_buffer = LiveFrameBuffer(max_frames=8)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_stream_loop,
        args=(camera, source_type, target, frame_buffer, stop_event),
        kwargs={"sample_interval_sec": 1.0},
        daemon=True,
    )
    capture_thread.start()

    label = source_description(camera, source_type, target)
    print(f"\nLive VLM QA started ({label}, provider={provider}).")
    print("Ask questions like:")
    print("- What objects are visible?")
    print("- Is there a person in front of the camera?")
    print("- What changed in the last few seconds?")
    print("- Is the camera facing a desk, street, or room?")
    print("\nType 'q' or 'quit' to stop.")

    while not frame_buffer.get_latest():
        time.sleep(0.1)

    try:
        while True:
            question = read_user_question()

            if question.lower() in {"q", "quit", "exit"}:
                break

            frames = frame_buffer.get_latest()

            print("VLM: thinking...")
            try:
                answer = ask_vlm_about_recent_frames(
                    provider=provider,
                    model=model,
                    question=question,
                    frames=frames,
                    max_frames_to_send=4,
                    openai_client=openai_client,
                    ollama_base_url=ollama_base_url,
                )
            except OllamaError as exc:
                print(f"\nVLM error: {exc}\n")
                continue

            print(f"\nVLM: {answer}\n")

    finally:
        stop_event.set()
        print("Stopped.")


if __name__ == "__main__":
    main()
