import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI

from src.config import Config
from src.utils import FrameItem, encode_frame_as_base64_jpeg

logger = logging.getLogger("vlm_smoke.vlm")

SOURCE_PROMPTS = {
    "rtmp": "a live GoPro camera stream",
    "webcam": "a live webcam feed",
    "video": "a video recording",
}


class VLMClient(ABC):
    @abstractmethod
    def answer_question(
        self,
        question: str,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None = None,
    ) -> str:
        ...


class OpenAIVLMClient(VLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        frame_source_type: str,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.feed_description = SOURCE_PROMPTS.get(
            frame_source_type, "a live camera feed"
        )

    def answer_question(
        self,
        question: str,
        frames: list[np.ndarray],
        frame_items: list[FrameItem] | None = None,
    ) -> str:
        if not frames:
            return "No frames are available yet. Wait a few seconds and ask again."

        content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    f"You are answering questions about {self.feed_description}. "
                    "Use only the visual evidence in the provided recent frames. "
                    "If the answer is uncertain or not visible, say that clearly. "
                    "Be concise but specific.\n\n"
                    f"User question: {question}"
                ),
            }
        ]

        for i, frame in enumerate(frames):
            if frame_items and i < len(frame_items):
                age_sec = time.time() - frame_items[i].timestamp
                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"Frame {i + 1}: captured about {age_sec:.1f} seconds ago."
                        ),
                    }
                )
            else:
                content.append(
                    {
                        "type": "input_text",
                        "text": f"Frame {i + 1}.",
                    }
                )

            b64 = encode_frame_as_base64_jpeg(frame)
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )

        logger.info(
            "Calling VLM model=%s with %d frame(s)", self.model, len(frames)
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        return response.output_text


def create_vlm_client(config: Config) -> VLMClient:
    if config.vlm_provider == "openai":
        return OpenAIVLMClient(
            api_key=config.openai_api_key,
            model=config.vlm_model,
            frame_source_type=config.frame_source_type,
        )

    raise NotImplementedError(
        f"VLM provider {config.vlm_provider!r} is not implemented in Phase 1"
    )
