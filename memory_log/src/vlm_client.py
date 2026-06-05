import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI
from providers.ollama import chat as ollama_chat

from src.config import Config
from src.utils import FrameItem, encode_frame_as_base64_jpeg, parse_vlm_memory_analysis

logger = logging.getLogger("memory_log.vlm")

MEMORY_ANALYSIS_PROMPT = """Analyze this image frame for a wearable visual memory system.

Return ONLY valid JSON with the following schema:
{
  "summary": "short natural-language description of the scene",
  "scene_type": "short snake_case scene category",
  "objects": ["object1", "object2"],
  "people_count": 0,
  "text_visible": ["visible text if any"],
  "should_store": true,
  "memory_reason": "why this frame should or should not be stored",
  "privacy_risk": "low|medium|high"
}

Guidelines:
- Use concise descriptions.
- Include only objects clearly visible.
- If the frame is blurry or meaningless, set should_store to false.
- If private documents, faces, IDs, screens with sensitive info, or personal information are visible, increase privacy_risk.
- Return JSON only. No markdown. No explanation."""

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

    @abstractmethod
    def analyze_frame_for_memory(self, frame: np.ndarray) -> dict | None:
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
            "Calling VLM model=%s for Q&A with %d frame(s)", self.model, len(frames)
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

    def analyze_frame_for_memory(self, frame: np.ndarray) -> dict | None:
        b64 = encode_frame_as_base64_jpeg(frame)

        content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    f"This frame is from {self.feed_description}.\n\n"
                    f"{MEMORY_ANALYSIS_PROMPT}"
                ),
            },
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            },
        ]

        logger.info("Calling VLM model=%s for memory analysis", self.model)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )

        raw_text = response.output_text
        return parse_vlm_memory_analysis(raw_text)


class OllamaVLMClient(VLMClient):
    def __init__(
        self,
        model: str,
        frame_source_type: str,
        base_url: str,
    ):
        self.model = model
        self.base_url = base_url
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

        prompt_parts = [
            (
                f"You are answering questions about {self.feed_description}. "
                "Use only the visual evidence in the provided recent frames. "
                "If the answer is uncertain or not visible, say that clearly. "
                "Be concise but specific."
            ),
            f"User question: {question}",
        ]
        images: list[str] = []

        for i, frame in enumerate(frames):
            if frame_items and i < len(frame_items):
                age_sec = time.time() - frame_items[i].timestamp
                prompt_parts.append(
                    f"Frame {i + 1}: captured about {age_sec:.1f} seconds ago."
                )
            else:
                prompt_parts.append(f"Frame {i + 1}.")

            images.append(encode_frame_as_base64_jpeg(frame))

        logger.info(
            "Calling Ollama model=%s for Q&A with %d frame(s)",
            self.model,
            len(frames),
        )
        return ollama_chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "\n\n".join(prompt_parts),
                    "images": images,
                }
            ],
            base_url=self.base_url,
        )

    def analyze_frame_for_memory(self, frame: np.ndarray) -> dict | None:
        b64 = encode_frame_as_base64_jpeg(frame)
        prompt = (
            f"This frame is from {self.feed_description}.\n\n"
            f"{MEMORY_ANALYSIS_PROMPT}"
        )

        logger.info("Calling Ollama model=%s for memory analysis", self.model)
        raw_text = ollama_chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [b64],
                }
            ],
            base_url=self.base_url,
        )
        return parse_vlm_memory_analysis(raw_text)


def create_vlm_client(config: Config) -> VLMClient:
    if config.vlm_provider == "openai":
        return OpenAIVLMClient(
            api_key=config.openai_api_key,
            model=config.vlm_model,
            frame_source_type=config.frame_source_type,
        )

    if config.vlm_provider == "ollama":
        return OllamaVLMClient(
            model=config.vlm_model,
            frame_source_type=config.frame_source_type,
            base_url=config.ollama_base_url,
        )

    raise ValueError(f"Unsupported VLM provider: {config.vlm_provider!r}")
