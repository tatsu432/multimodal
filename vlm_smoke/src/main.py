import logging
import sys
import time

from openai import OpenAIError
from providers.ollama import OllamaError

from src.config import Config
from src.frame_source import FrameSource, create_frame_source
from src.utils import save_query_frames
from src.vlm_client import VLMClient, create_vlm_client

logger = logging.getLogger("vlm_smoke.main")

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_repl(config: Config, source: FrameSource, vlm: VLMClient) -> None:
    config.frame_sample_dir.mkdir(parents=True, exist_ok=True)

    print("\nvlm_smoke: live visual QA started.")
    print(f"Frame source: {config.frame_source_type}")
    print("Example questions:")
    print("  - What do you see?")
    print("  - Is there a person?")
    print("  - What object is closest to the camera?")
    print("  - What text is visible?")
    print("\nWait a few seconds for frames to buffer before your first question.\n")

    while True:
        try:
            question = input(
                "Ask a question about the current view, or type 'q' to quit:\n> "
            ).strip()
        except EOFError:
            print()
            break

        if not question:
            continue

        if question.lower() in QUIT_COMMANDS:
            break

        try:
            _handle_question(config, source, vlm, question)
        except Exception as exc:
            logger.exception("Error handling question: %s", exc)
            print(f"\nError: {exc}\n")


def _handle_question(
    config: Config,
    source: FrameSource,
    vlm: VLMClient,
    question: str,
) -> None:
    num_frames = min(config.num_frames_per_query, config.frame_buffer_size)
    frames = source.get_recent(num_frames)

    if not frames:
        print(
            "No frames are available yet. Wait a few seconds and try again.\n"
        )
        logger.info("frame_read_ok=False num_frames_sent=0")
        return

    logger.info("frame_read_ok=True num_frames_sent=%d", len(frames))

    if config.save_queried_frames:
        try:
            saved_paths = save_query_frames(frames, config.frame_sample_dir)
            for path in saved_paths:
                logger.info("queried_frame_saved=%s", path)
        except (OSError, RuntimeError) as exc:
            logger.error("Failed to save queried frames: %s", exc)

    frame_items = None
    if hasattr(source, "get_recent_items"):
        frame_items = source.get_recent_items(num_frames)

    print("Assistant: thinking...")
    start = time.perf_counter()

    try:
        answer = vlm.answer_question(
            question=question,
            frames=frames,
            frame_items=frame_items,
        )
    except (OpenAIError, OllamaError) as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("VLM error after %.0f ms", latency_ms)
        print(f"\nAssistant: VLM error — {exc}\n")
        return
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.exception("Unexpected VLM error after %.0f ms", latency_ms)
        print(f"\nAssistant: Unexpected error — {exc}\n")
        return

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info("vlm_latency_ms=%.0f", latency_ms)
    print(f"\nAssistant: {answer}\n")


def main() -> None:
    setup_logging()
    config = Config.from_env()

    try:
        config.validate()
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    source = create_frame_source(config)
    vlm = create_vlm_client(config)

    try:
        source.start()
        run_repl(config, source, vlm)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        logger.info("Keyboard interrupt received")
    finally:
        source.stop()
        source.release()
        print("Stopped.")


if __name__ == "__main__":
    main()
