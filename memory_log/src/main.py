import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

from capture.stream_config import add_source_args, configure_decode_logging
from openai import OpenAIError
from providers.ollama import OllamaError, ensure_model

from src.config import Config
from src.frame_source import FrameSource, create_frame_source
from src.memory_writer import MemoryWriter
from src.vlm_client import VLMClient, create_vlm_client

logger = logging.getLogger("memory_log.main")

QUIT_COMMANDS = frozenset({"q", "quit", "exit"})


def read_user_question() -> str:
    """Prompt on stdout — fd 2 may be redirected to rtsp_decode.log for FFmpeg."""
    sys.stdout.write(
        "Ask a question about the current view, or type 'q' to quit:\n> "
    )
    sys.stdout.flush()
    return input().strip()


@dataclass
class RunStats:
    questions_asked: int = 0
    memories_written: int = 0
    vlm_failures: int = 0
    json_parse_failures: int = 0
    question_latencies: list[float] = field(default_factory=list)

    @property
    def average_vlm_latency_seconds(self) -> float:
        if not self.question_latencies:
            return 0.0
        return sum(self.question_latencies) / len(self.question_latencies)


def _format_objects(objects: list[str]) -> str:
    return ",".join(objects) if objects else "-"


def _log_memory_line(record, latency_sec: float) -> None:
    ts_short = record.timestamp
    if len(ts_short) > 19:
        ts_short = ts_short[:19] + record.timestamp[19:]

    print(
        f"[{ts_short}] "
        f"stored={str(record.should_store).lower()} "
        f"privacy={record.privacy_risk} "
        f"scene={record.scene_type} "
        f"objects={_format_objects(record.objects)} "
        f"latency={latency_sec:.2f}s"
    )


def _handle_question(
    config: Config,
    source: FrameSource,
    vlm: VLMClient,
    writer: MemoryWriter,
    question: str,
    stats: RunStats,
) -> None:
    num_frames = min(config.num_frames_per_query, config.frame_buffer_size)
    frames = source.get_recent(num_frames)

    if not frames:
        print(
            "No frames are available yet. Wait a few seconds and try again.\n"
        )
        return

    frame_items = None
    if hasattr(source, "get_recent_items"):
        frame_items = source.get_recent_items(num_frames)

    stats.questions_asked += 1
    question_start = time.perf_counter()

    print("Assistant: thinking...")
    try:
        answer = vlm.answer_question(
            question=question,
            frames=frames,
            frame_items=frame_items,
        )
    except (OpenAIError, OllamaError) as exc:
        stats.vlm_failures += 1
        logger.error("VLM Q&A error: %s", exc)
        print(f"\nAssistant: VLM error — {exc}\n")
        return
    except Exception as exc:
        stats.vlm_failures += 1
        logger.exception("Unexpected VLM Q&A error: %s", exc)
        print(f"\nAssistant: Unexpected error — {exc}\n")
        return

    print(f"\nAssistant: {answer}\n")

    try:
        analysis = vlm.analyze_frame_for_memory(frames[-1])
    except (OpenAIError, OllamaError) as exc:
        stats.vlm_failures += 1
        logger.error("VLM memory error: %s", exc)
        print(f"Memory: VLM error — {exc}\n")
        return
    except Exception as exc:
        stats.vlm_failures += 1
        logger.exception("Unexpected VLM memory error: %s", exc)
        print(f"Memory: Unexpected error — {exc}\n")
        return

    latency_sec = time.perf_counter() - question_start
    stats.question_latencies.append(latency_sec)

    if analysis is None:
        stats.json_parse_failures += 1
        logger.warning("Skipping memory: invalid or unparseable VLM JSON")
        print("Memory: skipped (could not parse structured response)\n")
        return

    try:
        record = writer.save_memory(frames[-1], analysis, user_question=question)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Failed to save memory: %s", exc)
        print(f"Memory: save error — {exc}\n")
        return

    if record.should_store:
        stats.memories_written += 1

    _log_memory_line(record, latency_sec)


def run_repl(
    config: Config,
    source: FrameSource,
    vlm: VLMClient,
    writer: MemoryWriter,
) -> RunStats:
    stats = RunStats()
    start_time = time.monotonic()

    print("\nmemory_log: question-driven visual memory started.")
    print(f"Frame source: {config.frame_source_type}")
    print(f"Memories append to: {config.memory_jsonl_path}")
    print("Example questions:")
    print("  - What do you see?")
    print("  - Is there a person?")
    print("  - What text is visible?")
    print("\nWait a few seconds for frames to buffer before your first question.")
    print("Memories are written only when you ask a question.\n")

    while True:
        if config.max_runtime_seconds is not None:
            elapsed = time.monotonic() - start_time
            if elapsed >= config.max_runtime_seconds:
                logger.info("MAX_RUNTIME_SECONDS reached (%.1fs)", elapsed)
                break

        try:
            question = read_user_question()
        except EOFError:
            print()
            break

        if not question:
            continue

        if question.lower() in QUIT_COMMANDS:
            break

        try:
            _handle_question(config, source, vlm, writer, question, stats)
        except Exception as exc:
            logger.exception("Error handling question: %s", exc)
            print(f"\nError: {exc}\n")

    return stats


def print_run_summary(stats: RunStats) -> None:
    print("\nRun summary:")
    print(f"- questions_asked: {stats.questions_asked}")
    print(f"- memories_written: {stats.memories_written}")
    print(f"- vlm_failures: {stats.vlm_failures}")
    print(f"- json_parse_failures: {stats.json_parse_failures}")
    print(
        f"- average_vlm_latency_seconds: {stats.average_vlm_latency_seconds:.2f}"
    )


def main() -> None:
    from dotenv import load_dotenv
    from src.config import PROJECT_ROOT

    load_dotenv(PROJECT_ROOT / ".env")
    configure_decode_logging()

    parser = argparse.ArgumentParser(
        description="Visual memory logging — GoPro/webcam/video or Tapo/phone cameras."
    )
    add_source_args(parser)
    args = parser.parse_args()

    config = Config.from_env()
    if args.camera:
        config.camera_preset_override = args.camera
    if args.url:
        config.camera_url_override = args.url

    try:
        config.validate()
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if config.vlm_provider == "ollama":
        try:
            ensure_model(config.vlm_model, base_url=config.ollama_base_url)
        except OllamaError as exc:
            logger.error("%s", exc)
            sys.exit(1)

    source = create_frame_source(config)
    vlm = create_vlm_client(config)
    writer = MemoryWriter(config)

    stats = RunStats()
    try:
        source.start()
        stats = run_repl(config, source, vlm, writer)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        logger.info("Keyboard interrupt received")
    finally:
        source.stop()
        source.release()
        print_run_summary(stats)
        print("Stopped.")


if __name__ == "__main__":
    main()
