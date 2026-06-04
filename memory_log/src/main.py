import logging
import sys
import time
from dataclasses import dataclass, field

from openai import OpenAIError

from src.config import Config
from src.frame_source import FrameSource, VideoFileFrameSource, create_frame_source
from src.memory_writer import MemoryWriter
from src.vlm_client import VLMClient, create_vlm_client

logger = logging.getLogger("memory_log.main")

LOOP_SLEEP_SEC = 0.02


@dataclass
class RunStats:
    frames_read: int = 0
    frames_sampled: int = 0
    memories_written: int = 0
    vlm_failures: int = 0
    json_parse_failures: int = 0
    vlm_latencies: list[float] = field(default_factory=list)

    @property
    def average_vlm_latency_seconds(self) -> float:
        if not self.vlm_latencies:
            return 0.0
        return sum(self.vlm_latencies) / len(self.vlm_latencies)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def run_memory_loop(
    config: Config,
    source: FrameSource,
    vlm: VLMClient,
    writer: MemoryWriter,
) -> RunStats:
    stats = RunStats()
    start_time = time.monotonic()
    last_sample_time = 0.0

    print("\nmemory_log: visual memory recording started.")
    print(f"Frame source: {config.frame_source_type}")
    print(
        f"Sampling every {config.frame_sample_interval_seconds:.1f}s "
        f"→ {config.memory_jsonl_path}"
    )
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if config.max_runtime_seconds is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= config.max_runtime_seconds:
                    logger.info("MAX_RUNTIME_SECONDS reached (%.1fs)", elapsed)
                    break

            if isinstance(source, VideoFileFrameSource) and source.stream_ended:
                logger.info("Video file ended; stopping")
                break

            ok, frame = source.read()
            stats.frames_read += 1

            if not ok or frame is None:
                time.sleep(LOOP_SLEEP_SEC)
                continue

            now = time.monotonic()
            if now - last_sample_time < config.frame_sample_interval_seconds:
                time.sleep(LOOP_SLEEP_SEC)
                continue

            last_sample_time = now
            stats.frames_sampled += 1

            vlm_start = time.perf_counter()
            try:
                analysis = vlm.analyze_frame_for_memory(frame)
            except OpenAIError as exc:
                stats.vlm_failures += 1
                logger.error("VLM API error: %s", exc)
                continue
            except Exception as exc:
                stats.vlm_failures += 1
                logger.exception("Unexpected VLM error: %s", exc)
                continue

            latency_sec = time.perf_counter() - vlm_start
            stats.vlm_latencies.append(latency_sec)

            if analysis is None:
                stats.json_parse_failures += 1
                logger.warning("Skipping frame: invalid or unparseable VLM JSON")
                continue

            try:
                record = writer.save_memory(frame, analysis)
            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("Failed to save memory: %s", exc)
                continue

            stats.memories_written += 1
            _log_memory_line(record, latency_sec)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        logger.info("Keyboard interrupt in memory loop")

    return stats


def print_run_summary(stats: RunStats) -> None:
    print("\nRun summary:")
    print(f"- frames_read: {stats.frames_read}")
    print(f"- frames_sampled: {stats.frames_sampled}")
    print(f"- memories_written: {stats.memories_written}")
    print(f"- vlm_failures: {stats.vlm_failures}")
    print(f"- json_parse_failures: {stats.json_parse_failures}")
    print(
        f"- average_vlm_latency_seconds: {stats.average_vlm_latency_seconds:.2f}"
    )


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
    writer = MemoryWriter(config)

    stats = RunStats()
    try:
        stats = run_memory_loop(config, source, vlm, writer)
    finally:
        source.release()
        print_run_summary(stats)
        print("Stopped.")


if __name__ == "__main__":
    main()
