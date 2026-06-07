"""Combined entry point: live Q&A + passive observation in one process.

The wearable "power-on" switch — starts the interactive REPL and the
background passive observer together, sharing a single camera source,
location server, geocode client, and memory DB connection.

Usage:
    uv run python -m src.run_all               # live QA + passive memory
    uv run python -m src.run_all --no-passive  # live QA only (same as src.main)
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from capture.stream_config import add_source_args, configure_decode_logging
from providers.ollama import OllamaError, ensure_model

from src.config import Config, PROJECT_ROOT
from src.db_writer import SQLiteWriter
from src.frame_source import create_frame_source
from src.geocode_client import GeocodeClient
from src.location import LocationSidecarStore
from src.location_server import LocationServer
from src.main import RunStats, print_run_summary, run_repl
from src.memory_db import open_db
from src.memory_writer import MemoryWriter
from src.passive_observer import PassiveObserver
from src.vlm_client import create_vlm_client

logger = logging.getLogger("memory_log.run_all")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    configure_decode_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Wearable assistant: live Q&A + passive background memory logging. "
            "Shares one camera, location server, and memory DB."
        )
    )
    add_source_args(parser)
    parser.add_argument(
        "--no-passive",
        action="store_true",
        default=False,
        help="Disable passive background observer (same as running src.main directly).",
    )
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

    # --- shared resources -------------------------------------------------------
    sidecar: LocationSidecarStore | None = None
    location_server: LocationServer | None = None
    geocode_client: GeocodeClient | None = None

    if config.location_server_enabled:
        sidecar = LocationSidecarStore(max_age_sec=config.location_gps_max_age_sec)
        location_server = LocationServer(
            store=sidecar,
            host=config.location_server_host,
            port=config.location_server_port,
            cert_path=config.location_server_cert,
            key_path=config.location_server_key,
        )
        location_server.start()

    if config.geocode_enabled:
        geocode_client = GeocodeClient(config)

    source = create_frame_source(config)
    vlm = create_vlm_client(config)
    writer = MemoryWriter(config)

    indexer = None
    if config.vector_search_enabled:
        try:
            from src.vector_index import create_memory_indexer
            indexer = create_memory_indexer(config)
        except Exception as exc:
            logger.warning("Could not create memory indexer: %s", exc)

    conn = None
    db_writer: SQLiteWriter | None = None
    try:
        conn = open_db(config.memory_db_path)
        db_writer = SQLiteWriter(conn, PROJECT_ROOT, indexer=indexer)
        logger.info("SQLite memory DB opened: %s", config.memory_db_path)
    except Exception as exc:
        logger.warning("Could not open SQLite memory DB (JSONL-only mode): %s", exc)

    # --- passive observer thread ------------------------------------------------
    passive_thread: threading.Thread | None = None
    stop_event = threading.Event()

    if not args.no_passive:
        if db_writer is None:
            logger.warning(
                "Passive observer disabled: SQLite DB unavailable (JSONL-only mode)."
            )
        else:
            observer = PassiveObserver(
                config,
                source,
                db_writer,
                sidecar,
                geocode_client,
                stop_event=stop_event,
                quiet=True,
            )
            passive_thread = threading.Thread(
                target=observer.run,
                name="passive-observer",
                daemon=True,
            )

    # --- start and run ----------------------------------------------------------
    stats = RunStats()
    try:
        source.start()
        if passive_thread is not None:
            passive_thread.start()
            logger.info(
                "Passive observer started (interval=%.0fs).",
                config.passive_observation_interval_sec,
            )
        stats = run_repl(
            config, source, vlm, writer, sidecar, geocode_client, db_writer=db_writer
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        logger.info("Keyboard interrupt received")
    finally:
        # Signal passive thread before releasing source/DB so it can exit cleanly.
        stop_event.set()
        if passive_thread is not None and passive_thread.is_alive():
            passive_thread.join(
                timeout=config.passive_observation_interval_sec + 2.0
            )
        source.stop()
        source.release()
        if location_server is not None:
            location_server.stop()
        if geocode_client is not None:
            geocode_client.close()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        print_run_summary(stats)
        print("Stopped.")


if __name__ == "__main__":
    main()
