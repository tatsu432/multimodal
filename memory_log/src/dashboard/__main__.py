"""Entry point for the memory_log dashboard.

Usage (from memory_log/):
    uv run python -m src.dashboard
    uv run python -m src.dashboard --host 0.0.0.0 --port 8800
    uv run python -m src.dashboard --no-grounding
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

from dotenv import load_dotenv

from capture.stream_config import configure_decode_logging
from src.config import Config, PROJECT_ROOT
from src.dashboard.server import DashboardServer
from src.dashboard.service import DashboardService

logger = logging.getLogger("memory_log.dashboard")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    configure_decode_logging()

    parser = argparse.ArgumentParser(description="memory_log browser dashboard")
    parser.add_argument("--host", help="Override DASHBOARD_HOST")
    parser.add_argument("--port", type=int, help="Override DASHBOARD_PORT")
    parser.add_argument(
        "--no-grounding",
        action="store_true",
        default=False,
        help="Disable visual grounding for LTM queries (no live camera required for LTM).",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if args.host:
        config.dashboard_host = args.host
    if args.port:
        config.dashboard_port = args.port

    try:
        config.validate()
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)

    service = DashboardService(config, no_grounding=args.no_grounding)
    server = DashboardServer(service, config)

    scheme = "https" if (config.dashboard_cert and config.dashboard_key) else "http"
    url = f"{scheme}://{config.dashboard_host}:{config.dashboard_port}/"

    stop_event = threading.Event()

    try:
        service.start()
        server.start()
        print(f"\nmemory_log dashboard running at: {url}")
        print("Press Ctrl-C to stop.\n")
        stop_event.wait()  # block until KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        server.stop()
        service.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
