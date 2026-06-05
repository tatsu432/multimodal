"""Subprocess WHEP worker — aiortc only; streams BGR frames on stdout."""

from __future__ import annotations

import argparse
import asyncio
import logging
import struct
import sys
from typing import Any

import numpy as np

from whep_client import (
    ice_servers_from_dicts,
    ice_servers_to_dicts,
    parse_ice_servers,
    run_whep_stream,
)

logger = logging.getLogger(__name__)

_FRAME_MAGIC = b"WFRM"
_ERROR_MAGIC = b"WERR"
_HEADER = struct.Struct(">4sIII")  # magic, height, width, payload_len


def _write_frame(frame: np.ndarray) -> None:
    payload = frame.tobytes()
    header = _HEADER.pack(_FRAME_MAGIC, frame.shape[0], frame.shape[1], len(payload))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _write_error(message: str) -> None:
    data = message.encode("utf-8")
    header = _HEADER.pack(_ERROR_MAGIC, 0, 0, len(data))
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


async def _run(url: str, ice_data: list[dict[str, Any]], timeout_sec: float) -> int:
    ice_servers = ice_servers_from_dicts(ice_data)
    stop = asyncio.Event()

    def on_frame(frame: np.ndarray) -> None:
        _write_frame(frame)

    try:
        await run_whep_stream(
            url,
            ice_servers=ice_servers,
            open_timeout_sec=timeout_sec,
            stop_event=stop,
            on_frame=on_frame,
        )
        return 0
    except Exception as exc:
        logger.exception("WHEP worker failed")
        _write_error(str(exc))
        return 1


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    parser = argparse.ArgumentParser(description="WHEP worker (stdout frame protocol)")
    parser.add_argument("--url", required=True, help="WHEP endpoint URL")
    parser.add_argument("--timeout", type=float, default=15.0, help="Open/ICE timeout")
    parser.add_argument(
        "--ice-servers",
        default="",
        help="Comma-separated ICE URLs (empty = fetch via OPTIONS)",
    )
    args = parser.parse_args()

    ice_data = ice_servers_to_dicts(parse_ice_servers(args.ice_servers or None))

    code = asyncio.run(_run(args.url, ice_data, args.timeout))
    sys.exit(code)


if __name__ == "__main__":
    main()
