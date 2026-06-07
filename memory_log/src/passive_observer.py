"""Passive observation writer — periodic background frame + location logging without VLM."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from capture.stream_config import add_source_args, configure_decode_logging
from providers.ollama import OllamaError

from src.config import Config, PROJECT_ROOT
from src.db_writer import SQLiteWriter, _to_utc_iso, _timezone_name
from src.frame_source import FrameSource, create_frame_source
from src.geocode_client import GeocodeClient
from src.location import (
    LocationSidecarStore,
    enrich_location_with_geocode,
    resolve_location,
)
from src.location_server import LocationServer
from src.memory_db import open_db
from src.utils import FrameItem, frame_capture_timestamp_iso, make_memory_id, relative_path, resize_frame, save_frame_image

logger = logging.getLogger("memory_log.passive_observer")

_THUMBNAIL_MAX_WIDTH = 128


def _compute_phash(frame: np.ndarray) -> str | None:
    try:
        import imagehash
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return str(imagehash.phash(pil_img))
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("phash computation failed: %s", exc)
        return None


def _save_thumbnail(frame: np.ndarray, directory: Path, obs_id: str) -> str | None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        thumb = resize_frame(frame, max_width=_THUMBNAIL_MAX_WIDTH)
        path = directory / f"{obs_id}_thumb.jpg"
        ok = cv2.imwrite(str(path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            return None
        return str(path)
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not save thumbnail: %s", exc)
        return None


class PassiveObserver:
    def __init__(
        self,
        config: Config,
        source: FrameSource,
        db_writer: SQLiteWriter,
        sidecar: LocationSidecarStore | None,
        geocode_client: GeocodeClient | None,
    ) -> None:
        self._config = config
        self._source = source
        self._db_writer = db_writer
        self._sidecar = sidecar
        self._geocode_client = geocode_client

    def run(self) -> None:
        interval = self._config.passive_observation_interval_sec
        print(f"\npassive_observer: logging every {interval:.0f}s. Ctrl+C to stop.")
        print(f"Frame source: {self._config.frame_source_type}")
        print(f"DB: {self._config.memory_db_path}\n")

        observations_written = 0
        last_tick = time.monotonic() - interval  # fire immediately on first loop

        while True:
            now = time.monotonic()
            if now - last_tick < interval:
                time.sleep(0.5)
                continue

            last_tick = now
            self._observe(observations_written)
            observations_written += 1

    def _observe(self, count: int) -> None:
        items: list[FrameItem] = []
        if hasattr(self._source, "get_recent_items"):
            items = self._source.get_recent_items(1)

        if not items:
            logger.debug("No frames available for passive observation")
            return

        frame_item = items[-1]
        frame = frame_item.frame
        frame_ts_local = frame_capture_timestamp_iso(frame_item.timestamp)
        frame_ts_utc = _to_utc_iso(frame_ts_local)

        memory_id, timestamp_local, _ = make_memory_id()
        obs_id = f"obs_{memory_id}"
        timestamp_utc = _to_utc_iso(timestamp_local)
        tz_name = _timezone_name(timestamp_local)

        location = resolve_location(
            self._config, self._sidecar, self._config.camera_source_key
        )
        location = enrich_location_with_geocode(
            self._config, location, self._geocode_client
        )

        frame_path: str | None = None
        thumbnail_path: str | None = None
        if self._config.passive_save_frames:
            try:
                saved = save_frame_image(
                    frame, self._config.passive_frame_dir, obs_id
                )
                frame_path = relative_path(saved, PROJECT_ROOT)
            except (OSError, RuntimeError) as exc:
                logger.warning("Could not save passive frame: %s", exc)

            thumb_abs = _save_thumbnail(
                frame, self._config.passive_frame_dir, obs_id
            )
            if thumb_abs:
                thumbnail_path = relative_path(Path(thumb_abs), PROJECT_ROOT)

        phash = _compute_phash(frame)

        try:
            self._db_writer.write_passive_observation(
                obs_id=obs_id,
                timestamp_utc=timestamp_utc,
                timestamp_local=timestamp_local,
                timezone_name=tz_name,
                camera_source=self._config.camera_source_key,
                location=location,
                frame_path=frame_path,
                thumbnail_path=thumbnail_path,
                frame_timestamp=frame_ts_local,
                phash=phash,
            )
            print(
                f"[{timestamp_local[:19]}] obs #{count + 1} "
                f"location={location.display_name()} "
                f"phash={phash or 'n/a'}"
            )
        except Exception as exc:
            logger.error("Failed to write passive observation: %s", exc)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    configure_decode_logging()

    parser = argparse.ArgumentParser(
        description="Passive observer — periodic frame + location logging without VLM."
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

    try:
        conn = open_db(config.memory_db_path)
        db_writer = SQLiteWriter(conn, PROJECT_ROOT)
    except Exception as exc:
        logger.error("Could not open SQLite DB: %s", exc)
        sys.exit(1)

    source = create_frame_source(config)

    try:
        source.start()
        observer = PassiveObserver(config, source, db_writer, sidecar, geocode_client)
        observer.run()
    except KeyboardInterrupt:
        print("\nPassive observer stopped.")
    finally:
        source.stop()
        source.release()
        if location_server is not None:
            location_server.stop()
        if geocode_client is not None:
            geocode_client.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
