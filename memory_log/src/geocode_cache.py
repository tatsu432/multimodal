import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("memory_log.geocode_cache")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS geocode_cache (
    cache_key TEXT PRIMARY KEY,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    full_address TEXT,
    city TEXT,
    prefecture TEXT,
    country TEXT,
    postal_code TEXT,
    geocode_provider TEXT NOT NULL,
    geocoded_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class CachedGeocode:
    full_address: str | None
    city: str | None
    prefecture: str | None
    country: str | None
    postal_code: str | None
    geocode_provider: str
    geocoded_at: str


def cache_key_for_coordinates(lat: float, lon: float, *, precision: int = 4) -> str:
    return f"{round(lat, precision):.{precision}f},{round(lon, precision):.{precision}f}"


class GeocodeCache:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._ensure_schema()
        self._conn.commit()

    def _ensure_schema(self) -> None:
        columns = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(geocode_cache)").fetchall()
        }
        if columns and "full_address" not in columns:
            logger.info("Migrating geocode cache to Japan-specific schema")
            self._conn.execute("DROP TABLE geocode_cache")
        self._conn.execute(CREATE_TABLE_SQL)

    def get(self, lat: float, lon: float) -> CachedGeocode | None:
        key = cache_key_for_coordinates(lat, lon)
        with self._lock:
            row = self._conn.execute(
                """
                SELECT full_address, city, prefecture, country, postal_code,
                       geocode_provider, geocoded_at
                FROM geocode_cache
                WHERE cache_key = ?
                """,
                (key,),
            ).fetchone()

        if row is None:
            return None

        return CachedGeocode(
            full_address=row[0],
            city=row[1],
            prefecture=row[2],
            country=row[3],
            postal_code=row[4],
            geocode_provider=row[5],
            geocoded_at=row[6],
        )

    def get_payload(self, lat: float, lon: float) -> dict | None:
        key = cache_key_for_coordinates(lat, lon)
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM geocode_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()

        if row is None:
            return None

        try:
            payload = json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning("Invalid payload_json for cache key %s", key)
            return None

        return payload if isinstance(payload, dict) else None

    def put(
        self,
        lat: float,
        lon: float,
        *,
        full_address: str | None,
        city: str | None,
        prefecture: str | None,
        country: str | None,
        postal_code: str | None,
        geocode_provider: str,
        payload: dict,
    ) -> CachedGeocode:
        key = cache_key_for_coordinates(lat, lon)
        geocoded_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        payload_json = json.dumps(payload, ensure_ascii=False)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO geocode_cache (
                    cache_key, lat, lon, full_address, city, prefecture, country,
                    postal_code, geocode_provider, geocoded_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    lat = excluded.lat,
                    lon = excluded.lon,
                    full_address = excluded.full_address,
                    city = excluded.city,
                    prefecture = excluded.prefecture,
                    country = excluded.country,
                    postal_code = excluded.postal_code,
                    geocode_provider = excluded.geocode_provider,
                    geocoded_at = excluded.geocoded_at,
                    payload_json = excluded.payload_json
                """,
                (
                    key,
                    lat,
                    lon,
                    full_address,
                    city,
                    prefecture,
                    country,
                    postal_code,
                    geocode_provider,
                    geocoded_at,
                    payload_json,
                ),
            )
            self._conn.commit()

        logger.debug("Cached geocode for %s", key)
        return CachedGeocode(
            full_address=full_address,
            city=city,
            prefecture=prefecture,
            country=country,
            postal_code=postal_code,
            geocode_provider=geocode_provider,
            geocoded_at=geocoded_at,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
