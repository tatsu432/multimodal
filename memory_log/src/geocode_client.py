import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

from src.config import Config
from src.geocode_cache import CachedGeocode, GeocodeCache

logger = logging.getLogger("memory_log.geocode_client")

USER_AGENT = "memory_log/0.1 (local wearable memory prototype)"
_JP_POSTAL_RE = re.compile(r"^\d{7}$")


@dataclass(frozen=True)
class GeocodeResult:
    full_address: str | None
    city: str | None
    prefecture: str | None
    country: str | None
    postal_code: str | None
    geocode_provider: str
    geocoded_at: str


def _first_address_value(address: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = address.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_jp_postal_code(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if _JP_POSTAL_RE.match(digits):
        return f"{digits[:3]}-{digits[3:]}"
    return value.strip()


def _prefecture_from_display_name(
    display_name: str | None,
    postal_code: str | None,
) -> str | None:
    """Nominatim often omits province/state for Japan; it appears in display_name."""
    if not display_name or not postal_code:
        return None

    parts = [part.strip() for part in display_name.split(",") if part.strip()]
    if len(parts) < 2:
        return None

    normalized_postal = postal_code.replace("-", "")
    for index, part in enumerate(parts):
        if normalized_postal in part.replace("-", "").replace(" ", ""):
            return parts[index - 1]
    return None


def parse_nominatim_response(payload: dict) -> GeocodeResult:
    """Map Nominatim/OSM address fields to Japan-oriented location metadata."""
    address_obj = payload.get("address")
    if not isinstance(address_obj, dict):
        address_obj = {}

    full_address = str(payload.get("display_name", "")).strip() or None
    postal_code = normalize_jp_postal_code(
        _first_address_value(address_obj, ("postcode",))
    )
    prefecture = _first_address_value(
        address_obj,
        ("province", "state", "region"),
    )
    if prefecture is None:
        prefecture = _prefecture_from_display_name(full_address, postal_code)
    city = _first_address_value(
        address_obj,
        (
            "city_district",
            "city",
            "town",
            "village",
            "municipality",
            "suburb",
            "neighbourhood",
            "quarter",
        ),
    )
    country = _first_address_value(address_obj, ("country",))

    return GeocodeResult(
        full_address=full_address,
        city=city,
        prefecture=prefecture,
        country=country,
        postal_code=postal_code,
        geocode_provider="nominatim",
        geocoded_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
    )


class GeocodeClient:
    def __init__(self, config: Config):
        self.config = config
        self.cache = GeocodeCache(config.geocode_cache_path)
        self._last_request_at = 0.0

    def close(self) -> None:
        self.cache.close()

    def _result_from_cached(
        self,
        lat: float,
        lon: float,
        cached: CachedGeocode,
    ) -> GeocodeResult:
        if cached.prefecture is not None:
            return GeocodeResult(
                full_address=cached.full_address,
                city=cached.city,
                prefecture=cached.prefecture,
                country=cached.country,
                postal_code=cached.postal_code,
                geocode_provider=cached.geocode_provider,
                geocoded_at=cached.geocoded_at,
            )

        payload = self.cache.get_payload(lat, lon)
        if payload is None:
            return GeocodeResult(
                full_address=cached.full_address,
                city=cached.city,
                prefecture=cached.prefecture,
                country=cached.country,
                postal_code=cached.postal_code,
                geocode_provider=cached.geocode_provider,
                geocoded_at=cached.geocoded_at,
            )

        reparsed = parse_nominatim_response(payload)
        if reparsed.prefecture is None:
            return GeocodeResult(
                full_address=cached.full_address,
                city=cached.city,
                prefecture=cached.prefecture,
                country=cached.country,
                postal_code=cached.postal_code,
                geocode_provider=cached.geocode_provider,
                geocoded_at=cached.geocoded_at,
            )

        self.cache.put(
            lat,
            lon,
            full_address=reparsed.full_address,
            city=reparsed.city,
            prefecture=reparsed.prefecture,
            country=reparsed.country,
            postal_code=reparsed.postal_code,
            geocode_provider=reparsed.geocode_provider,
            payload=payload,
        )
        return GeocodeResult(
            full_address=reparsed.full_address,
            city=reparsed.city,
            prefecture=reparsed.prefecture,
            country=reparsed.country,
            postal_code=reparsed.postal_code,
            geocode_provider=reparsed.geocode_provider,
            geocoded_at=cached.geocoded_at,
        )

    def reverse_geocode(self, lat: float, lon: float) -> GeocodeResult | None:
        cached = self.cache.get(lat, lon)
        if cached is not None:
            return self._result_from_cached(lat, lon, cached)

        if self.config.geocode_provider != "nominatim":
            logger.warning(
                "Unsupported GEOCODE_PROVIDER=%r — only nominatim is implemented",
                self.config.geocode_provider,
            )
            return None

        payload = self._fetch_nominatim(lat, lon)
        if payload is None:
            return None

        parsed = parse_nominatim_response(payload)
        self.cache.put(
            lat,
            lon,
            full_address=parsed.full_address,
            city=parsed.city,
            prefecture=parsed.prefecture,
            country=parsed.country,
            postal_code=parsed.postal_code,
            geocode_provider=parsed.geocode_provider,
            payload=payload,
        )
        return parsed

    def _throttle(self) -> None:
        min_interval = 1.0
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _fetch_nominatim(self, lat: float, lon: float) -> dict | None:
        self._throttle()

        params = urllib.parse.urlencode(
            {
                "lat": f"{lat:.6f}",
                "lon": f"{lon:.6f}",
                "format": "jsonv2",
                "addressdetails": "1",
                "accept-language": "ja,en",
            }
        )
        base = self.config.nominatim_base_url.rstrip("/")
        url = f"{base}/reverse?{params}"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.geocode_timeout_sec,
            ) as response:
                body = response.read().decode("utf-8")
            self._last_request_at = time.monotonic()
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.warning("Nominatim reverse geocode failed: %s", exc)
            return None

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            logger.warning("Nominatim returned invalid JSON: %s", exc)
            return None

        if not isinstance(payload, dict):
            logger.warning("Nominatim response is not a JSON object")
            return None

        if payload.get("error"):
            logger.warning("Nominatim error: %s", payload.get("error"))
            return None

        return payload
