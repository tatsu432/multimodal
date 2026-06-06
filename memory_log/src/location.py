import threading
import time

from src.config import Config
from src.schema import LocationInfo


class LocationSidecarStore:
    """Thread-safe store for the latest phone GPS fix posted by the sidecar page."""

    def __init__(self, max_age_sec: float):
        self.max_age_sec = max_age_sec
        self._lock = threading.Lock()
        self._lat: float | None = None
        self._lon: float | None = None
        self._label: str | None = None
        self._updated_at: float | None = None

    def update(
        self,
        lat: float,
        lon: float,
        *,
        label: str | None = None,
    ) -> None:
        with self._lock:
            self._lat = lat
            self._lon = lon
            if label:
                self._label = label.strip() or None
            self._updated_at = time.time()

    def get_fresh(self) -> tuple[float, float, str | None] | None:
        with self._lock:
            if self._lat is None or self._lon is None or self._updated_at is None:
                return None
            if time.time() - self._updated_at > self.max_age_sec:
                return None
            return self._lat, self._lon, self._label


def _config_location(
    label: str | None,
    lat: float | None,
    lon: float | None,
) -> LocationInfo | None:
    if label or lat is not None or lon is not None:
        return LocationInfo(label=label, lat=lat, lon=lon, source="config")
    return None


def _camera_config_location(config: Config, camera_source: str) -> LocationInfo | None:
    if camera_source in {"tapo-rtsp", "tapo-webrtc"}:
        label = config.tapo_location_label or config.location_label
        lat = (
            config.tapo_location_lat
            if config.tapo_location_lat is not None
            else config.location_lat
        )
        lon = (
            config.tapo_location_lon
            if config.tapo_location_lon is not None
            else config.location_lon
        )
        return _config_location(label, lat, lon)

    if camera_source == "phone-webrtc":
        label = config.phone_location_label or config.location_label
        lat = (
            config.phone_location_lat
            if config.phone_location_lat is not None
            else config.location_lat
        )
        lon = (
            config.phone_location_lon
            if config.phone_location_lon is not None
            else config.location_lon
        )
        return _config_location(label, lat, lon)

    return _config_location(
        config.location_label,
        config.location_lat,
        config.location_lon,
    )


def resolve_location(
    config: Config,
    sidecar: LocationSidecarStore | None,
    camera_source: str,
) -> LocationInfo:
    if camera_source == "phone-webrtc" and sidecar is not None:
        fresh = sidecar.get_fresh()
        if fresh is not None:
            lat, lon, posted_label = fresh
            label = posted_label or config.phone_location_label or config.location_label
            return LocationInfo(
                label=label,
                lat=lat,
                lon=lon,
                source="phone_gps",
            )

    configured = _camera_config_location(config, camera_source)
    if configured is not None:
        return configured

    return LocationInfo(source="manual_or_not_available")
