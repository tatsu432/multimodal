import json
import logging
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.location import LocationSidecarStore

logger = logging.getLogger("memory_log.location_server")

PHONE_LOCATION_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>memory_log phone location</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1.5rem; line-height: 1.5; }
    .ok { color: #0a0; }
    .err { color: #a00; }
    code { background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px; }
  </style>
</head>
<body>
  <h1>Phone GPS for memory_log</h1>
  <p>Keep this page open while you publish from your phone and ask questions in memory_log.</p>
  <p id="status">Requesting location permission…</p>
  <p id="coords"></p>
  <script>
    const statusEl = document.getElementById("status");
    const coordsEl = document.getElementById("coords");

    async function postLocation(position) {
      const payload = {
        lat: position.coords.latitude,
        lon: position.coords.longitude,
        accuracy: position.coords.accuracy,
      };
      const response = await fetch("/location", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("POST /location failed: " + response.status);
      }
      statusEl.className = "ok";
      statusEl.textContent = "Location sent to memory_log.";
      coordsEl.textContent =
        payload.lat.toFixed(6) + ", " + payload.lon.toFixed(6) +
        " (±" + Math.round(payload.accuracy) + " m)";
    }

    if (!navigator.geolocation) {
      statusEl.className = "err";
      statusEl.textContent = "Geolocation is not supported in this browser.";
    } else {
      navigator.geolocation.watchPosition(
        (position) => {
          postLocation(position).catch((err) => {
            statusEl.className = "err";
            statusEl.textContent = "Failed to send location: " + err.message;
          });
        },
        (err) => {
          statusEl.className = "err";
          statusEl.textContent = "Geolocation error: " + err.message;
        },
        { enableHighAccuracy: true, maximumAge: 5000, timeout: 15000 }
      );
    }
  </script>
</body>
</html>
"""


def _load_html(configured_path: Path | None) -> str:
    if configured_path is not None and configured_path.is_file():
        return configured_path.read_text(encoding="utf-8")
    fallback = (
        Path(__file__).resolve().parents[2]
        / "camera_test"
        / "phone_location.html"
    )
    if fallback.is_file():
        return fallback.read_text(encoding="utf-8")
    return PHONE_LOCATION_HTML


class LocationServer:
    def __init__(
        self,
        store: LocationSidecarStore,
        host: str,
        port: int,
        cert_path: Path | None,
        key_path: Path | None,
        html_path: Path | None = None,
    ):
        self.store = store
        self.host = host
        self.port = port
        self.cert_path = cert_path
        self.key_path = key_path
        self.html = _load_html(html_path)
        self._thread: threading.Thread | None = None
        self._httpd: ThreadingHTTPServer | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        store = self.store
        html = self.html

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                logger.debug(format % args)

            def _send_cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._send_cors()
                self.end_headers()

            def do_GET(self) -> None:
                if self.path not in {"/", "/index.html"}:
                    self.send_error(404)
                    return
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if self.path != "/location":
                    self.send_error(404)
                    return

                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    self.send_error(400, "Invalid JSON")
                    return

                try:
                    lat = float(data["lat"])
                    lon = float(data["lon"])
                except (KeyError, TypeError, ValueError):
                    self.send_error(400, "lat and lon are required numbers")
                    return

                label = data.get("label")
                store.update(lat, lon, label=str(label) if label else None)

                body = b'{"ok":true}'
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)

        if self.cert_path and self.key_path:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(self.cert_path), str(self.key_path))
            self._httpd.socket = context.wrap_socket(
                self._httpd.socket,
                server_side=True,
            )
            scheme = "https"
        else:
            scheme = "http"
            logger.warning(
                "Location sidecar running without TLS — phone browsers may block geolocation"
            )

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="location-sidecar",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Location sidecar listening on %s://%s:%d/",
            scheme,
            self.host,
            self.port,
        )
        print(
            f"[location] sidecar at {scheme}://<your-lan-ip>:{self.port}/ "
            f"(open on phone while publishing)"
        )

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
