"""DashboardServer — ThreadingHTTPServer serving the dashboard UI.

Routes:
  GET  /                  → static/index.html
  GET  /frame.mjpeg       → live MJPEG stream (~10 fps)
  GET  /frame.jpg         → single current JPEG (poster/fallback)
  POST /api/qa/stream     → SSE: live-QA tokens (text/event-stream)
  POST /api/ltm/stream    → SSE: LTM planner+retrieval+answer tokens
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.config import Config
from src.dashboard.sse import MJPEG_CONTENT_TYPE, mjpeg_part, sse_event

logger = logging.getLogger("memory_log.dashboard.server")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_MJPEG_FPS = 10
_MJPEG_INTERVAL = 1.0 / _MJPEG_FPS
_NO_FRAME_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
    b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
    b"\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e!"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00"
    b"\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
    b"\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03"
    b"\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05"
    b"\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1"
    b"\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
    b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97"
    b"\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
    b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7"
    b"\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5"
    b"\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd4P\x00"
    b"\x00\x00\x00\x1f\xff\xd9"
)


class DashboardServer:
    """HTTP(S) server hosting the dashboard on a daemon thread."""

    def __init__(self, service, config: Config) -> None:
        self._service = service
        self._config = config
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        service = self._service
        static_dir = _STATIC_DIR

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                logger.debug(format % args)

            # --- CORS helpers -------------------------------------------
            def _send_cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._send_cors()
                self.end_headers()

            # --- GET routes ---------------------------------------------
            def do_GET(self) -> None:
                path = self.path.split("?")[0]  # strip query string
                if path in ("/", "/index.html"):
                    self._serve_static("index.html", "text/html; charset=utf-8")
                elif path == "/frame.mjpeg":
                    self._serve_mjpeg()
                elif path == "/frame.jpg":
                    self._serve_single_frame()
                else:
                    self.send_error(404)

            def _serve_static(self, filename: str, content_type: str) -> None:
                fpath = static_dir / filename
                if not fpath.is_file():
                    self.send_error(404)
                    return
                body = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_single_frame(self) -> None:
                jpeg = service.get_frame_jpeg() or _NO_FRAME_JPEG
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(jpeg)

            def _serve_mjpeg(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", MJPEG_CONTENT_TYPE)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        t0 = time.monotonic()
                        jpeg = service.get_frame_jpeg() or _NO_FRAME_JPEG
                        self.wfile.write(mjpeg_part(jpeg))
                        self.wfile.flush()
                        elapsed = time.monotonic() - t0
                        sleep_for = _MJPEG_INTERVAL - elapsed
                        if sleep_for > 0:
                            time.sleep(sleep_for)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client disconnected
                except Exception as exc:
                    logger.debug("MJPEG stream ended: %s", exc)

            # --- POST routes --------------------------------------------
            def do_POST(self) -> None:
                path = self.path.split("?")[0]
                if path == "/api/qa/stream":
                    self._stream_sse(service.live_qa_stream, "question")
                elif path == "/api/ltm/stream":
                    self._stream_sse(service.ltm_stream, "query")
                else:
                    self.send_error(404)

            def _stream_sse(self, generator_fn, body_key: str) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                    text = data.get(body_key, "").strip()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.send_error(400, "Invalid JSON body")
                    return

                if not text:
                    self.send_error(400, f"'{body_key}' field is required")
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self._send_cors()
                self.end_headers()

                try:
                    for event_name, payload in generator_fn(text):
                        chunk = sse_event(event_name, payload)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client disconnected mid-stream
                except Exception as exc:
                    logger.error("SSE stream error: %s", exc)
                    try:
                        self.wfile.write(sse_event("error", {"message": str(exc)}))
                        self.wfile.flush()
                    except Exception:
                        pass

        self._httpd = ThreadingHTTPServer(
            (self._config.dashboard_host, self._config.dashboard_port), Handler
        )

        scheme = "http"
        if self._config.dashboard_cert and self._config.dashboard_key:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(
                str(self._config.dashboard_cert),
                str(self._config.dashboard_key),
            )
            self._httpd.socket = ctx.wrap_socket(
                self._httpd.socket, server_side=True
            )
            scheme = "https"

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="dashboard-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Dashboard listening on %s://%s:%d/",
            scheme,
            self._config.dashboard_host,
            self._config.dashboard_port,
        )

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
