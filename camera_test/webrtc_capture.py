"""WebRTC frame capture via WHEP — subprocess IPC wrapper (default) or in-process."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CAMERA_TEST_DIR = Path(__file__).resolve().parent
_WHEP_WORKER_SCRIPT = _CAMERA_TEST_DIR / "whep_worker.py"
_BROWSER_WHEP_WORKER_SCRIPT = _CAMERA_TEST_DIR / "browser_whep_worker.py"

_FRAME_MAGIC = b"WFRM"
_ERROR_MAGIC = b"WERR"
_HEADER = struct.Struct(">4sIII")


def _webrtc_ipc_mode() -> str:
    """subprocess (aiortc), browser (Chromium player), or inprocess."""
    mode = os.getenv("WEBRTC_IPC", "").strip().lower()
    if not mode:
        # aiortc WHEP often fails DTLS with MediaMTX on macOS; Chromium succeeds.
        mode = "browser" if sys.platform == "darwin" else "subprocess"
    return mode


def _use_subprocess_ipc() -> bool:
    return _webrtc_ipc_mode() in {"subprocess", "browser"}


def _build_worker_command(
    whep_url: str,
    open_timeout_sec: float,
    ice_servers_env: str | None,
) -> list[str]:
    """
    Build a command that runs a WHEP worker with the same deps as camera_test.

    Prefer `uv run` so the worker works even when the parent was started from an
    IDE (avoids wrong-interpreter / pip-list environment errors).
    """
    ipc = _webrtc_ipc_mode()
    worker_script = (
        _BROWSER_WHEP_WORKER_SCRIPT if ipc == "browser" else _WHEP_WORKER_SCRIPT
    )
    args = [
        "--url",
        whep_url,
        "--timeout",
        str(open_timeout_sec),
    ]
    if ipc != "browser":
        args.extend(["--ice-servers", ice_servers_env or ""])

    uv = shutil.which("uv")
    if uv:
        cmd = [
            uv,
            "run",
            "--directory",
            str(_CAMERA_TEST_DIR),
        ]
        if ipc == "browser":
            cmd.append("--extra")
            cmd.append("browser-webrtc")
        cmd.extend(["python", str(worker_script), *args])
        return cmd

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        py = Path(venv) / "bin" / "python3"
        if py.is_file():
            return [str(py), str(worker_script), *args]

    return [sys.executable, str(worker_script), *args]


class _InProcessWebRTCCapture:
    """aiortc in the same process (debug only; triggers macOS FFmpeg warning with cv2)."""

    def __init__(
        self,
        whep_url: str,
        *,
        ice_servers: list | None = None,
        open_timeout_sec: float = 15.0,
    ) -> None:
        from whep_client import run_whep_stream

        self._run_whep_stream = run_whep_stream
        self._whep_url = whep_url
        self._ice_servers = ice_servers
        self._open_timeout_sec = open_timeout_sec
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._opened = False
        self._stop = threading.Event()
        self._error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(
            target=self._run, name="webrtc-capture-inprocess", daemon=True
        )
        self._thread.start()
        self._wait_until_open()

    def _wait_until_open(self) -> None:
        deadline = time.time() + self._open_timeout_sec
        while time.time() < deadline:
            if self._opened:
                return
            if not self._thread.is_alive():
                break
            time.sleep(0.05)

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._opened = False

    def _set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame
            self._opened = True

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        stop_async = asyncio.Event()

        def watch() -> None:
            while not self._stop.is_set():
                time.sleep(0.05)
            self._loop.call_soon_threadsafe(stop_async.set)

        threading.Thread(target=watch, daemon=True).start()

        try:
            self._loop.run_until_complete(
                self._run_whep_stream(
                    self._whep_url,
                    ice_servers=self._ice_servers,
                    open_timeout_sec=self._open_timeout_sec,
                    stop_event=stop_async,
                    on_frame=self._set_frame,
                )
            )
        except Exception as exc:
            self._error = str(exc)
            logger.exception("In-process WebRTC capture failed")
        finally:
            self._loop.close()

    @property
    def last_error(self) -> str | None:
        return self._error


class _SubprocessWebRTCCapture:
    """WHEP via `python -m whep_worker` — parent never imports aiortc."""

    def __init__(
        self,
        whep_url: str,
        *,
        ice_servers_env: str | None = None,
        open_timeout_sec: float = 15.0,
    ) -> None:
        self._whep_url = whep_url
        self._open_timeout_sec = open_timeout_sec
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._opened = False
        self._error: str | None = None
        self._stop = threading.Event()
        self._stderr_tail = ""

        cmd = _build_worker_command(whep_url, open_timeout_sec, ice_servers_env)
        env = os.environ.copy()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_CAMERA_TEST_DIR),
            env=env,
        )

        self._stderr_reader = threading.Thread(
            target=self._stderr_loop, name="whep-stderr-reader", daemon=True
        )
        self._stdout_reader = threading.Thread(
            target=self._read_loop, name="whep-stdout-reader", daemon=True
        )
        self._stderr_reader.start()
        self._stdout_reader.start()
        self._wait_until_open()

    def _wait_until_open(self) -> None:
        deadline = time.time() + self._open_timeout_sec
        while time.time() < deadline:
            if self._opened:
                return
            if self._error is not None:
                return
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)

        if not self._opened and self._error is None:
            hint = self._stderr_tail.strip()
            self._error = (
                f"WHEP worker did not deliver a frame within {self._open_timeout_sec:.0f}s"
                + (f"\n{hint}" if hint else "")
            )

    def _stderr_loop(self) -> None:
        stderr = self._proc.stderr
        if stderr is None:
            return
        lines: list[str] = []
        for raw in iter(stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                lines.append(line)
                if len(lines) > 40:
                    lines.pop(0)
                self._stderr_tail = "\n".join(lines)

    def _read_loop(self) -> None:
        stdout = self._proc.stdout
        if stdout is None:
            self._error = "WHEP worker stdout not available"
            return

        while not self._stop.is_set():
            header_bytes = stdout.read(_HEADER.size)
            if not header_bytes or len(header_bytes) < _HEADER.size:
                if self._proc.poll() is not None and not self._opened:
                    if self._error is None:
                        hint = self._stderr_tail.strip()
                        self._error = "WHEP worker exited before delivering frames"
                        if hint:
                            self._error += f"\n{hint}"
                break

            magic, height, width, length = _HEADER.unpack(header_bytes)
            payload = stdout.read(length)
            if len(payload) < length:
                break

            if magic == _ERROR_MAGIC:
                self._error = payload.decode("utf-8", errors="replace")
                break

            if magic == _FRAME_MAGIC and height > 0 and width > 0:
                frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
                with self._lock:
                    self._latest_frame = frame.copy()
                    self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def release(self) -> None:
        self._stop.set()
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._opened = False

    @property
    def last_error(self) -> str | None:
        return self._error


class WebRTCCapture:
    """VideoCapture-like reader for a WHEP WebRTC endpoint."""

    def __init__(
        self,
        whep_url: str,
        *,
        ice_servers_env: str | None = None,
        open_timeout_sec: float = 15.0,
    ) -> None:
        if _use_subprocess_ipc():
            self._impl = _SubprocessWebRTCCapture(
                whep_url,
                ice_servers_env=ice_servers_env,
                open_timeout_sec=open_timeout_sec,
            )
        else:
            from whep_client import parse_ice_servers

            self._impl = _InProcessWebRTCCapture(
                whep_url,
                ice_servers=parse_ice_servers(ice_servers_env),
                open_timeout_sec=open_timeout_sec,
            )

    def isOpened(self) -> bool:
        return self._impl.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._impl.read()

    def release(self) -> None:
        self._impl.release()

    @property
    def last_error(self) -> str | None:
        return self._impl.last_error


