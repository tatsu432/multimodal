"""WebRTC frame capture via WHEP (WebRTC-HTTP Egress Protocol)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any
from urllib.parse import urljoin

import httpx
import numpy as np
from aiortc import (
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
)

logger = logging.getLogger(__name__)


def parse_ice_servers(value: str | None) -> list[RTCIceServer]:
    if not value or not value.strip():
        return [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]

    servers: list[RTCIceServer] = []
    for part in value.split(","):
        url = part.strip()
        if url:
            servers.append(RTCIceServer(urls=[url]))
    return servers or [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]


async def _wait_for_ice_gathering(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


async def _negotiate_whep(
    client: httpx.AsyncClient,
    whep_url: str,
    pc: RTCPeerConnection,
) -> str | None:
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await _wait_for_ice_gathering(pc)

    if pc.localDescription is None:
        raise RuntimeError("WebRTC local description was not created")

    response = await client.post(
        whep_url,
        content=pc.localDescription.sdp,
        headers={"Content-Type": "application/sdp"},
        timeout=30.0,
    )

    if response.status_code == 201:
        answer = RTCSessionDescription(sdp=response.text, type="answer")
        await pc.setRemoteDescription(answer)
        location = response.headers.get("Location")
        return urljoin(whep_url, location) if location else None

    if response.status_code == 406:
        # WHEP counter-offer: answer the server's SDP offer via PATCH.
        remote = RTCSessionDescription(sdp=response.text, type="offer")
        await pc.setRemoteDescription(remote)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_gathering(pc)

        location = response.headers.get("Location")
        if not location:
            raise RuntimeError(
                "WHEP counter-offer did not include a session Location header"
            )

        session_url = urljoin(whep_url, location)
        patch = await client.patch(
            session_url,
            content=pc.localDescription.sdp if pc.localDescription else "",
            headers={"Content-Type": "application/sdp"},
            timeout=30.0,
        )
        if patch.status_code not in {200, 204}:
            raise RuntimeError(
                f"WHEP PATCH failed ({patch.status_code}): {patch.text}"
            )
        return session_url

    body = response.text.strip()
    hint = ""
    if "no stream is available" in body.lower() or response.status_code in {404, 503}:
        hint = (
            "\n\nMediaMTX has no active stream on this path. "
            "The path name in WEBRTC_URL must match the key under paths: in mediamtx.yml "
            "(e.g. tapo → http://localhost:8889/tapo/whep, not .../live/whep). "
            "Confirm the browser player works first (http://localhost:8889/<path>/)."
        )
    raise RuntimeError(
        f"WHEP negotiation failed ({response.status_code}): {body}{hint}"
    )


class WebRTCCapture:
    """VideoCapture-like reader for a WHEP WebRTC endpoint."""

    def __init__(
        self,
        whep_url: str,
        *,
        ice_servers: list[RTCIceServer] | None = None,
        open_timeout_sec: float = 15.0,
    ) -> None:
        self._whep_url = whep_url
        self._ice_servers = ice_servers or [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
        self._open_timeout_sec = open_timeout_sec
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._opened = False
        self._stop = threading.Event()
        self._error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pc: RTCPeerConnection | None = None
        self._session_url: str | None = None
        self._thread = threading.Thread(target=self._run, name="webrtc-capture", daemon=True)
        self._thread.start()

        deadline = time.time() + open_timeout_sec
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
        if self._loop is not None and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            try:
                future.result(timeout=5.0)
            except Exception:
                logger.exception("Error while shutting down WebRTC capture")
        self._thread.join(timeout=5.0)
        self._opened = False

    def _set_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame
            self._opened = True

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:
            self._error = str(exc)
            logger.exception("WebRTC capture thread failed")
        finally:
            self._loop.close()

    @property
    def last_error(self) -> str | None:
        return self._error

    async def _main(self) -> None:
        configuration = RTCConfiguration(iceServers=self._ice_servers)
        self._pc = RTCPeerConnection(configuration=configuration)
        pc = self._pc
        video_track: Any | None = None
        track_ready = asyncio.Event()

        @pc.on("track")
        def on_track(track: Any) -> None:
            nonlocal video_track
            if track.kind == "video" and video_track is None:
                video_track = track
                track_ready.set()

        pc.addTransceiver("video", direction="recvonly")

        async with httpx.AsyncClient() as client:
            self._session_url = await _negotiate_whep(client, self._whep_url, pc)

            try:
                await asyncio.wait_for(track_ready.wait(), timeout=self._open_timeout_sec)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"No WebRTC video track received within {self._open_timeout_sec:.0f}s"
                ) from exc

            assert video_track is not None

            while not self._stop.is_set():
                try:
                    frame = await asyncio.wait_for(video_track.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    if not self._stop.is_set():
                        logger.exception("WebRTC frame read failed")
                    break

                self._set_frame(frame.to_ndarray(format="bgr24"))

    async def _shutdown(self) -> None:
        if self._session_url:
            try:
                async with httpx.AsyncClient() as client:
                    await client.delete(self._session_url, timeout=10.0)
            except Exception:
                logger.exception("Failed to delete WHEP session")

        if self._pc is not None:
            await self._pc.close()
            self._pc = None
