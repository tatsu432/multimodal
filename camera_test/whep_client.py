"""Core WHEP/WebRTC client logic (aiortc). Used by whep_worker and in-process mode."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable
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

_LINK_ICE_SERVER_RE = re.compile(
    r'^<(.+?)>; rel="ice-server"'
    r'(?:; username="(.*?)"; credential="(.*?)"; credential-type="password")?',
)


def parse_ice_servers(value: str | None) -> list[RTCIceServer]:
    if not value or not value.strip():
        return []

    servers: list[RTCIceServer] = []
    for part in value.split(","):
        url = part.strip()
        if url:
            servers.append(RTCIceServer(urls=[url]))
    return servers


def ice_servers_to_dicts(servers: list[RTCIceServer]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for server in servers:
        entry: dict[str, Any] = {"urls": list(server.urls)}
        if server.username:
            entry["username"] = server.username
        if server.credential:
            entry["credential"] = server.credential
        if server.credentialType:
            entry["credentialType"] = server.credentialType
        out.append(entry)
    return out


def ice_servers_from_dicts(data: list[dict[str, Any]]) -> list[RTCIceServer]:
    servers: list[RTCIceServer] = []
    for entry in data:
        kwargs: dict[str, Any] = {"urls": entry["urls"]}
        if entry.get("username"):
            kwargs["username"] = entry["username"]
        if entry.get("credential"):
            kwargs["credential"] = entry["credential"]
        if entry.get("credentialType"):
            kwargs["credentialType"] = entry["credentialType"]
        servers.append(RTCIceServer(**kwargs))
    return servers


async def fetch_whep_ice_servers(
    client: httpx.AsyncClient,
    whep_url: str,
) -> list[RTCIceServer]:
    try:
        response = await client.options(whep_url, timeout=10.0)
    except httpx.HTTPError as exc:
        logger.warning("WHEP OPTIONS failed (%s); falling back to env ICE servers", exc)
        return []

    servers: list[RTCIceServer] = []
    for value in response.headers.get_list("link"):
        match = _LINK_ICE_SERVER_RE.match(value)
        if not match:
            continue

        kwargs: dict[str, Any] = {"urls": match.group(1)}
        if match.group(2) is not None:
            kwargs["username"] = match.group(2)
            kwargs["credential"] = match.group(3)
            kwargs["credentialType"] = "password"
        servers.append(RTCIceServer(**kwargs))

    return servers


def resolve_ice_servers(
    whep_servers: list[RTCIceServer],
    env_servers: list[RTCIceServer],
) -> list[RTCIceServer]:
    if env_servers:
        return env_servers
    if whep_servers:
        return whep_servers
    return [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]


def connection_state_summary(pc: RTCPeerConnection) -> str:
    return (
        f"iceGathering={pc.iceGatheringState}, "
        f"ice={pc.iceConnectionState}, "
        f"connection={pc.connectionState}"
    )


async def wait_for_ice_gathering(pc: RTCPeerConnection) -> None:
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def on_ice_gathering_state_change() -> None:
        if pc.iceGatheringState == "complete":
            done.set()

    await done.wait()


async def wait_for_ice_connected(
    pc: RTCPeerConnection,
    timeout_sec: float,
) -> None:
    if pc.iceConnectionState in {"connected", "completed"}:
        return

    done = asyncio.Event()
    failed = False

    @pc.on("iceconnectionstatechange")
    def on_ice_connection_state_change() -> None:
        nonlocal failed
        state = pc.iceConnectionState
        if state in {"connected", "completed"}:
            done.set()
        elif state == "failed":
            failed = True
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"ICE connection timed out after {timeout_sec:.0f}s "
            f"({connection_state_summary(pc)})"
        ) from exc

    if failed:
        raise RuntimeError(
            f"ICE connection failed ({connection_state_summary(pc)})"
        )


async def negotiate_whep(
    client: httpx.AsyncClient,
    whep_url: str,
    pc: RTCPeerConnection,
) -> str | None:
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await wait_for_ice_gathering(pc)

    if pc.localDescription is None:
        raise RuntimeError("WebRTC local description was not created")

    if "candidate" not in pc.localDescription.sdp:
        raise RuntimeError(
            "SDP offer contains no ICE candidates. aiortc requires non-trickle ICE — "
            "wait for iceGatheringState=complete before POSTing to WHEP."
        )

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
        remote = RTCSessionDescription(sdp=response.text, type="offer")
        await pc.setRemoteDescription(remote)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_for_ice_gathering(pc)

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
            "(e.g. tapo → http://localhost:8889/tapo/whep). "
            "Confirm the browser player works first (http://localhost:8889/<path>/)."
        )
    raise RuntimeError(
        f"WHEP negotiation failed ({response.status_code}): {body}{hint}"
    )


async def run_whep_stream(
    whep_url: str,
    *,
    ice_servers: list[RTCIceServer] | None = None,
    open_timeout_sec: float = 15.0,
    stop_event: asyncio.Event | None = None,
    on_frame: Callable[[np.ndarray], None] | None = None,
) -> None:
    """
    Connect to a WHEP endpoint and invoke on_frame for each decoded BGR frame.

    Raises RuntimeError on negotiation or connection failure.
    """
    stop = stop_event or asyncio.Event()
    video_track: Any | None = None
    track_ready = asyncio.Event()

    async with httpx.AsyncClient() as client:
        whep_ice = await fetch_whep_ice_servers(client, whep_url)
        resolved = resolve_ice_servers(whep_ice, ice_servers or [])
        logger.info(
            "WHEP ICE servers: %s",
            [getattr(s, "urls", s) for s in resolved],
        )

        configuration = RTCConfiguration(iceServers=resolved)
        pc = RTCPeerConnection(configuration=configuration)
        session_url: str | None = None

        try:
            @pc.on("track")
            def on_track(track: Any) -> None:
                nonlocal video_track
                if track.kind == "video" and video_track is None:
                    video_track = track
                    track_ready.set()

            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")

            session_url = await negotiate_whep(client, whep_url, pc)
            await wait_for_ice_connected(pc, open_timeout_sec)

            try:
                await asyncio.wait_for(track_ready.wait(), timeout=open_timeout_sec)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    "No WebRTC video track received within "
                    f"{open_timeout_sec:.0f}s ({connection_state_summary(pc)})"
                ) from exc

            assert video_track is not None

            while not stop.is_set():
                try:
                    frame = await asyncio.wait_for(video_track.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    if not stop.is_set():
                        logger.exception("WebRTC frame read failed")
                    break

                if on_frame is not None:
                    on_frame(frame.to_ndarray(format="bgr24"))
        finally:
            if session_url:
                try:
                    await client.delete(session_url, timeout=10.0)
                except Exception:
                    logger.exception("Failed to delete WHEP session")
            await pc.close()
