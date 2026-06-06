"""Core WHEP/WebRTC client logic (aiortc). Used by whep_worker and in-process mode."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import ssl
import subprocess
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import certifi
import httpx
import numpy as np
from aiortc import (
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
)

logger = logging.getLogger(__name__)

_MKCERT_ROOT_CANDIDATES = (
    Path.home() / "Library/Application Support/mkcert/rootCA.pem",  # macOS
    Path.home() / ".local/share/mkcert/rootCA.pem",  # Linux
)


def _find_mkcert_root() -> Path | None:
    try:
        caroot = subprocess.run(
            ["mkcert", "-CAROOT"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        if caroot:
            candidate = Path(caroot) / "rootCA.pem"
            if candidate.is_file():
                return candidate
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        pass

    for candidate in _MKCERT_ROOT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def resolve_whep_ssl_verify() -> bool | str | ssl.SSLContext:
    """
    httpx ``verify`` value for WHEP HTTP(S) calls.

    Chrome trusts mkcert via the OS keychain; Python uses certifi and ignores
    mkcert unless we add its root CA (or you set WEBRTC_CA_FILE).
    """
    verify_env = os.getenv("WEBRTC_SSL_VERIFY", "").strip().lower()
    if verify_env in {"0", "false", "no"}:
        return False

    ca_file = os.getenv("WEBRTC_CA_FILE", "").strip()
    if ca_file:
        return ca_file

    mkcert_root = _find_mkcert_root()
    if mkcert_root is None:
        return True

    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.load_verify_locations(str(mkcert_root))
    return ctx


def create_whep_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=resolve_whep_ssl_verify())


_LINK_ICE_SERVER_RE = re.compile(
    r'^<([^>]+)>;\s*rel=["\']?ice-server["\']?'
    r'(?:;\s*username="(.*?)";\s*credential="(.*?)";\s*credential-type="password")?',
    re.IGNORECASE,
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


def _is_local_whep(whep_url: str) -> bool:
    host = (urlparse(whep_url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def resolve_ice_servers(
    whep_servers: list[RTCIceServer],
    env_servers: list[RTCIceServer],
    *,
    whep_url: str | None = None,
) -> list[RTCIceServer]:
    if env_servers:
        return env_servers
    if whep_servers:
        return whep_servers
    # Local MediaMTX exposes host candidates via webrtcAdditionalHosts; STUN adds
    # srflx candidates that can confuse aiortc ICE/DTLS with localhost WHEP.
    if whep_url and _is_local_whep(whep_url):
        return []
    return [RTCIceServer(urls=["stun:stun.l.google.com:19302"])]


@dataclass
class _OfferIceData:
    ice_ufrag: str
    ice_pwd: str
    medias: list[str] = field(default_factory=list)


def _parse_offer_sdp(sdp: str) -> _OfferIceData:
    """Parse ICE credentials and m= lines (MediaMTX trickle-ice-sdpfrag format)."""
    data = _OfferIceData(ice_ufrag="", ice_pwd="")
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            data.medias.append(line[2:])
        elif not data.ice_ufrag and line.startswith("a=ice-ufrag:"):
            data.ice_ufrag = line[len("a=ice-ufrag:") :]
        elif not data.ice_pwd and line.startswith("a=ice-pwd:"):
            data.ice_pwd = line[len("a=ice-pwd:") :]
    return data


def _sdp_is_bundled(sdp: str) -> bool:
    return any(
        line.startswith("a=group:BUNDLE")
        for line in sdp.replace("\r\n", "\n").split("\n")
    )


def _extract_candidates_from_sdp(
    sdp: str,
    *,
    bundled_only: bool = True,
) -> list[tuple[int, str]]:
    """Return (media_index, candidate_line) pairs from a local SDP."""
    is_bundled = _sdp_is_bundled(sdp)
    mid = -1
    out: list[tuple[int, str]] = []
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("m="):
            mid += 1
        elif line.startswith("a=candidate:"):
            if bundled_only and is_bundled and mid != 0:
                continue
            out.append((mid, line[2:]))
    return out


def _generate_trickle_sdp_fragment(
    offer_data: _OfferIceData,
    candidates: list[tuple[int, str]],
) -> str:
    by_media: dict[int, list[str]] = {}
    for mid, candidate in candidates:
        by_media.setdefault(mid, []).append(candidate)

    lines = [
        f"a=ice-ufrag:{offer_data.ice_ufrag}",
        f"a=ice-pwd:{offer_data.ice_pwd}",
    ]
    for mid, media in enumerate(offer_data.medias):
        media_candidates = by_media.get(mid)
        if not media_candidates:
            continue
        lines.append(f"m={media}")
        lines.append(f"a=mid:{mid}")
        lines.extend(f"a={candidate}" for candidate in media_candidates)
    return "\r\n".join(lines) + "\r\n"


async def _patch_trickle_candidates(
    client: httpx.AsyncClient,
    session_url: str,
    offer_data: _OfferIceData,
    candidates: list[tuple[int, str]],
) -> None:
    """Send candidates one PATCH at a time (MediaMTX reader.js behaviour)."""
    if not candidates:
        return
    # Bundled sessions only need the BUNDLE master (mid 0) in the fragment.
    if _sdp_is_bundled("\n".join(f"m={m}" for m in offer_data.medias)):
        offer_data = _OfferIceData(
            ice_ufrag=offer_data.ice_ufrag,
            ice_pwd=offer_data.ice_pwd,
            medias=offer_data.medias[:1],
        )
        candidates = [(0, cand) for mid, cand in candidates if mid == 0]

    seen: set[str] = set()
    for mid, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        body = _generate_trickle_sdp_fragment(offer_data, [(mid, candidate)])
        response = await client.patch(
            session_url,
            content=body,
            headers={
                "Content-Type": "application/trickle-ice-sdpfrag",
                "If-Match": "*",
            },
            timeout=30.0,
        )
        if response.status_code not in {200, 204}:
            raise RuntimeError(
                f"WHEP trickle-ICE PATCH failed ({response.status_code}): "
                f"{response.text.strip()}"
            )


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
    """Wait until WebRTC peer connection is ready (ICE + DTLS)."""
    if pc.connectionState == "connected":
        return

    done = asyncio.Event()
    failed = False

    @pc.on("connectionstatechange")
    def on_connection_state_change() -> None:
        nonlocal failed
        if pc.connectionState == "connected":
            done.set()
        elif pc.connectionState == "failed":
            failed = True
            done.set()

    @pc.on("iceconnectionstatechange")
    def on_ice_connection_state_change() -> None:
        nonlocal failed
        if pc.iceConnectionState == "failed":
            failed = True
            done.set()
        elif (
            pc.iceConnectionState in {"connected", "completed"}
            and pc.connectionState == "connected"
        ):
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_sec)
    except asyncio.TimeoutError as exc:
        hint = ""
        if (
            pc.iceConnectionState in {"connected", "completed"}
            and pc.connectionState == "connecting"
        ):
            hint = (
                "\n\nICE completed but DTLS never finished (connection stuck at 'connecting'). "
                "For MediaMTX + aiortc, ensure trickle-ICE PATCH is reaching the WHEP session "
                "(see whep_client.negotiate_whep). With phone + webrtcEncryption, WHEP may "
                "still fail — use the RTSP relay: rtsp://127.0.0.1:8554/phone."
            )
        raise RuntimeError(
            f"WebRTC peer connection timed out after {timeout_sec:.0f}s "
            f"({connection_state_summary(pc)}). "
            "If ice stays 'checking', ensure MediaMTX UDP 8189 is reachable or set "
            "webrtcLocalTCPAddress in mediamtx.yml (see mediamtx-tapo.example.yml)."
            f"{hint}"
        ) from exc

    if failed:
        raise RuntimeError(
            f"WebRTC peer connection failed ({connection_state_summary(pc)})"
        )


async def negotiate_whep(
    client: httpx.AsyncClient,
    whep_url: str,
    pc: RTCPeerConnection,
) -> str | None:
    """
    WHEP offer/answer with MediaMTX-compatible trickle ICE.

    MediaMTX's browser reader POSTs the offer before all candidates are known,
    then PATCHes ``application/trickle-ice-sdpfrag`` to the session URL. aiortc
    gathers synchronously, so we POST a candidate-free offer, set local+remote
    descriptions, then PATCH the gathered candidates (reader.js flow).
    """
    offer = await pc.createOffer()
    post_sdp = offer.sdp
    post_offer_data = _parse_offer_sdp(post_sdp)

    response = await client.post(
        whep_url,
        content=post_sdp,
        headers={"Content-Type": "application/sdp"},
        timeout=30.0,
    )

    if response.status_code == 201:
        session_url = response.headers.get("Location")
        session_url = urljoin(whep_url, session_url) if session_url else None
        answer = RTCSessionDescription(sdp=response.text, type="answer")

        await pc.setLocalDescription(offer)
        if pc.localDescription is None:
            raise RuntimeError("WebRTC local description was not created")

        candidates = _extract_candidates_from_sdp(pc.localDescription.sdp)
        await pc.setRemoteDescription(answer)

        # PATCH m= lines must match the POSTed offer (MediaMTX session state), not the
        # post-gather local SDP where ports differ from the placeholder 9.
        if session_url and candidates:
            await _patch_trickle_candidates(
                client, session_url, post_offer_data, candidates
            )
        return session_url

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

    async with create_whep_http_client() as client:
        whep_ice = await fetch_whep_ice_servers(client, whep_url)
        resolved = resolve_ice_servers(
            whep_ice, ice_servers or [], whep_url=whep_url
        )
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
            # Match MediaMTX embedded reader.js (SCTP m=application in BUNDLE).
            pc.createDataChannel("")

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
