"""Diagnose WHEP connectivity to MediaMTX (OPTIONS, POST, ICE states)."""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx
from aiortc import RTCPeerConnection, RTCConfiguration

from whep_client import (
    connection_state_summary,
    create_whep_http_client,
    fetch_whep_ice_servers,
    negotiate_whep,
    parse_ice_servers,
    resolve_ice_servers,
    wait_for_ice_connected,
)


async def probe(whep_url: str, ice_env: str | None, timeout_sec: float) -> int:
    print(f"WHEP URL: {whep_url}")
    print()

    async with create_whep_http_client() as client:
        print("=== OPTIONS (ICE servers) ===")
        try:
            opt = await client.options(whep_url, timeout=10.0)
            print(f"HTTP {opt.status_code}")
            links = opt.headers.get_list("link")
            if links:
                for link in links:
                    print(f"  Link: {link}")
            else:
                print("  (no Link headers)")
        except httpx.HTTPError as exc:
            print(f"OPTIONS failed: {exc}")
            return 1

        whep_ice = await fetch_whep_ice_servers(client, whep_url)
        env_ice = parse_ice_servers(ice_env)
        ice_servers = resolve_ice_servers(whep_ice, env_ice)
        print(f"Resolved ICE servers: {[s.urls for s in ice_servers]}")
        print()

        print("=== POST (WHEP offer/answer) ===")
        pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        pc.addTransceiver("video", direction="recvonly")
        pc.addTransceiver("audio", direction="recvonly")
        session_url: str | None = None

        try:
            session_url = await negotiate_whep(client, whep_url, pc)
            sdp = pc.localDescription.sdp if pc.localDescription else ""
            print(f"Offer SDP length: {len(sdp)} bytes")
            print(f"Offer contains ICE candidates: {'candidate' in sdp}")
            print(f"WHEP session: {session_url or '(none)'}")
            print(f"After negotiate: {connection_state_summary(pc)}")
            print()

            print("=== ICE connection ===")
            await wait_for_ice_connected(pc, timeout_sec)
            print(f"ICE connected: {connection_state_summary(pc)}")
            print()
            print("SUCCESS: WHEP negotiation and ICE connection OK")
            return 0
        except Exception as exc:
            print(f"FAILED: {exc}")
            print(f"State: {connection_state_summary(pc)}")
            return 1
        finally:
            if session_url:
                try:
                    await client.delete(session_url, timeout=10.0)
                except Exception:
                    pass
            await pc.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe MediaMTX WHEP endpoint (OPTIONS, POST, ICE)."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8889/tapo/whep",
        help="WHEP endpoint URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="WebRTC connection timeout (seconds)",
    )
    parser.add_argument(
        "--ice-servers",
        default=None,
        help="Override ICE servers (comma-separated), else use OPTIONS",
    )
    args = parser.parse_args()

    code = asyncio.run(probe(args.url, args.ice_servers, args.timeout))
    sys.exit(code)


if __name__ == "__main__":
    main()
