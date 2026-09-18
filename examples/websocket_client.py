#!/usr/bin/env python3
"""Independent WebSocket client for verifying the HandEvent protocol.

Usage:
    python examples/websocket_client.py
    python examples/websocket_client.py --url ws://127.0.0.1:8766
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

DEFAULT_URL = "ws://127.0.0.1:8766"


async def _listen(url: str) -> None:
    import websockets

    async with websockets.connect(url) as ws:
        print(f"Connected to {url}", flush=True)
        async for message in ws:
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                print(f"← (raw) {message}", flush=True)
                continue
            event_type = payload.get("type", "?")
            print(f"← {event_type} {payload}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HandEvent WebSocket test client")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"default: {DEFAULT_URL}")
    args = parser.parse_args(argv)
    try:
        asyncio.run(_listen(args.url))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
