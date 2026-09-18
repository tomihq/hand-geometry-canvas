"""CLI for hand_interaction — run as an independent WebSocket event publisher."""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from hand_interaction.api import create_hand_gesture
from hand_interaction.transport.websocket_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WebSocketServer,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hand-gesture",
        description="Hand interaction tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve",
        help="Start live hand tracking and publish HandEvents over WebSocket",
    )
    serve.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST})",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"WebSocket port (default: {DEFAULT_PORT})",
    )
    serve.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    serve.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Max hands to track (default: 2)",
    )
    return parser


def _cmd_serve(args: argparse.Namespace) -> int:
    server = WebSocketServer(host=args.host, port=args.port)
    gesture = create_hand_gesture(
        live=True,
        device_index=args.device_index,
        num_hands=args.num_hands,
    )
    gesture.on_event(server.publish)

    stop = threading.Event()

    def _request_stop(*_args: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    server.start()
    print(f"WebSocket listening on {server.url}", flush=True)
    gesture.start()
    try:
        stop.wait()
    finally:
        gesture.stop()
        server.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _cmd_serve(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
