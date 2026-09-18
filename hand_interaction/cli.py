"""CLI for hand_interaction — run as an independent WebSocket event publisher."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass, field

from hand_interaction.api import create_hand_gesture
from hand_interaction.live import LiveHandSource
from hand_interaction.transport.websocket_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    WebSocketServer,
)
from hand_interaction.types import (
    HandEvent,
    HandMove,
    HandRotate,
    PinchEnd,
    PinchStart,
    Vector2,
)

PREVIEW_WINDOW = "hand-gesture serve"


@dataclass
class _PreviewHand:
    position: Vector2 = field(default_factory=lambda: Vector2(0.5, 0.5))
    pinching: bool = False
    last_type: str = ""


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
    serve.add_argument(
        "--preview",
        action="store_true",
        help="Show live camera window while publishing events",
    )
    return parser


def _apply_preview_event(state: dict[str, _PreviewHand], event: HandEvent) -> None:
    hand = state.setdefault(event.hand_id, _PreviewHand())
    hand.last_type = event.type
    if isinstance(event, (PinchStart, PinchEnd, HandMove)):
        hand.position = event.position
    if isinstance(event, PinchStart):
        hand.pinching = True
    elif isinstance(event, PinchEnd):
        hand.pinching = False
    elif isinstance(event, HandRotate):
        pass


def _draw_preview(
    frame,
    state: dict[str, _PreviewHand],
    ws_url: str,
):
    import cv2

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.putText(
        overlay,
        f"WS {ws_url}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 220, 40),
        2,
        cv2.LINE_AA,
    )
    for hand_id, hand in state.items():
        # Event coords: y origin at bottom → OpenCV y origin at top.
        x = int(hand.position.x * w)
        y = int((1.0 - hand.position.y) * h)
        color = (0, 80, 255) if hand.pinching else (255, 180, 40)
        cv2.circle(overlay, (x, y), 14 if hand.pinching else 10, color, -1)
        cv2.putText(
            overlay,
            f"{hand_id} {hand.last_type}",
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay


def _run_preview_loop(
    source: LiveHandSource,
    state: dict[str, _PreviewHand],
    state_lock: threading.Lock,
    ws_url: str,
    stop: threading.Event,
) -> None:
    import cv2

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    while not stop.is_set():
        err = source.poll_error()
        if err is not None:
            raise err
        frame = source.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        with state_lock:
            snapshot = {
                hid: _PreviewHand(
                    position=h.position,
                    pinching=h.pinching,
                    last_type=h.last_type,
                )
                for hid, h in state.items()
            }
        display = _draw_preview(frame, snapshot, ws_url)
        cv2.imshow(PREVIEW_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop.set()
            break
    cv2.destroyWindow(PREVIEW_WINDOW)


def _cmd_serve(args: argparse.Namespace) -> int:
    server = WebSocketServer(host=args.host, port=args.port)
    gesture = create_hand_gesture(
        live=True,
        device_index=args.device_index,
        num_hands=args.num_hands,
    )

    preview_state: dict[str, _PreviewHand] = {}
    preview_lock = threading.Lock()

    def on_event(event: HandEvent) -> None:
        server.publish(event)
        if args.preview:
            with preview_lock:
                _apply_preview_event(preview_state, event)

    gesture.on_event(on_event)

    stop = threading.Event()

    def _request_stop(*_args: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    server.start()
    print(f"WebSocket listening on {server.url}", flush=True)
    if args.preview:
        print("Preview on (press q to quit)", flush=True)
    gesture.start()
    try:
        if args.preview:
            source = gesture.source
            if not isinstance(source, LiveHandSource):
                raise RuntimeError("preview requires LiveHandSource")
            _run_preview_loop(source, preview_state, preview_lock, server.url, stop)
        else:
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
