"""Hand Geometry Canvas — entry point.

Pipeline:
  Camera → HandTracker (≤2 hands) → MultiHandGestureDetector → InteractionEngine → Canvas

Each hand has its own gesture state machine (not OS threads). MediaPipe returns
both hands in one frame; sessions are keyed by handedness (Left / Right).
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from hand_canvas.camera import Camera
from hand_canvas.canvas import Canvas
from hand_canvas.gestures import GestureState, MultiHandGestureDetector
from hand_canvas.hand_tracker import Hand, HandTracker
from hand_canvas.interaction import InteractionEngine, InteractionState
from hand_canvas.geometry import Point


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# BGR colors per handedness
HAND_COLORS = {
    "Left": ((0, 255, 255), (255, 128, 0)),   # index cyan, thumb orange
    "Right": ((255, 100, 255), (100, 255, 100)),  # index magenta, thumb green
}
DEFAULT_COLORS = ((200, 200, 200), (160, 160, 160))


def _to_px(point: Point, width: int, height: int) -> tuple[int, int]:
    return int(point.x * width), int(point.y * height)


def _colors_for(handedness: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    for key, colors in HAND_COLORS.items():
        if handedness.startswith(key):
            return colors
    return DEFAULT_COLORS


def _draw_hand(image: np.ndarray, hand: Hand) -> None:
    h, w = image.shape[:2]
    idx_color, thumb_color = _colors_for(hand.handedness)
    pts = [_to_px(p, w, h) for p in hand.landmarks]

    for a, b in HAND_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(image, pts[a], pts[b], (180, 180, 180), 1)
    for pt in pts:
        cv2.circle(image, pt, 3, (200, 200, 200), -1)

    idx = _to_px(hand.index_tip, w, h)
    thb = _to_px(hand.thumb_tip, w, h)
    cv2.circle(image, idx, 8, idx_color, -1)
    cv2.circle(image, thb, 8, thumb_color, -1)
    cv2.line(image, idx, thb, idx_color, 2)

    label_pos = (idx[0] + 10, idx[1] - 10)
    cv2.putText(
        image,
        hand.handedness,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        idx_color,
        1,
        cv2.LINE_AA,
    )


def draw_debug(
    image: np.ndarray,
    hands: list[Hand],
    gesture_states: dict[str, GestureState],
    interaction: InteractionEngine,
) -> None:
    h, w = image.shape[:2]
    debug = interaction.debug

    for hand in hands:
        _draw_hand(image, hand)

    for session in debug.sessions.values():
        if session.drag_origin is not None:
            origin_px = _to_px(session.drag_origin, w, h)
            cv2.circle(image, origin_px, 6, (0, 0, 255), 2)
            cv2.putText(
                image,
                "P0",
                (origin_px[0] + 8, origin_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    lines = [f"Hands: {len(hands)}"]
    for pid, gstate in sorted(gesture_states.items()):
        session = debug.sessions.get(pid)
        if session is not None and session.state != InteractionState.IDLE:
            label = session.state.value
            obj = session.selected_id or "-"
        else:
            label = gstate.value
            obj = "-"
        lines.append(f"{pid}: {label}  obj={obj}")

    y = 28
    for line in lines:
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24


def run() -> int:
    camera = Camera(device_index=0, mirror=True)
    tracker: HandTracker | None = None
    try:
        camera.open()
        tracker = HandTracker(num_hands=2)
        gestures = MultiHandGestureDetector()
        interaction = InteractionEngine()
        canvas = Canvas()

        print("Hand Geometry Canvas running (2 hands). Press 'q' to quit.")

        while True:
            frame = camera.read()
            hands = tracker.process(frame)
            events = gestures.update(hands)
            interaction.handle(events, canvas)

            display = canvas.render(
                frame.width,
                frame.height,
                background=frame.image,
                selected_ids=interaction.selected_ids(),
            )
            draw_debug(display, hands, gestures.states, interaction)

            cv2.imshow("Hand Geometry Canvas", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tracker is not None:
            tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
