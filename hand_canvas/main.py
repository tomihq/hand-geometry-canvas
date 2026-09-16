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
from hand_canvas.constants import (
    SHOW_DEBUG_OVERLAY,
    WINDOW_HEIGHT,
    WINDOW_NAME,
    WINDOW_WIDTH,
)
from hand_canvas.gestures import GestureState, MultiHandGestureDetector
from hand_canvas.gpu_renderer import GpuCanvasRenderer
from hand_canvas.hand_tracker import Hand, HandTracker
from hand_canvas.interaction import InteractionEngine, InteractionState, trash_zone
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


def draw_trash_zone(image: np.ndarray, armed: bool, undo_depth: int) -> None:
    """Drop target for deleting a held figure; glows red once armed."""
    h, w = image.shape[:2]
    zone = trash_zone()
    x1, y1 = int(zone.x * w), int(zone.y * h)
    x2, y2 = int((zone.x + zone.width) * w), int((zone.y + zone.height) * h)

    color = (60, 60, 235) if armed else (150, 150, 150)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.30 if armed else 0.15, image, 0.70 if armed else 0.85, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if armed else 1)

    # Bin glyph: lid plus body
    cx = (x1 + x2) // 2
    bin_w = max(int((x2 - x1) * 0.34), 8)
    lid_y = y1 + int((y2 - y1) * 0.30)
    body_bottom = y2 - int((y2 - y1) * 0.22)
    cv2.line(image, (cx - bin_w, lid_y), (cx + bin_w, lid_y), color, 2)
    cv2.rectangle(
        image,
        (cx - int(bin_w * 0.75), lid_y + 3),
        (cx + int(bin_w * 0.75), body_bottom),
        color,
        2,
    )

    label = "DROP" if armed else (f"undo z ({undo_depth})" if undo_depth else "trash")
    cv2.putText(
        image,
        label,
        (x1 + 6, y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


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
    pinch_dists: dict[str, float] | None = None,
    pinch_thresholds: dict[str, tuple[float, float]] | None = None,
    fist_scores: dict[str, int] | None = None,
) -> None:
    h, w = image.shape[:2]
    debug = interaction.debug
    pinch_dists = pinch_dists or {}
    pinch_thresholds = pinch_thresholds or {}
    fist_scores = fist_scores or {}

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
    if debug.lock_owner:
        lines.append(f"Lock: {debug.lock_owner} → {debug.selected_id or '-'}")
    for pid, gstate in sorted(gesture_states.items()):
        session = debug.sessions.get(pid)
        dist = pinch_dists.get(pid)
        thr = pinch_thresholds.get(pid)
        fist = fist_scores.get(pid)
        parts = []
        if dist is not None and thr is not None:
            parts.append(f"d={dist:.3f}/{thr[0]:.3f}")
        if fist is not None:
            parts.append(f"fist={fist}/4")
        dist_txt = f"  {'  '.join(parts)}" if parts else ""
        if session is not None and session.state != InteractionState.IDLE:
            role = "owner" if session.is_owner else "resize"
            label = f"{session.state.value}({role})"
            obj = session.selected_id or "-"
        else:
            label = gstate.value
            obj = "-"
        lines.append(f"{pid}: {label}{dist_txt}  obj={obj}")

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
    camera = Camera(
        device_index=0,
        mirror=True,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
    tracker: HandTracker | None = None
    gpu: GpuCanvasRenderer | None = None
    try:
        camera.open()
        tracker = HandTracker(num_hands=2)
        gestures = MultiHandGestureDetector()
        interaction = InteractionEngine()
        canvas = Canvas()
        gpu = GpuCanvasRenderer.try_create(WINDOW_WIDTH, WINDOW_HEIGHT)

        # WINDOW_GUI_NORMAL: no toolbar/buttons. WINDOW_NORMAL: resizable.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

        show_debug = SHOW_DEBUG_OVERLAY
        print(
            "Hand Geometry Canvas running (2 hands).\n"
            "  Delete: grab a figure with a fist and open your hand over the "
            "trash zone (bottom-right).\n"
            "  Clear all: swipe an open palm sideways across the frame.\n"
            "  Keys: 'z' undo delete, 'd' debug overlay, 'q' quit."
        )

        while True:
            frame = camera.read()
            hands = tracker.process(frame)
            events = gestures.update(hands)
            interaction.handle(events, canvas)

            selected = interaction.selected_ids()
            if gpu is not None:
                display = gpu.render(canvas, frame.image, selected_ids=selected)
            else:
                display = canvas.render(
                    frame.width,
                    frame.height,
                    background=frame.image,
                    selected_ids=selected,
                )
            draw_trash_zone(
                display,
                armed=bool(interaction.pending_delete_ids(canvas)),
                undo_depth=canvas.trash_depth,
            )
            if show_debug:
                draw_debug(
                    display,
                    hands,
                    gestures.states,
                    interaction,
                    pinch_dists=gestures.pinch_dists,
                    pinch_thresholds=gestures.pinch_thresholds,
                    fist_scores=gestures.fist_scores,
                )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
            if key == ord("z"):
                restored = canvas.restore_last()
                if restored:
                    print(f"Restored {len(restored)} figure(s).")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tracker is not None:
            tracker.close()
        if gpu is not None:
            gpu.close()
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
