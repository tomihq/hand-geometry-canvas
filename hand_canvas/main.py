"""Hand Geometry Canvas — entry point.

Pipeline:
  Camera → HandTracker (≤2 hands) → MultiHandGestureDetector → InteractionEngine → Canvas

Each hand has its own gesture state machine (not OS threads). MediaPipe returns
both hands in one frame; sessions are keyed by handedness (Left / Right).
"""

from __future__ import annotations

import sys
import time
from collections import deque

import cv2
import numpy as np

from hand_canvas.camera import Camera
from hand_canvas.canvas import Canvas
from hand_canvas.constants import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    SHOW_DEBUG_OVERLAY,
    SHOW_FPS,
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


def _col(image: np.ndarray, bgr: tuple[int, int, int]):
    """A 3-tuple on a BGRA overlay leaves alpha at 0, i.e. invisible."""
    return (*bgr, 255) if image.shape[2] == 4 else bgr


def draw_trash_zone(image: np.ndarray, armed: bool, undo_depth: int) -> None:
    """Drop target for deleting a held figure; glows red once armed."""
    h, w = image.shape[:2]
    zone = trash_zone()
    x1, y1 = int(zone.x * w), int(zone.y * h)
    x2, y2 = int((zone.x + zone.width) * w), int((zone.y + zone.height) * h)

    color = (60, 60, 235) if armed else (150, 150, 150)
    fill = 0.30 if armed else 0.15
    if image.shape[2] == 4:
        # Premultiplied so the GPU blend matches what cv2 does on a clear layer.
        alpha = int(255 * fill)
        b, g, r = color
        premultiplied = (int(b * fill), int(g * fill), int(r * fill), alpha)
        cv2.rectangle(image, (x1, y1), (x2, y2), premultiplied, -1)
    else:
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, fill, image, 1.0 - fill, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), _col(image, color), 2 if armed else 1)

    # Bin glyph: lid plus body
    cx = (x1 + x2) // 2
    bin_w = max(int((x2 - x1) * 0.34), 8)
    lid_y = y1 + int((y2 - y1) * 0.30)
    body_bottom = y2 - int((y2 - y1) * 0.22)
    cv2.line(image, (cx - bin_w, lid_y), (cx + bin_w, lid_y), _col(image, color), 2)
    cv2.rectangle(
        image,
        (cx - int(bin_w * 0.75), lid_y + 3),
        (cx + int(bin_w * 0.75), body_bottom),
        _col(image, color),
        2,
    )

    label = "DROP" if armed else (f"undo z ({undo_depth})" if undo_depth else "trash")
    cv2.putText(
        image,
        label,
        (x1 + 6, y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        _col(image, color),
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
            cv2.line(image, pts[a], pts[b], _col(image, (180, 180, 180)), 1)
    for pt in pts:
        cv2.circle(image, pt, 3, _col(image, (200, 200, 200)), -1)

    idx = _to_px(hand.index_tip, w, h)
    thb = _to_px(hand.thumb_tip, w, h)
    cv2.circle(image, idx, 8, _col(image, idx_color), -1)
    cv2.circle(image, thb, 8, _col(image, thumb_color), -1)
    cv2.line(image, idx, thb, _col(image, idx_color), 2)

    label_pos = (idx[0] + 10, idx[1] - 10)
    cv2.putText(
        image,
        hand.handedness,
        label_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        _col(image, idx_color),
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
            cv2.circle(image, origin_px, 6, _col(image, (0, 0, 255)), 2)
            cv2.putText(
                image,
                "P0",
                (origin_px[0] + 8, origin_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                _col(image, (0, 0, 255)),
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
            parts.append(f"d={dist:.3f} on<{thr[0]:.2f} off>{thr[1]:.2f}")
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
            _col(image, (0, 0, 0)),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _col(image, (255, 255, 255)),
            1,
            cv2.LINE_AA,
        )
        y += 24


class FpsMeter:
    """Frame rate over a sliding window, plus a throttle for reporting it."""

    def __init__(self, window: int = 60) -> None:
        self._stamps: deque[float] = deque(maxlen=window)
        self._last_report = 0.0
        self.value = 0.0

    def tick(self) -> None:
        self._stamps.append(time.perf_counter())
        span = self._stamps[-1] - self._stamps[0]
        if len(self._stamps) >= 2 and span > 0:
            self.value = (len(self._stamps) - 1) / span

    def due(self, every: float = 0.4) -> bool:
        now = time.perf_counter()
        if now - self._last_report < every:
            return False
        self._last_report = now
        return True


class Hud:
    """The HUD is the only thing still rasterized on the CPU, so it is cached
    and rebuilt only when its contents actually change. With the debug overlay
    off that means once, instead of every frame."""

    def __init__(self) -> None:
        self._layer: np.ndarray | None = None
        self._key: tuple | None = None
        self.version = 0

    def layer(
        self,
        width: int,
        height: int,
        armed: bool,
        undo_depth: int,
        debug: tuple | None,
    ) -> np.ndarray:
        key = (width, height, armed, undo_depth)
        if debug is None and self._layer is not None and key == self._key:
            return self._layer

        layer = np.zeros((height, width, 4), dtype=np.uint8)
        draw_trash_zone(layer, armed=armed, undo_depth=undo_depth)
        if debug is not None:
            hands, gestures, interaction = debug
            draw_debug(
                layer,
                hands,
                gestures.states,
                interaction,
                pinch_dists=gestures.pinch_dists,
                pinch_thresholds=gestures.pinch_thresholds,
                fist_scores=gestures.fist_scores,
            )

        self._layer = layer
        self._key = key
        self.version += 1
        return layer


def run() -> int:
    camera = Camera(
        device_index=CAMERA_INDEX,
        mirror=True,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )
    tracker: HandTracker | None = None
    gpu: GpuCanvasRenderer | None = None
    try:
        camera.open()
        print(f"Camera: {camera.describe()}")
        tracker = HandTracker(num_hands=2)
        gestures = MultiHandGestureDetector()
        interaction = InteractionEngine()
        canvas = Canvas()
        gpu = GpuCanvasRenderer.try_create(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME)

        if gpu is None:
            # WINDOW_GUI_NORMAL: no toolbar/buttons. WINDOW_NORMAL: resizable.
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

        show_debug = SHOW_DEBUG_OVERLAY
        hud = Hud()
        fps = FpsMeter()
        print(
            "Hand Geometry Canvas running (2 hands).\n"
            "  Delete: grab a figure with a fist and take it to the trash zone "
            "(bottom-right). It settles into your hand as you approach, and "
            "anything inside the zone is deleted.\n"
            "  Clear all: swipe an open palm sideways across the frame.\n"
            "  Keys: 'z' undo delete, 'd' debug overlay, 'q' quit."
        )

        while True:
            frame = camera.read()
            hands = tracker.process(frame)
            events = gestures.update(hands)
            interaction.handle(events, canvas)

            selected = interaction.selected_ids()
            armed = bool(interaction.pending_delete_ids(canvas))

            if gpu is not None:
                overlay = hud.layer(
                    frame.width,
                    frame.height,
                    armed,
                    canvas.trash_depth,
                    (hands, gestures, interaction) if show_debug else None,
                )
                gpu.present(
                    canvas,
                    frame.image,
                    selected_ids=selected,
                    overlay=overlay,
                    overlay_version=hud.version,
                )
                if gpu.should_close:
                    break
                keys = gpu.poll_keys()
            else:
                display = canvas.render(
                    frame.width,
                    frame.height,
                    background=frame.image,
                    selected_ids=selected,
                )
                draw_trash_zone(display, armed=armed, undo_depth=canvas.trash_depth)
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
                keys = [chr(key)] if 32 <= key <= 126 else []

            if "q" in keys:
                break
            if "d" in keys:
                show_debug = not show_debug
            if "z" in keys:
                restored = canvas.restore_last()
                if restored:
                    print(f"Restored {len(restored)} figure(s).")

            fps.tick()
            if SHOW_FPS and gpu is not None and fps.due():
                gpu.set_title(f"{WINDOW_NAME} — {fps.value:.0f} fps")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        if tracker is not None:
            tracker.close()
        if gpu is not None:
            gpu.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
