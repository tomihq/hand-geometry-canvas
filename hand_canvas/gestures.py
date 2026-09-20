"""Canvas gestures: shared pinch/fist detector + sweep-to-clear.

Pinch and fist live in ``hand_interaction.gestures``. This module maps those
events onto canvas Pointer*/Grab* types and adds the open-palm wipe.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from hand_canvas.constants import (
    FIST_CURL_RATIO_OFF,
    SWEEP_COOLDOWN_FRAMES,
    SWEEP_HORIZONTAL_RATIO,
    SWEEP_MIN_TRAVEL,
    SWEEP_TRAVEL_FRAMES,
)
from hand_canvas.events import (
    GrabDown,
    GrabMove,
    GrabUp,
    PointerDown,
    PointerEvent,
    PointerMove,
    PointerUp,
    SweepClear,
)
from hand_canvas.geometry import Point
from hand_canvas.hand_tracker import Hand
from hand_interaction.gestures import (
    GestureHand,
    GestureState,
    MultiHandGestureDetector as SharedGestureDetector,
)
from hand_interaction.types import (
    GrabEnd,
    GrabMove as SharedGrabMove,
    GrabStart,
    HandEvent,
    PinchEnd,
    PinchMove,
    PinchStart,
)


def _to_point(x: float, y: float) -> Point:
    return Point(x, y)


def _hand_to_gesture(hand: Hand) -> GestureHand:
    landmarks = np.array(
        [(p.x, p.y, 0.0) for p in hand.landmarks], dtype=np.float64
    )
    return GestureHand(hand_id=hand.handedness, landmarks=landmarks)


def _map_event(event: HandEvent) -> PointerEvent | None:
    if isinstance(event, PinchStart):
        return PointerDown(
            position=_to_point(event.position.x, event.position.y),
            pointer_id=event.hand_id,
        )
    if isinstance(event, PinchMove):
        return PointerMove(
            position=_to_point(event.position.x, event.position.y),
            pointer_id=event.hand_id,
        )
    if isinstance(event, PinchEnd):
        return PointerUp(
            position=_to_point(event.position.x, event.position.y),
            pointer_id=event.hand_id,
        )
    if isinstance(event, GrabStart):
        return GrabDown(
            position=_to_point(event.position.x, event.position.y),
            pointer_id=event.hand_id,
            fingers_together=event.fingers_together,
        )
    if isinstance(event, SharedGrabMove):
        return GrabMove(
            position=_to_point(event.position.x, event.position.y),
            pointer_id=event.hand_id,
            fingers_together=event.fingers_together,
        )
    if isinstance(event, GrabEnd):
        return GrabUp(
            position=_to_point(event.position.x, event.position.y),
            pointer_id=event.hand_id,
        )
    return None


class _SweepTracker:
    """Open palm carried sideways across a good chunk of the frame."""

    def __init__(self) -> None:
        self._palm_track: deque[tuple[Point, bool]] = deque(
            maxlen=SWEEP_TRAVEL_FRAMES
        )
        self._cooldown = 0

    def reset(self) -> None:
        self._palm_track.clear()
        self._cooldown = 0

    def update(self, hand: Hand | None, busy: bool) -> SweepClear | None:
        if self._cooldown > 0:
            self._cooldown -= 1

        if hand is None or busy:
            self._palm_track.clear()
            return None

        pts = np.array(
            [(p.x, p.y, 0.0) for p in hand.landmarks], dtype=np.float64
        )
        palm_ids = np.array([0, 5, 9, 13, 17])
        tip_ids = np.array([8, 12, 16, 20])
        palm = pts[palm_ids].mean(axis=0)[:2]
        from hand_interaction.pose import hand_scale

        scale = hand_scale(pts)
        curls = np.linalg.norm(pts[tip_ids, :2] - palm, axis=1) / scale
        loose_curls = int(np.count_nonzero(curls <= FIST_CURL_RATIO_OFF))
        palm_open = loose_curls == 0
        pos = Point(float(palm[0]), float(palm[1]))
        self._palm_track.append((pos, palm_open))

        if self._cooldown > 0 or not self._swept():
            return None

        self._cooldown = SWEEP_COOLDOWN_FRAMES
        self._palm_track.clear()
        return SweepClear(position=pos, pointer_id=hand.handedness)

    def _swept(self) -> bool:
        if len(self._palm_track) < self._palm_track.maxlen:
            return False
        if not all(palm_open for _pos, palm_open in self._palm_track):
            return False

        xs = np.array([pos.x for pos, _open in self._palm_track])
        ys = np.array([pos.y for pos, _open in self._palm_track])
        dx = float(xs[-1] - xs[0])
        dy = float(ys[-1] - ys[0])

        if abs(dx) < SWEEP_MIN_TRAVEL:
            return False
        if abs(dx) < SWEEP_HORIZONTAL_RATIO * abs(dy):
            return False

        steps = np.diff(xs)
        total = float(np.sum(np.abs(steps)))
        if total <= 0.0:
            return False
        return float(np.sum(steps * np.sign(dx))) / total > 0.8


class MultiHandGestureDetector:
    """Shared pinch/fist detector plus per-hand sweep-to-clear."""

    def __init__(self) -> None:
        self._shared = SharedGestureDetector()
        self._sweeps: dict[str, _SweepTracker] = {}

    def _sweep(self, pointer_id: str) -> _SweepTracker:
        if pointer_id not in self._sweeps:
            self._sweeps[pointer_id] = _SweepTracker()
        return self._sweeps[pointer_id]

    def update(
        self, hands: list[Hand], now: float | None = None
    ) -> list[PointerEvent]:
        gesture_hands = [_hand_to_gesture(h) for h in hands]
        shared_events = self._shared.update(gesture_hands, now=now)

        events: list[PointerEvent] = []
        for event in shared_events:
            mapped = _map_event(event)
            if mapped is not None:
                events.append(mapped)

        busy_ids = {
            hid
            for hid, state in self._shared.states.items()
            if state in (GestureState.PINCHING, GestureState.FIST)
        }
        seen: set[str] = set()
        for hand in hands:
            pid = hand.handedness
            seen.add(pid)
            sweep = self._sweep(pid).update(hand, busy=pid in busy_ids)
            if sweep is not None:
                events.append(sweep)

        for pid, tracker in list(self._sweeps.items()):
            if pid not in seen:
                tracker.update(None, busy=False)

        return events

    @property
    def states(self) -> dict[str, GestureState]:
        return self._shared.states

    @property
    def pinch_dists(self) -> dict[str, float]:
        return self._shared.pinch_dists

    @property
    def pinch_thresholds(self) -> dict[str, tuple[float, float]]:
        return self._shared.pinch_thresholds

    @property
    def fist_scores(self) -> dict[str, int]:
        return self._shared.fist_scores
