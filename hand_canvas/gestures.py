"""Gesture state machine → pointer / grab events.

Interprets hand landmarks only. Does not touch canvas or geometry shapes.

- Pinch → create / stretch / resize (Pointer*). Uses thumb–index midpoint.
- Closed fist → select + move (Grab*); open hand releases.

Pinch distance is scaled by hand size and must stay under threshold for several
frames before PointerDown, so a resting hand does not spawn points.
"""

from __future__ import annotations

from enum import Enum

from hand_canvas.constants import (
    FIST_END_CURLS,
    FIST_START_CURLS,
    PINCH_CONFIRM_FRAMES,
    PINCH_END_MAX,
    PINCH_END_MIN,
    PINCH_END_RATIO,
    PINCH_START_MAX,
    PINCH_START_MIN,
    PINCH_START_RATIO,
)
from hand_canvas.events import (
    GrabDown,
    GrabMove,
    GrabUp,
    PointerDown,
    PointerEvent,
    PointerMove,
    PointerUp,
)
from hand_canvas.geometry import Point, distance
from hand_canvas.hand_tracker import (
    FINGER_PAIRS,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    WRIST,
    Hand,
)

_FIST_FINGER_PAIRS = FINGER_PAIRS[1:]


class GestureState(Enum):
    IDLE = "IDLE"
    POINTING = "POINTING"
    PINCHING = "PINCHING"
    FIST = "FIST"


def _is_index_extended(hand: Hand) -> bool:
    tip = hand.landmarks[INDEX_TIP]
    pip = hand.landmarks[INDEX_PIP]
    wrist = hand.landmarks[WRIST]
    return distance(tip, wrist) > distance(pip, wrist)


def _curled_count(hand: Hand, pairs: tuple[tuple[int, int], ...]) -> int:
    wrist = hand.landmarks[WRIST]
    curled = 0
    for tip_i, pip_i in pairs:
        tip = hand.landmarks[tip_i]
        pip = hand.landmarks[pip_i]
        if distance(tip, wrist) <= distance(pip, wrist) * 1.08:
            curled += 1
    return curled


def _palm_point(hand: Hand) -> Point:
    return hand.landmarks[MIDDLE_MCP]


def _pinch_point(hand: Hand) -> Point:
    a, b = hand.index_tip, hand.thumb_tip
    return Point(x=(a.x + b.x) * 0.5, y=(a.y + b.y) * 0.5)


def _hand_scale(hand: Hand) -> float:
    return max(distance(hand.landmarks[WRIST], hand.landmarks[MIDDLE_MCP]), 1e-3)


def _pinch_thresholds(hand: Hand) -> tuple[float, float]:
    scale = _hand_scale(hand)
    start = min(max(PINCH_START_RATIO * scale, PINCH_START_MIN), PINCH_START_MAX)
    end = min(max(PINCH_END_RATIO * scale, PINCH_END_MIN), PINCH_END_MAX)
    if end <= start:
        end = start + 0.02
    return start, end


def _tag(event: PointerEvent, pointer_id: str) -> PointerEvent:
    cls = type(event)
    return cls(position=event.position, pointer_id=pointer_id)


class GestureDetector:
    def __init__(
        self,
        fist_start_curls: int = FIST_START_CURLS,
        fist_end_curls: int = FIST_END_CURLS,
        confirm_frames: int = PINCH_CONFIRM_FRAMES,
    ) -> None:
        self._fist_start = fist_start_curls
        self._fist_end = fist_end_curls
        self._confirm_frames = max(1, confirm_frames)
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._was_fisting = False
        self._pinch_hold = 0
        self._last_cursor = Point(0.5, 0.5)
        self._last_palm = Point(0.5, 0.5)
        self.last_pinch_dist = 1.0
        self.last_pinch_start = PINCH_START_MAX
        self.last_pinch_end = PINCH_END_MAX

    def update(self, hand: Hand | None) -> list[PointerEvent]:
        if hand is None:
            events: list[PointerEvent] = []
            if self._was_pinching:
                events.append(PointerUp(position=self._last_cursor))
                self._was_pinching = False
            if self._was_fisting:
                events.append(GrabUp(position=self._last_palm))
                self._was_fisting = False
            self._pinch_hold = 0
            self.state = GestureState.IDLE
            self.last_pinch_dist = 1.0
            return events

        index = hand.index_tip
        palm = _palm_point(hand)
        pinch_pos = _pinch_point(hand)
        pinch_dist = distance(hand.index_tip, hand.thumb_tip)
        pinch_start, pinch_end = _pinch_thresholds(hand)
        self.last_pinch_dist = pinch_dist
        self.last_pinch_start = pinch_start
        self.last_pinch_end = pinch_end
        pointing = _is_index_extended(hand)
        fist_curls = _curled_count(hand, _FIST_FINGER_PAIRS)

        events: list[PointerEvent] = []

        if self._was_fisting:
            if pinch_dist < pinch_start:
                events.append(GrabUp(position=palm))
                self._was_fisting = False
                self._pinch_hold = self._confirm_frames
                events.append(PointerDown(position=pinch_pos))
                events.append(PointerMove(position=pinch_pos))
                self._was_pinching = True
                self._last_cursor = pinch_pos
                self.state = GestureState.PINCHING
                return events
            if fist_curls <= self._fist_end:
                events.append(GrabUp(position=palm))
                self._was_fisting = False
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
                return events
            events.append(GrabMove(position=palm))
            self._last_palm = palm
            self.state = GestureState.FIST
            return events

        if self._was_pinching:
            self._last_cursor = pinch_pos
            self._last_palm = palm
            if pinch_dist > pinch_end:
                events.append(PointerUp(position=pinch_pos))
                self._was_pinching = False
                self._pinch_hold = 0
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            else:
                events.append(PointerMove(position=pinch_pos))
                self.state = GestureState.PINCHING
            return events

        if pinch_dist < pinch_start:
            self._pinch_hold += 1
            self._last_cursor = pinch_pos
            self._last_palm = palm
            if self._pinch_hold >= self._confirm_frames:
                events.append(PointerDown(position=pinch_pos))
                events.append(PointerMove(position=pinch_pos))
                self._was_pinching = True
                self.state = GestureState.PINCHING
            else:
                # Warm-up: show intent without creating yet
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            return events

        self._pinch_hold = 0

        if fist_curls >= self._fist_start and pinch_dist > pinch_end:
            events.append(GrabDown(position=palm))
            events.append(GrabMove(position=palm))
            self._was_fisting = True
            self._last_palm = palm
            self.state = GestureState.FIST
            return events

        self._last_cursor = index
        self._last_palm = palm
        if pointing:
            events.append(PointerMove(position=index))
            self.state = GestureState.POINTING
        else:
            self.state = GestureState.IDLE
        return events


class MultiHandGestureDetector:
    """One independent gesture state machine per handedness label."""

    def __init__(self) -> None:
        self._detectors: dict[str, GestureDetector] = {}

    def _get(self, pointer_id: str) -> GestureDetector:
        if pointer_id not in self._detectors:
            self._detectors[pointer_id] = GestureDetector()
        return self._detectors[pointer_id]

    def update(self, hands: list[Hand]) -> list[PointerEvent]:
        events: list[PointerEvent] = []
        seen: set[str] = set()

        for hand in hands:
            pointer_id = hand.handedness
            seen.add(pointer_id)
            detector = self._get(pointer_id)
            for event in detector.update(hand):
                events.append(_tag(event, pointer_id))

        for pointer_id, detector in list(self._detectors.items()):
            if pointer_id not in seen:
                for event in detector.update(None):
                    events.append(_tag(event, pointer_id))

        return events

    @property
    def states(self) -> dict[str, GestureState]:
        return {pid: det.state for pid, det in self._detectors.items()}

    @property
    def pinch_dists(self) -> dict[str, float]:
        return {pid: det.last_pinch_dist for pid, det in self._detectors.items()}

    @property
    def pinch_thresholds(self) -> dict[str, tuple[float, float]]:
        return {
            pid: (det.last_pinch_start, det.last_pinch_end)
            for pid, det in self._detectors.items()
        }
