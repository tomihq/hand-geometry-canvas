"""Gesture state machine → pointer / grab events.

Interprets hand landmarks only. Does not touch canvas or geometry shapes.

- Pinch → create / stretch / resize (Pointer*)
- Closed fist → select + move figures (Grab*); opening the hand releases
"""

from __future__ import annotations

from enum import Enum

from hand_canvas.constants import (
    FIST_END_CURLS,
    FIST_START_CURLS,
    PINCH_END_THRESHOLD,
    PINCH_START_THRESHOLD,
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


def _curled_finger_count(hand: Hand) -> int:
    """How many non-thumb fingers look curled (tip closer to wrist than PIP)."""
    wrist = hand.landmarks[WRIST]
    curled = 0
    for tip_i, pip_i in FINGER_PAIRS:
        tip = hand.landmarks[tip_i]
        pip = hand.landmarks[pip_i]
        if distance(tip, wrist) <= distance(pip, wrist) * 1.08:
            curled += 1
    return curled


def _palm_point(hand: Hand) -> Point:
    return hand.landmarks[MIDDLE_MCP]


def _tag(event: PointerEvent, pointer_id: str) -> PointerEvent:
    cls = type(event)
    return cls(position=event.position, pointer_id=pointer_id)


class GestureDetector:
    def __init__(
        self,
        pinch_start: float = PINCH_START_THRESHOLD,
        pinch_end: float = PINCH_END_THRESHOLD,
        fist_start_curls: int = FIST_START_CURLS,
        fist_end_curls: int = FIST_END_CURLS,
    ) -> None:
        if pinch_end <= pinch_start:
            raise ValueError("PINCH_END_THRESHOLD must be > PINCH_START_THRESHOLD")
        self._pinch_start = pinch_start
        self._pinch_end = pinch_end
        self._fist_start = fist_start_curls
        self._fist_end = fist_end_curls
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._was_fisting = False
        self._last_cursor = Point(0.5, 0.5)
        self._last_palm = Point(0.5, 0.5)

    def update(self, hand: Hand | None) -> list[PointerEvent]:
        if hand is None:
            events: list[PointerEvent] = []
            if self._was_pinching:
                events.append(PointerUp(position=self._last_cursor))
                self._was_pinching = False
            if self._was_fisting:
                events.append(GrabUp(position=self._last_palm))
                self._was_fisting = False
            self.state = GestureState.IDLE
            return events

        cursor = hand.index_tip
        palm = _palm_point(hand)
        self._last_cursor = cursor
        self._last_palm = palm
        pinch_dist = distance(hand.index_tip, hand.thumb_tip)
        pointing = _is_index_extended(hand)
        curls = _curled_finger_count(hand)

        events: list[PointerEvent] = []

        # Fist has priority over pinch (select / move)
        if self._was_fisting:
            if curls <= self._fist_end:
                events.append(GrabUp(position=palm))
                self._was_fisting = False
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            else:
                events.append(GrabMove(position=palm))
                self.state = GestureState.FIST
                return events
        elif curls >= self._fist_start:
            if self._was_pinching:
                events.append(PointerUp(position=cursor))
                self._was_pinching = False
            events.append(GrabDown(position=palm))
            events.append(GrabMove(position=palm))
            self._was_fisting = True
            self.state = GestureState.FIST
            return events

        # Pinch (create / stretch / resize) — only when not fisting
        if self._was_pinching:
            if pinch_dist > self._pinch_end:
                events.append(PointerUp(position=cursor))
                self._was_pinching = False
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            else:
                events.append(PointerMove(position=cursor))
                self.state = GestureState.PINCHING
        else:
            if pinch_dist < self._pinch_start and pointing:
                events.append(PointerDown(position=cursor))
                events.append(PointerMove(position=cursor))
                self._was_pinching = True
                self.state = GestureState.PINCHING
            elif pointing:
                events.append(PointerMove(position=cursor))
                self.state = GestureState.POINTING
            else:
                self.state = GestureState.IDLE

        return events


class MultiHandGestureDetector:
    """One independent gesture state machine per handedness label."""

    def __init__(
        self,
        pinch_start: float = PINCH_START_THRESHOLD,
        pinch_end: float = PINCH_END_THRESHOLD,
    ) -> None:
        self._pinch_start = pinch_start
        self._pinch_end = pinch_end
        self._detectors: dict[str, GestureDetector] = {}

    def _get(self, pointer_id: str) -> GestureDetector:
        if pointer_id not in self._detectors:
            self._detectors[pointer_id] = GestureDetector(
                pinch_start=self._pinch_start,
                pinch_end=self._pinch_end,
            )
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
