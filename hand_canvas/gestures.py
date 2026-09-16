"""Gesture state machine → pointer events.

Interprets hand landmarks only. Does not touch canvas or geometry shapes.

One GestureDetector instance per hand (logical stream). No OS threads needed:
MediaPipe already returns both hands per frame; each detector keeps its own
pinch hysteresis / state independently.
"""

from __future__ import annotations

from enum import Enum

from hand_canvas.constants import PINCH_END_THRESHOLD, PINCH_START_THRESHOLD
from hand_canvas.events import PointerDown, PointerEvent, PointerMove, PointerUp
from hand_canvas.geometry import Point, distance
from hand_canvas.hand_tracker import INDEX_PIP, INDEX_TIP, WRIST, Hand


class GestureState(Enum):
    IDLE = "IDLE"
    POINTING = "POINTING"
    PINCHING = "PINCHING"


def _is_index_extended(hand: Hand) -> bool:
    """Index tip farther from wrist than PIP joint (simple extension check)."""
    tip = hand.landmarks[INDEX_TIP]
    pip = hand.landmarks[INDEX_PIP]
    wrist = hand.landmarks[WRIST]
    return distance(tip, wrist) > distance(pip, wrist)


def _tag(event: PointerEvent, pointer_id: str) -> PointerEvent:
    if isinstance(event, PointerDown):
        return PointerDown(position=event.position, pointer_id=pointer_id)
    if isinstance(event, PointerUp):
        return PointerUp(position=event.position, pointer_id=pointer_id)
    return PointerMove(position=event.position, pointer_id=pointer_id)


class GestureDetector:
    def __init__(
        self,
        pinch_start: float = PINCH_START_THRESHOLD,
        pinch_end: float = PINCH_END_THRESHOLD,
    ) -> None:
        if pinch_end <= pinch_start:
            raise ValueError("PINCH_END_THRESHOLD must be > PINCH_START_THRESHOLD")
        self._pinch_start = pinch_start
        self._pinch_end = pinch_end
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._last_cursor = Point(0.5, 0.5)

    def update(self, hand: Hand | None) -> list[PointerEvent]:
        if hand is None:
            events: list[PointerEvent] = []
            if self._was_pinching:
                events.append(PointerUp(position=self._last_cursor))
                self._was_pinching = False
            self.state = GestureState.IDLE
            return events

        cursor = hand.index_tip
        self._last_cursor = cursor
        pinch_dist = distance(hand.index_tip, hand.thumb_tip)
        pointing = _is_index_extended(hand)

        events = []

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
