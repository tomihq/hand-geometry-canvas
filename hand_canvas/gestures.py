"""Gesture state machine → pointer / grab events.

- Pinch → create / stretch / resize (Pointer*). Midpoint of thumb–index.
- Closed fist → select + move (Grab*); open hand releases.

Fist is detected by fingertip proximity to the palm (not thumb–index distance),
so a closed hand is never mistaken for a pinch. Grab needs a short hold
(~FIST_CONFIRM_FRAMES) before firing — about 300–500ms at typical FPS.
"""

from __future__ import annotations

from enum import Enum

from hand_canvas.constants import (
    FIST_CONFIRM_FRAMES,
    FIST_SCORE_END,
    FIST_SCORE_START,
    FIST_TIP_PALM_RATIO,
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
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    WRIST,
    Hand,
)

_PALM_LANDMARKS = (0, 5, 9, 13, 17)
_FINGERTIPS = (8, 12, 16, 20)


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


def _hand_scale(hand: Hand) -> float:
    return max(distance(hand.landmarks[WRIST], hand.landmarks[MIDDLE_MCP]), 1e-3)


def _palm_point(hand: Hand) -> Point:
    """Stable grab cursor = average of wrist + finger MCPs."""
    xs = sum(hand.landmarks[i].x for i in _PALM_LANDMARKS) / len(_PALM_LANDMARKS)
    ys = sum(hand.landmarks[i].y for i in _PALM_LANDMARKS) / len(_PALM_LANDMARKS)
    return Point(xs, ys)


def _pinch_point(hand: Hand) -> Point:
    a, b = hand.index_tip, hand.thumb_tip
    return Point(x=(a.x + b.x) * 0.5, y=(a.y + b.y) * 0.5)


def _fist_score(hand: Hand) -> int:
    """How many fingertips sit near the palm (closed hand)."""
    palm = _palm_point(hand)
    limit = FIST_TIP_PALM_RATIO * _hand_scale(hand)
    return sum(
        1
        for tip_i in _FINGERTIPS
        if distance(hand.landmarks[tip_i], palm) <= limit
    )


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
        pinch_confirm: int = PINCH_CONFIRM_FRAMES,
        fist_confirm: int = FIST_CONFIRM_FRAMES,
    ) -> None:
        self._pinch_confirm = max(1, pinch_confirm)
        self._fist_confirm = max(1, fist_confirm)
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._was_fisting = False
        self._pinch_hold = 0
        self._fist_hold = 0
        self._last_cursor = Point(0.5, 0.5)
        self._last_palm = Point(0.5, 0.5)
        self.last_pinch_dist = 1.0
        self.last_pinch_start = PINCH_START_MAX
        self.last_pinch_end = PINCH_END_MAX
        self.last_fist_score = 0

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
            self._fist_hold = 0
            self.state = GestureState.IDLE
            self.last_pinch_dist = 1.0
            self.last_fist_score = 0
            return events

        index = hand.index_tip
        palm = _palm_point(hand)
        pinch_pos = _pinch_point(hand)
        pinch_dist = distance(hand.index_tip, hand.thumb_tip)
        pinch_start, pinch_end = _pinch_thresholds(hand)
        score = _fist_score(hand)
        pointing = _is_index_extended(hand)

        self.last_pinch_dist = pinch_dist
        self.last_pinch_start = pinch_start
        self.last_pinch_end = pinch_end
        self.last_fist_score = score
        self._last_palm = palm

        fist_pose = score >= FIST_SCORE_START
        fist_open = score <= FIST_SCORE_END

        events: list[PointerEvent] = []

        # --- Already grabbing ---
        if self._was_fisting:
            if fist_open:
                events.append(GrabUp(position=palm))
                self._was_fisting = False
                self._fist_hold = 0
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            else:
                events.append(GrabMove(position=palm))
                self.state = GestureState.FIST
            return events

        # --- Closing into a fist (wins over pinch — tips near each other is normal) ---
        if fist_pose:
            if self._was_pinching:
                events.append(PointerUp(position=self._last_cursor))
                self._was_pinching = False
            self._pinch_hold = 0
            self._fist_hold += 1
            self._last_cursor = palm
            if self._fist_hold >= self._fist_confirm:
                if not self._was_fisting:
                    events.append(GrabDown(position=palm))
                events.append(GrabMove(position=palm))
                self._was_fisting = True
                self.state = GestureState.FIST
            else:
                # Holding closed — show FIST intent without grabbing yet
                self.state = GestureState.FIST
            return events

        self._fist_hold = 0

        # --- Active pinch ---
        if self._was_pinching:
            self._last_cursor = pinch_pos
            if pinch_dist > pinch_end:
                events.append(PointerUp(position=pinch_pos))
                self._was_pinching = False
                self._pinch_hold = 0
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            else:
                events.append(PointerMove(position=pinch_pos))
                self.state = GestureState.PINCHING
            return events

        # --- Start pinch (only when clearly not a fist) ---
        if pinch_dist < pinch_start:
            self._pinch_hold += 1
            self._last_cursor = pinch_pos
            if self._pinch_hold >= self._pinch_confirm:
                events.append(PointerDown(position=pinch_pos))
                events.append(PointerMove(position=pinch_pos))
                self._was_pinching = True
                self.state = GestureState.PINCHING
            else:
                self.state = GestureState.POINTING if pointing else GestureState.IDLE
            return events

        self._pinch_hold = 0
        self._last_cursor = index
        if pointing:
            events.append(PointerMove(position=index))
            self.state = GestureState.POINTING
        else:
            self.state = GestureState.IDLE
        return events


class MultiHandGestureDetector:
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

    @property
    def fist_scores(self) -> dict[str, int]:
        return {pid: det.last_fist_score for pid, det in self._detectors.items()}
