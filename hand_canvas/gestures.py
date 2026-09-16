"""Gesture state machine → pointer / grab events.

Interprets hand landmarks only. Does not touch canvas or geometry shapes.

- Pinch → create / stretch / resize (Pointer*). Midpoint of thumb–index.
- Closed fist → select + move (Grab*); open hand releases.

Landmarks are noisy, so nothing here compares raw distances:

  1. Every measure is a ratio of hand size, making it independent of how far
     the hand is from the camera. Hand size itself is the *median* of several
     palm spans, so one bad landmark cannot skew it.
  2. Measures are median-filtered over a short frame window, so a single
     outlier frame can never start or stop a gesture.
  3. Thresholds come in ON/OFF pairs. Between them lies a margin band where
     the previous state simply holds, instead of flapping on borderline values.
     A pinch scales its own OFF threshold off the gap it is holding, so the
     band is as wide for a loose pinch as for a tight one.

Both gestures must be deliberate, so a relaxed or half-closed hand does nothing:
  - Grab: fingertips tucked onto the palm (index included), held briefly.
  - Pinch: thumb and index tips touching, out away from the palm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from hand_canvas.constants import (
    FIST_CURL_RATIO_OFF,
    FIST_CURL_RATIO_ON,
    FIST_FINGERS_OFF,
    FIST_FINGERS_ON,
    FIST_TIPS_PALM_MAX,
    GESTURE_WINDOW_FRAMES,
    INDEX_EXTENDED_RATIO,
    MIN_HAND_SCALE,
    PINCH_BASELINE_FRAMES,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    PINCH_RELEASE_FACTOR,
    PINCH_RELEASE_MAX,
    PINCH_TIPS_PALM_MIN,
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
from hand_canvas.hand_tracker import (
    INDEX_PIP,
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
    Hand,
)

# Landmark groups (MediaPipe hand topology)
_PALM_IDS = np.array([0, 5, 9, 13, 17])
_MCP_IDS = np.array([5, 9, 13, 17])
_TIP_IDS = np.array([8, 12, 16, 20])


class GestureState(Enum):
    IDLE = "IDLE"
    POINTING = "POINTING"
    PINCHING = "PINCHING"
    FIST = "FIST"


@dataclass(frozen=True)
class HandMetrics:
    """Scale-free descriptors of a hand pose (all ratios of hand size)."""

    scale: float
    pinch: float
    tips_palm: float
    curls: np.ndarray
    index_extended: float
    palm: Point
    pinch_point: Point
    index_point: Point


def _landmark_array(hand: Hand) -> np.ndarray:
    return np.array([(p.x, p.y) for p in hand.landmarks], dtype=np.float64)


def _hand_scale(points: np.ndarray) -> float:
    """Median of several palm spans — robust to any single bad landmark."""
    spans = np.linalg.norm(points[_MCP_IDS] - points[WRIST], axis=1)
    spans = np.append(spans, np.linalg.norm(points[5] - points[17]))
    return max(float(np.median(spans)), MIN_HAND_SCALE)


def _metrics(hand: Hand) -> HandMetrics:
    points = _landmark_array(hand)
    scale = _hand_scale(points)

    palm = points[_PALM_IDS].mean(axis=0)
    pinch_point = (points[THUMB_TIP] + points[INDEX_TIP]) * 0.5

    pinch = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP])) / scale
    tips_palm = float(np.linalg.norm(pinch_point - palm)) / scale
    curls = np.linalg.norm(points[_TIP_IDS] - palm, axis=1) / scale

    wrist = points[WRIST]
    pip_span = max(float(np.linalg.norm(points[INDEX_PIP] - wrist)), MIN_HAND_SCALE)
    index_extended = float(np.linalg.norm(points[INDEX_TIP] - wrist)) / pip_span

    return HandMetrics(
        scale=scale,
        pinch=pinch,
        tips_palm=tips_palm,
        curls=curls,
        index_extended=index_extended,
        palm=Point(float(palm[0]), float(palm[1])),
        pinch_point=Point(float(pinch_point[0]), float(pinch_point[1])),
        index_point=Point(
            float(points[INDEX_TIP][0]), float(points[INDEX_TIP][1])
        ),
    )


def _tag(event: PointerEvent, pointer_id: str) -> PointerEvent:
    cls = type(event)
    return cls(position=event.position, pointer_id=pointer_id)


class GestureDetector:
    """One hand's gesture state, driven by median-filtered pose ratios."""

    def __init__(self, window: int = GESTURE_WINDOW_FRAMES) -> None:
        self._window = max(1, window)
        self._history: deque[HandMetrics] = deque(maxlen=self._window)
        self._palm_track: deque[tuple[Point, bool]] = deque(
            maxlen=SWEEP_TRAVEL_FRAMES
        )
        self._pinch_baseline: deque[float] = deque(maxlen=PINCH_BASELINE_FRAMES)
        self._sweep_cooldown = 0
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._was_fisting = False
        self._last_cursor = Point(0.5, 0.5)
        self._last_palm = Point(0.5, 0.5)
        self.last_pinch_ratio = 1.0
        self.last_release_ratio = PINCH_RATIO_OFF
        self.last_fist_score = 0

    def _smoothed(self) -> HandMetrics:
        """Median over the window: one outlier frame cannot decide anything."""
        recent = list(self._history)
        latest = recent[-1]
        if len(recent) == 1:
            return latest
        return HandMetrics(
            scale=float(np.median([m.scale for m in recent])),
            pinch=float(np.median([m.pinch for m in recent])),
            tips_palm=float(np.median([m.tips_palm for m in recent])),
            curls=np.median(np.stack([m.curls for m in recent]), axis=0),
            index_extended=float(np.median([m.index_extended for m in recent])),
            # Positions stay live so the cursor never lags behind the hand
            palm=latest.palm,
            pinch_point=latest.pinch_point,
            index_point=latest.index_point,
        )

    def _pinch_release_ratio(self) -> float:
        """The gap that counts as letting go of the pinch being held.

        Measured against the gap the pinch has actually been holding, because
        the same absolute band means very different things: a pinch entered at
        0.30 is a couple of centimetres from PINCH_RATIO_OFF, so relaxing the
        fingers mid-drag released it and committed the figure half-drawn.
        """
        if not self._pinch_baseline:
            return PINCH_RATIO_OFF
        held = float(np.median(self._pinch_baseline))
        return min(
            PINCH_RELEASE_MAX, max(PINCH_RATIO_OFF, held * PINCH_RELEASE_FACTOR)
        )

    def _swept(self) -> bool:
        """Open palm carried sideways across a good chunk of the frame."""
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

        # Mostly one-way travel: a swipe, not a hand waving back and forth.
        steps = np.diff(xs)
        total = float(np.sum(np.abs(steps)))
        if total <= 0.0:
            return False
        return float(np.sum(steps * np.sign(dx))) / total > 0.8

    def _release_all(self) -> list[PointerEvent]:
        events: list[PointerEvent] = []
        if self._was_pinching:
            events.append(PointerUp(position=self._last_cursor))
            self._was_pinching = False
        if self._was_fisting:
            events.append(GrabUp(position=self._last_palm))
            self._was_fisting = False
        return events

    def update(self, hand: Hand | None) -> list[PointerEvent]:
        if hand is None:
            events = self._release_all()
            self._history.clear()
            self._palm_track.clear()
            self._pinch_baseline.clear()
            self.state = GestureState.IDLE
            self.last_pinch_ratio = 1.0
            self.last_release_ratio = PINCH_RATIO_OFF
            self.last_fist_score = 0
            return events

        self._history.append(_metrics(hand))
        m = self._smoothed()
        # A gesture may only *start* once the window is full, which is the
        # hold that separates intent from a hand passing through the pose.
        settled = len(self._history) >= self._window

        palm = m.palm
        pinch_pos = m.pinch_point
        self._last_palm = palm
        self.last_pinch_ratio = m.pinch
        self.last_fist_score = int(np.count_nonzero(m.curls <= FIST_CURL_RATIO_ON))

        index_extended = m.index_extended > INDEX_EXTENDED_RATIO
        tips_over_palm = m.tips_palm < FIST_TIPS_PALM_MAX
        tips_reaching_out = m.tips_palm > PINCH_TIPS_PALM_MIN

        # Closed hand: enough fingertips tucked onto the palm, index included
        closed_hand = (
            self.last_fist_score >= FIST_FINGERS_ON
            and tips_over_palm
            and not index_extended
        )
        loose_curls = int(np.count_nonzero(m.curls <= FIST_CURL_RATIO_OFF))
        hand_opened = loose_curls <= FIST_FINGERS_OFF
        # Sweep needs a flat open palm: every finger clearly away from the palm
        self._palm_track.append((palm, loose_curls == 0))
        if self._sweep_cooldown > 0:
            self._sweep_cooldown -= 1

        pinch_pose = (
            m.pinch < PINCH_RATIO_ON and tips_reaching_out and not closed_hand
        )
        # Only frames that unambiguously read as held feed the reference, so a
        # hand on its way open cannot drag its own exit threshold up with it.
        if self._was_pinching and m.pinch <= PINCH_RATIO_OFF:
            self._pinch_baseline.append(m.pinch)
        self.last_release_ratio = self._pinch_release_ratio()
        pinch_released = m.pinch > self.last_release_ratio

        events: list[PointerEvent] = []

        # --- Holding a grab ---
        if self._was_fisting:
            # Reaching out into a pinch hands control over to the pinch cursor
            if pinch_pose:
                events.append(GrabUp(position=palm))
                self._was_fisting = False
                events.extend(self._begin_pinch(pinch_pos))
                return events
            if hand_opened:
                events.append(GrabUp(position=palm))
                self._was_fisting = False
                self.state = (
                    GestureState.POINTING if index_extended else GestureState.IDLE
                )
                return events
            events.append(GrabMove(position=palm))
            self._last_cursor = palm
            self.state = GestureState.FIST
            return events

        # --- Holding a pinch ---
        if self._was_pinching:
            # Closing the hand upgrades the pinch into a grab
            if closed_hand:
                events.append(PointerUp(position=pinch_pos))
                self._was_pinching = False
                events.extend(self._begin_grab(palm))
                return events
            self._last_cursor = pinch_pos
            if pinch_released:
                events.append(PointerUp(position=pinch_pos))
                self._was_pinching = False
                self.state = (
                    GestureState.POINTING if index_extended else GestureState.IDLE
                )
            else:
                events.append(PointerMove(position=pinch_pos))
                self.state = GestureState.PINCHING
            return events

        # --- Starting a gesture (closed hand wins: it is never a pinch) ---
        if closed_hand:
            self.state = GestureState.FIST
            if settled:
                events.extend(self._begin_grab(palm))
            return events

        if pinch_pose:
            if settled:
                events.extend(self._begin_pinch(pinch_pos))
            else:
                self.state = (
                    GestureState.POINTING if index_extended else GestureState.IDLE
                )
            return events

        if self._sweep_cooldown == 0 and self._swept():
            self._sweep_cooldown = SWEEP_COOLDOWN_FRAMES
            self._palm_track.clear()
            events.append(SweepClear(position=palm))
            self.state = GestureState.IDLE
            return events

        self._last_cursor = m.index_point
        if index_extended:
            events.append(PointerMove(position=m.index_point))
            self.state = GestureState.POINTING
        else:
            self.state = GestureState.IDLE
        return events

    def _begin_pinch(self, pinch_pos: Point) -> list[PointerEvent]:
        self._pinch_baseline.clear()
        self._was_pinching = True
        self._last_cursor = pinch_pos
        self.state = GestureState.PINCHING
        return [
            PointerDown(position=pinch_pos),
            PointerMove(position=pinch_pos),
        ]

    def _begin_grab(self, palm: Point) -> list[PointerEvent]:
        self._was_fisting = True
        self._last_cursor = palm
        self.state = GestureState.FIST
        return [GrabDown(position=palm), GrabMove(position=palm)]


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
        """Smoothed thumb–index gap as a fraction of hand size."""
        return {pid: det.last_pinch_ratio for pid, det in self._detectors.items()}

    @property
    def pinch_thresholds(self) -> dict[str, tuple[float, float]]:
        """Entry threshold plus the live exit, which follows the held gap."""
        return {
            pid: (PINCH_RATIO_ON, det.last_release_ratio)
            for pid, det in self._detectors.items()
        }

    @property
    def fist_scores(self) -> dict[str, int]:
        return {pid: det.last_fist_score for pid, det in self._detectors.items()}
