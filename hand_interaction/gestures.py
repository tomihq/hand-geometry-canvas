"""Gesture state machine → pinch / grab events (canvas model, no sweep).

Interprets hand landmarks only. Does not know about figures or rendering.

- Pinch → PinchStart / PinchMove / PinchEnd (thumb–index midpoint)
- Closed fist → GrabStart / GrabMove / GrabEnd; open hand releases

Landmarks are noisy, so nothing here compares raw distances:

  1. Every measure is a ratio of hand size (median of palm spans).
  2. Measures are median-filtered over a short frame window.
  3. Thresholds come in ON/OFF pairs with a hold-to-start window.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from hand_interaction.constants import (
    FIST_CURL_RATIO_OFF,
    FIST_CURL_RATIO_ON,
    FIST_FINGERS_ON,
    FIST_OPEN_CURL_MIN,
    FIST_OPEN_FINGERS_MIN,
    FIST_OPEN_TIPS_PALM_MIN,
    FIST_RELEASE_FRAMES,
    FIST_TIPS_PALM_MAX,
    GESTURE_WINDOW_FRAMES,
    HAND_LOST_GRACE_FRAMES,
    INDEX_EXTENDED_RATIO,
    INDEX_PIP,
    INDEX_TIP,
    MCP_IDS,
    MIN_HAND_SCALE,
    PALM_IDS,
    PINCH_BASELINE_FRAMES,
    PINCH_CONTACT_RATIO,
    PINCH_FREE_FINGERS_MIN,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    PINCH_RELEASE_FACTOR,
    PINCH_RELEASE_MAX,
    PINCH_TIPS_PALM_MIN,
    TIP_IDS,
    THUMB_TIP,
    WRIST,
)
from hand_interaction.types import (
    GrabEnd,
    GrabMove,
    GrabStart,
    HandEvent,
    PinchEnd,
    PinchMove,
    PinchStart,
    Vector2,
)

_PALM_IDS = np.array(PALM_IDS)
_MCP_IDS = np.array(MCP_IDS)
_TIP_IDS = np.array(TIP_IDS)


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
    palm: Vector2
    pinch_point: Vector2
    index_point: Vector2


@dataclass(frozen=True)
class GestureHand:
    """One hand for gesture detection (image-space landmarks)."""

    hand_id: str
    landmarks: np.ndarray  # (21, 3) or (21, 2+)

    def __post_init__(self) -> None:
        pts = np.asarray(self.landmarks, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] != 21 or pts.shape[1] < 2:
            raise ValueError(
                f"expected landmarks shape (21, ≥2), got {pts.shape}"
            )
        object.__setattr__(self, "landmarks", pts)


def _xy(points: np.ndarray) -> np.ndarray:
    return np.asarray(points[:, :2], dtype=np.float64)


def _hand_scale(points: np.ndarray) -> float:
    """Median of several palm spans — robust to any single bad landmark."""
    xy = _xy(points)
    spans = np.linalg.norm(xy[_MCP_IDS] - xy[WRIST], axis=1)
    spans = np.append(spans, np.linalg.norm(xy[5] - xy[17]))
    return max(float(np.median(spans)), MIN_HAND_SCALE)


def _metrics(landmarks: np.ndarray) -> HandMetrics:
    points = _xy(landmarks)
    scale = _hand_scale(landmarks)

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
        palm=Vector2(float(palm[0]), float(palm[1])),
        pinch_point=Vector2(float(pinch_point[0]), float(pinch_point[1])),
        index_point=Vector2(
            float(points[INDEX_TIP][0]), float(points[INDEX_TIP][1])
        ),
    )


class GestureDetector:
    """One hand's gesture state, driven by median-filtered pose ratios."""

    def __init__(self, window: int = GESTURE_WINDOW_FRAMES) -> None:
        self._window = max(1, window)
        self._history: deque[HandMetrics] = deque(maxlen=self._window)
        self._pinch_baseline: deque[float] = deque(maxlen=PINCH_BASELINE_FRAMES)
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._was_fisting = False
        self._handover_frames = 0
        self._lost_frames = 0
        self._last_cursor = Vector2(0.5, 0.5)
        self._last_palm = Vector2(0.5, 0.5)
        self.last_pinch_ratio = 1.0
        self.last_release_ratio = PINCH_RATIO_OFF
        self.last_fist_score = 0

    @property
    def last_cursor(self) -> Vector2:
        """Active interaction cursor (pinch midpoint or palm while fisting)."""
        return self._last_cursor

    def reset(self) -> None:
        self._history.clear()
        self._pinch_baseline.clear()
        self.state = GestureState.IDLE
        self._was_pinching = False
        self._was_fisting = False
        self._handover_frames = 0
        self._lost_frames = 0
        self.last_pinch_ratio = 1.0
        self.last_release_ratio = PINCH_RATIO_OFF
        self.last_fist_score = 0

    def _smoothed(self) -> HandMetrics:
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
            palm=latest.palm,
            pinch_point=latest.pinch_point,
            index_point=latest.index_point,
        )

    def _pinch_release_ratio(self) -> float:
        if not self._pinch_baseline:
            return PINCH_RATIO_OFF
        held = float(np.median(self._pinch_baseline))
        return min(
            PINCH_RELEASE_MAX, max(PINCH_RATIO_OFF, held * PINCH_RELEASE_FACTOR)
        )

    @staticmethod
    def _hand_clearly_open(m: HandMetrics) -> bool:
        open_fingers = int(np.count_nonzero(m.curls >= FIST_OPEN_CURL_MIN))
        return (
            open_fingers >= FIST_OPEN_FINGERS_MIN
            and m.tips_palm >= FIST_OPEN_TIPS_PALM_MIN
        )

    def _release_all(self, hand_id: str) -> list[HandEvent]:
        events: list[HandEvent] = []
        if self._was_pinching:
            events.append(PinchEnd(hand_id=hand_id, position=self._last_cursor))
            self._was_pinching = False
        if self._was_fisting:
            events.append(GrabEnd(hand_id=hand_id, position=self._last_palm))
            self._was_fisting = False
        return events

    def update(
        self, landmarks: np.ndarray | None, hand_id: str
    ) -> list[HandEvent]:
        if landmarks is None:
            if (self._was_fisting or self._was_pinching) and (
                self._lost_frames < HAND_LOST_GRACE_FRAMES
            ):
                self._lost_frames += 1
                return []
            events = self._release_all(hand_id)
            self.reset()
            return events

        self._lost_frames = 0
        live = _metrics(landmarks)
        self._history.append(live)
        m = self._smoothed()
        settled = len(self._history) >= self._window

        palm = m.palm
        pinch_pos = m.pinch_point
        self._last_palm = palm
        self.last_pinch_ratio = m.pinch
        self.last_fist_score = int(np.count_nonzero(m.curls <= FIST_CURL_RATIO_ON))

        index_extended = m.index_extended > INDEX_EXTENDED_RATIO
        tips_over_palm = m.tips_palm < FIST_TIPS_PALM_MAX
        free_fingers_out = float(m.curls[1:].min()) > PINCH_FREE_FINGERS_MIN
        tips_reaching_out = m.tips_palm > PINCH_TIPS_PALM_MIN or free_fingers_out
        fingers_together = m.pinch < PINCH_CONTACT_RATIO

        closed_hand = (
            self.last_fist_score >= FIST_FINGERS_ON
            and tips_over_palm
            and not index_extended
        )

        pinch_pose = (
            m.pinch < PINCH_RATIO_ON and tips_reaching_out and not closed_hand
        )
        if self._was_pinching and m.pinch <= PINCH_RATIO_OFF:
            self._pinch_baseline.append(m.pinch)
        self.last_release_ratio = self._pinch_release_ratio()
        pinch_released = m.pinch > self.last_release_ratio

        events: list[HandEvent] = []

        if self._was_fisting:
            self._handover_frames = self._handover_frames + 1 if pinch_pose else 0

            if self._hand_clearly_open(live):
                events.append(GrabEnd(hand_id=hand_id, position=palm))
                self._was_fisting = False
                self.state = (
                    GestureState.POINTING
                    if live.index_extended > INDEX_EXTENDED_RATIO
                    else GestureState.IDLE
                )
                return events

            if self._handover_frames >= FIST_RELEASE_FRAMES:
                events.append(GrabEnd(hand_id=hand_id, position=palm))
                self._was_fisting = False
                events.extend(self._begin_pinch(hand_id, pinch_pos))
                return events
            events.append(
                GrabMove(
                    hand_id=hand_id,
                    position=palm,
                    fingers_together=fingers_together,
                )
            )
            self._last_cursor = palm
            self.state = GestureState.FIST
            return events

        if self._was_pinching:
            if closed_hand:
                events.append(PinchEnd(hand_id=hand_id, position=pinch_pos))
                self._was_pinching = False
                events.extend(self._begin_grab(hand_id, palm, fingers_together))
                return events
            self._last_cursor = pinch_pos
            if pinch_released:
                events.append(PinchEnd(hand_id=hand_id, position=pinch_pos))
                self._was_pinching = False
                self.state = (
                    GestureState.POINTING if index_extended else GestureState.IDLE
                )
            else:
                events.append(PinchMove(hand_id=hand_id, position=pinch_pos))
                self.state = GestureState.PINCHING
            return events

        if closed_hand:
            self.state = GestureState.FIST
            if settled:
                events.extend(self._begin_grab(hand_id, palm, fingers_together))
            return events

        if pinch_pose:
            if settled:
                events.extend(self._begin_pinch(hand_id, pinch_pos))
            else:
                self.state = (
                    GestureState.POINTING if index_extended else GestureState.IDLE
                )
            return events

        self._last_cursor = m.index_point
        self.state = (
            GestureState.POINTING if index_extended else GestureState.IDLE
        )
        return events

    def _begin_pinch(self, hand_id: str, pinch_pos: Vector2) -> list[HandEvent]:
        self._pinch_baseline.clear()
        self._was_pinching = True
        self._last_cursor = pinch_pos
        self.state = GestureState.PINCHING
        return [
            PinchStart(hand_id=hand_id, position=pinch_pos),
            PinchMove(hand_id=hand_id, position=pinch_pos),
        ]

    def _begin_grab(
        self,
        hand_id: str,
        palm: Vector2,
        fingers_together: bool = False,
    ) -> list[HandEvent]:
        self._was_fisting = True
        self._handover_frames = 0
        self._last_cursor = palm
        self.state = GestureState.FIST
        return [
            GrabStart(
                hand_id=hand_id,
                position=palm,
                fingers_together=fingers_together,
            ),
            GrabMove(
                hand_id=hand_id,
                position=palm,
                fingers_together=fingers_together,
            ),
        ]


class MultiHandGestureDetector:
    """One independent gesture state machine per hand_id."""

    def __init__(self) -> None:
        self._detectors: dict[str, GestureDetector] = {}

    def reset(self) -> None:
        self._detectors.clear()

    def _get(self, hand_id: str) -> GestureDetector:
        if hand_id not in self._detectors:
            self._detectors[hand_id] = GestureDetector()
        return self._detectors[hand_id]

    def update(
        self, hands: list[GestureHand], now: float | None = None
    ) -> list[HandEvent]:
        _ = time.perf_counter() if now is None else now
        events: list[HandEvent] = []
        seen: set[str] = set()

        for hand in hands:
            seen.add(hand.hand_id)
            detector = self._get(hand.hand_id)
            events.extend(detector.update(hand.landmarks, hand.hand_id))

        for hand_id, detector in list(self._detectors.items()):
            if hand_id not in seen:
                events.extend(detector.update(None, hand_id))

        return events

    @property
    def states(self) -> dict[str, GestureState]:
        return {hid: det.state for hid, det in self._detectors.items()}

    @property
    def pinch_dists(self) -> dict[str, float]:
        return {hid: det.last_pinch_ratio for hid, det in self._detectors.items()}

    @property
    def pinch_thresholds(self) -> dict[str, tuple[float, float]]:
        return {
            hid: (PINCH_RATIO_ON, det.last_release_ratio)
            for hid, det in self._detectors.items()
        }

    @property
    def fist_scores(self) -> dict[str, int]:
        return {hid: det.last_fist_score for hid, det in self._detectors.items()}

    @property
    def cursors(self) -> dict[str, Vector2]:
        return {hid: det.last_cursor for hid, det in self._detectors.items()}
