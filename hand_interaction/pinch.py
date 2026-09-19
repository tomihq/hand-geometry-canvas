"""Pinch detector: OPEN ↔ PINCHING with 3D ratio hysteresis.

Emits only transition events (pinch.start / pinch.end) with abstract Vector2
positions. Fist/sweep stay in the canvas app — this is the shared pinch core.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np

from hand_interaction.constants import (
    GESTURE_WINDOW_FRAMES,
    INDEX_TIP,
    PINCH_BASELINE_FRAMES,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    PINCH_RELEASE_FACTOR,
    PINCH_RELEASE_MAX,
    THUMB_TIP,
)
from hand_interaction.pose import hand_scale, to_hybrid_position
from hand_interaction.types import HandEvent, PinchEnd, PinchStart, Vector2


class PinchPhase(Enum):
    OPEN = "OPEN"
    PINCHING = "PINCHING"


def pinch_ratio_3d(landmarks: np.ndarray) -> float:
    """‖thumb − index‖₃ / hand_scale."""
    pts = np.asarray(landmarks, dtype=np.float64)
    scale = hand_scale(pts)
    gap = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
    return gap / scale


def pinch_midpoint_2d(landmarks: np.ndarray) -> Vector2:
    """Thumb–index midpoint in public hybrid screen space (x, y')."""
    pts = np.asarray(landmarks, dtype=np.float64)
    mid = 0.5 * (pts[THUMB_TIP] + pts[INDEX_TIP])
    hybrid = to_hybrid_position(mid, hand_scale(pts))
    return Vector2(hybrid.x, hybrid.y)


class PinchDetector:
    """One hand's pinch state machine (median window + ON/OFF hysteresis)."""

    def __init__(self, window: int = GESTURE_WINDOW_FRAMES) -> None:
        self._window = max(1, window)
        self._ratios: deque[float] = deque(maxlen=self._window)
        self._baseline: deque[float] = deque(maxlen=PINCH_BASELINE_FRAMES)
        self.phase = PinchPhase.OPEN
        self.last_ratio = 1.0
        self.last_release_ratio = PINCH_RATIO_OFF
        self._last_position = Vector2(0.5, 0.5)

    @property
    def last_position(self) -> Vector2:
        """Most recent thumb–index midpoint (hybrid screen space)."""
        return self._last_position

    def reset(self) -> None:
        self._ratios.clear()
        self._baseline.clear()
        self.phase = PinchPhase.OPEN
        self.last_ratio = 1.0
        self.last_release_ratio = PINCH_RATIO_OFF

    def _smoothed_ratio(self) -> float:
        return float(np.median(self._ratios))

    def _release_ratio(self) -> float:
        if not self._baseline:
            return PINCH_RATIO_OFF
        held = float(np.median(self._baseline))
        return min(
            PINCH_RELEASE_MAX, max(PINCH_RATIO_OFF, held * PINCH_RELEASE_FACTOR)
        )

    def update(self, landmarks: np.ndarray, hand_id: str) -> list[HandEvent]:
        """Feed one frame of landmarks; return 0–1 transition events."""
        pts = np.asarray(landmarks, dtype=np.float64)
        if pts.shape != (21, 3):
            raise ValueError(f"expected landmarks shape (21, 3), got {pts.shape}")

        ratio = pinch_ratio_3d(pts)
        position = pinch_midpoint_2d(pts)
        self._ratios.append(ratio)
        self.last_ratio = self._smoothed_ratio()
        self._last_position = position
        settled = len(self._ratios) >= self._window

        events: list[HandEvent] = []

        if self.phase is PinchPhase.PINCHING:
            if ratio <= PINCH_RATIO_OFF:
                self._baseline.append(ratio)
            self.last_release_ratio = self._release_ratio()
            if self.last_ratio > self.last_release_ratio:
                self.phase = PinchPhase.OPEN
                self._baseline.clear()
                events.append(PinchEnd(hand_id=hand_id, position=position))
            return events

        # OPEN → only enter after the median window is full (deliberate hold).
        self.last_release_ratio = PINCH_RATIO_OFF
        if settled and self.last_ratio < PINCH_RATIO_ON:
            self.phase = PinchPhase.PINCHING
            self._baseline.clear()
            self._baseline.append(ratio)
            events.append(PinchStart(hand_id=hand_id, position=position))
        return events

    def force_end(self, hand_id: str) -> list[HandEvent]:
        """Emit pinch.end if currently pinching (e.g. hand lost after grace)."""
        if self.phase is not PinchPhase.PINCHING:
            return []
        self.phase = PinchPhase.OPEN
        self._baseline.clear()
        self._ratios.clear()
        return [PinchEnd(hand_id=hand_id, position=self._last_position)]
