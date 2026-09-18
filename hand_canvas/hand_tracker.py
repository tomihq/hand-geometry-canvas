"""Hand tracking for the canvas app — thin adapter over hand_interaction.

MediaPipe lives in ``hand_interaction.landmarker``. This module keeps the
canvas-facing ``Hand`` / ``Point`` types and 2D one-euro smoothing used by
gestures.py.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from hand_canvas.constants import (
    SMOOTH_BETA,
    SMOOTH_DERIVATIVE_CUTOFF,
    SMOOTH_MIN_CUTOFF,
    TRACKING_MAX_WIDTH,
    USE_GPU_INFERENCE,
)
from hand_canvas.geometry import Point
from hand_interaction.camera import Frame
from hand_interaction.constants import INDEX_TIP, THUMB_TIP, WRIST
from hand_interaction.landmarker import MediaPipeLandmarker
from hand_interaction.pose import LandmarkHand

# Re-export landmark indices gestures.py imports from here.
INDEX_PIP = 6
MIDDLE_MCP = 9
FINGER_PAIRS = ((8, 6), (12, 10), (16, 14), (20, 18))

NOMINAL_DT = 1.0 / 30.0
MIN_DT = 1.0 / 240.0
MAX_DT = 0.2


def _cutoff_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class Smoother:
    """One-euro filter for normalized 2D points (canvas gesture cursor)."""

    def __init__(
        self,
        min_cutoff: float = SMOOTH_MIN_CUTOFF,
        beta: float = SMOOTH_BETA,
        derivative_cutoff: float = SMOOTH_DERIVATIVE_CUTOFF,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivative_cutoff = derivative_cutoff
        self._previous: list[Point] | None = None
        self._speeds: list[Point] | None = None

    def reset(self) -> None:
        self._previous = None
        self._speeds = None

    def smooth(self, points: list[Point], dt: float = NOMINAL_DT) -> list[Point]:
        if (
            self._previous is None
            or self._speeds is None
            or len(self._previous) != len(points)
        ):
            self._previous = list(points)
            self._speeds = [Point(0.0, 0.0) for _ in points]
            return list(points)

        speed_alpha = _cutoff_alpha(self._derivative_cutoff, dt)
        smoothed: list[Point] = []
        speeds: list[Point] = []
        for current, prev, prev_speed in zip(points, self._previous, self._speeds):
            speed = Point(
                x=speed_alpha * (current.x - prev.x) / dt
                + (1.0 - speed_alpha) * prev_speed.x,
                y=speed_alpha * (current.y - prev.y) / dt
                + (1.0 - speed_alpha) * prev_speed.y,
            )
            alpha = _cutoff_alpha(
                self._min_cutoff + self._beta * math.hypot(speed.x, speed.y), dt
            )
            smoothed.append(
                Point(
                    x=alpha * current.x + (1.0 - alpha) * prev.x,
                    y=alpha * current.y + (1.0 - alpha) * prev.y,
                )
            )
            speeds.append(speed)

        self._previous = smoothed
        self._speeds = speeds
        return smoothed


@dataclass
class Hand:
    index_tip: Point
    thumb_tip: Point
    landmarks: list[Point]
    handedness: str = "Unknown"


def _landmark_hand_to_points(hand: LandmarkHand) -> list[Point]:
    # Canvas gestures use image-space y (origin top); landmarker stores raw MP y.
    return [Point(x=float(p[0]), y=float(p[1])) for p in hand.landmarks]


class HandTracker:
    """Canvas adapter: MediaPipeLandmarker → smoothed 2D Hand list."""

    def __init__(
        self,
        model_path: Path | None = None,
        num_hands: int = 2,
        use_gpu: bool = USE_GPU_INFERENCE,
    ) -> None:
        self._landmarker = MediaPipeLandmarker(
            model_path=model_path,
            num_hands=num_hands,
            use_gpu=use_gpu,
            tracking_max_width=TRACKING_MAX_WIDTH,
        )
        self._smoothers: dict[str, Smoother] = {}
        self._last_frame_s: float | None = None
        self.delegate = self._landmarker.delegate

    def _frame_dt(self) -> float:
        now = time.perf_counter()
        previous = self._last_frame_s
        self._last_frame_s = now
        if previous is None:
            return NOMINAL_DT
        return min(max(now - previous, MIN_DT), MAX_DT)

    def process(self, frame: Frame) -> list[Hand]:
        dt = self._frame_dt()
        detected = self._landmarker.detect(frame)
        if not detected:
            for smoother in self._smoothers.values():
                smoother.reset()
            return []

        hands: list[Hand] = []
        seen: set[str] = set()
        for hand in detected:
            label = hand.hand_id
            seen.add(label)
            points = _landmark_hand_to_points(hand)
            smoother = self._smoothers.setdefault(label, Smoother())
            smoothed = smoother.smooth(points, dt)
            hands.append(
                Hand(
                    index_tip=smoothed[INDEX_TIP],
                    thumb_tip=smoothed[THUMB_TIP],
                    landmarks=smoothed,
                    handedness=label,
                )
            )

        for label in list(self._smoothers):
            if label not in seen:
                self._smoothers[label].reset()

        return hands

    def close(self) -> None:
        self._landmarker.close()


# Back-compat aliases if anything imported ensure_model from here.
from hand_interaction.landmarker import ensure_model  # noqa: E402

__all__ = [
    "FINGER_PAIRS",
    "Hand",
    "HandTracker",
    "INDEX_PIP",
    "INDEX_TIP",
    "MIDDLE_MCP",
    "Smoother",
    "THUMB_TIP",
    "WRIST",
    "ensure_model",
]
