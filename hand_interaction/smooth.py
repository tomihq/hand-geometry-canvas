"""One-euro smoothing for Vector3 (shared vision infra)."""

from __future__ import annotations

import math

from hand_interaction.constants import (
    NOMINAL_DT,
    SMOOTH_BETA,
    SMOOTH_DERIVATIVE_CUTOFF,
    SMOOTH_MIN_CUTOFF_XY,
    SMOOTH_MIN_CUTOFF_Z,
)
from hand_interaction.types import Vector3


def _cutoff_alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroVector3:
    """One-euro filter with a stricter cutoff on z (relative depth)."""

    def __init__(
        self,
        min_cutoff_xy: float = SMOOTH_MIN_CUTOFF_XY,
        min_cutoff_z: float = SMOOTH_MIN_CUTOFF_Z,
        beta: float = SMOOTH_BETA,
        derivative_cutoff: float = SMOOTH_DERIVATIVE_CUTOFF,
    ) -> None:
        self._min_cutoff_xy = min_cutoff_xy
        self._min_cutoff_z = min_cutoff_z
        self._beta = beta
        self._derivative_cutoff = derivative_cutoff
        self._previous: Vector3 | None = None
        self._speed: Vector3 | None = None

    def reset(self) -> None:
        self._previous = None
        self._speed = None

    def smooth(self, sample: Vector3, dt: float = NOMINAL_DT) -> Vector3:
        if self._previous is None or self._speed is None:
            self._previous = sample
            self._speed = Vector3(0.0, 0.0, 0.0)
            return sample

        prev = self._previous
        prev_speed = self._speed
        speed_alpha = _cutoff_alpha(self._derivative_cutoff, dt)

        speed = Vector3(
            speed_alpha * (sample.x - prev.x) / dt + (1.0 - speed_alpha) * prev_speed.x,
            speed_alpha * (sample.y - prev.y) / dt + (1.0 - speed_alpha) * prev_speed.y,
            speed_alpha * (sample.z - prev.z) / dt + (1.0 - speed_alpha) * prev_speed.z,
        )
        speed_xy = math.hypot(speed.x, speed.y)
        alpha_xy = _cutoff_alpha(self._min_cutoff_xy + self._beta * speed_xy, dt)
        alpha_z = _cutoff_alpha(self._min_cutoff_z + self._beta * abs(speed.z), dt)

        smoothed = Vector3(
            alpha_xy * sample.x + (1.0 - alpha_xy) * prev.x,
            alpha_xy * sample.y + (1.0 - alpha_xy) * prev.y,
            alpha_z * sample.z + (1.0 - alpha_z) * prev.z,
        )
        self._previous = smoothed
        self._speed = speed
        return smoothed
