"""Pinch helpers — thin re-exports; full state machine lives in gestures.py."""

from __future__ import annotations

import numpy as np

from hand_interaction.constants import INDEX_TIP, THUMB_TIP
from hand_interaction.gestures import GestureDetector, GestureState
from hand_interaction.pose import hand_scale
from hand_interaction.types import Vector2

# Back-compat names used by older tests / imports.
PinchPhase = GestureState  # PINCHING vs others; OPEN ≈ IDLE
PinchDetector = GestureDetector


def pinch_ratio_3d(landmarks: np.ndarray) -> float:
    """‖thumb − index‖₃ / hand_scale."""
    pts = np.asarray(landmarks, dtype=np.float64)
    scale = hand_scale(pts)
    gap = float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP]))
    return gap / scale


def pinch_midpoint_2d(landmarks: np.ndarray) -> Vector2:
    """Thumb–index midpoint in image space (y origin at top)."""
    pts = np.asarray(landmarks, dtype=np.float64)
    mid = 0.5 * (pts[THUMB_TIP] + pts[INDEX_TIP])
    return Vector2(float(mid[0]), float(mid[1]))


__all__ = [
    "GestureDetector",
    "GestureState",
    "PinchDetector",
    "PinchPhase",
    "pinch_midpoint_2d",
    "pinch_ratio_3d",
]
