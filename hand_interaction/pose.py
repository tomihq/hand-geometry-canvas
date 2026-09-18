"""Estimate HandPose (3D) from 21 hand landmarks — no MediaPipe dependency."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hand_interaction.constants import (
    INDEX_MCP,
    MAX_DT,
    MCP_IDS,
    MIDDLE_MCP,
    MIN_DT,
    MIN_HAND_SCALE,
    NOMINAL_DT,
    ORIENTATION_SLERP_RATE,
    PALM_IDS,
    PINKY_MCP,
    WRIST,
)
from hand_interaction.math3d import (
    quat_from_rotation_matrix,
    quat_identity,
    quat_normalize,
    quat_slerp,
    vec3_cross,
    vec3_dot,
    vec3_normalize,
    vec3_norm,
    vec3_scale,
    vec3_sub,
)
from hand_interaction.smooth import OneEuroVector3
from hand_interaction.types import HandPose, Quaternion, Vector3


@dataclass(frozen=True)
class LandmarkHand:
    """One hand's landmarks in MediaPipe image space (x, y, z).

    ``x,y`` typically in [0, 1] with origin top-left (pre y-flip).
    ``z`` is MediaPipe's relative depth (same order of magnitude as x/y).
    """

    landmarks: np.ndarray  # shape (21, 3)
    hand_id: str
    handedness: str = "Right"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        pts = np.asarray(self.landmarks, dtype=np.float64)
        if pts.shape != (21, 3):
            raise ValueError(f"expected landmarks shape (21, 3), got {pts.shape}")
        object.__setattr__(self, "landmarks", pts)


def hand_scale(points: np.ndarray) -> float:
    """Median palm span — robust to a single bad landmark."""
    wrist = points[WRIST]
    spans = [float(np.linalg.norm(points[i] - wrist)) for i in MCP_IDS]
    spans.append(float(np.linalg.norm(points[INDEX_MCP] - points[PINKY_MCP])))
    return max(float(np.median(spans)), MIN_HAND_SCALE)


def palm_center_raw(points: np.ndarray) -> np.ndarray:
    return points[list(PALM_IDS)].mean(axis=0)


def to_hybrid_position(raw: np.ndarray, scale: float) -> Vector3:
    """Image landmark → public hybrid frame (y-up, z relative to hand size)."""
    return Vector3(
        x=float(raw[0]),
        y=1.0 - float(raw[1]),
        z=float(raw[2]) / scale,
    )


def orientation_from_landmarks(
    points: np.ndarray, handedness: str
) -> Quaternion:
    """Build a right-handed palm basis → quaternion (wxyz).

    Axes (columns of R): X = right, Y = up, Z = forward.
    """
    wrist = points[WRIST]
    forward = vec3_normalize(
        Vector3(
            float(points[MIDDLE_MCP][0] - wrist[0]),
            float(points[MIDDLE_MCP][1] - wrist[1]),
            float(points[MIDDLE_MCP][2] - wrist[2]),
        )
    )
    right = Vector3(
        float(points[INDEX_MCP][0] - points[PINKY_MCP][0]),
        float(points[INDEX_MCP][1] - points[PINKY_MCP][1]),
        float(points[INDEX_MCP][2] - points[PINKY_MCP][2]),
    )
    # Keep a consistent right-handed frame for both hands.
    if handedness.lower().startswith("left"):
        right = Vector3(-right.x, -right.y, -right.z)

    if vec3_norm(forward) < MIN_HAND_SCALE or vec3_norm(right) < MIN_HAND_SCALE:
        return quat_identity()

    # Gram–Schmidt: orthogonalize right against forward, then up = forward × right.
    right = vec3_sub(right, vec3_scale(forward, vec3_dot(right, forward)))
    right = vec3_normalize(right)
    if vec3_norm(right) < MIN_HAND_SCALE:
        return quat_identity()

    up = vec3_normalize(vec3_cross(forward, right))
    if vec3_norm(up) < MIN_HAND_SCALE:
        return quat_identity()

    # Columns of R are the basis axes in world/image space.
    matrix = (
        (right.x, up.x, forward.x),
        (right.y, up.y, forward.y),
        (right.z, up.z, forward.z),
    )
    return quat_from_rotation_matrix(matrix)


def clamp_dt(dt: float) -> float:
    return min(max(dt, MIN_DT), MAX_DT)


class PoseEstimator:
    """Stateful estimator: landmarks → smoothed HandPose per hand_id."""

    def __init__(self) -> None:
        self._palm_filters: dict[str, OneEuroVector3] = {}
        self._orientations: dict[str, Quaternion] = {}

    def reset(self, hand_id: str | None = None) -> None:
        if hand_id is None:
            self._palm_filters.clear()
            self._orientations.clear()
            return
        self._palm_filters.pop(hand_id, None)
        self._orientations.pop(hand_id, None)

    def estimate(
        self, hand: LandmarkHand, dt: float = NOMINAL_DT
    ) -> HandPose:
        dt = clamp_dt(dt)
        points = hand.landmarks
        scale = hand_scale(points)
        raw_palm = palm_center_raw(points)
        measured_palm = to_hybrid_position(raw_palm, scale)
        measured_orient = orientation_from_landmarks(points, hand.handedness)

        filt = self._palm_filters.setdefault(hand.hand_id, OneEuroVector3())
        palm = filt.smooth(measured_palm, dt)

        prev_q = self._orientations.get(hand.hand_id)
        if prev_q is None:
            orientation = measured_orient
        else:
            # Frame-rate aware blend: alpha = 1 - exp(-rate * dt)
            alpha = 1.0 - math.exp(-ORIENTATION_SLERP_RATE * dt)
            orientation = quat_normalize(quat_slerp(prev_q, measured_orient, alpha))
        self._orientations[hand.hand_id] = orientation

        confidence = float(min(1.0, max(0.0, hand.confidence)))
        return HandPose(
            hand_id=hand.hand_id,
            palm=palm,
            orientation=orientation,
            confidence=confidence,
        )
