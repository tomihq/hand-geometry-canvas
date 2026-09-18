"""Tests for 3D HandPose estimation from synthetic landmarks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_interaction.constants import (
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    WRIST,
)
from hand_interaction.math3d import (
    delta_rotation_local,
    quat_angle,
    quat_from_axis_angle,
    quat_multiply,
    vec3_sub,
)
from hand_interaction.pose import (
    LandmarkHand,
    PoseEstimator,
    hand_scale,
    orientation_from_landmarks,
    to_hybrid_position,
)
from hand_interaction.types import Quaternion, Vector3


def _flat_hand(
    *,
    origin: tuple[float, float, float] = (0.5, 0.5, 0.0),
    scale: float = 0.1,
    handedness: str = "Right",
) -> np.ndarray:
    """21 landmarks: palm in image plane, fingers along -y (up on screen = smaller y)."""
    pts = np.zeros((21, 3), dtype=np.float64)
    ox, oy, oz = origin
    # Wrist at origin; MCPs spread along +x; fingers toward smaller y.
    pts[WRIST] = (ox, oy, oz)
    pts[INDEX_MCP] = (ox + 0.4 * scale, oy - 0.3 * scale, oz)
    pts[MIDDLE_MCP] = (ox + 0.1 * scale, oy - 0.35 * scale, oz)
    pts[13] = (ox - 0.15 * scale, oy - 0.3 * scale, oz)  # ring MCP
    pts[PINKY_MCP] = (ox - 0.4 * scale, oy - 0.25 * scale, oz)
    # Tips roughly above MCPs (smaller y).
    for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        pts[tip] = pts[mcp] + np.array([0.0, -0.5 * scale, 0.0])
    pts[4] = pts[WRIST] + np.array([0.5 * scale, -0.2 * scale, 0.0])  # thumb tip
    # Fill remaining joints as midpoints so shape is (21,3) coherent.
    for i in range(21):
        if np.allclose(pts[i], 0.0) and i not in (0,):
            pts[i] = pts[WRIST] + np.array([0.01 * i, -0.01 * i, 0.0])
    # Re-apply known joints after fill.
    pts[WRIST] = (ox, oy, oz)
    pts[INDEX_MCP] = (ox + 0.4 * scale, oy - 0.3 * scale, oz)
    pts[MIDDLE_MCP] = (ox + 0.1 * scale, oy - 0.35 * scale, oz)
    pts[13] = (ox - 0.15 * scale, oy - 0.3 * scale, oz)
    pts[PINKY_MCP] = (ox - 0.4 * scale, oy - 0.25 * scale, oz)
    for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        pts[tip] = pts[mcp] + np.array([0.0, -0.5 * scale, 0.0])
    pts[4] = pts[WRIST] + np.array([0.5 * scale, -0.2 * scale, 0.0])
    _ = handedness  # chirality handled by estimator via label
    return pts


def _rotate_landmarks(pts: np.ndarray, q: Quaternion) -> np.ndarray:
    """Rotate all landmarks around the wrist using quaternion q."""
    # q ⊗ (0, v) ⊗ q⁻¹
    w, x, y, z = q.w, q.x, q.y, q.z
    # Rotation matrix from quaternion
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    R = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    wrist = pts[WRIST].copy()
    out = pts.copy()
    for i in range(21):
        out[i] = wrist + R @ (pts[i] - wrist)
    return out


def test_hand_scale_positive() -> None:
    pts = _flat_hand()
    assert hand_scale(pts) > 0.01


def test_hybrid_position_flips_y() -> None:
    raw = np.array([0.25, 0.75, 0.05])
    pos = to_hybrid_position(raw, scale=0.1)
    assert pos.x == pytest.approx(0.25)
    assert pos.y == pytest.approx(0.25)  # 1 - 0.75
    assert pos.z == pytest.approx(0.5)


def test_pose_stable_when_still() -> None:
    pts = _flat_hand()
    hand = LandmarkHand(pts, hand_id="Right", handedness="Right")
    est = PoseEstimator()
    a = est.estimate(hand)
    b = est.estimate(hand)
    assert a.palm.x == pytest.approx(b.palm.x, abs=1e-6)
    assert a.palm.y == pytest.approx(b.palm.y, abs=1e-6)
    assert a.confidence == pytest.approx(1.0)


def test_pose_moves_when_palm_translates() -> None:
    est = PoseEstimator()
    p0 = est.estimate(
        LandmarkHand(_flat_hand(origin=(0.4, 0.5, 0.0)), "Right", "Right")
    )
    p1 = est.estimate(
        LandmarkHand(_flat_hand(origin=(0.6, 0.5, 0.0)), "Right", "Right")
    )
    delta = vec3_sub(p1.palm, p0.palm)
    assert delta.x > 0.05
    assert abs(delta.y) < 0.05


def test_orientation_distinguishes_axis_rotations() -> None:
    base = _flat_hand()
    q_yaw = quat_from_axis_angle(Vector3(0.0, 1.0, 0.0), 0.4)
    q_pitch = quat_from_axis_angle(Vector3(1.0, 0.0, 0.0), 0.4)
    q_roll = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.4)

    o0 = orientation_from_landmarks(base, "Right")
    o_yaw = orientation_from_landmarks(_rotate_landmarks(base, q_yaw), "Right")
    o_pitch = orientation_from_landmarks(_rotate_landmarks(base, q_pitch), "Right")
    o_roll = orientation_from_landmarks(_rotate_landmarks(base, q_roll), "Right")

    d_yaw = quat_angle(delta_rotation_local(o0, o_yaw))
    d_pitch = quat_angle(delta_rotation_local(o0, o_pitch))
    d_roll = quat_angle(delta_rotation_local(o0, o_roll))
    assert d_yaw > 0.15
    assert d_pitch > 0.15
    assert d_roll > 0.15
    # Rotations about different axes produce different orientations.
    assert quat_angle(delta_rotation_local(o_yaw, o_pitch)) > 0.1
    assert quat_angle(delta_rotation_local(o_yaw, o_roll)) > 0.1


def test_left_right_chirality_keeps_right_handed_delta() -> None:
    pts = _flat_hand()
    # Same geometry labeled Left vs Right should flip the lateral axis handling.
    q_right = orientation_from_landmarks(pts, "Right")
    q_left = orientation_from_landmarks(pts, "Left")
    # They should differ (lateral flip) but both be unit quaternions.
    assert abs(q_right.w) <= 1.0
    assert abs(q_left.w) <= 1.0
    assert quat_angle(delta_rotation_local(q_right, q_left)) > 0.01


def test_smoothing_reduces_jitter() -> None:
    rng = np.random.default_rng(0)
    base = _flat_hand(origin=(0.5, 0.5, 0.0))
    est = PoseEstimator()
    # Warm up
    est.estimate(LandmarkHand(base, "Right", "Right"))

    noisy_palms = []
    smooth_palms = []
    for _ in range(30):
        noise = rng.normal(0.0, 0.01, size=(21, 3))
        noisy = base + noise
        # Measure raw hybrid palm without filter for comparison
        from hand_interaction.pose import hand_scale, palm_center_raw, to_hybrid_position

        raw = to_hybrid_position(palm_center_raw(noisy), hand_scale(noisy))
        pose = est.estimate(LandmarkHand(noisy, "Right", "Right"))
        noisy_palms.append(raw)
        smooth_palms.append(pose.palm)

    def _std_x(samples: list) -> float:
        xs = [p.x for p in samples]
        return float(np.std(xs))

    assert _std_x(smooth_palms) < _std_x(noisy_palms)


def test_landmark_hand_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        LandmarkHand(np.zeros((20, 3)), "Right")
