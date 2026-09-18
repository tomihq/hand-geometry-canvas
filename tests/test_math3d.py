"""Tests for Vector2/3 and Quaternion math."""

from __future__ import annotations

import math

import pytest

from hand_interaction.math3d import (
    delta_rotation_local,
    project_delta_to_screen_angle,
    quat_angle,
    quat_from_axis_angle,
    quat_from_rotation_matrix,
    quat_identity,
    quat_inverse,
    quat_multiply,
    quat_norm,
    quat_normalize,
    quat_slerp,
    vec2_add,
    vec2_norm,
    vec2_sub,
    vec3_add,
    vec3_cross,
    vec3_dot,
    vec3_norm,
    vec3_normalize,
    vec3_scale,
    vec3_sub,
)
from hand_interaction.types import Quaternion, Vector2, Vector3


def _approx_quat(a: Quaternion, b: Quaternion, rel: float = 1e-6) -> None:
    # Quaternions q and -q represent the same rotation.
    same = (
        a.w == pytest.approx(b.w, rel=rel, abs=1e-9)
        and a.x == pytest.approx(b.x, rel=rel, abs=1e-9)
        and a.y == pytest.approx(b.y, rel=rel, abs=1e-9)
        and a.z == pytest.approx(b.z, rel=rel, abs=1e-9)
    )
    opposite = (
        a.w == pytest.approx(-b.w, rel=rel, abs=1e-9)
        and a.x == pytest.approx(-b.x, rel=rel, abs=1e-9)
        and a.y == pytest.approx(-b.y, rel=rel, abs=1e-9)
        and a.z == pytest.approx(-b.z, rel=rel, abs=1e-9)
    )
    assert same or opposite


def test_vector2_ops() -> None:
    a = Vector2(1.0, 2.0)
    b = Vector2(0.5, -1.0)
    assert vec2_add(a, b) == Vector2(1.5, 1.0)
    assert vec2_sub(a, b) == Vector2(0.5, 3.0)
    assert vec2_norm(Vector2(3.0, 4.0)) == pytest.approx(5.0)


def test_vector3_ops() -> None:
    a = Vector3(1.0, 0.0, 0.0)
    b = Vector3(0.0, 1.0, 0.0)
    assert vec3_add(a, b) == Vector3(1.0, 1.0, 0.0)
    assert vec3_sub(a, b) == Vector3(1.0, -1.0, 0.0)
    assert vec3_scale(a, 2.0) == Vector3(2.0, 0.0, 0.0)
    assert vec3_dot(a, b) == pytest.approx(0.0)
    assert vec3_cross(a, b) == Vector3(0.0, 0.0, 1.0)
    assert vec3_norm(Vector3(0.0, 3.0, 4.0)) == pytest.approx(5.0)
    n = vec3_normalize(Vector3(0.0, 3.0, 4.0))
    assert vec3_norm(n) == pytest.approx(1.0)


def test_quat_identity_and_normalize() -> None:
    q = quat_identity()
    assert quat_norm(q) == pytest.approx(1.0)
    dirty = Quaternion(2.0, 0.0, 0.0, 0.0)
    assert quat_norm(quat_normalize(dirty)) == pytest.approx(1.0)


def test_quat_multiply_identity() -> None:
    q = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), math.pi / 3)
    _approx_quat(quat_multiply(q, quat_identity()), q)
    _approx_quat(quat_multiply(quat_identity(), q), q)


def test_quat_inverse_roundtrip() -> None:
    q = quat_from_axis_angle(Vector3(1.0, 1.0, 0.0), 0.7)
    inv = quat_inverse(q)
    _approx_quat(quat_multiply(q, inv), quat_identity())
    _approx_quat(quat_multiply(inv, q), quat_identity())


def test_axis_angle_yaw_pitch_roll() -> None:
    yaw = quat_from_axis_angle(Vector3(0.0, 1.0, 0.0), math.pi / 2)
    pitch = quat_from_axis_angle(Vector3(1.0, 0.0, 0.0), math.pi / 2)
    roll = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), math.pi / 2)
    assert quat_angle(yaw) == pytest.approx(math.pi / 2)
    assert quat_angle(pitch) == pytest.approx(math.pi / 2)
    assert quat_angle(roll) == pytest.approx(math.pi / 2)
    # Distinct axes → distinct quaternions (not the same rotation).
    assert not (
        yaw.x == pytest.approx(pitch.x)
        and yaw.y == pytest.approx(pitch.y)
        and yaw.z == pytest.approx(pitch.z)
    )


def test_delta_rotation_local_small_and_inverse() -> None:
    prev = quat_identity()
    curr = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.2)
    delta = delta_rotation_local(prev, curr)
    assert quat_angle(delta) == pytest.approx(0.2, rel=1e-5)

    # Going back: local delta from curr to prev is the inverse rotation.
    back = delta_rotation_local(curr, prev)
    _approx_quat(quat_multiply(delta, back), quat_identity())


def test_delta_rotation_local_composed() -> None:
    a = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.3)
    b = quat_from_axis_angle(Vector3(0.0, 1.0, 0.0), 0.4)
    # q_curr = q_prev ⊗ Δ_local  ⇒  Δ_local = q_prev⁻¹ ⊗ q_curr
    curr = quat_multiply(a, b)
    delta = delta_rotation_local(a, curr)
    _approx_quat(delta, b)


def test_slerp_endpoints_and_midpoint() -> None:
    a = quat_identity()
    b = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), math.pi / 2)
    _approx_quat(quat_slerp(a, b, 0.0), a)
    _approx_quat(quat_slerp(a, b, 1.0), b)
    mid = quat_slerp(a, b, 0.5)
    assert quat_angle(mid) == pytest.approx(math.pi / 4, rel=1e-5)
    assert quat_norm(mid) == pytest.approx(1.0)


def test_slerp_near_parallel() -> None:
    a = quat_identity()
    b = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 1e-4)
    mid = quat_slerp(a, b, 0.5)
    assert quat_norm(mid) == pytest.approx(1.0)
    assert quat_angle(mid) == pytest.approx(5e-5, rel=0.05, abs=1e-6)


def test_quat_from_rotation_matrix_identity() -> None:
    m = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    _approx_quat(quat_from_rotation_matrix(m), quat_identity())


def test_quat_from_rotation_matrix_90deg_z() -> None:
    # 90° about Z: x→y, y→-x
    m = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    q = quat_from_rotation_matrix(m)
    expected = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), math.pi / 2)
    _approx_quat(q, expected)
    assert quat_angle(q) == pytest.approx(math.pi / 2)


def test_project_delta_to_screen_angle() -> None:
    delta = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.35)
    assert project_delta_to_screen_angle(delta) == pytest.approx(0.35, rel=1e-5)
    delta_neg = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), -0.35)
    assert project_delta_to_screen_angle(delta_neg) == pytest.approx(-0.35, rel=1e-5)
