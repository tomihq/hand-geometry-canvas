"""Thin math adapter: public Vector/Quaternion types ↔ numpy / scipy.

Vector ops use numpy. Rotations use ``scipy.spatial.transform.Rotation``
(scalar-last xyzw internally; our public Quaternion stays wxyz).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from hand_interaction.types import Quaternion, Vector2, Vector3

_EPS = 1e-12


# --- adapters ----------------------------------------------------------------


def _as2(v: Vector2) -> np.ndarray:
    return np.asarray([v.x, v.y], dtype=np.float64)


def _vec2(a: np.ndarray) -> Vector2:
    return Vector2(float(a[0]), float(a[1]))


def _as3(v: Vector3) -> np.ndarray:
    return np.asarray([v.x, v.y, v.z], dtype=np.float64)


def _vec3(a: np.ndarray) -> Vector3:
    return Vector3(float(a[0]), float(a[1]), float(a[2]))


def _as_rotation(q: Quaternion) -> Rotation:
    """Public wxyz → scipy Rotation (xyzw)."""
    return Rotation.from_quat([q.x, q.y, q.z, q.w])


def _from_rotation(r: Rotation) -> Quaternion:
    """scipy Rotation → public wxyz (shortest-arc, w >= 0 when possible)."""
    x, y, z, w = r.as_quat()
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    return Quaternion(float(w), float(x), float(y), float(z))


# --- Vector2 -----------------------------------------------------------------


def vec2_add(a: Vector2, b: Vector2) -> Vector2:
    return _vec2(_as2(a) + _as2(b))


def vec2_sub(a: Vector2, b: Vector2) -> Vector2:
    return _vec2(_as2(a) - _as2(b))


def vec2_scale(v: Vector2, s: float) -> Vector2:
    return _vec2(_as2(v) * s)


def vec2_norm(v: Vector2) -> float:
    return float(np.linalg.norm(_as2(v)))


# --- Vector3 -----------------------------------------------------------------


def vec3_add(a: Vector3, b: Vector3) -> Vector3:
    return _vec3(_as3(a) + _as3(b))


def vec3_sub(a: Vector3, b: Vector3) -> Vector3:
    return _vec3(_as3(a) - _as3(b))


def vec3_scale(v: Vector3, s: float) -> Vector3:
    return _vec3(_as3(v) * s)


def vec3_dot(a: Vector3, b: Vector3) -> float:
    return float(np.dot(_as3(a), _as3(b)))


def vec3_cross(a: Vector3, b: Vector3) -> Vector3:
    return _vec3(np.cross(_as3(a), _as3(b)))


def vec3_norm(v: Vector3) -> float:
    return float(np.linalg.norm(_as3(v)))


def vec3_normalize(v: Vector3) -> Vector3:
    a = _as3(v)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        return Vector3(0.0, 0.0, 0.0)
    return _vec3(a / n)


# --- Quaternion (via scipy.Rotation) ----------------------------------------


def quat_identity() -> Quaternion:
    return _from_rotation(Rotation.identity())


def quat_dot(a: Quaternion, b: Quaternion) -> float:
    return float(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z)


def quat_norm(q: Quaternion) -> float:
    return float(np.linalg.norm([q.w, q.x, q.y, q.z]))


def quat_normalize(q: Quaternion) -> Quaternion:
    n = quat_norm(q)
    if n < _EPS:
        return quat_identity()
    return Quaternion(q.w / n, q.x / n, q.y / n, q.z / n)


def quat_conjugate(q: Quaternion) -> Quaternion:
    return Quaternion(q.w, -q.x, -q.y, -q.z)


def quat_inverse(q: Quaternion) -> Quaternion:
    return _from_rotation(_as_rotation(q).inv())


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product a ⊗ b (apply b first, then a)."""
    return _from_rotation(_as_rotation(a) * _as_rotation(b))


def quat_angle(q: Quaternion) -> float:
    """Rotation angle in radians, in [0, π]."""
    return float(_as_rotation(q).magnitude())


def quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation; t in [0, 1]. Shortest arc."""
    ra, rb = _as_rotation(a), _as_rotation(b)
    # Same hemisphere so Slerp takes the short path.
    if np.dot(ra.as_quat(), rb.as_quat()) < 0.0:
        q = rb.as_quat()
        rb = Rotation.from_quat(-q)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([ra, rb]))
    return _from_rotation(slerp(t))


def quat_from_rotation_matrix(m: tuple[tuple[float, float, float], ...]) -> Quaternion:
    """Build a quaternion from a 3×3 rotation matrix (row-major)."""
    return _from_rotation(Rotation.from_matrix(np.asarray(m, dtype=np.float64)))


def quat_from_axis_angle(axis: Vector3, angle: float) -> Quaternion:
    """Unit quaternion rotating ``angle`` radians around ``axis``."""
    a = _as3(axis)
    n = float(np.linalg.norm(a))
    if n < _EPS:
        return quat_identity()
    return _from_rotation(Rotation.from_rotvec((a / n) * angle))


def delta_rotation_local(previous: Quaternion, current: Quaternion) -> Quaternion:
    """Local-frame delta: Δq = q_prev⁻¹ ⊗ q_curr."""
    return _from_rotation(_as_rotation(previous).inv() * _as_rotation(current))


def project_delta_to_screen_angle(delta: Quaternion) -> float:
    """Signed radians around screen +Z from a local delta quaternion."""
    # Rotation vector z-component is exact for pure Z twists; good abstract signal.
    return float(_as_rotation(delta).as_rotvec()[2])
