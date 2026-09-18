"""Vector and quaternion math for 3D hand pose (wxyz quaternions)."""

from __future__ import annotations

import math

from hand_interaction.types import Quaternion, Vector2, Vector3

_EPS = 1e-12


# --- Vector2 -----------------------------------------------------------------


def vec2_add(a: Vector2, b: Vector2) -> Vector2:
    return Vector2(a.x + b.x, a.y + b.y)


def vec2_sub(a: Vector2, b: Vector2) -> Vector2:
    return Vector2(a.x - b.x, a.y - b.y)


def vec2_scale(v: Vector2, s: float) -> Vector2:
    return Vector2(v.x * s, v.y * s)


def vec2_norm(v: Vector2) -> float:
    return math.hypot(v.x, v.y)


# --- Vector3 -----------------------------------------------------------------


def vec3_add(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(a.x + b.x, a.y + b.y, a.z + b.z)


def vec3_sub(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(a.x - b.x, a.y - b.y, a.z - b.z)


def vec3_scale(v: Vector3, s: float) -> Vector3:
    return Vector3(v.x * s, v.y * s, v.z * s)


def vec3_dot(a: Vector3, b: Vector3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def vec3_cross(a: Vector3, b: Vector3) -> Vector3:
    return Vector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def vec3_norm(v: Vector3) -> float:
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def vec3_normalize(v: Vector3) -> Vector3:
    n = vec3_norm(v)
    if n < _EPS:
        return Vector3(0.0, 0.0, 0.0)
    return vec3_scale(v, 1.0 / n)


# --- Quaternion --------------------------------------------------------------


def quat_identity() -> Quaternion:
    return Quaternion(1.0, 0.0, 0.0, 0.0)


def quat_dot(a: Quaternion, b: Quaternion) -> float:
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z


def quat_norm(q: Quaternion) -> float:
    return math.sqrt(quat_dot(q, q))


def quat_normalize(q: Quaternion) -> Quaternion:
    n = quat_norm(q)
    if n < _EPS:
        return quat_identity()
    inv = 1.0 / n
    return Quaternion(q.w * inv, q.x * inv, q.y * inv, q.z * inv)


def quat_conjugate(q: Quaternion) -> Quaternion:
    return Quaternion(q.w, -q.x, -q.y, -q.z)


def quat_inverse(q: Quaternion) -> Quaternion:
    """Inverse for a (near-)unit quaternion: conjugate / |q|²."""
    n2 = quat_dot(q, q)
    if n2 < _EPS:
        return quat_identity()
    inv = 1.0 / n2
    return Quaternion(q.w * inv, -q.x * inv, -q.y * inv, -q.z * inv)


def quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Hamilton product a ⊗ b (apply b first, then a)."""
    return Quaternion(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    )


def quat_angle(q: Quaternion) -> float:
    """Rotation angle in radians of a (near-)unit quaternion, in [0, π]."""
    q = quat_normalize(q)
    # Clamp for numerical safety: angle = 2 * acos(|w|)
    w = min(1.0, max(-1.0, abs(q.w)))
    return 2.0 * math.acos(w)


def quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    """Spherical linear interpolation; t in [0, 1]. Takes shortest arc."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = quat_dot(a, b)

    # Same hemisphere.
    if dot < 0.0:
        b = Quaternion(-b.w, -b.x, -b.y, -b.z)
        dot = -dot

    if dot > 0.9995:
        # Nearly parallel — fall back to normalized lerp.
        return quat_normalize(
            Quaternion(
                a.w + t * (b.w - a.w),
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
            )
        )

    dot = min(1.0, max(-1.0, dot))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta
    return Quaternion(
        w1 * a.w + w2 * b.w,
        w1 * a.x + w2 * b.x,
        w1 * a.y + w2 * b.y,
        w1 * a.z + w2 * b.z,
    )


def quat_from_rotation_matrix(m: tuple[tuple[float, float, float], ...]) -> Quaternion:
    """Build a quaternion from a 3×3 rotation matrix (row-major, right-handed).

    ``m[i]`` is row i; columns are the basis vectors expressed in world space
    when interpreting rows as (right, up, forward) or any orthonormal triad
    passed in consistently by the caller.
    """
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return quat_normalize(
            Quaternion(
                0.25 / s,
                (m21 - m12) * s,
                (m02 - m20) * s,
                (m10 - m01) * s,
            )
        )
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return quat_normalize(
            Quaternion(
                (m21 - m12) / s,
                0.25 * s,
                (m01 + m10) / s,
                (m02 + m20) / s,
            )
        )
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return quat_normalize(
            Quaternion(
                (m02 - m20) / s,
                (m01 + m10) / s,
                0.25 * s,
                (m12 + m21) / s,
            )
        )
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return quat_normalize(
        Quaternion(
            (m10 - m01) / s,
            (m02 + m20) / s,
            (m12 + m21) / s,
            0.25 * s,
        )
    )


def quat_from_axis_angle(axis: Vector3, angle: float) -> Quaternion:
    """Unit quaternion rotating ``angle`` radians around ``axis``."""
    axis = vec3_normalize(axis)
    if vec3_norm(axis) < _EPS:
        return quat_identity()
    half = 0.5 * angle
    s = math.sin(half)
    return Quaternion(math.cos(half), axis.x * s, axis.y * s, axis.z * s)


def delta_rotation_local(previous: Quaternion, current: Quaternion) -> Quaternion:
    """Local-frame delta: Δq = normalize(q_prev⁻¹ ⊗ q_curr)."""
    return quat_normalize(quat_multiply(quat_inverse(previous), current))


def project_delta_to_screen_angle(delta: Quaternion) -> float:
    """Signed radians around screen +Z from a local delta quaternion.

    Uses the twist of Δq about +Z (atan2 of the z-component after projecting
    out xy twist). Suitable for abstract HandRotate events.
    """
    d = quat_normalize(delta)
    # Ensure shortest arc representation (w >= 0).
    if d.w < 0.0:
        d = Quaternion(-d.w, -d.x, -d.y, -d.z)
    # Rotation about Z: q ≈ (cos(θ/2), 0, 0, sin(θ/2))
    return 2.0 * math.atan2(d.z, d.w)
