"""Smoke tests for the OpenCV demo helpers (no camera)."""

from __future__ import annotations

import numpy as np

from demo.hand_pose_demo import HandHud, apply_event, quat_to_euler_deg, update_pose_hud
from hand_interaction.math3d import quat_from_axis_angle, quat_identity
from hand_interaction.types import (
    HandPose,
    PinchEnd,
    PinchStart,
    Vector2,
    Vector3,
)


def test_apply_pinch_events() -> None:
    hud: dict[str, HandHud] = {}
    apply_event(hud, PinchStart("Right", Vector2(0.4, 0.5)))
    assert hud["Right"].pinching is True
    assert hud["Right"].gesture == "PINCHING"
    apply_event(hud, PinchEnd("Right", Vector2(0.4, 0.5)))
    assert hud["Right"].pinching is False
    assert hud["Right"].gesture == "OPEN"


def test_quat_to_euler_identity() -> None:
    pose = HandPose(
        hand_id="Left",
        palm=Vector3(0.5, 0.5, 0.0),
        orientation=quat_identity(),
        confidence=1.0,
    )
    yaw, pitch, roll = quat_to_euler_deg(pose)
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6 and abs(roll) < 1e-6


def test_update_pose_hud() -> None:
    hud: dict[str, HandHud] = {}
    q = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.5)
    pose = HandPose(
        hand_id="Right",
        palm=Vector3(0.3, 0.7, 0.1),
        orientation=q,
        confidence=0.85,
    )
    update_pose_hud(hud, pose)
    assert hud["Right"].confidence == 0.85
    assert hud["Right"].palm_xyz[0] == 0.3
