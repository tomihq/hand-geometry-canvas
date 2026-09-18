"""Tests for MediaPipe landmarker adapter helpers (no live camera required)."""

from __future__ import annotations

import pytest

from hand_interaction.interaction import InteractionEngine
from hand_interaction.live import process_landmark_hands
from hand_interaction.pose import LandmarkHand, PoseEstimator
from hand_interaction.types import HandMove, HandPose
from tests.test_pinch import _hand_with_gap


def test_process_landmark_hands_emits_move() -> None:
    poses_out: list[HandPose] = []
    estimator = PoseEstimator()
    engine = InteractionEngine(move_epsilon=1e-6)

    open_hand = LandmarkHand(
        landmarks=_hand_with_gap(0.9, scale=0.1),
        hand_id="Right",
        handedness="Right",
    )
    shifted = open_hand.landmarks.copy()
    shifted[:, 0] += 0.1
    moved = LandmarkHand(
        landmarks=shifted, hand_id="Right", handedness="Right"
    )

    process_landmark_hands(
        [open_hand],
        pose_estimator=estimator,
        engine=engine,
        on_pose=poses_out.append,
    )
    events = process_landmark_hands(
        [moved],
        pose_estimator=estimator,
        engine=engine,
        on_pose=poses_out.append,
    )

    assert len(poses_out) == 2
    moves = [e for e in events if isinstance(e, HandMove)]
    assert len(moves) == 1
    # Palm one-euro absorbs part of a one-frame jump; direction must still match.
    assert moves[0].delta_position.x > 0.02


def test_canvas_camera_reexports() -> None:
    from hand_canvas.camera import Camera, Frame
    from hand_interaction.camera import Camera as Cam2
    from hand_interaction.camera import Frame as Frame2

    assert Camera is Cam2
    assert Frame is Frame2


def test_canvas_hand_tracker_exports_indices() -> None:
    from hand_canvas.hand_tracker import (
        INDEX_PIP,
        INDEX_TIP,
        THUMB_TIP,
        WRIST,
        Hand,
        HandTracker,
    )

    assert INDEX_TIP == 8
    assert INDEX_PIP == 6
    assert THUMB_TIP == 4
    assert WRIST == 0
    assert HandTracker is not None
    assert Hand is not None


def test_create_hand_gesture_live_flag_imports() -> None:
    from hand_interaction import create_hand_gesture
    from hand_interaction.live import LiveHandSource

    g = create_hand_gesture(live=True)
    assert isinstance(g.source, LiveHandSource)
