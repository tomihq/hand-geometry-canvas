"""Tests for InteractionEngine: move / rotate / pinch / grace."""

from __future__ import annotations

import pytest

from hand_interaction.interaction import InteractionEngine, TrackedHand
from hand_interaction.math3d import quat_from_axis_angle, quat_identity
from hand_interaction.types import (
    HandMove,
    HandPose,
    HandRotate,
    PinchEnd,
    PinchStart,
    Quaternion,
    ResizeEnd,
    ResizeMove,
    ResizeStart,
    Vector2,
    Vector3,
)
from tests.test_pinch import _hand_at, _hand_with_gap


def _pose(
    hand_id: str,
    x: float,
    y: float,
    *,
    z: float = 0.0,
    orientation: Quaternion | None = None,
    confidence: float = 1.0,
) -> HandPose:
    return HandPose(
        hand_id=hand_id,
        palm=Vector3(x, y, z),
        orientation=orientation or quat_identity(),
        confidence=confidence,
    )


def _tracked(
    pose: HandPose,
    gap_ratio: float = 0.9,
    *,
    ox: float | None = None,
    oy: float | None = None,
) -> TrackedHand:
    if ox is None:
        ox = pose.palm.x
    if oy is None:
        oy = pose.palm.y
    # Image y is top-origin; public palm.y is bottom-origin.
    landmarks = _hand_at(ox, 1.0 - oy, gap_ratio)
    return TrackedHand(pose=pose, landmarks=landmarks)


def test_move_positive_and_negative_deltas() -> None:
    eng = InteractionEngine(move_epsilon=1e-6)
    events = eng.update([_tracked(_pose("Right", 0.4, 0.5))])
    assert not any(isinstance(e, HandMove) for e in events)

    events = eng.update([_tracked(_pose("Right", 0.55, 0.5))])
    moves = [e for e in events if isinstance(e, HandMove)]
    assert len(moves) == 1
    assert moves[0].delta_position.x == pytest.approx(0.15)
    assert moves[0].position == Vector2(0.55, 0.5)

    events = eng.update([_tracked(_pose("Right", 0.50, 0.5))])
    moves = [e for e in events if isinstance(e, HandMove)]
    assert len(moves) == 1
    assert moves[0].delta_position.x == pytest.approx(-0.05)


def test_stable_pose_emits_no_move() -> None:
    eng = InteractionEngine()
    eng.update([_tracked(_pose("Right", 0.5, 0.5))])
    events = eng.update([_tracked(_pose("Right", 0.5, 0.5))])
    assert not any(isinstance(e, HandMove) for e in events)


def test_rotate_emits_projected_float() -> None:
    eng = InteractionEngine(rotate_epsilon=1e-6)
    eng.update([_tracked(_pose("Right", 0.5, 0.5))])
    q = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.25)
    events = eng.update([_tracked(_pose("Right", 0.5, 0.5, orientation=q))])
    rotates = [e for e in events if isinstance(e, HandRotate)]
    assert len(rotates) == 1
    assert rotates[0].delta_rotation == pytest.approx(0.25, rel=1e-4)
    assert isinstance(rotates[0].delta_rotation, float)


def test_small_rotation_within_deadzone_silent() -> None:
    eng = InteractionEngine(rotate_epsilon=0.05)
    eng.update([_tracked(_pose("Right", 0.5, 0.5))])
    q = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.01)
    events = eng.update([_tracked(_pose("Right", 0.5, 0.5, orientation=q))])
    assert not any(isinstance(e, HandRotate) for e in events)


def test_pinch_integrated() -> None:
    eng = InteractionEngine()
    events: list = []
    for _ in range(5):
        events.extend(
            eng.update([_tracked(_pose("Right", 0.5, 0.5), gap_ratio=0.2)])
        )
    starts = [e for e in events if isinstance(e, PinchStart)]
    assert len(starts) == 1
    assert isinstance(starts[0].position, Vector2)


def test_grace_holds_pinch_then_ends() -> None:
    eng = InteractionEngine(grace_frames=3)
    for _ in range(5):
        eng.update([_tracked(_pose("Right", 0.5, 0.5), gap_ratio=0.2)])

    for _ in range(3):
        events = eng.update([])
        assert not any(isinstance(e, PinchEnd) for e in events)

    events = eng.update([])
    ends = [e for e in events if isinstance(e, PinchEnd)]
    assert len(ends) == 1


def test_multi_hand_independent() -> None:
    eng = InteractionEngine(move_epsilon=1e-6)
    eng.update(
        [
            _tracked(_pose("Left", 0.3, 0.5)),
            _tracked(_pose("Right", 0.7, 0.5)),
        ]
    )
    events = eng.update(
        [
            _tracked(_pose("Left", 0.35, 0.5)),
            _tracked(_pose("Right", 0.65, 0.5)),
        ]
    )
    moves = {e.hand_id: e for e in events if isinstance(e, HandMove)}
    assert set(moves) == {"Left", "Right"}
    assert moves["Left"].delta_position.x == pytest.approx(0.05)
    assert moves["Right"].delta_position.x == pytest.approx(-0.05)


def test_owner_pinch_then_helper_emits_resize() -> None:
    """Canvas lock model: first pinch = owner; second = helper resize cursor."""
    eng = InteractionEngine(resize_epsilon=1e-6)

    # Owner (Right) pinches alone first.
    for _ in range(5):
        eng.update([_tracked(_pose("Right", 0.7, 0.5), gap_ratio=0.2)])
    assert eng._owner_hand_id == "Right"

    # Helper (Left) joins while owner still holds.
    events: list = []
    for _ in range(5):
        events.extend(
            eng.update(
                [
                    _tracked(_pose("Left", 0.3, 0.5), gap_ratio=0.2),
                    _tracked(_pose("Right", 0.7, 0.5), gap_ratio=0.2),
                ]
            )
        )
    starts = [e for e in events if isinstance(e, ResizeStart)]
    assert len(starts) == 1
    assert starts[0].owner_hand_id == "Right"
    assert starts[0].helper_hand_id == "Left"

    # Helper cursor moves → resize.move follows helper position (not distance).
    moved = eng.update(
        [
            _tracked(_pose("Left", 0.2, 0.5), gap_ratio=0.2),
            _tracked(_pose("Right", 0.7, 0.5), gap_ratio=0.2),
        ]
    )
    resizes = [e for e in moved if isinstance(e, ResizeMove)]
    assert len(resizes) == 1
    assert resizes[0].helper_hand_id == "Left"
    assert resizes[0].delta_position.x != 0.0

    # Helper releases → resize.end; owner keeps pinch.
    ended: list = []
    for _ in range(5):
        ended.extend(
            eng.update(
                [
                    _tracked(_pose("Left", 0.2, 0.5), gap_ratio=0.9),
                    _tracked(_pose("Right", 0.7, 0.5), gap_ratio=0.2),
                ]
            )
        )
    assert any(isinstance(e, ResizeEnd) for e in ended)
    assert eng._owner_hand_id == "Right"


def test_events_do_not_expose_3d_types() -> None:
    eng = InteractionEngine(move_epsilon=1e-6, rotate_epsilon=1e-6)
    eng.update([_tracked(_pose("Right", 0.5, 0.5))])
    q = quat_from_axis_angle(Vector3(0.0, 0.0, 1.0), 0.2)
    events = eng.update(
        [_tracked(_pose("Right", 0.6, 0.5, orientation=q), gap_ratio=0.2)]
    )
    for event in events:
        for value in vars(event).values():
            assert not isinstance(value, Vector3)
            assert not isinstance(value, Quaternion)
