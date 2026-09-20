"""Tests for InteractionEngine: move / pinch / grab / grace."""

from __future__ import annotations

import pytest

from hand_interaction.interaction import InteractionEngine, TrackedHand
from hand_interaction.math3d import quat_identity
from hand_interaction.types import (
    HandMove,
    HandPose,
    PinchEnd,
    PinchStart,
    Quaternion,
    ResizeEnd,
    ResizeMove,
    ResizeStart,
    Vector2,
    Vector3,
)
from tests.test_pinch import _open_hand, _pinch_hand


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
    *,
    pinching: bool = False,
    ox: float | None = None,
    oy_image: float | None = None,
) -> TrackedHand:
    if ox is None:
        ox = pose.palm.x
    # pose.palm.y is hybrid (bottom origin); landmarks use image y.
    if oy_image is None:
        oy_image = 1.0 - pose.palm.y
    landmarks = _pinch_hand(0.2, ox=ox, oy=oy_image) if pinching else _open_hand(
        ox=ox, oy=oy_image
    )
    return TrackedHand(pose=pose, landmarks=landmarks)


def test_move_positive_and_negative_deltas() -> None:
    eng = InteractionEngine(move_epsilon=1e-6)
    events = eng.update([_tracked(_pose("Right", 0.4, 0.5))])
    assert not any(isinstance(e, HandMove) for e in events)

    events = eng.update([_tracked(_pose("Right", 0.55, 0.5))])
    moves = [e for e in events if isinstance(e, HandMove)]
    assert len(moves) == 1
    # HandMove uses image y (top origin): hybrid 0.5 → image 0.5
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


def test_pinch_integrated() -> None:
    eng = InteractionEngine()
    events: list = []
    for _ in range(5):
        events.extend(eng.update([_tracked(_pose("Right", 0.5, 0.5), pinching=True)]))
    starts = [e for e in events if isinstance(e, PinchStart)]
    assert len(starts) == 1
    assert isinstance(starts[0].position, Vector2)


def test_grace_holds_pinch_then_ends() -> None:
    eng = InteractionEngine()
    for _ in range(5):
        eng.update([_tracked(_pose("Right", 0.5, 0.5), pinching=True)])

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
        eng.update([_tracked(_pose("Right", 0.7, 0.5), pinching=True)])
    assert eng._owner_hand_id == "Right"

    # Helper (Left) joins while owner still holds.
    events: list = []
    for _ in range(5):
        events.extend(
            eng.update(
                [
                    _tracked(_pose("Left", 0.3, 0.5), pinching=True),
                    _tracked(_pose("Right", 0.7, 0.5), pinching=True),
                ]
            )
        )
    starts = [e for e in events if isinstance(e, ResizeStart)]
    assert len(starts) == 1
    assert starts[0].owner_hand_id == "Right"
    assert starts[0].helper_hand_id == "Left"

    # Helper cursor moves → resize.move follows helper position.
    moved = eng.update(
        [
            _tracked(_pose("Left", 0.2, 0.5), pinching=True, ox=0.2),
            _tracked(_pose("Right", 0.7, 0.5), pinching=True, ox=0.7),
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
                    _tracked(_pose("Left", 0.2, 0.5), pinching=False, ox=0.2),
                    _tracked(_pose("Right", 0.7, 0.5), pinching=True, ox=0.7),
                ]
            )
        )
    assert any(isinstance(e, ResizeEnd) for e in ended)
    assert eng._owner_hand_id == "Right"


def test_events_do_not_expose_3d_types() -> None:
    eng = InteractionEngine(move_epsilon=1e-6)
    eng.update([_tracked(_pose("Right", 0.5, 0.5))])
    events = eng.update(
        [_tracked(_pose("Right", 0.6, 0.5), pinching=True)]
    )
    for event in events:
        for value in vars(event).values():
            assert not isinstance(value, Vector3)
            assert not isinstance(value, Quaternion)
