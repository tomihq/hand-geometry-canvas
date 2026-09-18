"""Contract tests for public pose and interaction event types."""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

import pytest

from hand_interaction import (
    HandEvent,
    HandMove,
    HandPose,
    HandRotate,
    PinchEnd,
    PinchStart,
    Quaternion,
    Vector2,
    Vector3,
)


def test_vector2_and_vector3_are_frozen() -> None:
    v2 = Vector2(0.1, 0.2)
    v3 = Vector3(0.1, 0.2, 0.3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        v2.x = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        v3.z = 1.0  # type: ignore[misc]


def test_quaternion_and_hand_pose() -> None:
    orientation = Quaternion(1.0, 0.0, 0.0, 0.0)
    pose = HandPose(
        hand_id="Left",
        palm=Vector3(0.5, 0.5, 0.0),
        orientation=orientation,
        confidence=0.9,
    )
    assert pose.hand_id == "Left"
    assert pose.palm.z == 0.0
    assert pose.orientation.w == 1.0
    assert pose.confidence == 0.9
    with pytest.raises(dataclasses.FrozenInstanceError):
        pose.confidence = 0.1  # type: ignore[misc]


def test_pinch_events_carry_vector2_and_type() -> None:
    pos = Vector2(0.4, 0.6)
    start = PinchStart(hand_id="Right", position=pos)
    end = PinchEnd(hand_id="Right", position=pos)
    assert start.type == "pinch.start"
    assert end.type == "pinch.end"
    assert start.position == pos
    assert isinstance(start.position, Vector2)


def test_hand_move_is_abstract_2d() -> None:
    event = HandMove(
        hand_id="Left",
        position=Vector2(0.5, 0.5),
        delta_position=Vector2(0.01, -0.02),
    )
    assert event.type == "hand.move"
    assert event.delta_position.x == pytest.approx(0.01)
    hints = get_type_hints(HandMove)
    assert hints["position"] is Vector2
    assert hints["delta_position"] is Vector2
    assert Vector3 not in hints.values()
    assert Quaternion not in hints.values()


def test_hand_rotate_is_scalar_angle() -> None:
    event = HandRotate(hand_id="Left", delta_rotation=0.15)
    assert event.type == "hand.rotate"
    assert event.delta_rotation == pytest.approx(0.15)
    hints = get_type_hints(HandRotate)
    assert hints["delta_rotation"] is float
    assert Quaternion not in hints.values()
    assert Vector3 not in hints.values()


def test_hand_event_union_accepts_all_variants() -> None:
    events: list[HandEvent] = [
        PinchStart("Left", Vector2(0.1, 0.2)),
        PinchEnd("Left", Vector2(0.1, 0.2)),
        HandMove("Left", Vector2(0.1, 0.2), Vector2(0.0, 0.0)),
        HandRotate("Left", 0.0),
    ]
    assert [e.type for e in events] == [
        "pinch.start",
        "pinch.end",
        "hand.move",
        "hand.rotate",
    ]


def test_public_exports() -> None:
    import hand_interaction as hi

    for name in (
        "Vector2",
        "Vector3",
        "Quaternion",
        "HandPose",
        "PinchStart",
        "PinchEnd",
        "HandMove",
        "HandRotate",
        "HandEvent",
    ):
        assert hasattr(hi, name)
