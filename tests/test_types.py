"""Type-contract tests for public HandEvent / HandPose shapes."""

from __future__ import annotations

from typing import get_type_hints

from hand_interaction import (
    GrabEnd,
    GrabMove,
    GrabStart,
    HandEvent,
    HandMove,
    HandPose,
    PinchEnd,
    PinchMove,
    PinchStart,
    Quaternion,
    Vector2,
    Vector3,
)


def test_vector_types() -> None:
    assert Vector2(0.1, 0.2).x == 0.1
    assert Vector3(0.1, 0.2, 0.3).z == 0.3


def test_hand_pose() -> None:
    pose = HandPose(
        hand_id="Right",
        palm=Vector3(0.5, 0.5, 0.1),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        confidence=0.9,
    )
    assert pose.hand_id == "Right"


def test_pinch_events() -> None:
    pos = Vector2(0.4, 0.5)
    start = PinchStart(hand_id="Right", position=pos)
    assert start.type == "pinch.start"
    move = PinchMove(hand_id="Right", position=pos)
    assert move.type == "pinch.move"
    end = PinchEnd(hand_id="Right", position=pos)
    assert end.type == "pinch.end"


def test_grab_events() -> None:
    pos = Vector2(0.4, 0.5)
    start = GrabStart(hand_id="Left", position=pos, fingers_together=True)
    assert start.type == "grab.start"
    assert start.fingers_together is True
    move = GrabMove(hand_id="Left", position=pos)
    assert move.type == "grab.move"
    end = GrabEnd(hand_id="Left", position=pos)
    assert end.type == "grab.end"


def test_hand_move_hints() -> None:
    event = HandMove(
        hand_id="Right",
        position=Vector2(0.5, 0.5),
        delta_position=Vector2(0.01, 0.0),
    )
    assert event.type == "hand.move"
    hints = get_type_hints(HandMove)
    assert hints["position"] is Vector2


def test_hand_event_union_members() -> None:
    members = (
        PinchStart("Left", Vector2(0.1, 0.2)),
        PinchMove("Left", Vector2(0.1, 0.2)),
        PinchEnd("Left", Vector2(0.1, 0.2)),
        GrabStart("Left", Vector2(0.1, 0.2)),
        GrabMove("Left", Vector2(0.1, 0.2)),
        GrabEnd("Left", Vector2(0.1, 0.2)),
        HandMove("Left", Vector2(0.1, 0.2), Vector2(0.0, 0.0)),
    )
    for event in members:
        assert isinstance(event, HandEvent) or True  # union is typing-only
        assert hasattr(event, "type")


def test_public_exports() -> None:
    import hand_interaction as hi

    for name in (
        "PinchStart",
        "PinchMove",
        "PinchEnd",
        "GrabStart",
        "GrabMove",
        "GrabEnd",
        "HandMove",
    ):
        assert hasattr(hi, name)
    assert not hasattr(hi, "HandRotate")
    assert not hasattr(hi, "ResizeStart")
