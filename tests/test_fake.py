"""Tests for FakeHandSource subscribe / emit / unsubscribe."""

from __future__ import annotations

from hand_interaction.fake import FakeHandSource
from hand_interaction.math3d import quat_identity
from hand_interaction.types import (
    GrabStart,
    HandMove,
    HandPose,
    PinchEnd,
    PinchStart,
    Vector2,
    Vector3,
)


def test_emit_reaches_subscribers() -> None:
    fake = FakeHandSource()
    received: list = []
    fake.on_event(received.append)

    start = PinchStart("Right", Vector2(0.4, 0.5))
    move = HandMove("Right", Vector2(0.5, 0.5), Vector2(0.1, 0.0))
    grab = GrabStart("Right", Vector2(0.5, 0.5))
    end = PinchEnd("Right", Vector2(0.5, 0.5))
    fake.emit_sequence([start, move, grab, end])

    assert received == [start, move, grab, end]


def test_unsubscribe_stops_delivery() -> None:
    fake = FakeHandSource()
    received: list = []
    unsub = fake.on_event(received.append)
    fake.emit(PinchStart("Left", Vector2(0.1, 0.2)))
    unsub()
    fake.emit(PinchEnd("Left", Vector2(0.1, 0.2)))
    assert len(received) == 1
    assert isinstance(received[0], PinchStart)


def test_multiple_listeners() -> None:
    fake = FakeHandSource()
    a: list = []
    b: list = []
    fake.on_event(a.append)
    fake.on_event(b.append)
    event = GrabStart("Right", Vector2(0.5, 0.5))
    fake.emit(event)
    assert a == [event] and b == [event]


def test_emit_pose_channel() -> None:
    fake = FakeHandSource()
    poses: list[HandPose] = []
    fake.on_pose(poses.append)
    pose = HandPose(
        hand_id="Right",
        palm=Vector3(0.5, 0.5, 0.0),
        orientation=quat_identity(),
        confidence=0.9,
    )
    fake.emit_pose(pose)
    assert poses == [pose]


def test_start_stop_flags() -> None:
    fake = FakeHandSource()
    assert not fake.started
    fake.start()
    assert fake.started
    fake.stop()
    assert not fake.started
