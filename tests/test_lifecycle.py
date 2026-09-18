"""Lifecycle tests for create_hand_gesture / on_event / on_pose."""

from __future__ import annotations

from hand_interaction import FakeHandSource, create_hand_gesture
from hand_interaction.math3d import quat_identity
from hand_interaction.types import (
    HandMove,
    HandPose,
    PinchStart,
    Vector2,
    Vector3,
)


def test_create_defaults_to_fake() -> None:
    gesture = create_hand_gesture()
    assert isinstance(gesture.source, FakeHandSource)


def test_start_stop() -> None:
    fake = FakeHandSource()
    gesture = create_hand_gesture(source=fake)
    assert not fake.started
    gesture.start()
    assert fake.started
    gesture.stop()
    assert not fake.started


def test_on_event_subscribe_unsubscribe() -> None:
    fake = FakeHandSource()
    gesture = create_hand_gesture(source=fake)
    received: list = []
    unsub = gesture.on_event(received.append)
    fake.emit(PinchStart("Right", Vector2(0.2, 0.3)))
    unsub()
    fake.emit(HandMove("Right", Vector2(0.3, 0.3), Vector2(0.1, 0.0)))
    assert len(received) == 1
    assert isinstance(received[0], PinchStart)


def test_on_pose_channel() -> None:
    fake = FakeHandSource()
    gesture = create_hand_gesture(source=fake)
    poses: list[HandPose] = []
    unsub = gesture.on_pose(poses.append)
    pose = HandPose(
        hand_id="Left",
        palm=Vector3(0.4, 0.5, 0.1),
        orientation=quat_identity(),
        confidence=0.8,
    )
    fake.emit_pose(pose)
    unsub()
    fake.emit_pose(pose)
    assert poses == [pose]


def test_injected_fake_flows_events() -> None:
    fake = FakeHandSource()
    gesture = create_hand_gesture(source=fake)
    events: list = []
    gesture.on_event(events.append)
    gesture.start()
    fake.emit_sequence(
        [
            PinchStart("Right", Vector2(0.5, 0.5)),
            HandMove("Right", Vector2(0.55, 0.5), Vector2(0.05, 0.0)),
        ]
    )
    gesture.stop()
    assert [e.type for e in events] == ["pinch.start", "hand.move"]
