"""Serialization tests for HandEvent → JSON."""

from __future__ import annotations

import json

import pytest

from hand_interaction.transport.serialize import event_to_dict, event_to_json
from hand_interaction.types import (
    HandMove,
    HandRotate,
    PinchEnd,
    PinchStart,
    ResizeEnd,
    ResizeMove,
    ResizeStart,
    Vector2,
)


def test_pinch_start() -> None:
    event = PinchStart("Right", Vector2(0.2, 0.3))
    assert event_to_dict(event) == {
        "type": "pinch.start",
        "hand_id": "Right",
        "position": {"x": 0.2, "y": 0.3},
    }


def test_pinch_end() -> None:
    event = PinchEnd("Left", Vector2(0.1, 0.9))
    assert event_to_dict(event) == {
        "type": "pinch.end",
        "hand_id": "Left",
        "position": {"x": 0.1, "y": 0.9},
    }


def test_hand_move() -> None:
    event = HandMove("Right", Vector2(0.42, 0.31), Vector2(0.01, -0.02))
    assert event_to_dict(event) == {
        "type": "hand.move",
        "hand_id": "Right",
        "position": {"x": 0.42, "y": 0.31},
        "delta_position": {"x": 0.01, "y": -0.02},
    }


def test_hand_rotate() -> None:
    event = HandRotate("Left", 0.15)
    assert event_to_dict(event) == {
        "type": "hand.rotate",
        "hand_id": "Left",
        "delta_rotation": 0.15,
    }


def test_resize_start() -> None:
    event = ResizeStart("Right", "Left", Vector2(0.3, 0.5))
    assert event_to_dict(event) == {
        "type": "resize.start",
        "owner_hand_id": "Right",
        "helper_hand_id": "Left",
        "position": {"x": 0.3, "y": 0.5},
    }


def test_resize_move() -> None:
    event = ResizeMove("Right", "Left", Vector2(0.25, 0.5), Vector2(-0.05, 0.0))
    assert event_to_dict(event) == {
        "type": "resize.move",
        "owner_hand_id": "Right",
        "helper_hand_id": "Left",
        "position": {"x": 0.25, "y": 0.5},
        "delta_position": {"x": -0.05, "y": 0.0},
    }


def test_resize_end() -> None:
    event = ResizeEnd("Right", "Left")
    assert event_to_dict(event) == {
        "type": "resize.end",
        "owner_hand_id": "Right",
        "helper_hand_id": "Left",
    }


def test_event_to_json_roundtrip_shape() -> None:
    event = HandMove("Right", Vector2(0.5, 0.5), Vector2(0.0, 0.0))
    payload = json.loads(event_to_json(event))
    assert payload == event_to_dict(event)
    assert "," in event_to_json(event)  # compact, no spaces required beyond separators


def test_unsupported_type_raises() -> None:
    with pytest.raises(TypeError):
        event_to_dict(object())  # type: ignore[arg-type]
