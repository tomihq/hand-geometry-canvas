"""Tests for HandEvent JSON serialization."""

from __future__ import annotations

import json

from hand_interaction.transport.serialize import event_to_dict, event_to_json
from hand_interaction.types import (
    GrabStart,
    HandMove,
    PinchEnd,
    PinchMove,
    PinchStart,
    Vector2,
)


def test_pinch_start_dict() -> None:
    event = PinchStart("Right", Vector2(0.2, 0.3))
    data = event_to_dict(event)
    assert data == {
        "type": "pinch.start",
        "hand_id": "Right",
        "position": {"x": 0.2, "y": 0.3},
    }


def test_pinch_move_dict() -> None:
    event = PinchMove("Right", Vector2(0.25, 0.3))
    data = event_to_dict(event)
    assert data["type"] == "pinch.move"
    assert data["position"] == {"x": 0.25, "y": 0.3}


def test_pinch_end_dict() -> None:
    event = PinchEnd("Left", Vector2(0.1, 0.9))
    assert event_to_dict(event)["type"] == "pinch.end"


def test_grab_start_dict() -> None:
    event = GrabStart("Right", Vector2(0.4, 0.5), fingers_together=True)
    data = event_to_dict(event)
    assert data["type"] == "grab.start"
    assert data["fingers_together"] is True


def test_hand_move_dict() -> None:
    event = HandMove("Right", Vector2(0.42, 0.31), Vector2(0.01, -0.02))
    data = event_to_dict(event)
    assert data["type"] == "hand.move"
    assert data["delta_position"] == {"x": 0.01, "y": -0.02}


def test_event_to_json_roundtrip_keys() -> None:
    event = HandMove("Right", Vector2(0.5, 0.5), Vector2(0.0, 0.0))
    parsed = json.loads(event_to_json(event))
    assert parsed["type"] == "hand.move"
    assert "position" in parsed
