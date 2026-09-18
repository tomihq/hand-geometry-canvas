"""JSON serialization for HandEvent over WebSocket."""

from __future__ import annotations

import json
from typing import Any

from hand_interaction.types import (
    HandEvent,
    HandMove,
    HandRotate,
    PinchEnd,
    PinchStart,
    Vector2,
)


def _vector2_to_dict(v: Vector2) -> dict[str, float]:
    return {"x": v.x, "y": v.y}


def event_to_dict(event: HandEvent) -> dict[str, Any]:
    """Convert a HandEvent to a plain dict suitable for JSON."""
    if isinstance(event, PinchStart):
        return {
            "type": event.type,
            "hand_id": event.hand_id,
            "position": _vector2_to_dict(event.position),
        }
    if isinstance(event, PinchEnd):
        return {
            "type": event.type,
            "hand_id": event.hand_id,
            "position": _vector2_to_dict(event.position),
        }
    if isinstance(event, HandMove):
        return {
            "type": event.type,
            "hand_id": event.hand_id,
            "position": _vector2_to_dict(event.position),
            "delta_position": _vector2_to_dict(event.delta_position),
        }
    if isinstance(event, HandRotate):
        return {
            "type": event.type,
            "hand_id": event.hand_id,
            "delta_rotation": event.delta_rotation,
        }
    raise TypeError(f"unsupported HandEvent: {type(event)!r}")


def event_to_json(event: HandEvent) -> str:
    """Serialize a HandEvent to a compact JSON string."""
    return json.dumps(event_to_dict(event), separators=(",", ":"))
