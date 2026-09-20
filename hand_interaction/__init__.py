"""Hand interaction library — camera → pose → abstract interaction events.

Public surface:
  - create_hand_gesture() → on_event / on_pose / start / stop
  - HandPose (3D) and HandEvent (2D/abstract) types
  - FakeHandSource for tests without a camera
"""

from __future__ import annotations

from hand_interaction.api import HandGesture, create_hand_gesture
from hand_interaction.fake import FakeHandSource
from hand_interaction.types import (
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

__version__ = "0.1.0"

__all__ = [
    "FakeHandSource",
    "GrabEnd",
    "GrabMove",
    "GrabStart",
    "HandEvent",
    "HandGesture",
    "HandMove",
    "HandPose",
    "PinchEnd",
    "PinchMove",
    "PinchStart",
    "Quaternion",
    "Vector2",
    "Vector3",
    "create_hand_gesture",
    "__version__",
]
