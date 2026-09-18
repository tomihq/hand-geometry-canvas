"""Hand interaction library — camera → pose → abstract interaction events.

Public surface:
  - HandPose (3D) via on_pose
  - HandEvent (2D/abstract) via on_event
  - FakeHandSource for tests without a camera
"""

from __future__ import annotations

from hand_interaction.fake import FakeHandSource
from hand_interaction.types import (
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

__version__ = "0.1.0"

__all__ = [
    "FakeHandSource",
    "HandEvent",
    "HandMove",
    "HandPose",
    "HandRotate",
    "PinchEnd",
    "PinchStart",
    "Quaternion",
    "Vector2",
    "Vector3",
    "__version__",
]
