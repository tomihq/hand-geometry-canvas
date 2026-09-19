"""Public data contract: 3D pose types and abstract 2D interaction events.

HandPose carries full 3D information for consumers that need it.
HandEvent stays 2D/abstract so a canvas client never has to know Vector3
or Quaternion just to move a rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Vector2:
    """Normalized screen-space point (mirrored camera; y origin at bottom)."""

    x: float
    y: float


@dataclass(frozen=True)
class Vector3:
    """Hybrid hand-space point: x/y normalized image, z relative depth."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    """Unit quaternion in wxyz order."""

    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandPose:
    """Full 3D hand pose — palm position, orientation, tracking confidence."""

    hand_id: str
    palm: Vector3
    orientation: Quaternion
    confidence: float


@dataclass(frozen=True)
class PinchStart:
    hand_id: str
    position: Vector2
    type: Literal["pinch.start"] = field(default="pinch.start", init=False)


@dataclass(frozen=True)
class PinchEnd:
    hand_id: str
    position: Vector2
    type: Literal["pinch.end"] = field(default="pinch.end", init=False)


@dataclass(frozen=True)
class HandMove:
    hand_id: str
    position: Vector2
    delta_position: Vector2
    type: Literal["hand.move"] = field(default="hand.move", init=False)


@dataclass(frozen=True)
class HandRotate:
    """Screen-axis rotation in radians (projected from internal 3D Δq)."""

    hand_id: str
    delta_rotation: float
    type: Literal["hand.rotate"] = field(default="hand.rotate", init=False)


@dataclass(frozen=True)
class ResizeStart:
    """Second hand joined while the first still pinches (canvas helper latch).

    Mirrors hand_canvas lock model: owner holds, helper drives resize via
    cursor position — the consumer applies corner-follow on its own figure.
    """

    owner_hand_id: str
    helper_hand_id: str
    position: Vector2
    type: Literal["resize.start"] = field(default="resize.start", init=False)


@dataclass(frozen=True)
class ResizeMove:
    """Helper hand moved while owner still pinches — follow this cursor to resize."""

    owner_hand_id: str
    helper_hand_id: str
    position: Vector2
    delta_position: Vector2
    type: Literal["resize.move"] = field(default="resize.move", init=False)


@dataclass(frozen=True)
class ResizeEnd:
    """Owner or helper released pinch — end of helper resize session."""

    owner_hand_id: str
    helper_hand_id: str
    type: Literal["resize.end"] = field(default="resize.end", init=False)


HandEvent = (
    PinchStart
    | PinchEnd
    | HandMove
    | HandRotate
    | ResizeStart
    | ResizeMove
    | ResizeEnd
)
