"""Public data contract: 3D pose types and abstract 2D interaction events.

HandPose carries full 3D information for consumers that need it.
HandEvent stays 2D/abstract so a canvas client never has to know Vector3
or Quaternion just to move a rectangle.

Gesture events match the canvas model (minus sweep-to-clear):
  - Pinch* — thumb–index create / stretch cursor
  - Grab*  — closed fist select + move
  - Resize* — bimanual helper latch (owner holds; helper cursor drives resize)
  - HandMove — palm tracking for other apps (not gated on a gesture)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Vector2:
    """Normalized image-space point (mirrored camera; y origin at top)."""

    x: float
    y: float


@dataclass(frozen=True)
class Vector3:
    """Hybrid hand-space point: x/y normalized image, z relative depth.

    ``y`` here is flipped vs ``Vector2`` (origin at bottom) for 3D consumers.
    """

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
class PinchMove:
    hand_id: str
    position: Vector2
    type: Literal["pinch.move"] = field(default="pinch.move", init=False)


@dataclass(frozen=True)
class PinchEnd:
    hand_id: str
    position: Vector2
    type: Literal["pinch.end"] = field(default="pinch.end", init=False)


@dataclass(frozen=True)
class GrabStart:
    """Fist closed — start selecting / moving.

    ``fingers_together`` reports thumb–index pressed together, which in
    projection a closed fist and a camera-aimed pinch can share.
    """

    hand_id: str
    position: Vector2
    fingers_together: bool = False
    type: Literal["grab.start"] = field(default="grab.start", init=False)


@dataclass(frozen=True)
class GrabMove:
    hand_id: str
    position: Vector2
    fingers_together: bool = False
    type: Literal["grab.move"] = field(default="grab.move", init=False)


@dataclass(frozen=True)
class GrabEnd:
    """Hand opened — release the grab."""

    hand_id: str
    position: Vector2
    type: Literal["grab.end"] = field(default="grab.end", init=False)


@dataclass(frozen=True)
class HandMove:
    """Palm cursor tracking (always-on; not tied to pinch/grab)."""

    hand_id: str
    position: Vector2
    delta_position: Vector2
    type: Literal["hand.move"] = field(default="hand.move", init=False)


@dataclass(frozen=True)
class ResizeStart:
    """Second hand joined while the first still holds (canvas helper latch).

    Owner holds via pinch or fist; helper cursor drives resize — the consumer
    applies corner-follow on its own figure.
    """

    owner_hand_id: str
    helper_hand_id: str
    position: Vector2
    type: Literal["resize.start"] = field(default="resize.start", init=False)


@dataclass(frozen=True)
class ResizeMove:
    """Helper hand moved while owner still holds — follow this cursor to resize."""

    owner_hand_id: str
    helper_hand_id: str
    position: Vector2
    delta_position: Vector2
    type: Literal["resize.move"] = field(default="resize.move", init=False)


@dataclass(frozen=True)
class ResizeEnd:
    """Owner or helper released — end of helper resize session."""

    owner_hand_id: str
    helper_hand_id: str
    type: Literal["resize.end"] = field(default="resize.end", init=False)


HandEvent = (
    PinchStart
    | PinchMove
    | PinchEnd
    | GrabStart
    | GrabMove
    | GrabEnd
    | HandMove
    | ResizeStart
    | ResizeMove
    | ResizeEnd
)
