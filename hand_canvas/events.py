"""Pointer / grab events independent of vision implementation.

Pinch → PointerDown / PointerMove / PointerUp (create, stretch, resize).
Closed fist → GrabDown / GrabMove / GrabUp (select + move figures).
"""

from __future__ import annotations

from dataclasses import dataclass

from hand_canvas.geometry import Point


@dataclass(frozen=True)
class PointerMove:
    position: Point
    pointer_id: str = "default"


@dataclass(frozen=True)
class PointerDown:
    position: Point
    pointer_id: str = "default"


@dataclass(frozen=True)
class PointerUp:
    position: Point
    pointer_id: str = "default"


@dataclass(frozen=True)
class GrabDown:
    """Fist closed — start selecting / moving a figure."""

    position: Point
    pointer_id: str = "default"


@dataclass(frozen=True)
class GrabMove:
    position: Point
    pointer_id: str = "default"


@dataclass(frozen=True)
class GrabUp:
    """Hand opened — release the grabbed figure."""

    position: Point
    pointer_id: str = "default"


@dataclass(frozen=True)
class SweepClear:
    """Open palm swiped sideways across the frame — wipe the canvas."""

    position: Point
    pointer_id: str = "default"


PointerEvent = (
    PointerMove
    | PointerDown
    | PointerUp
    | GrabDown
    | GrabMove
    | GrabUp
    | SweepClear
)
