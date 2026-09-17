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
    """Fist closed — start selecting / moving a figure.

    ``fingers_together`` reports that the thumb and index tips are pressed
    together, which in projection a closed fist and a pinch made towards the
    camera share. The pose alone cannot say which was meant, so the flag is
    passed on and whatever is under the hand settles it.
    """

    position: Point
    pointer_id: str = "default"
    fingers_together: bool = False


@dataclass(frozen=True)
class GrabMove:
    position: Point
    pointer_id: str = "default"
    fingers_together: bool = False


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
