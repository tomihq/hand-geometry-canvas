"""Pointer events independent of vision implementation."""

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


PointerEvent = PointerMove | PointerDown | PointerUp
