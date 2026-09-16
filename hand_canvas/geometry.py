"""Pure geometry types and helpers — no OpenCV / MediaPipe dependency."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum

from hand_canvas.constants import MIN_SHAPE_SIZE


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass
class Transform:
    """Future-ready transform; MVP uses position + scale for rectangles."""

    position: Point
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0


@dataclass
class PointShape:
    position: Point
    id: str = field(default_factory=lambda: f"point-{uuid.uuid4().hex[:8]}")
    z: int = 0


@dataclass
class Rectangle:
    x: float
    y: float
    width: float
    height: float
    id: str = field(default_factory=lambda: f"rect-{uuid.uuid4().hex[:8]}")
    z: int = 0
    transform: Transform | None = None

    def __post_init__(self) -> None:
        if self.transform is None:
            self.transform = Transform(
                position=Point(self.x, self.y),
                scale_x=self.width,
                scale_y=self.height,
                rotation=0.0,
            )


Shape = PointShape | Rectangle


class Corner(Enum):
    TL = "tl"
    TR = "tr"
    BL = "bl"
    BR = "br"


def distance(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def shape_area(shape: Shape) -> float:
    if isinstance(shape, PointShape):
        return 0.0
    return max(shape.width, 0.0) * max(shape.height, 0.0)


def is_visible_figure(shape: Shape) -> bool:
    """A finished figure: big enough on both axes to read as a shape on screen.

    Bare points and slivers fail this — they are previews mid-gesture, not
    something the canvas should keep or paint once the hand lets go.
    """
    if isinstance(shape, PointShape):
        return False
    return shape.width >= MIN_SHAPE_SIZE and shape.height >= MIN_SHAPE_SIZE


def rectangle_from_points(
    start: Point,
    current: Point,
    shape_id: str | None = None,
    z: int = 0,
) -> Rectangle:
    left = min(start.x, current.x)
    top = min(start.y, current.y)
    width = abs(current.x - start.x)
    height = abs(current.y - start.y)
    kwargs: dict = {
        "x": left,
        "y": top,
        "width": width,
        "height": height,
        "z": z,
    }
    if shape_id is not None:
        kwargs["id"] = shape_id
    return Rectangle(**kwargs)


def point_in_rectangle(point: Point, rect: Rectangle, padding: float = 0.0) -> bool:
    return (
        rect.x - padding <= point.x <= rect.x + rect.width + padding
        and rect.y - padding <= point.y <= rect.y + rect.height + padding
    )


def translate_rectangle(rect: Rectangle, dx: float, dy: float) -> Rectangle:
    return Rectangle(
        x=rect.x + dx,
        y=rect.y + dy,
        width=rect.width,
        height=rect.height,
        id=rect.id,
        z=rect.z,
    )


def rectangle_corners(rect: Rectangle) -> dict[Corner, Point]:
    return {
        Corner.TL: Point(rect.x, rect.y),
        Corner.TR: Point(rect.x + rect.width, rect.y),
        Corner.BL: Point(rect.x, rect.y + rect.height),
        Corner.BR: Point(rect.x + rect.width, rect.y + rect.height),
    }


def opposite_corner(corner: Corner) -> Corner:
    return {
        Corner.TL: Corner.BR,
        Corner.TR: Corner.BL,
        Corner.BL: Corner.TR,
        Corner.BR: Corner.TL,
    }[corner]


def nearest_corner(
    point: Point, rect: Rectangle, radius: float
) -> Corner | None:
    best: Corner | None = None
    best_dist = radius
    for corner, pos in rectangle_corners(rect).items():
        d = distance(point, pos)
        if d <= best_dist:
            best_dist = d
            best = corner
    return best


def nearest_corner_any(point: Point, rect: Rectangle) -> Corner:
    """Always return the closest corner (no hit-radius limit)."""
    corners = rectangle_corners(rect)
    return min(corners.keys(), key=lambda c: distance(point, corners[c]))


def nearest_corner_excluding(
    point: Point,
    rect: Rectangle,
    excluded: set[Corner],
) -> Corner | None:
    """Closest corner that is not in ``excluded``. None if all are excluded."""
    corners = {
        corner: pos
        for corner, pos in rectangle_corners(rect).items()
        if corner not in excluded
    }
    if not corners:
        return None
    return min(corners.keys(), key=lambda c: distance(point, corners[c]))


def resize_rectangle_from_corner(
    rect: Rectangle,
    active_corner: Corner,
    cursor: Point,
) -> Rectangle:
    """Move one corner to cursor; keep the opposite corner fixed."""
    corners = rectangle_corners(rect)
    anchor = corners[opposite_corner(active_corner)]
    return rectangle_from_points(anchor, cursor, shape_id=rect.id, z=rect.z)
