"""Free flight for thrown figures — what happens after the hand lets go.

A figure leaves the hand carrying the hand's own velocity, so how far it goes
is decided by how hard it was thrown. Drag is exponential rather than linear,
which makes that relationship easy to feel: the travel works out to roughly
``speed * FLING_DECAY_TAU``, so twice the throw is twice the distance.

The canvas edges are walls. A figure bounces off them having lost most of its
energy, so it settles somewhere inside rather than parking against a border or
rattling between two of them.

Knows nothing about hands or gestures: callers hand it a velocity and step it
once per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hand_canvas.constants import (
    FLING_BOUNCE_RESTITUTION,
    FLING_DECAY_TAU,
    FLING_MAX_SPEED,
    FLING_MIN_SPEED,
    FLING_STOP_SPEED,
)
from hand_canvas.geometry import Rectangle

if TYPE_CHECKING:
    from hand_canvas.canvas import Canvas


@dataclass
class Flight:
    """Velocity of one figure in flight, in frame widths per second."""

    vx: float
    vy: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


def _bounce(low: float, size: float, velocity: float) -> tuple[float, float]:
    """Reflect one axis off the canvas edges, keeping the figure inside.

    The reflected position is clamped as well as mirrored. Dragging is not
    fenced in, so a figure can be released already hanging off the edge, and
    mirroring a large overshoot would fling it clean out the opposite side.
    """
    limit = 1.0 - size
    if limit <= 0.0:
        # Bigger than the canvas on this axis: no inside left to bounce within,
        # so centre it and give up the velocity.
        return limit * 0.5, 0.0
    if low < 0.0:
        return (
            min(-low * FLING_BOUNCE_RESTITUTION, limit),
            -velocity * FLING_BOUNCE_RESTITUTION,
        )
    if low > limit:
        return (
            max(limit - (low - limit) * FLING_BOUNCE_RESTITUTION, 0.0),
            -velocity * FLING_BOUNCE_RESTITUTION,
        )
    return low, velocity


def _advance(shape: Rectangle, flight: Flight, dt: float) -> tuple[Rectangle, Flight]:
    x, vx = _bounce(shape.x + flight.vx * dt, shape.width, flight.vx)
    y, vy = _bounce(shape.y + flight.vy * dt, shape.height, flight.vy)
    moved = Rectangle(
        x=x,
        y=y,
        width=shape.width,
        height=shape.height,
        id=shape.id,
        z=shape.z,
    )
    return moved, Flight(vx, vy)


class MomentumField:
    """The set of figures currently coasting, advanced one frame at a time."""

    def __init__(self) -> None:
        self._flights: dict[str, Flight] = {}

    def launch(self, shape_id: str, vx: float, vy: float) -> bool:
        """Send a figure off at the given velocity. False if it was a drop.

        Slow releases are the common case — putting a figure down — and those
        should stay exactly where they were left, so they never fly.
        """
        self._flights.pop(shape_id, None)
        flight = Flight(vx, vy)
        speed = flight.speed
        if speed < FLING_MIN_SPEED:
            return False
        if speed > FLING_MAX_SPEED:
            scale = FLING_MAX_SPEED / speed
            flight = Flight(vx * scale, vy * scale)
        self._flights[shape_id] = flight
        return True

    def cancel(self, shape_id: str) -> None:
        """Catch a figure in mid-air, or forget one that no longer exists."""
        self._flights.pop(shape_id, None)

    def clear(self) -> None:
        self._flights.clear()

    @property
    def flying_ids(self) -> set[str]:
        return set(self._flights)

    def step(self, canvas: Canvas, dt: float) -> None:
        if not self._flights or dt <= 0.0:
            return
        decay = math.exp(-dt / FLING_DECAY_TAU)
        for shape_id, flight in list(self._flights.items()):
            shape = canvas.get(shape_id)
            if not isinstance(shape, Rectangle):
                del self._flights[shape_id]
                continue
            moved, flight = _advance(shape, flight, dt)
            canvas.update(moved)
            flight = Flight(flight.vx * decay, flight.vy * decay)
            if flight.speed < FLING_STOP_SPEED:
                del self._flights[shape_id]
            else:
                self._flights[shape_id] = flight
