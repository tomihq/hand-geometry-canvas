"""Interaction engine: pointer events → canvas shape mutations.

Does not know about MediaPipe or cameras.

Supports multiple simultaneous pointers (one session per pointer_id).

Lock model (two hands on one figure):
  - First hand to grab a figure becomes the owner (lock): can move or resize.
  - Second hand on a locked figure cannot steal the lock; it only resizes
    (preferably from corners; body grab still maps to nearest-corner resize).
  - Move uses incremental deltas so it composes with the other hand's resize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from hand_canvas.constants import CORNER_HIT_RADIUS, HIT_RADIUS, MIN_RECT_SIZE
from hand_canvas.events import PointerDown, PointerEvent, PointerMove, PointerUp
from hand_canvas.geometry import (
    Corner,
    Point,
    PointShape,
    Rectangle,
    nearest_corner,
    nearest_corner_any,
    rectangle_from_points,
    resize_rectangle_from_corner,
    translate_rectangle,
)

if TYPE_CHECKING:
    from hand_canvas.canvas import Canvas


class InteractionState(Enum):
    IDLE = "IDLE"
    STRETCHING = "STRETCHING"
    MOVING = "MOVING"
    RESIZING = "RESIZING"


@dataclass
class PointerSession:
    state: InteractionState = InteractionState.IDLE
    selected_id: str | None = None
    drag_origin: Point | None = None
    grab_pointer: Point | None = None
    last_pointer: Point | None = None
    shape_origin: Point | None = None
    active_corner: Corner | None = None
    shape_size: tuple[float, float] | None = None
    is_owner: bool = False


@dataclass
class InteractionDebug:
    state: InteractionState
    selected_id: str | None
    drag_origin: Point | None
    sessions: dict[str, PointerSession] = field(default_factory=dict)
    lock_owner: str | None = None


class InteractionEngine:
    def __init__(
        self,
        hit_radius: float = HIT_RADIUS,
        corner_radius: float = CORNER_HIT_RADIUS,
    ) -> None:
        self._hit_radius = hit_radius
        self._corner_radius = corner_radius
        self._sessions: dict[str, PointerSession] = {}

    def _session(self, pointer_id: str) -> PointerSession:
        if pointer_id not in self._sessions:
            self._sessions[pointer_id] = PointerSession()
        return self._sessions[pointer_id]

    def selected_ids(self) -> set[str]:
        return {
            s.selected_id
            for s in self._sessions.values()
            if s.selected_id is not None and s.state != InteractionState.IDLE
        }

    def _owner_of(self, shape_id: str) -> str | None:
        for pid, session in self._sessions.items():
            if (
                session.is_owner
                and session.selected_id == shape_id
                and session.state != InteractionState.IDLE
            ):
                return pid
        return None

    @property
    def debug(self) -> InteractionDebug:
        active = [
            (pid, s)
            for pid, s in self._sessions.items()
            if s.state != InteractionState.IDLE
        ]
        lock_owner = None
        selected = None
        state = InteractionState.IDLE
        drag_origin = None
        if active:
            # Prefer showing the owner session in the summary line
            owner_first = sorted(active, key=lambda item: (not item[1].is_owner))
            pid, session = owner_first[0]
            state = session.state
            selected = session.selected_id
            drag_origin = session.drag_origin
            if session.selected_id is not None:
                lock_owner = self._owner_of(session.selected_id)
        return InteractionDebug(
            state=state,
            selected_id=selected,
            drag_origin=drag_origin,
            sessions=dict(self._sessions),
            lock_owner=lock_owner,
        )

    def handle(self, events: list[PointerEvent], canvas: Canvas) -> None:
        for event in events:
            if isinstance(event, PointerDown):
                self._on_down(event.position, event.pointer_id, canvas)
            elif isinstance(event, PointerMove):
                self._on_move(event.position, event.pointer_id, canvas)
            elif isinstance(event, PointerUp):
                self._on_up(event.pointer_id)

    def _begin_resize(
        self,
        session: PointerSession,
        shape: Rectangle,
        position: Point,
        *,
        is_owner: bool,
        force_corner: bool,
    ) -> None:
        if force_corner:
            corner = nearest_corner_any(position, shape)
        else:
            corner = nearest_corner(position, shape, self._corner_radius)
            if corner is None:
                corner = nearest_corner_any(position, shape)
        session.selected_id = shape.id
        session.is_owner = is_owner
        session.active_corner = corner
        session.drag_origin = None
        session.grab_pointer = Point(position.x, position.y)
        session.last_pointer = Point(position.x, position.y)
        session.shape_origin = Point(shape.x, shape.y)
        session.shape_size = (shape.width, shape.height)
        session.state = InteractionState.RESIZING

    def _on_down(self, position: Point, pointer_id: str, canvas: Canvas) -> None:
        session = self._session(pointer_id)
        hit = canvas.hit_test(position.x, position.y, self._hit_radius)

        if isinstance(hit, PointShape):
            # Points are only stretched by a free (owning) grab
            if self._owner_of(hit.id) is not None:
                return
            canvas.bring_to_front(hit.id)
            session.selected_id = hit.id
            session.is_owner = True
            session.drag_origin = Point(hit.position.x, hit.position.y)
            session.grab_pointer = None
            session.last_pointer = Point(position.x, position.y)
            session.shape_origin = None
            session.active_corner = None
            session.shape_size = None
            session.state = InteractionState.STRETCHING
            return

        if isinstance(hit, Rectangle):
            owner = self._owner_of(hit.id)

            # Another hand already holds the lock → dimension changes only
            if owner is not None and owner != pointer_id:
                self._begin_resize(
                    session,
                    hit,
                    position,
                    is_owner=False,
                    force_corner=True,
                )
                return

            canvas.bring_to_front(hit.id)
            corner = nearest_corner(position, hit, self._corner_radius)
            if corner is not None:
                self._begin_resize(
                    session,
                    hit,
                    position,
                    is_owner=True,
                    force_corner=False,
                )
                return

            # Body grab → owner lock + move
            session.selected_id = hit.id
            session.is_owner = True
            session.active_corner = None
            session.drag_origin = None
            session.grab_pointer = Point(position.x, position.y)
            session.last_pointer = Point(position.x, position.y)
            session.shape_origin = Point(hit.x, hit.y)
            session.shape_size = (hit.width, hit.height)
            session.state = InteractionState.MOVING
            return

        # Empty space: place a point only
        canvas.add(PointShape(position=Point(position.x, position.y)))
        self._clear_session(pointer_id)

    def _on_move(self, position: Point, pointer_id: str, canvas: Canvas) -> None:
        session = self._sessions.get(pointer_id)
        if session is None or session.selected_id is None:
            return

        if session.state == InteractionState.STRETCHING:
            if session.drag_origin is None:
                return
            origin = session.drag_origin
            existing = canvas.get(session.selected_id)
            z = existing.z if existing is not None else 0
            dx = abs(position.x - origin.x)
            dy = abs(position.y - origin.y)
            if dx < MIN_RECT_SIZE and dy < MIN_RECT_SIZE:
                canvas.update(
                    PointShape(
                        position=Point(origin.x, origin.y),
                        id=session.selected_id,
                        z=z,
                    )
                )
                return
            canvas.update(
                rectangle_from_points(
                    origin, position, shape_id=session.selected_id, z=z
                )
            )
            return

        if session.state == InteractionState.RESIZING:
            shape = canvas.get(session.selected_id)
            if not isinstance(shape, Rectangle) or session.active_corner is None:
                return
            resized = resize_rectangle_from_corner(
                shape, session.active_corner, position
            )
            if resized.width < MIN_RECT_SIZE or resized.height < MIN_RECT_SIZE:
                return
            canvas.update(resized)
            session.last_pointer = Point(position.x, position.y)
            return

        if session.state == InteractionState.MOVING:
            if session.last_pointer is None:
                return
            shape = canvas.get(session.selected_id)
            if not isinstance(shape, Rectangle):
                return
            # Incremental translate so concurrent corner-resize is preserved
            dx = position.x - session.last_pointer.x
            dy = position.y - session.last_pointer.y
            session.last_pointer = Point(position.x, position.y)
            if dx == 0.0 and dy == 0.0:
                return
            canvas.update(translate_rectangle(shape, dx, dy))

    def _on_up(self, pointer_id: str) -> None:
        self._clear_session(pointer_id)

    def _clear_session(self, pointer_id: str) -> None:
        self._sessions[pointer_id] = PointerSession()
