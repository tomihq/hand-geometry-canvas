"""Interaction engine: pointer events → canvas shape mutations.

Does not know about MediaPipe or cameras.

Supports multiple simultaneous pointers (one session per pointer_id).

Pinch-create events from both hands are queued and applied in parallel: hit-tests
use a pre-batch canvas snapshot so one create cannot steal the other hand's
empty-space target. Newly created points stay selected (STRETCHING) by default.

Lock model (two hands on one figure):
  - First hand to grab a figure becomes the owner (lock): can move or resize.
  - Second hand cannot steal the lock and cannot move the figure.
  - Second hand only resizes, and cannot take over a corner already held by
    the other hand — that corner stays pinned to the owner's finger.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    Shape,
    nearest_corner,
    nearest_corner_excluding,
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


@dataclass(frozen=True)
class _DownPlan:
    pointer_id: str
    position: Point
    hit: Shape | None


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

    def _held_corners(
        self, shape_id: str, exclude_pointer: str | None = None
    ) -> set[Corner]:
        held: set[Corner] = set()
        for pid, session in self._sessions.items():
            if exclude_pointer is not None and pid == exclude_pointer:
                continue
            if (
                session.selected_id == shape_id
                and session.state == InteractionState.RESIZING
                and session.active_corner is not None
            ):
                held.add(session.active_corner)
        return held

    def _owner_session(self, shape_id: str) -> PointerSession | None:
        owner_id = self._owner_of(shape_id)
        if owner_id is None:
            return None
        return self._sessions.get(owner_id)

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
            owner_first = sorted(active, key=lambda item: (not item[1].is_owner))
            _pid, session = owner_first[0]
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
        """Queue same-frame events and apply downs/moves in parallel batches."""
        downs = [e for e in events if isinstance(e, PointerDown)]
        moves = [e for e in events if isinstance(e, PointerMove)]
        ups = [e for e in events if isinstance(e, PointerUp)]

        if downs:
            self._handle_downs_parallel(downs, canvas)
        if moves:
            self._handle_moves_parallel(moves, canvas)
        for up in ups:
            self._on_up(up.pointer_id)

    def _handle_downs_parallel(
        self, downs: list[PointerDown], canvas: Canvas
    ) -> None:
        # Snapshot hits BEFORE any mutation so both empty pinches create points.
        plans = [
            _DownPlan(
                pointer_id=event.pointer_id,
                position=event.position,
                hit=canvas.hit_test(
                    event.position.x, event.position.y, self._hit_radius
                ),
            )
            for event in downs
        ]

        creates = [p for p in plans if p.hit is None]
        grabs = [p for p in plans if p.hit is not None]

        if creates:
            with ThreadPoolExecutor(max_workers=max(len(creates), 1)) as pool:
                points = list(
                    pool.map(
                        lambda plan: PointShape(
                            position=Point(plan.position.x, plan.position.y)
                        ),
                        creates,
                    )
                )
            for plan, point in zip(creates, points):
                canvas.add(point)
                self._select_new_point(plan.pointer_id, point, plan.position)

        for plan in grabs:
            self._apply_grab(plan.pointer_id, plan.position, plan.hit, canvas)

    def _select_new_point(
        self, pointer_id: str, point: PointShape, position: Point
    ) -> None:
        """Keep the freshly created point selected for this hand."""
        session = self._session(pointer_id)
        session.selected_id = point.id
        session.is_owner = True
        session.drag_origin = Point(point.position.x, point.position.y)
        session.grab_pointer = None
        session.last_pointer = Point(position.x, position.y)
        session.shape_origin = None
        session.active_corner = None
        session.shape_size = None
        session.state = InteractionState.STRETCHING

    def _apply_grab(
        self,
        pointer_id: str,
        position: Point,
        hit: Shape,
        canvas: Canvas,
    ) -> None:
        session = self._session(pointer_id)

        if isinstance(hit, PointShape):
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
            if owner is not None and owner != pointer_id:
                self._begin_resize(
                    session,
                    hit,
                    position,
                    is_owner=False,
                    pointer_id=pointer_id,
                    prefer_near_corner=False,
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
                    pointer_id=pointer_id,
                    prefer_near_corner=True,
                )
                return

            session.selected_id = hit.id
            session.is_owner = True
            session.active_corner = None
            session.drag_origin = None
            session.grab_pointer = Point(position.x, position.y)
            session.last_pointer = Point(position.x, position.y)
            session.shape_origin = Point(hit.x, hit.y)
            session.shape_size = (hit.width, hit.height)
            session.state = InteractionState.MOVING

    def _begin_resize(
        self,
        session: PointerSession,
        shape: Rectangle,
        position: Point,
        *,
        is_owner: bool,
        pointer_id: str,
        prefer_near_corner: bool,
    ) -> bool:
        excluded = self._held_corners(shape.id, exclude_pointer=pointer_id)

        corner: Corner | None = None
        if prefer_near_corner and not excluded:
            corner = nearest_corner(position, shape, self._corner_radius)

        if corner is None or corner in excluded:
            corner = nearest_corner_excluding(position, shape, excluded)

        if corner is None:
            return False

        session.selected_id = shape.id
        session.is_owner = is_owner
        session.active_corner = corner
        session.drag_origin = None
        session.grab_pointer = Point(position.x, position.y)
        session.last_pointer = Point(position.x, position.y)
        session.shape_origin = Point(shape.x, shape.y)
        session.shape_size = (shape.width, shape.height)
        session.state = InteractionState.RESIZING
        return True

    def _pin_owner_corner(self, shape: Rectangle, shape_id: str) -> Rectangle:
        owner = self._owner_session(shape_id)
        if (
            owner is None
            or owner.state != InteractionState.RESIZING
            or owner.active_corner is None
            or owner.last_pointer is None
        ):
            return shape
        return resize_rectangle_from_corner(
            shape, owner.active_corner, owner.last_pointer
        )

    def _compute_move(
        self, position: Point, pointer_id: str, canvas: Canvas
    ) -> tuple[Shape, PointerSession] | None:
        """Pure-ish move computation against current canvas (no write yet)."""
        session = self._sessions.get(pointer_id)
        if session is None or session.selected_id is None:
            return None

        if session.state == InteractionState.STRETCHING:
            if session.drag_origin is None:
                return None
            origin = session.drag_origin
            existing = canvas.get(session.selected_id)
            z = existing.z if existing is not None else 0
            dx = abs(position.x - origin.x)
            dy = abs(position.y - origin.y)
            session.last_pointer = Point(position.x, position.y)
            if dx < MIN_RECT_SIZE and dy < MIN_RECT_SIZE:
                return (
                    PointShape(
                        position=Point(origin.x, origin.y),
                        id=session.selected_id,
                        z=z,
                    ),
                    session,
                )
            return (
                rectangle_from_points(
                    origin, position, shape_id=session.selected_id, z=z
                ),
                session,
            )

        if session.state == InteractionState.RESIZING:
            shape = canvas.get(session.selected_id)
            if not isinstance(shape, Rectangle) or session.active_corner is None:
                return None

            held_by_others = self._held_corners(
                session.selected_id, exclude_pointer=pointer_id
            )
            active = session.active_corner
            if active in held_by_others:
                free = nearest_corner_excluding(position, shape, held_by_others)
                if free is None:
                    return None
                active = free
                session.active_corner = free

            session.last_pointer = Point(position.x, position.y)
            resized = resize_rectangle_from_corner(shape, active, position)
            if resized.width < MIN_RECT_SIZE or resized.height < MIN_RECT_SIZE:
                return None
            if not session.is_owner:
                resized = self._pin_owner_corner(resized, session.selected_id)
            return resized, session

        if session.state == InteractionState.MOVING:
            if session.last_pointer is None:
                return None
            shape = canvas.get(session.selected_id)
            if not isinstance(shape, Rectangle):
                return None
            dx = position.x - session.last_pointer.x
            dy = position.y - session.last_pointer.y
            session.last_pointer = Point(position.x, position.y)
            if dx == 0.0 and dy == 0.0:
                return None
            return translate_rectangle(shape, dx, dy), session

        return None

    def _handle_moves_parallel(
        self, moves: list[PointerMove], canvas: Canvas
    ) -> None:
        def compute(event: PointerMove) -> Shape | None:
            result = self._compute_move(event.position, event.pointer_id, canvas)
            return result[0] if result is not None else None

        if len(moves) == 1:
            shape = compute(moves[0])
            if shape is not None:
                canvas.update(shape)
            return

        with ThreadPoolExecutor(max_workers=len(moves)) as pool:
            results = list(pool.map(compute, moves))

        # Commit by shape id — last write wins only across different ids usually
        by_id: dict[str, Shape] = {}
        for shape in results:
            if shape is not None:
                by_id[shape.id] = shape
        for shape in by_id.values():
            canvas.update(shape)

    def _on_up(self, pointer_id: str) -> None:
        self._clear_session(pointer_id)

    def _clear_session(self, pointer_id: str) -> None:
        self._sessions[pointer_id] = PointerSession()
