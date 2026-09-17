"""Interaction engine: pointer events → canvas shape mutations.

Does not know about MediaPipe or cameras.

Supports multiple simultaneous pointers (one session per pointer_id).

Pinch-create events from both hands are queued and applied in parallel: hit-tests
use a pre-batch canvas snapshot so one create cannot steal the other hand's
empty-space target. Newly created points stay selected (STRETCHING) by default.

Gestures:
  - Pinch → create point / stretch / resize
  - Closed fist on a figure → select + move; open hand → release
  - Letting go of a figure mid-swing throws it: it keeps travelling at the
    hand's speed, slows down, and bounces off the edges of the canvas
  - Near the trash zone a held figure rides on the hand, sized to fit; the
    zone deletes anything inside it, dropped there or carried in

Lock model (two hands on one figure):
  - First hand to grab a figure becomes the owner (lock): can move.
  - Second hand: any action that starts inside the locked figure latches on
    and follows that hand's movement to resize (gesture type does not matter).
  - Second hand cannot steal the lock or move the figure.
  - A corner held by the owner stays pinned to the owner's finger.
"""

from __future__ import annotations

import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from hand_canvas.constants import (
    CORNER_HIT_RADIUS,
    FLING_MIN_SPAN,
    FLING_SAMPLE_WINDOW,
    FLING_TRAIL_SAMPLES,
    GRAB_HIT_PADDING,
    HELPER_HIT_PADDING,
    HIT_RADIUS,
    MIN_SHAPE_SIZE,
    TRASH_EVICT_GAP,
    TRASH_FIT_MARGIN,
    TRASH_PULL_MIN_SCALE,
    TRASH_PULL_RADIUS,
    TRASH_ZONE_H,
    TRASH_ZONE_W,
    TRASH_ZONE_X,
    TRASH_ZONE_Y,
)
from hand_canvas.events import (
    GrabDown,
    GrabMove,
    GrabUp,
    PointerDown,
    PointerEvent,
    PointerMove,
    PointerUp,
    SweepClear,
)
from hand_canvas.geometry import (
    Corner,
    Point,
    PointShape,
    Rectangle,
    Shape,
    is_visible_figure,
    nearest_corner,
    nearest_corner_excluding,
    point_in_rectangle,
    rectangle_from_points,
    resize_rectangle_from_corner,
    translate_rectangle,
)
from hand_canvas.momentum import MomentumField

if TYPE_CHECKING:
    from hand_canvas.canvas import Canvas


# Longest frame the flight physics will integrate in one go. A hitch would
# otherwise teleport a figure across the canvas in a single step.
MAX_STEP = 0.1


def trash_zone() -> Rectangle:
    """Drop zone for deleting a held figure (normalized frame coords)."""
    return Rectangle(
        x=TRASH_ZONE_X,
        y=TRASH_ZONE_Y,
        width=TRASH_ZONE_W,
        height=TRASH_ZONE_H,
        id="trash-zone",
    )


def _shape_center(shape: Shape) -> Point:
    if isinstance(shape, Rectangle):
        return Point(shape.x + shape.width * 0.5, shape.y + shape.height * 0.5)
    return shape.position


def _distance_to_trash(position: Point) -> float:
    """Gap between a point and the trash zone; 0 once it is inside."""
    zone = trash_zone()
    dx = max(zone.x - position.x, 0.0, position.x - (zone.x + zone.width))
    dy = max(zone.y - position.y, 0.0, position.y - (zone.y + zone.height))
    return math.hypot(dx, dy)


def _rect_at(center: Point, width: float, height: float, like: Rectangle) -> Rectangle:
    return Rectangle(
        x=center.x - width * 0.5,
        y=center.y - height * 0.5,
        width=width,
        height=height,
        id=like.id,
        z=like.z,
    )


def _evicted(shape: Rectangle) -> Rectangle:
    """Park a binned figure beside the zone, so undo cannot feed it back in."""
    zone = trash_zone()
    center = _shape_center(shape)
    if not point_in_rectangle(center, zone, padding=0.0):
        return shape
    half = shape.width * 0.5
    x = max(zone.x - half - TRASH_EVICT_GAP, half)
    return _rect_at(Point(x, center.y), shape.width, shape.height, shape)


def _fit_in_trash_scale(width: float, height: float) -> float:
    """How much a figure has to shrink to drop inside the bin, at most."""
    zone = trash_zone()
    fit = min(
        zone.width * TRASH_FIT_MARGIN / max(width, 1e-6),
        zone.height * TRASH_FIT_MARGIN / max(height, 1e-6),
    )
    return max(min(fit, 1.0), TRASH_PULL_MIN_SCALE)


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
    # Size and hand-relative position of the figure before the bin reached for
    # it. Both are restored exactly, so carrying it away — or binning it and
    # undoing — gives back the real figure rather than the shrunken husk.
    full_size: tuple[float, float] | None = None
    full_offset: Point | None = None
    # Where the hand has been while dragging, so a release can read off how
    # fast the figure was being carried and throw it at that speed.
    trail: deque[tuple[float, Point]] = field(
        default_factory=lambda: deque(maxlen=FLING_TRAIL_SAMPLES)
    )


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
        self._momentum = MomentumField()
        self._now = 0.0
        self._dt = 0.0

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

    def handle(
        self, events: list[PointerEvent], canvas: Canvas, now: float | None = None
    ) -> None:
        """Queue same-frame events and apply downs/moves in parallel batches."""
        self._tick(now)
        downs = [e for e in events if isinstance(e, PointerDown)]
        moves = [e for e in events if isinstance(e, PointerMove)]
        ups = [e for e in events if isinstance(e, PointerUp)]
        grab_downs = [e for e in events if isinstance(e, GrabDown)]
        grab_moves = [e for e in events if isinstance(e, GrabMove)]
        grab_ups = [e for e in events if isinstance(e, GrabUp)]
        sweeps = [e for e in events if isinstance(e, SweepClear)]

        # A hand can let go and start a new gesture in the same frame: opening
        # a fist passes through a pinch pose on the way out. That pointer's
        # release has to land first, or the new gesture latches onto the very
        # session the release was meant to end.
        restarting = {e.pointer_id for e in downs} | {e.pointer_id for e in grab_downs}
        for up in [e for e in grab_ups if e.pointer_id in restarting]:
            self._on_grab_up(up.pointer_id, canvas)
        grab_ups = [e for e in grab_ups if e.pointer_id not in restarting]

        if downs:
            self._handle_downs_parallel(downs, canvas)
        if grab_downs:
            self._handle_grab_downs_parallel(grab_downs, canvas)
        # Acquire with fist before composing moves (helper may start RESIZING)
        if grab_moves:
            self._acquire_from_grab_moves(grab_moves, canvas)
        # Owner MOVING first, then helper RESIZING on the updated shape
        combined_moves = list(moves) + [
            PointerMove(position=m.position, pointer_id=m.pointer_id)
            for m in grab_moves
        ]
        if combined_moves:
            self._handle_moves_composed(combined_moves, canvas)
        for up in ups:
            self._on_pointer_up(up.pointer_id, canvas)
        for up in grab_ups:
            self._on_grab_up(up.pointer_id, canvas)
        if sweeps:
            self._clear_canvas(canvas)
        self._empty_trash_zone(canvas)

    def _tick(self, now: float | None) -> None:
        moment = time.perf_counter() if now is None else now
        # The first frame has no previous stamp to measure against, and a long
        # stall is not a time step the physics should integrate over.
        self._dt = 0.0 if self._now == 0.0 else min(moment - self._now, MAX_STEP)
        self._now = moment

    def advance(self, canvas: Canvas) -> None:
        """Move whatever is still in flight on by one frame."""
        self._momentum.step(canvas, self._dt)
        self._empty_trash_zone(canvas)

    @property
    def flying_ids(self) -> set[str]:
        """Figures still coasting from a throw."""
        return self._momentum.flying_ids

    def _selected_rect_under(
        self, position: Point, pointer_id: str, canvas: Canvas
    ) -> Rectangle | None:
        """Rectangle another hand already holds, if ``position`` is over it."""
        for pid, session in self._sessions.items():
            if pid == pointer_id or session.selected_id is None:
                continue
            if session.state == InteractionState.IDLE:
                continue
            shape = canvas.get(session.selected_id)
            if isinstance(shape, Rectangle) and point_in_rectangle(
                position, shape, padding=HELPER_HIT_PADDING
            ):
                return shape
        return None

    def _resolve_pinch_target(
        self, position: Point, pointer_id: str, canvas: Canvas
    ) -> Shape | None:
        """Hit-test for pinch; locked selection by the other hand wins."""
        locked = self._selected_rect_under(position, pointer_id, canvas)
        if locked is not None:
            return locked

        return canvas.hit_test(position.x, position.y, self._hit_radius)

    def _handle_downs_parallel(
        self, downs: list[PointerDown], canvas: Canvas
    ) -> None:
        # Snapshot hits BEFORE any mutation so both empty pinches create points.
        plans = [
            _DownPlan(
                pointer_id=event.pointer_id,
                position=event.position,
                hit=self._resolve_pinch_target(
                    event.position, event.pointer_id, canvas
                ),
            )
            for event in downs
        ]

        creates = [p for p in plans if p.hit is None]
        pinch_hits = [p for p in plans if p.hit is not None]

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

        for plan in pinch_hits:
            self._apply_pinch_hit(plan.pointer_id, plan.position, plan.hit, canvas)

    def _resolve_grab_target(
        self, position: Point, canvas: Canvas, pointer_id: str | None = None
    ) -> Shape | None:
        """Hit-test for fist grab; prefer a figure the other hand already holds."""
        if pointer_id is not None:
            locked = self._selected_rect_under(position, pointer_id, canvas)
            if locked is not None:
                return locked

        best: Shape | None = None
        best_key = (-1.0, -1)
        for shape in canvas.shapes:
            if isinstance(shape, Rectangle) and point_in_rectangle(
                position, shape, padding=GRAB_HIT_PADDING
            ):
                key = (shape.width * shape.height, shape.z)
                if key > best_key:
                    best = shape
                    best_key = key
        if best is not None:
            return best
        return canvas.hit_test(position.x, position.y, self._hit_radius)

    def _handle_grab_downs_parallel(
        self, grabs: list[GrabDown], canvas: Canvas
    ) -> None:
        plans = [
            (
                _DownPlan(
                    pointer_id=event.pointer_id,
                    position=event.position,
                    hit=self._resolve_grab_target(
                        event.position, canvas, event.pointer_id
                    ),
                ),
                event.fingers_together,
            )
            for event in grabs
        ]
        for plan, fingers_together in plans:
            self._apply_fist_grab(
                plan.pointer_id, plan.position, plan.hit, canvas, fingers_together
            )

    def _acquire_from_grab_moves(
        self, moves: list[GrabMove], canvas: Canvas
    ) -> None:
        """Start move/resize/draw from an ongoing fist when not already busy."""
        for move in moves:
            session = self._sessions.get(move.pointer_id)
            # Anything but idle is already an interaction this hand owns. A
            # drawing session in particular must not be re-acquired: it would
            # lay down a fresh figure on every frame the fist is held.
            if session is not None and session.state != InteractionState.IDLE:
                continue
            hit = self._resolve_grab_target(
                move.position, canvas, move.pointer_id
            )
            self._apply_fist_grab(
                move.pointer_id, move.position, hit, canvas, move.fingers_together
            )

    def _begin_draw(self, pointer_id: str, position: Point, canvas: Canvas) -> None:
        """Start a figure at ``position``, as a pinch on empty space would.

        Nothing is committed yet: what sits on the canvas is a preview, and it
        survives the release only if the hand moved far enough to make a figure
        out of it. So a hand that closes over the canvas and stays put draws
        nothing, which is what keeps this from firing on every stray fist.
        """
        point = PointShape(position=Point(position.x, position.y))
        canvas.add(point)
        self._select_new_point(pointer_id, point, position)

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

    def _start_helper_follow(
        self,
        pointer_id: str,
        position: Point,
        canvas: Canvas,
    ) -> bool:
        """If another hand holds a rect and we start inside it, follow to resize."""
        locked = self._selected_rect_under(position, pointer_id, canvas)
        if locked is None:
            return False
        return self._begin_resize(
            self._session(pointer_id),
            locked,
            position,
            is_owner=False,
            pointer_id=pointer_id,
            prefer_near_corner=False,
        )

    def _apply_pinch_hit(
        self,
        pointer_id: str,
        position: Point,
        hit: Shape,
        canvas: Canvas,
    ) -> None:
        """Pinch on shapes: stretch points or resize (no move)."""
        session = self._session(pointer_id)
        # This hand is no longer carrying the figure towards the bin
        self._restore_pulled(session, canvas)

        # Locked by the other hand → follow this hand (any pinch inside)
        if self._start_helper_follow(pointer_id, position, canvas):
            return

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
            # Body pinch no longer moves — use closed fist instead

    def _apply_fist_grab(
        self,
        pointer_id: str,
        position: Point,
        hit: Shape | None,
        canvas: Canvas,
        fingers_together: bool = False,
    ) -> None:
        """Fist: own+move, or if another hand already holds it → follow/resize."""
        # Locked by the other hand and we started inside → follow movement
        if self._start_helper_follow(pointer_id, position, canvas):
            return

        # A pinch aimed at the camera is indistinguishable from a fist, so the
        # hand alone cannot say which was meant — but a fist over empty space
        # means nothing, while a pinch there means draw. Nothing under the hand
        # and the tips pressed together is therefore read as the pinch.
        if hit is None and fingers_together:
            self._begin_draw(pointer_id, position, canvas)
            return

        if not isinstance(hit, Rectangle):
            return

        owner = self._owner_of(hit.id)
        if owner is not None and owner != pointer_id:
            return

        session = self._session(pointer_id)
        # Closing a fist on a figure still coasting catches it out of the air.
        self._momentum.cancel(hit.id)
        canvas.bring_to_front(hit.id)
        session.selected_id = hit.id
        session.is_owner = True
        session.active_corner = None
        session.drag_origin = None
        session.trail.clear()
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

        self._momentum.cancel(shape.id)
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
            # Both sides must clear the floor: a drag along one axis alone would
            # otherwise commit a zero-thickness sliver.
            if dx < MIN_SHAPE_SIZE or dy < MIN_SHAPE_SIZE:
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
            if resized.width < MIN_SHAPE_SIZE or resized.height < MIN_SHAPE_SIZE:
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
            session.trail.append((self._now, Point(position.x, position.y)))
            moved = translate_rectangle(shape, dx, dy)
            pulled = self._trash_pull(moved, position, session)
            if pulled is None and dx == 0.0 and dy == 0.0:
                return None
            return pulled or moved, session

        return None

    def _throw(self, session: PointerSession, canvas: Canvas) -> None:
        """Hand the released figure the speed it was being carried at."""
        if session.selected_id is None:
            return
        shape = canvas.get(session.selected_id)
        if not isinstance(shape, Rectangle):
            return
        # A figure the bin has reached for is not tracking the hand one to one,
        # so its trail says nothing about how fast it is really going.
        if session.full_size is not None:
            return
        vx, vy = self._release_velocity(session)
        self._momentum.launch(shape.id, vx, vy)

    def _release_velocity(self, session: PointerSession) -> tuple[float, float]:
        """The fastest the hand was going shortly before it let go.

        Not the speed at the instant of release: opening a fist is only
        confirmed a few frames after the fingers start to move, and by then the
        arm has begun to slow down, so the reading there is close to nothing.
        Taking the peak of the recent window gives the speed of the swing and
        ignores the tail where the arm was already stopping.
        """
        samples = [
            (stamp, position)
            for stamp, position in session.trail
            if self._now - stamp <= FLING_SAMPLE_WINDOW
        ]
        best = (0.0, 0.0)
        best_speed = 0.0
        for index, (start_t, start_pos) in enumerate(samples):
            for end_t, end_pos in samples[index + 1 :]:
                span = end_t - start_t
                # Dividing by a span of a millisecond turns jitter into speed.
                if span < FLING_MIN_SPAN:
                    continue
                vx = (end_pos.x - start_pos.x) / span
                vy = (end_pos.y - start_pos.y) / span
                speed = math.hypot(vx, vy)
                if speed > best_speed:
                    best_speed = speed
                    best = (vx, vy)
                break
        return best

    def _trash_pull(
        self, shape: Rectangle, hand: Point, session: PointerSession
    ) -> Rectangle | None:
        """Near the bin the figure settles into your hand, sized to fit in it.

        Nothing is deleted here. The point is to let you aim: once the figure
        rides on the palm, putting your hand in the bin puts the figure in the
        bin, and it is still yours until you open your hand.
        """
        gap = _distance_to_trash(hand)

        if gap >= TRASH_PULL_RADIUS:
            return self._unpulled(shape, hand, session)

        if session.full_size is None:
            center = _shape_center(shape)
            session.full_size = (shape.width, shape.height)
            session.full_offset = Point(center.x - hand.x, center.y - hand.y)

        width, height = session.full_size
        offset = session.full_offset or Point(0.0, 0.0)
        # 0 at the edge of the bin's reach, 1 at its mouth.
        pull = 1.0 - gap / TRASH_PULL_RADIUS
        scale = 1.0 - pull * (1.0 - _fit_in_trash_scale(width, height))
        return _rect_at(
            Point(hand.x + offset.x * (1.0 - pull), hand.y + offset.y * (1.0 - pull)),
            width * scale,
            height * scale,
            shape,
        )

    def _unpulled(
        self, shape: Rectangle, hand: Point, session: PointerSession
    ) -> Rectangle | None:
        """The figure as if the bin had never reached for it."""
        if session.full_size is None:
            return None
        width, height = session.full_size
        offset = session.full_offset or Point(0.0, 0.0)
        session.full_size = None
        session.full_offset = None
        return _rect_at(
            Point(hand.x + offset.x, hand.y + offset.y), width, height, shape
        )

    def _restore_pulled(self, session: PointerSession, canvas: Canvas) -> None:
        """Hand back the real figure, so nothing is kept or binned as a husk."""
        if session.full_size is None or session.selected_id is None:
            return
        shape = canvas.get(session.selected_id)
        hand = session.last_pointer
        if not isinstance(shape, Rectangle) or hand is None:
            return
        restored = self._unpulled(shape, hand, session)
        if restored is not None:
            canvas.update(restored)

    def _handle_moves_composed(
        self, moves: list[PointerMove], canvas: Canvas
    ) -> None:
        """Apply owner moves first, then helper resizes on the updated shape."""
        by_pointer: dict[str, PointerMove] = {}
        for event in moves:
            by_pointer[event.pointer_id] = event

        stretching: list[PointerMove] = []
        moving: list[PointerMove] = []
        resizing: list[PointerMove] = []
        for event in by_pointer.values():
            session = self._sessions.get(event.pointer_id)
            if session is None:
                continue
            if session.state == InteractionState.STRETCHING:
                stretching.append(event)
            elif session.state == InteractionState.MOVING:
                moving.append(event)
            elif session.state == InteractionState.RESIZING:
                resizing.append(event)

        def apply_batch(batch: list[PointerMove]) -> None:
            for event in batch:
                result = self._compute_move(
                    event.position, event.pointer_id, canvas
                )
                if result is not None:
                    canvas.update(result[0])

        apply_batch(stretching)
        apply_batch(moving)
        apply_batch(resizing)

    def _on_pointer_up(self, pointer_id: str, canvas: Canvas) -> None:
        session = self._sessions.get(pointer_id)
        # Fist owns MOVING; pinch-up must not release a closed-hand grab
        if session is not None and session.state == InteractionState.MOVING:
            return
        if session is not None:
            self._restore_pulled(session, canvas)
        self._drop_unfinished(session, canvas)
        self._clear_session(pointer_id)

    def _on_grab_up(self, pointer_id: str, canvas: Canvas) -> None:
        """Opening the hand drops the figure — into the bin if it is in one."""
        session = self._sessions.get(pointer_id)
        if session is None:
            self._clear_session(pointer_id)
            return

        # Read the bin before restoring: the restore pulls the figure back out
        # of the hand, which would move it off the zone it was dropped into.
        binned = False
        was_moving = session.state == InteractionState.MOVING
        if was_moving and session.selected_id is not None:
            shape = canvas.get(session.selected_id)
            binned = shape is not None and self._in_trash(session, shape)

        # Either way the figure gets its real size back: deleted whole so undo
        # returns the figure, or dropped whole so the canvas keeps the figure.
        self._restore_pulled(session, canvas)

        if binned and session.selected_id is not None:
            self._bin(session.selected_id, canvas)
            return

        if was_moving:
            self._throw(session, canvas)

        self._drop_unfinished(session, canvas)
        self._clear_session(pointer_id)

    def _empty_trash_zone(self, canvas: Canvas) -> None:
        """The bin eats whatever is inside it, held or not.

        Opening a hand is not always read cleanly, and a drop that misfires
        used to leave the figure sitting in the bin untouched. So the release
        is not the only way in: being inside the zone is enough, which also
        means a figure riding on a hand that reaches in goes with it.
        """
        drawing = {
            session.selected_id
            for session in self._sessions.values()
            if session.state == InteractionState.STRETCHING
        }
        # A figure passing through in mid-air is not being thrown away, it is
        # just passing through. It gets eaten only if it comes to rest inside.
        spared = drawing | self._momentum.flying_ids
        zone = trash_zone()
        doomed = [
            shape.id
            for shape in canvas.shapes
            if isinstance(shape, Rectangle)
            and shape.id not in spared
            and point_in_rectangle(_shape_center(shape), zone, padding=0.0)
        ]
        for shape_id in doomed:
            self._bin(shape_id, canvas)

    def _bin(self, shape_id: str, canvas: Canvas) -> None:
        """Delete a figure as the bin should: whole, and undoable to beside it."""
        self._momentum.cancel(shape_id)
        for session in self._sessions.values():
            if session.selected_id == shape_id:
                self._restore_pulled(session, canvas)
        shape = canvas.get(shape_id)
        if isinstance(shape, Rectangle):
            canvas.update(_evicted(shape))
        canvas.discard([shape_id])
        self._release_sessions_holding(shape_id)

    def _release_sessions_holding(self, shape_id: str) -> None:
        for pointer_id, session in list(self._sessions.items()):
            if session.selected_id == shape_id:
                self._clear_session(pointer_id)

    def _drop_unfinished(self, session: PointerSession | None, canvas: Canvas) -> None:
        """Release with nothing worth keeping: remove the preview, don't commit.

        A pinch that never got dragged leaves a bare point, and a drag along a
        single axis leaves a sliver. Both read as stray specks on the canvas.
        """
        if session is None or session.selected_id is None:
            return
        shape = canvas.get(session.selected_id)
        if shape is None or is_visible_figure(shape):
            return
        if self._other_session_holds(session.selected_id, session):
            return
        canvas.remove(shape.id)

    def _other_session_holds(self, shape_id: str, owner: PointerSession) -> bool:
        return any(
            session is not owner and session.selected_id == shape_id
            for session in self._sessions.values()
        )

    def _in_trash(self, session: PointerSession, shape: Shape) -> bool:
        """The figure is sitting in the bin, or the hand holding it is.

        Testing the hand matters: a figure can be far wider than the zone, so
        its center may never make it in no matter where you drag from.
        """
        zone = trash_zone()
        if point_in_rectangle(_shape_center(shape), zone, padding=0.0):
            return True
        hand = session.last_pointer
        return hand is not None and point_in_rectangle(hand, zone, padding=0.0)

    def pending_delete_ids(self, canvas: Canvas) -> set[str]:
        """Held figures on their way into the bin, for the warning highlight.

        Reaching the zone deletes on the spot, so arming has to happen during
        the approach — by the time a figure is inside, it is already gone.
        """
        armed: set[str] = set()
        for session in self._sessions.values():
            if session.state != InteractionState.MOVING or session.selected_id is None:
                continue
            shape = canvas.get(session.selected_id)
            if shape is None:
                continue
            if session.full_size is not None or self._in_trash(session, shape):
                armed.add(shape.id)
        return armed

    def _clear_canvas(self, canvas: Canvas) -> None:
        """Wipe every figure as one undoable batch and drop all sessions."""
        self._momentum.clear()
        for session in self._sessions.values():
            self._restore_pulled(session, canvas)
        ids = [shape.id for shape in canvas.shapes]
        if not ids:
            return
        canvas.discard(ids)
        for pointer_id in list(self._sessions):
            self._clear_session(pointer_id)

    def _clear_session(self, pointer_id: str) -> None:
        self._sessions[pointer_id] = PointerSession()
