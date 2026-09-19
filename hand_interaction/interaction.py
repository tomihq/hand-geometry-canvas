"""Project 3D HandPose streams into abstract 2D HandEvents.

Consumers get Vector2 moves and float rotations — never Vector3/Quaternion
on the event channel. Pinch uses landmarks; pose drives move/rotate.

Bimanual resize follows the hand_canvas lock model (no figures here):
  - First hand to pinch is the owner (grab / move via pinch + hand.move).
  - Second hand to pinch while the owner still holds becomes the helper.
  - Helper cursor motion emits resize.* for the consumer to corner-follow.
  - Helper cannot steal ownership; session ends when either pinch releases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_interaction.constants import (
    HAND_LOST_GRACE_FRAMES,
    MOVE_EPSILON,
    RESIZE_EPSILON,
    ROTATE_EPSILON,
)
from hand_interaction.math3d import (
    delta_rotation_local,
    project_delta_to_screen_angle,
    vec2_norm,
    vec2_sub,
)
from hand_interaction.pinch import PinchDetector, PinchPhase
from hand_interaction.types import (
    HandEvent,
    HandMove,
    HandPose,
    HandRotate,
    PinchEnd,
    PinchStart,
    ResizeEnd,
    ResizeMove,
    ResizeStart,
    Vector2,
    Vector3,
)


@dataclass(frozen=True)
class TrackedHand:
    """One observed hand this frame: smoothed pose + raw landmarks for pinch."""

    pose: HandPose
    landmarks: np.ndarray  # (21, 3)

    def __post_init__(self) -> None:
        pts = np.asarray(self.landmarks, dtype=np.float64)
        if pts.shape != (21, 3):
            raise ValueError(f"expected landmarks (21, 3), got {pts.shape}")
        object.__setattr__(self, "landmarks", pts)


def _palm_2d(palm: Vector3) -> Vector2:
    return Vector2(palm.x, palm.y)


@dataclass
class _HandState:
    previous_pose: HandPose | None = None
    pinch: PinchDetector | None = None
    lost_frames: int = 0


@dataclass
class _ResizeSession:
    """Owner holds; helper cursor drives resize (canvas-style latch)."""

    owner_hand_id: str
    helper_hand_id: str
    last_helper_position: Vector2


class InteractionEngine:
    """Multi-hand pose → abstract HandEvent list (move / rotate / pinch / resize)."""

    def __init__(
        self,
        move_epsilon: float = MOVE_EPSILON,
        rotate_epsilon: float = ROTATE_EPSILON,
        resize_epsilon: float = RESIZE_EPSILON,
        grace_frames: int = HAND_LOST_GRACE_FRAMES,
    ) -> None:
        self._move_eps = move_epsilon
        self._rotate_eps = rotate_epsilon
        self._resize_eps = resize_epsilon
        self._grace = grace_frames
        self._hands: dict[str, _HandState] = {}
        self._owner_hand_id: str | None = None
        self._resize: _ResizeSession | None = None

    def reset(self) -> None:
        self._hands.clear()
        self._owner_hand_id = None
        self._resize = None

    def _state(self, hand_id: str) -> _HandState:
        if hand_id not in self._hands:
            self._hands[hand_id] = _HandState(pinch=PinchDetector())
        return self._hands[hand_id]

    def update(self, hands: list[TrackedHand]) -> list[HandEvent]:
        """Process one frame. Absent hands count toward grace, then pinch.end."""
        events: list[HandEvent] = []
        seen: set[str] = set()

        for tracked in hands:
            hand_id = tracked.pose.hand_id
            seen.add(hand_id)
            state = self._state(hand_id)
            state.lost_frames = 0
            events.extend(self._update_present(state, tracked))

        for hand_id, state in list(self._hands.items()):
            if hand_id in seen:
                continue
            events.extend(self._update_missing(hand_id, state))

        events.extend(self._update_owner_and_resize())
        return events

    def _update_present(
        self, state: _HandState, tracked: TrackedHand
    ) -> list[HandEvent]:
        events: list[HandEvent] = []
        pose = tracked.pose
        assert state.pinch is not None

        pinch_events = state.pinch.update(tracked.landmarks, pose.hand_id)
        for event in pinch_events:
            if isinstance(event, PinchStart) and self._owner_hand_id is None:
                # First pinch wins the lock (hand_canvas owner).
                self._owner_hand_id = event.hand_id
            elif isinstance(event, PinchEnd) and event.hand_id == self._owner_hand_id:
                self._owner_hand_id = None
        events.extend(pinch_events)

        prev = state.previous_pose
        if prev is not None:
            events.extend(self._move_events(pose, prev))
            events.extend(self._rotate_events(pose, prev))

        state.previous_pose = pose
        return events

    def _move_events(self, pose: HandPose, prev: HandPose) -> list[HandEvent]:
        position = _palm_2d(pose.palm)
        delta = vec2_sub(position, _palm_2d(prev.palm))
        if vec2_norm(delta) <= self._move_eps:
            return []
        return [
            HandMove(
                hand_id=pose.hand_id,
                position=position,
                delta_position=delta,
            )
        ]

    def _rotate_events(self, pose: HandPose, prev: HandPose) -> list[HandEvent]:
        delta_q = delta_rotation_local(prev.orientation, pose.orientation)
        angle = project_delta_to_screen_angle(delta_q)
        if abs(angle) <= self._rotate_eps:
            return []
        return [HandRotate(hand_id=pose.hand_id, delta_rotation=angle)]

    def _update_missing(self, hand_id: str, state: _HandState) -> list[HandEvent]:
        """Grace: hold (no move/rotate). After grace: pinch.end and clear."""
        assert state.pinch is not None
        if state.pinch.phase is PinchPhase.PINCHING and state.lost_frames < self._grace:
            state.lost_frames += 1
            return []

        events = state.pinch.force_end(hand_id)
        if any(isinstance(e, PinchEnd) and e.hand_id == self._owner_hand_id for e in events):
            self._owner_hand_id = None
        del self._hands[hand_id]
        return events

    def _pinching_ids(self) -> list[str]:
        ids: list[str] = []
        for hand_id, state in self._hands.items():
            assert state.pinch is not None
            if state.pinch.phase is PinchPhase.PINCHING:
                ids.append(hand_id)
        return ids

    def _pinch_position(self, hand_id: str) -> Vector2 | None:
        state = self._hands.get(hand_id)
        if state is None or state.pinch is None:
            return None
        if state.pinch.phase is not PinchPhase.PINCHING:
            return None
        return state.pinch.last_position

    def _end_resize(self) -> list[HandEvent]:
        if self._resize is None:
            return []
        session = self._resize
        self._resize = None
        return [
            ResizeEnd(
                owner_hand_id=session.owner_hand_id,
                helper_hand_id=session.helper_hand_id,
            )
        ]

    def _update_owner_and_resize(self) -> list[HandEvent]:
        """Second pinch while owner holds → helper resize (canvas lock model)."""
        events: list[HandEvent] = []
        pinching = self._pinching_ids()
        pinching_set = set(pinching)

        if self._owner_hand_id is not None and self._owner_hand_id not in pinching_set:
            events.extend(self._end_resize())
            self._owner_hand_id = None

        # Remaining pinch after owner release keeps the lock (still holding).
        if self._owner_hand_id is None and pinching:
            self._owner_hand_id = pinching[0]

        owner = self._owner_hand_id
        if owner is None:
            events.extend(self._end_resize())
            return events

        helpers = [hid for hid in pinching if hid != owner]
        if not helpers:
            events.extend(self._end_resize())
            return events

        # One helper at a time; cannot steal the lock.
        helper = helpers[0]
        if self._resize is not None and self._resize.helper_hand_id not in pinching_set:
            events.extend(self._end_resize())

        helper_pos = self._pinch_position(helper)
        if helper_pos is None:
            events.extend(self._end_resize())
            return events

        if self._resize is None or self._resize.helper_hand_id != helper:
            if self._resize is not None:
                events.extend(self._end_resize())
            self._resize = _ResizeSession(
                owner_hand_id=owner,
                helper_hand_id=helper,
                last_helper_position=helper_pos,
            )
            events.append(
                ResizeStart(
                    owner_hand_id=owner,
                    helper_hand_id=helper,
                    position=helper_pos,
                )
            )
            return events

        delta = vec2_sub(helper_pos, self._resize.last_helper_position)
        if vec2_norm(delta) > self._resize_eps:
            events.append(
                ResizeMove(
                    owner_hand_id=owner,
                    helper_hand_id=helper,
                    position=helper_pos,
                    delta_position=delta,
                )
            )
            self._resize.last_helper_position = helper_pos
        return events
