"""Project HandPose + landmarks into abstract 2D HandEvents.

Consumers get Vector2 pinch/grab/move/resize — never Vector3/Quaternion on
the event channel. Gestures match the canvas model (pinch + fist); HandMove
is always-on palm tracking for other apps.

Bimanual resize follows the hand_canvas lock model (no figures here):
  - First hand to pinch or fist is the owner.
  - Second hand to pinch/fist while the owner still holds becomes the helper.
  - Helper cursor motion emits resize.* for the consumer to corner-follow.
  - Helper cannot steal ownership; session ends when either hand releases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_interaction.constants import MOVE_EPSILON, RESIZE_EPSILON
from hand_interaction.gestures import GestureHand, GestureState, MultiHandGestureDetector
from hand_interaction.math3d import vec2_norm, vec2_sub
from hand_interaction.types import (
    HandEvent,
    HandMove,
    HandPose,
    ResizeEnd,
    ResizeMove,
    ResizeStart,
    Vector2,
)

_HOLDING = frozenset({GestureState.PINCHING, GestureState.FIST})


@dataclass(frozen=True)
class TrackedHand:
    """One observed hand this frame: smoothed pose + raw landmarks for gestures."""

    pose: HandPose
    landmarks: np.ndarray  # (21, 3)

    def __post_init__(self) -> None:
        pts = np.asarray(self.landmarks, dtype=np.float64)
        if pts.shape != (21, 3):
            raise ValueError(f"expected landmarks (21, 3), got {pts.shape}")
        object.__setattr__(self, "landmarks", pts)


def _palm_image(x: float, y_hybrid: float) -> Vector2:
    """HandPose.palm.y is bottom-origin; public HandEvent uses top-origin."""
    return Vector2(x, 1.0 - y_hybrid)


@dataclass
class _ResizeSession:
    """Owner holds; helper cursor drives resize (canvas-style latch)."""

    owner_hand_id: str
    helper_hand_id: str
    last_helper_position: Vector2


class InteractionEngine:
    """Multi-hand pose + landmarks → HandEvent list (move / pinch / grab / resize)."""

    def __init__(
        self,
        move_epsilon: float = MOVE_EPSILON,
        resize_epsilon: float = RESIZE_EPSILON,
    ) -> None:
        self._move_eps = move_epsilon
        self._resize_eps = resize_epsilon
        self._gestures = MultiHandGestureDetector()
        self._previous_pose: dict[str, HandPose] = {}
        self._owner_hand_id: str | None = None
        self._resize: _ResizeSession | None = None

    def reset(self) -> None:
        self._gestures.reset()
        self._previous_pose.clear()
        self._owner_hand_id = None
        self._resize = None

    @property
    def states(self):
        return self._gestures.states

    @property
    def pinch_dists(self) -> dict[str, float]:
        return self._gestures.pinch_dists

    @property
    def pinch_thresholds(self) -> dict[str, tuple[float, float]]:
        return self._gestures.pinch_thresholds

    @property
    def fist_scores(self) -> dict[str, int]:
        return self._gestures.fist_scores

    def update(self, hands: list[TrackedHand]) -> list[HandEvent]:
        """Process one frame. Absent hands are released after gesture grace."""
        events: list[HandEvent] = []
        gesture_hands = [
            GestureHand(hand_id=h.pose.hand_id, landmarks=h.landmarks) for h in hands
        ]
        events.extend(self._gestures.update(gesture_hands))
        events.extend(self._update_owner_and_resize())

        seen: set[str] = set()
        for tracked in hands:
            hand_id = tracked.pose.hand_id
            seen.add(hand_id)
            prev = self._previous_pose.get(hand_id)
            if prev is not None:
                events.extend(self._move_events(tracked.pose, prev))
            self._previous_pose[hand_id] = tracked.pose

        for hand_id in list(self._previous_pose):
            if hand_id not in seen:
                del self._previous_pose[hand_id]

        return events

    def _move_events(self, pose: HandPose, prev: HandPose) -> list[HandEvent]:
        position = _palm_image(pose.palm.x, pose.palm.y)
        prev_pos = _palm_image(prev.palm.x, prev.palm.y)
        delta = vec2_sub(position, prev_pos)
        if vec2_norm(delta) <= self._move_eps:
            return []
        return [
            HandMove(
                hand_id=pose.hand_id,
                position=position,
                delta_position=delta,
            )
        ]

    def _holding_ids(self) -> list[str]:
        return [
            hid
            for hid, state in self._gestures.states.items()
            if state in _HOLDING
        ]

    def _cursor(self, hand_id: str) -> Vector2 | None:
        return self._gestures.cursors.get(hand_id)

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
        """Second hold while owner holds → helper resize (canvas lock model)."""
        events: list[HandEvent] = []
        holding = self._holding_ids()
        holding_set = set(holding)

        if self._owner_hand_id is not None and self._owner_hand_id not in holding_set:
            events.extend(self._end_resize())
            self._owner_hand_id = None

        # Remaining hold after owner release keeps the lock.
        if self._owner_hand_id is None and holding:
            self._owner_hand_id = holding[0]

        owner = self._owner_hand_id
        if owner is None:
            events.extend(self._end_resize())
            return events

        helpers = [hid for hid in holding if hid != owner]
        if not helpers:
            events.extend(self._end_resize())
            return events

        # One helper at a time; cannot steal the lock.
        helper = helpers[0]
        if self._resize is not None and self._resize.helper_hand_id not in holding_set:
            events.extend(self._end_resize())

        helper_pos = self._cursor(helper)
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
