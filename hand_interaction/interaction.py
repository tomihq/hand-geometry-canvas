"""Project 3D HandPose streams into abstract 2D HandEvents.

Consumers get Vector2 moves and float rotations — never Vector3/Quaternion
on the event channel. Pinch uses landmarks; pose drives move/rotate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_interaction.constants import (
    HAND_LOST_GRACE_FRAMES,
    MOVE_EPSILON,
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
    Quaternion,
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


class InteractionEngine:
    """Multi-hand pose → abstract HandEvent list (move / rotate / pinch)."""

    def __init__(
        self,
        move_epsilon: float = MOVE_EPSILON,
        rotate_epsilon: float = ROTATE_EPSILON,
        grace_frames: int = HAND_LOST_GRACE_FRAMES,
    ) -> None:
        self._move_eps = move_epsilon
        self._rotate_eps = rotate_epsilon
        self._grace = grace_frames
        self._hands: dict[str, _HandState] = {}

    def reset(self) -> None:
        self._hands.clear()

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

        return events

    def _update_present(
        self, state: _HandState, tracked: TrackedHand
    ) -> list[HandEvent]:
        events: list[HandEvent] = []
        pose = tracked.pose
        assert state.pinch is not None

        events.extend(state.pinch.update(tracked.landmarks, pose.hand_id))

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
            # Hold: do not clear previous_pose; deltas stay zero next reappear.
            return []

        events = state.pinch.force_end(hand_id)
        del self._hands[hand_id]
        return events
