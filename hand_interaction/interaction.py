"""Project HandPose + landmarks into abstract 2D HandEvents.

Consumers get Vector2 pinch/grab/move — never Vector3/Quaternion on the
event channel. Gestures match the canvas model (pinch + fist); HandMove is
always-on palm tracking for other apps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hand_interaction.constants import MOVE_EPSILON
from hand_interaction.gestures import GestureHand, MultiHandGestureDetector
from hand_interaction.math3d import vec2_norm, vec2_sub
from hand_interaction.types import HandEvent, HandMove, HandPose, Vector2


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


class InteractionEngine:
    """Multi-hand pose + landmarks → HandEvent list (move / pinch / grab)."""

    def __init__(self, move_epsilon: float = MOVE_EPSILON) -> None:
        self._move_eps = move_epsilon
        self._gestures = MultiHandGestureDetector()
        self._previous_pose: dict[str, HandPose] = {}

    def reset(self) -> None:
        self._gestures.reset()
        self._previous_pose.clear()

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
