"""Fake hand event source for tests and headless demos (no camera)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from hand_interaction.types import HandEvent, HandPose

EventHandler: TypeAlias = Callable[[HandEvent], None]
PoseHandler: TypeAlias = Callable[[HandPose], None]
Unsubscribe: TypeAlias = Callable[[], None]


class FakeHandSource:
    """Injectable event/pose bus with the same subscribe shape as the real API."""

    def __init__(self) -> None:
        self._event_handlers: list[EventHandler] = []
        self._pose_handlers: list[PoseHandler] = []
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def on_event(self, handler: EventHandler) -> Unsubscribe:
        self._event_handlers.append(handler)

        def unsubscribe() -> None:
            try:
                self._event_handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def on_pose(self, handler: PoseHandler) -> Unsubscribe:
        self._pose_handlers.append(handler)

        def unsubscribe() -> None:
            try:
                self._pose_handlers.remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def emit(self, event: HandEvent) -> None:
        """Push one interaction event to all current subscribers."""
        for handler in list(self._event_handlers):
            handler(event)

    def emit_pose(self, pose: HandPose) -> None:
        """Push one 3D pose to all current pose subscribers."""
        for handler in list(self._pose_handlers):
            handler(pose)

    def emit_sequence(self, events: list[HandEvent]) -> None:
        for event in events:
            self.emit(event)
