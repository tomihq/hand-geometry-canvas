"""Public HandGesture API — framework-agnostic event subscription."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hand_interaction.fake import (
    EventHandler,
    FakeHandSource,
    PoseHandler,
    Unsubscribe,
)
from hand_interaction.types import HandEvent, HandPose


@runtime_checkable
class HandEventSource(Protocol):
    """Backend that can start/stop and fan out events and poses."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def on_event(self, handler: EventHandler) -> Unsubscribe: ...

    def on_pose(self, handler: PoseHandler) -> Unsubscribe: ...


class HandGesture:
    """Camera-agnostic session: subscribe to abstract events and/or 3D poses."""

    def __init__(self, source: HandEventSource) -> None:
        self._source = source

    @property
    def source(self) -> HandEventSource:
        return self._source

    def start(self) -> None:
        self._source.start()

    def stop(self) -> None:
        self._source.stop()

    def on_event(self, handler: EventHandler) -> Unsubscribe:
        """Subscribe to abstract HandEvent stream. Returns unsubscribe()."""
        return self._source.on_event(handler)

    def on_pose(self, handler: PoseHandler) -> Unsubscribe:
        """Subscribe to HandPose 3D stream (opt-in). Returns unsubscribe()."""
        return self._source.on_pose(handler)


def create_hand_gesture(
    *,
    source: HandEventSource | None = None,
    live: bool = False,
    device_index: int = 0,
    num_hands: int = 2,
) -> HandGesture:
    """Create a hand-gesture session.

    - Default: :class:`FakeHandSource` (no camera) — safe for tests.
    - ``live=True``: :class:`LiveHandSource` (camera + MediaPipe).
    - Or pass an explicit ``source``.
    """
    if source is not None:
        return HandGesture(source)
    if live:
        from hand_interaction.live import LiveHandSource

        return HandGesture(
            LiveHandSource(device_index=device_index, num_hands=num_hands)
        )
    return HandGesture(FakeHandSource())


__all__ = [
    "HandEventSource",
    "HandGesture",
    "create_hand_gesture",
    "EventHandler",
    "PoseHandler",
    "Unsubscribe",
    "HandEvent",
    "HandPose",
]
