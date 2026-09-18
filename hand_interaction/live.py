"""Live MediaPipe-backed HandEventSource (camera → events / poses)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from hand_interaction.camera import Camera
from hand_interaction.fake import EventHandler, PoseHandler, Unsubscribe
from hand_interaction.interaction import InteractionEngine, TrackedHand
from hand_interaction.landmarker import MediaPipeLandmarker
from hand_interaction.pose import PoseEstimator
from hand_interaction.types import HandEvent, HandPose


class LiveHandSource:
    """Runs a capture+detect loop on a background thread and fans out events."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        num_hands: int = 2,
        model_path: Path | None = None,
        mirror: bool = True,
        camera: Camera | None = None,
        landmarker: MediaPipeLandmarker | None = None,
    ) -> None:
        self._camera = camera or Camera(device_index=device_index, mirror=mirror)
        self._owns_camera = camera is None
        self._landmarker = landmarker
        self._model_path = model_path
        self._num_hands = num_hands
        self._owns_landmarker = landmarker is None

        self._pose_estimator = PoseEstimator()
        self._engine = InteractionEngine()
        self._event_handlers: list[EventHandler] = []
        self._pose_handlers: list[PoseHandler] = []

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._error: BaseException | None = None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def landmarker(self) -> MediaPipeLandmarker | None:
        return self._landmarker

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

    def start(self) -> None:
        if self._started:
            return
        if self._landmarker is None:
            self._landmarker = MediaPipeLandmarker(
                model_path=self._model_path,
                num_hands=self._num_hands,
            )
            self._owns_landmarker = True
        if self._owns_camera:
            self._camera.open()
        self._stop.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._loop, name="live-hand-source", daemon=True
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._owns_camera:
            self._camera.release()
        if self._owns_landmarker and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._pose_estimator.reset()
        self._engine.reset()
        self._started = False

    def poll_error(self) -> BaseException | None:
        return self._error

    def _loop(self) -> None:
        assert self._landmarker is not None
        try:
            while not self._stop.is_set():
                frame = self._camera.read(timeout=2.0)
                detected = self._landmarker.detect(frame)
                tracked: list[TrackedHand] = []
                for hand in detected:
                    pose = self._pose_estimator.estimate(hand)
                    self._emit_pose(pose)
                    tracked.append(TrackedHand(pose=pose, landmarks=hand.landmarks))
                for event in self._engine.update(tracked):
                    self._emit_event(event)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def _emit_event(self, event: HandEvent) -> None:
        for handler in list(self._event_handlers):
            handler(event)

    def _emit_pose(self, pose: HandPose) -> None:
        for handler in list(self._pose_handlers):
            handler(pose)


def process_landmark_hands(
    hands: list,
    *,
    pose_estimator: PoseEstimator | None = None,
    engine: InteractionEngine | None = None,
    on_pose: Callable[[HandPose], None] | None = None,
) -> list[HandEvent]:
    """Pure helper: LandmarkHand list → events (for tests / offline)."""
    poses = pose_estimator or PoseEstimator()
    interaction = engine or InteractionEngine()
    tracked: list[TrackedHand] = []
    for hand in hands:
        pose = poses.estimate(hand)
        if on_pose is not None:
            on_pose(pose)
        tracked.append(TrackedHand(pose=pose, landmarks=hand.landmarks))
    return interaction.update(tracked)
