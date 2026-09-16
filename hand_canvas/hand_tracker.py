"""Hand tracking via MediaPipe Hand Landmarker.

Converts MediaPipe landmarks into app-owned types with normalized coordinates.
Does not know about geometry shapes or interaction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hand_canvas.camera import Frame
from hand_canvas.constants import (
    SMOOTH_ALPHA,
    TRACKING_MAX_WIDTH,
    USE_GPU_INFERENCE,
)
from hand_canvas.geometry import Point

# MediaPipe hand landmark indices
INDEX_TIP = 8
INDEX_PIP = 6
THUMB_TIP = 4
WRIST = 0
MIDDLE_MCP = 9

# Finger tip / PIP pairs for fist detection (index, middle, ring, pinky)
FINGER_PAIRS = ((8, 6), (12, 10), (16, 14), (20, 18))

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class Smoother:
    """Exponential moving average for normalized points."""

    def __init__(self, alpha: float = SMOOTH_ALPHA) -> None:
        self.alpha = alpha
        self._previous: list[Point] | None = None

    def reset(self) -> None:
        self._previous = None

    def smooth(self, points: list[Point]) -> list[Point]:
        if self._previous is None or len(self._previous) != len(points):
            self._previous = list(points)
            return list(points)

        smoothed: list[Point] = []
        for current, prev in zip(points, self._previous):
            smoothed.append(
                Point(
                    x=self.alpha * current.x + (1.0 - self.alpha) * prev.x,
                    y=self.alpha * current.y + (1.0 - self.alpha) * prev.y,
                )
            )
        self._previous = smoothed
        return smoothed


@dataclass
class Hand:
    index_tip: Point
    thumb_tip: Point
    landmarks: list[Point]
    handedness: str = "Unknown"


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Download the Hand Landmarker model if missing."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        print(f"Downloading hand landmarker model to {model_path} ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to download model. Place hand_landmarker.task at {model_path}"
        ) from exc
    return model_path


def _handedness_label(result: mp_vision.HandLandmarkerResult, index: int) -> str:
    if not result.handedness or index >= len(result.handedness):
        return f"Hand{index}"
    categories = result.handedness[index]
    if not categories:
        return f"Hand{index}"
    name = categories[0].category_name or f"Hand{index}"
    return name


class HandTracker:
    def __init__(
        self,
        model_path: Path | None = None,
        smooth_alpha: float = SMOOTH_ALPHA,
        num_hands: int = 2,
        use_gpu: bool = USE_GPU_INFERENCE,
    ) -> None:
        path = ensure_model(model_path or DEFAULT_MODEL_PATH)
        self._smoothers: dict[str, Smoother] = {}
        self._smooth_alpha = smooth_alpha
        self._start_ns = time.perf_counter_ns()
        self._last_timestamp_ms = -1

        delegates = [mp_python.BaseOptions.Delegate.CPU]
        if use_gpu:
            delegates.insert(0, mp_python.BaseOptions.Delegate.GPU)

        self._landmarker = None
        self.delegate = "none"
        for delegate in delegates:
            try:
                self._landmarker = self._build(path, num_hands, delegate)
            except Exception as exc:  # noqa: BLE001
                print(f"Hand tracking on {delegate.name} unavailable: {exc}")
                continue
            self.delegate = delegate.name
            print(f"Hand tracking on {self.delegate}")
            break

        if self._landmarker is None:
            raise RuntimeError("Could not initialize the hand landmarker")

    def _build(self, path: Path, num_hands: int, delegate) -> mp_vision.HandLandmarker:
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(path), delegate=delegate
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
        try:
            # Run once here so a broken delegate raises now rather than mid-loop,
            # and so the first real frame does not pay for lazy initialization.
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=blank),
                self._next_timestamp_ms(),
            )
        except Exception:
            landmarker.close()
            raise
        return landmarker

    def _next_timestamp_ms(self) -> int:
        """VIDEO mode needs real, strictly increasing timestamps to track well."""
        elapsed = (time.perf_counter_ns() - self._start_ns) // 1_000_000
        timestamp = max(elapsed, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        return timestamp

    def process(self, frame: Frame) -> list[Hand]:
        image = frame.image
        if TRACKING_MAX_WIDTH and image.shape[1] > TRACKING_MAX_WIDTH:
            scale = TRACKING_MAX_WIDTH / image.shape[1]
            image = cv2.resize(
                image,
                (TRACKING_MAX_WIDTH, max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        # cvtColor is ~100x faster than numpy's [:, :, ::-1] + ascontiguousarray.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        if not result.hand_landmarks:
            for smoother in self._smoothers.values():
                smoother.reset()
            return []

        hands: list[Hand] = []
        seen: set[str] = set()
        for i, raw in enumerate(result.hand_landmarks):
            label = _handedness_label(result, i)
            # Avoid key collision if MediaPipe reports two Unknowns
            if label in seen:
                label = f"{label}-{i}"
            seen.add(label)

            points = [Point(x=lm.x, y=lm.y) for lm in raw]
            smoother = self._smoothers.setdefault(
                label, Smoother(alpha=self._smooth_alpha)
            )
            smoothed = smoother.smooth(points)
            hands.append(
                Hand(
                    index_tip=smoothed[INDEX_TIP],
                    thumb_tip=smoothed[THUMB_TIP],
                    landmarks=smoothed,
                    handedness=label,
                )
            )

        for label in list(self._smoothers):
            if label not in seen:
                self._smoothers[label].reset()

        return hands

    def close(self) -> None:
        self._landmarker.close()
