"""Hand tracking via MediaPipe Hand Landmarker.

Converts MediaPipe landmarks into app-owned types with normalized coordinates.
Does not know about geometry shapes or interaction.
"""

from __future__ import annotations

import math
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
    SMOOTH_BETA,
    SMOOTH_DERIVATIVE_CUTOFF,
    SMOOTH_MIN_CUTOFF,
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


# Frame spacing bounds for the filter. A stall — window drag, the first frame
# after a hiccup — must not be read as a real time step, or the filter takes
# the resulting jump for speed and stops smoothing altogether.
NOMINAL_DT = 1.0 / 30.0
MIN_DT = 1.0 / 240.0
MAX_DT = 0.2


def _cutoff_alpha(cutoff: float, dt: float) -> float:
    """EMA weight for a first-order low pass at ``cutoff`` Hz, sampled at dt."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class Smoother:
    """One-euro filter for normalized points: smoothing that yields to speed.

    The cutoff is not fixed — it opens up in proportion to how fast the point
    is moving. Standing still, the hand is filtered hard and reads steady;
    thrown across the frame, it is barely filtered at all and the lag that
    used to leave a dragged figure trailing behind the palm goes away.
    """

    def __init__(
        self,
        min_cutoff: float = SMOOTH_MIN_CUTOFF,
        beta: float = SMOOTH_BETA,
        derivative_cutoff: float = SMOOTH_DERIVATIVE_CUTOFF,
    ) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._derivative_cutoff = derivative_cutoff
        self._previous: list[Point] | None = None
        self._speeds: list[Point] | None = None

    def reset(self) -> None:
        self._previous = None
        self._speeds = None

    def smooth(self, points: list[Point], dt: float = NOMINAL_DT) -> list[Point]:
        if (
            self._previous is None
            or self._speeds is None
            or len(self._previous) != len(points)
        ):
            self._previous = list(points)
            self._speeds = [Point(0.0, 0.0) for _ in points]
            return list(points)

        speed_alpha = _cutoff_alpha(self._derivative_cutoff, dt)
        smoothed: list[Point] = []
        speeds: list[Point] = []
        for current, prev, prev_speed in zip(points, self._previous, self._speeds):
            speed = Point(
                x=speed_alpha * (current.x - prev.x) / dt
                + (1.0 - speed_alpha) * prev_speed.x,
                y=speed_alpha * (current.y - prev.y) / dt
                + (1.0 - speed_alpha) * prev_speed.y,
            )
            alpha = _cutoff_alpha(
                self._min_cutoff + self._beta * math.hypot(speed.x, speed.y), dt
            )
            smoothed.append(
                Point(
                    x=alpha * current.x + (1.0 - alpha) * prev.x,
                    y=alpha * current.y + (1.0 - alpha) * prev.y,
                )
            )
            speeds.append(speed)

        self._previous = smoothed
        self._speeds = speeds
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
        num_hands: int = 2,
        use_gpu: bool = USE_GPU_INFERENCE,
    ) -> None:
        path = ensure_model(model_path or DEFAULT_MODEL_PATH)
        self._smoothers: dict[str, Smoother] = {}
        self._start_ns = time.perf_counter_ns()
        self._last_timestamp_ms = -1
        self._last_frame_s: float | None = None

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

    def _frame_dt(self) -> float:
        """Seconds since the last processed frame, clamped to a sane range."""
        now = time.perf_counter()
        previous = self._last_frame_s
        self._last_frame_s = now
        if previous is None:
            return NOMINAL_DT
        return min(max(now - previous, MIN_DT), MAX_DT)

    def process(self, frame: Frame) -> list[Hand]:
        dt = self._frame_dt()
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
            smoother = self._smoothers.setdefault(label, Smoother())
            smoothed = smoother.smooth(points, dt)
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
