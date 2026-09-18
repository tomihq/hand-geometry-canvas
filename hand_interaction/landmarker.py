"""MediaPipe Hand Landmarker → LandmarkHand (21×3), no app knowledge."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from hand_interaction.camera import Frame
from hand_interaction.constants import (
    MAX_DT,
    MIN_DT,
    NOMINAL_DT,
    TRACKING_MAX_WIDTH,
    USE_GPU_INFERENCE,
)
from hand_interaction.pose import LandmarkHand
from hand_interaction.smooth import OneEuroVector3
from hand_interaction.types import Vector3

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
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
    return categories[0].category_name or f"Hand{index}"


def _confidence(result: mp_vision.HandLandmarkerResult, index: int) -> float:
    if not result.handedness or index >= len(result.handedness):
        return 0.0
    categories = result.handedness[index]
    if not categories:
        return 0.0
    score = categories[0].score
    return float(score) if score is not None else 0.0


class _LandmarkSmoother:
    """One-euro per landmark (x,y,z) keyed by hand_id."""

    def __init__(self) -> None:
        self._filters: list[OneEuroVector3] | None = None

    def reset(self) -> None:
        self._filters = None

    def smooth(self, points: np.ndarray, dt: float) -> np.ndarray:
        if self._filters is None or len(self._filters) != len(points):
            self._filters = [OneEuroVector3() for _ in range(len(points))]
            return points.copy()
        out = np.empty_like(points)
        for i, (pt, filt) in enumerate(zip(points, self._filters)):
            s = filt.smooth(Vector3(float(pt[0]), float(pt[1]), float(pt[2])), dt)
            out[i] = (s.x, s.y, s.z)
        return out


class MediaPipeLandmarker:
    """Detect hands in a Frame; returns LandmarkHand with xyz landmarks."""

    def __init__(
        self,
        model_path: Path | None = None,
        num_hands: int = 2,
        use_gpu: bool = USE_GPU_INFERENCE,
        tracking_max_width: int = TRACKING_MAX_WIDTH,
    ) -> None:
        path = ensure_model(model_path or DEFAULT_MODEL_PATH)
        self._tracking_max_width = tracking_max_width
        self._smoothers: dict[str, _LandmarkSmoother] = {}
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
        elapsed = (time.perf_counter_ns() - self._start_ns) // 1_000_000
        timestamp = max(elapsed, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        return timestamp

    def _frame_dt(self) -> float:
        now = time.perf_counter()
        previous = self._last_frame_s
        self._last_frame_s = now
        if previous is None:
            return NOMINAL_DT
        return min(max(now - previous, MIN_DT), MAX_DT)

    def detect(self, frame: Frame) -> list[LandmarkHand]:
        dt = self._frame_dt()
        image = frame.image
        if self._tracking_max_width and image.shape[1] > self._tracking_max_width:
            scale = self._tracking_max_width / image.shape[1]
            image = cv2.resize(
                image,
                (
                    self._tracking_max_width,
                    max(1, round(image.shape[0] * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        if not result.hand_landmarks:
            for smoother in self._smoothers.values():
                smoother.reset()
            return []

        hands: list[LandmarkHand] = []
        seen: set[str] = set()
        for i, raw in enumerate(result.hand_landmarks):
            label = _handedness_label(result, i)
            if label in seen:
                label = f"{label}-{i}"
            seen.add(label)

            pts = np.array([(lm.x, lm.y, lm.z) for lm in raw], dtype=np.float64)
            smoother = self._smoothers.setdefault(label, _LandmarkSmoother())
            smoothed = smoother.smooth(pts, dt)
            hands.append(
                LandmarkHand(
                    landmarks=smoothed,
                    hand_id=label,
                    handedness=label.split("-")[0],
                    confidence=_confidence(result, i),
                )
            )

        for label in list(self._smoothers):
            if label not in seen:
                self._smoothers[label].reset()

        return hands

    def close(self) -> None:
        self._landmarker.close()
