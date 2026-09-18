"""Webcam capture — no knowledge of hands or geometry.

Capture runs on its own thread so the driver wait never blocks inference.
``read()`` hands back the newest frame and skips whatever piled up in between.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from hand_interaction.constants import (
    CAMERA_BUFFER_SIZE,
    CAMERA_FOURCC,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)


@dataclass
class Frame:
    image: np.ndarray
    width: int
    height: int


class Camera:
    def __init__(
        self,
        device_index: int = CAMERA_INDEX,
        mirror: bool = True,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
        fourcc: str = CAMERA_FOURCC,
        buffer_size: int = CAMERA_BUFFER_SIZE,
    ) -> None:
        self._device_index = device_index
        self._mirror = mirror
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self._buffer_size = buffer_size
        self._cap: cv2.VideoCapture | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._consumed_seq = 0
        self._error: BaseException | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._device_index}")

        if self._fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps:
            cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        self._cap = cap
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, name="camera", daemon=True)
        self._thread.start()

    def describe(self) -> str:
        if self._cap is None:
            return "camera closed"
        code = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = (
            "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) if code else "?"
        )
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return f"{width}x{height} {fourcc} @ {fps:g}fps"

    def _pump(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok or image is None:
                with self._condition:
                    self._error = RuntimeError("Failed to read frame from camera")
                    self._condition.notify_all()
                return
            if self._mirror:
                image = cv2.flip(image, 1)
            with self._condition:
                self._latest = image
                self._seq += 1
                self._condition.notify_all()

    def read(self, timeout: float = 5.0) -> Frame:
        if self._cap is None:
            raise RuntimeError("Camera is not open")

        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._seq != self._consumed_seq or self._error is not None,
                timeout,
            )
            if self._error is not None:
                raise self._error
            if not ready or self._latest is None:
                raise RuntimeError("Timed out waiting for a camera frame")
            image = self._latest
            self._consumed_seq = self._seq

        height, width = image.shape[:2]
        return Frame(image=image, width=width, height=height)

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
