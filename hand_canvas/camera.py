"""Webcam capture — no knowledge of hands or geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    width: int
    height: int


class Camera:
    def __init__(self, device_index: int = 0, mirror: bool = True) -> None:
        self._device_index = device_index
        self._mirror = mirror
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._device_index}")

    def read(self) -> Frame:
        if self._cap is None:
            raise RuntimeError("Camera is not open")
        ok, image = self._cap.read()
        if not ok or image is None:
            raise RuntimeError("Failed to read frame from camera")
        if self._mirror:
            image = cv2.flip(image, 1)
        height, width = image.shape[:2]
        return Frame(image=image, width=width, height=height)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
