"""2D canvas — shape store + CPU render fallback.

GPU compositing lives in ``gpu_renderer.GpuCanvasRenderer`` (OpenGL).
"""

from __future__ import annotations

import cv2
import numpy as np

from hand_canvas.geometry import (
    Point,
    PointShape,
    Rectangle,
    Shape,
    distance,
    is_visible_figure,
    point_in_rectangle,
    rectangle_corners,
    shape_area,
)


class Canvas:
    def __init__(self) -> None:
        self._shapes: list[Shape] = []
        self._next_z = 1
        # Deletions are grouped, so undoing a sweep brings back the whole wipe
        self._trash: list[list[Shape]] = []

    @property
    def shapes(self) -> list[Shape]:
        return self._shapes

    @property
    def trash_depth(self) -> int:
        return len(self._trash)

    def add(self, shape: Shape) -> None:
        shape.z = self._next_z
        self._next_z += 1
        self._shapes.append(shape)

    def update(self, shape: Shape) -> None:
        for i, existing in enumerate(self._shapes):
            if existing.id == shape.id:
                shape.z = existing.z
                self._shapes[i] = shape
                return
        self.add(shape)

    def remove(self, shape_id: str) -> None:
        self._shapes = [s for s in self._shapes if s.id != shape_id]

    def discard(self, shape_ids: list[str]) -> list[Shape]:
        """Delete shapes as one undoable batch. Returns what was removed."""
        wanted = set(shape_ids)
        removed = [s for s in self._shapes if s.id in wanted]
        if not removed:
            return []
        self._shapes = [s for s in self._shapes if s.id not in wanted]
        self._trash.append(removed)
        return removed

    def restore_last(self) -> list[Shape]:
        """Undo the most recent deletion batch, back on top of the stack."""
        if not self._trash:
            return []
        batch = self._trash.pop()
        for shape in batch:
            shape.z = self._next_z
            self._next_z += 1
            self._shapes.append(shape)
        return batch

    def get(self, shape_id: str) -> Shape | None:
        for shape in self._shapes:
            if shape.id == shape_id:
                return shape
        return None

    def bring_to_front(self, shape_id: str) -> None:
        shape = self.get(shape_id)
        if shape is None:
            return
        shape.z = self._next_z
        self._next_z += 1

    def get_near_point(
        self, x: float, y: float, radius: float
    ) -> PointShape | None:
        target = Point(x, y)
        candidates: list[PointShape] = []
        for shape in self._shapes:
            if isinstance(shape, PointShape):
                if distance(target, shape.position) <= radius:
                    candidates.append(shape)
        if not candidates:
            return None
        candidates.sort(key=lambda s: (shape_area(s), s.z), reverse=True)
        return candidates[0]

    def hit_test(self, x: float, y: float, point_radius: float) -> Shape | None:
        """Pick the largest shape under the pointer; break ties with higher z."""
        target = Point(x, y)
        hits: list[Shape] = []

        for shape in self._shapes:
            if isinstance(shape, PointShape):
                if distance(target, shape.position) <= point_radius:
                    hits.append(shape)
            elif isinstance(shape, Rectangle):
                # Exact bounds only — no padding so empty-space create isn't blocked
                if point_in_rectangle(target, shape, padding=0.0):
                    hits.append(shape)

        if not hits:
            return None

        # Largest area first; if equal (or both points), higher z wins.
        hits.sort(key=lambda s: (shape_area(s), s.z), reverse=True)
        return hits[0]

    def render(
        self,
        width: int,
        height: int,
        background: np.ndarray | None = None,
        selected_ids: set[str] | None = None,
    ) -> np.ndarray:
        if background is not None:
            image = background.copy()
        else:
            image = np.zeros((height, width, 3), dtype=np.uint8)

        selected_ids = selected_ids or set()
        # Draw low-z first so higher z appears on top
        ordered = sorted(self._shapes, key=lambda s: s.z)

        for shape in ordered:
            # Only real figures get painted; a point or sliver is a live preview
            # of a gesture in progress, so it shows only while a hand holds it.
            if not is_visible_figure(shape) and shape.id not in selected_ids:
                continue

            if isinstance(shape, PointShape):
                cx = int(shape.position.x * width)
                cy = int(shape.position.y * height)
                cv2.circle(image, (cx, cy), 8, (0, 220, 255), -1)
                cv2.circle(image, (cx, cy), 10, (255, 255, 255), 1)
            elif isinstance(shape, Rectangle):
                x1 = int(shape.x * width)
                y1 = int(shape.y * height)
                x2 = int((shape.x + shape.width) * width)
                y2 = int((shape.y + shape.height) * height)
                color = (60, 180, 255) if shape.id in selected_ids else (80, 200, 120)
                overlay = image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Corner handles for resize affordance
                for corner_pt in rectangle_corners(shape).values():
                    hx = int(corner_pt.x * width)
                    hy = int(corner_pt.y * height)
                    handle_color = (0, 255, 255) if shape.id in selected_ids else (220, 220, 220)
                    cv2.rectangle(
                        image,
                        (hx - 5, hy - 5),
                        (hx + 5, hy + 5),
                        handle_color,
                        -1,
                    )
                    cv2.rectangle(
                        image,
                        (hx - 5, hy - 5),
                        (hx + 5, hy + 5),
                        (40, 40, 40),
                        1,
                    )

        return image
