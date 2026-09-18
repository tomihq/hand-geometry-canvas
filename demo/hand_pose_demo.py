"""Independent OpenCV demo: camera → hand pose → HUD (no canvas / Jarvis).

Run:
  python -m demo.hand_pose_demo
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hand_interaction.camera import Camera
from hand_interaction.constants import (
    CAMERA_FPS,
    CAMERA_FOURCC,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
)
from hand_interaction.interaction import InteractionEngine, TrackedHand
from hand_interaction.landmarker import MediaPipeLandmarker
from hand_interaction.pose import PoseEstimator
from hand_interaction.types import (
    HandEvent,
    HandMove,
    HandPose,
    HandRotate,
    PinchEnd,
    PinchStart,
    Vector2,
)

WINDOW_NAME = "Hand Interaction Demo"
WINDOW_WIDTH = CAMERA_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT
CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


@dataclass
class HandHud:
    pinching: bool = False
    position: Vector2 = field(default_factory=lambda: Vector2(0.5, 0.5))
    delta_rotation: float = 0.0
    confidence: float = 0.0
    palm_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gesture: str = "OPEN"


def quat_to_euler_deg(pose: HandPose) -> tuple[float, float, float]:
    """Display-only yaw/pitch/roll in degrees (not part of the public event API)."""
    q = pose.orientation
    roll, pitch, yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler(
        "xyz", degrees=True
    )
    return float(yaw), float(pitch), float(roll)


def apply_event(hud: dict[str, HandHud], event: HandEvent) -> None:
    state = hud.setdefault(event.hand_id, HandHud())
    if isinstance(event, PinchStart):
        state.pinching = True
        state.gesture = "PINCHING"
        state.position = event.position
    elif isinstance(event, PinchEnd):
        state.pinching = False
        state.gesture = "OPEN"
        state.position = event.position
    elif isinstance(event, HandMove):
        state.position = event.position
    elif isinstance(event, HandRotate):
        state.delta_rotation = event.delta_rotation


def update_pose_hud(hud: dict[str, HandHud], pose: HandPose) -> None:
    state = hud.setdefault(pose.hand_id, HandHud())
    state.confidence = pose.confidence
    state.palm_xyz = (pose.palm.x, pose.palm.y, pose.palm.z)
    state.euler_deg = quat_to_euler_deg(pose)
    if not state.pinching:
        state.position = Vector2(pose.palm.x, pose.palm.y)


def _to_px(x: float, y_image: float, w: int, h: int) -> tuple[int, int]:
    return int(x * w), int(y_image * h)


def draw_skeleton(image: np.ndarray, landmarks: np.ndarray) -> None:
    h, w = image.shape[:2]
    pts = [_to_px(float(p[0]), float(p[1]), w, h) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (80, 200, 80), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        color = (0, 255, 255) if i in (INDEX_TIP, THUMB_TIP, WRIST) else (200, 200, 200)
        cv2.circle(image, p, 4, color, -1, cv2.LINE_AA)


def draw_hud(image: np.ndarray, hud: dict[str, HandHud]) -> None:
    y = 28
    cv2.putText(
        image,
        "Hand Interaction Demo  |  q quit",
        (12, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    y += 28
    if not hud:
        cv2.putText(
            image,
            "No hand",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return

    for hand_id, state in hud.items():
        yaw, pitch, roll = state.euler_deg
        lines = [
            f"{hand_id}  gesture={state.gesture}",
            f"  PINCH={'YES' if state.pinching else 'no'}",
            f"  POSITION  x={state.position.x:.3f}  y={state.position.y:.3f}",
            (
                f"  PALM3D    x={state.palm_xyz[0]:.3f} "
                f"y={state.palm_xyz[1]:.3f} z={state.palm_xyz[2]:.3f}"
            ),
            (
                f"  ROTATION  yaw={yaw:+.1f} pitch={pitch:+.1f} "
                f"roll={roll:+.1f}  d={state.delta_rotation:+.3f}rad"
            ),
            f"  CONFIDENCE {state.confidence:.2f}",
        ]
        for line in lines:
            cv2.putText(
                image,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 255) if state.pinching else (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            y += 22
        y += 8


def run(device_index: int = CAMERA_INDEX) -> int:
    # Same capture knobs as hand_canvas (MJPG 1280x720 @ 60) so the preview
    # is not a low-res YUYV fallback that looks soft and choppy.
    camera = Camera(
        device_index=device_index,
        mirror=True,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        fourcc=CAMERA_FOURCC,
    )
    landmarker: MediaPipeLandmarker | None = None
    try:
        camera.open()
        print(f"Camera: {camera.describe()}")
        landmarker = MediaPipeLandmarker(num_hands=2)
        poses = PoseEstimator()
        engine = InteractionEngine()
        hud: dict[str, HandHud] = {}

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        while True:
            frame = camera.read()
            image = frame.image.copy()
            detected = landmarker.detect(frame)
            tracked: list[TrackedHand] = []
            seen: set[str] = set()

            for hand in detected:
                seen.add(hand.hand_id)
                draw_skeleton(image, hand.landmarks)
                pose = poses.estimate(hand)
                update_pose_hud(hud, pose)
                tracked.append(TrackedHand(pose=pose, landmarks=hand.landmarks))

            for event in engine.update(tracked):
                apply_event(hud, event)

            for hand_id in list(hud):
                if hand_id not in seen:
                    del hud[hand_id]
                    poses.reset(hand_id)

            draw_hud(image, hud)
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        return 0
    finally:
        if landmarker is not None:
            landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
