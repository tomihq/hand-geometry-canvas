"""Tunable thresholds for pose estimation and pinch detection."""

# Guard against a degenerate hand dividing by ~zero.
MIN_HAND_SCALE = 1e-3

# One-euro filter: xy vs z (depth is noisier → lower cutoff).
SMOOTH_MIN_CUTOFF_XY = 1.5
SMOOTH_MIN_CUTOFF_Z = 0.8
SMOOTH_BETA = 2.5
SMOOTH_DERIVATIVE_CUTOFF = 1.0

# Orientation slerp rate toward the measured quat (1 = snap, 0 = freeze).
# Interpreted as alpha at nominal 30 fps; scaled with dt.
ORIENTATION_SLERP_RATE = 12.0  # Hz-ish blend strength

NOMINAL_DT = 1.0 / 30.0
MIN_DT = 1.0 / 240.0
MAX_DT = 0.2

# MediaPipe-style landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
MCP_IDS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# --- Pinch (3D ratio / hand size; ON/OFF hysteresis) -------------------------
# Median window also acts as the hold required to *start* a pinch.
GESTURE_WINDOW_FRAMES = 5

# Thumb–index gap / hand size. Tips touching ~0.15; relaxed thumb past ~0.4.
PINCH_RATIO_ON = 0.32
PINCH_RATIO_OFF = 0.55

# Exit threshold scales with the gap actually held while pinching.
PINCH_RELEASE_FACTOR = 1.4
PINCH_RELEASE_MAX = 0.85
PINCH_BASELINE_FRAMES = 20

# --- Interaction event deadzones (sensitive) --------------------------------
MOVE_EPSILON = 5e-4  # ‖Δposition‖ in normalized screen units
ROTATE_EPSILON = 0.005  # ~0.3° in radians (projected screen angle)

# Frames a held pinch survives with no hand before force-end.
HAND_LOST_GRACE_FRAMES = 3

# --- Capture / MediaPipe ----------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60
CAMERA_FOURCC = "MJPG"
CAMERA_BUFFER_SIZE = 2

USE_GPU_INFERENCE = True
TRACKING_MAX_WIDTH = 640
