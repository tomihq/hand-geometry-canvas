"""Tunable thresholds for pose estimation and gesture detection."""

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
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
RING_MCP = 13
PINKY_MCP = 17

PALM_IDS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
MCP_IDS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
TIP_IDS = (8, 12, 16, 20)

# --- Pinch / fist shared window ---------------------------------------------
# Median window also acts as the hold required to *start* a gesture.
GESTURE_WINDOW_FRAMES = 5

# Thumb–index gap / hand size. Tips touching ~0.15; relaxed thumb past ~0.4.
PINCH_RATIO_ON = 0.32
PINCH_RATIO_OFF = 0.55

# Exit threshold scales with the gap actually held while pinching.
PINCH_RELEASE_FACTOR = 1.4
PINCH_RELEASE_MAX = 0.85
PINCH_BASELINE_FRAMES = 20

# --- Pinch pose filters (distinguish pinch vs fist) -------------------------
# Where the thumb–index midpoint sits relative to the palm center. A pinch
# reaches out (0.8–1.2); a closed hand keeps the tips over the palm (~0.2).
PINCH_TIPS_PALM_MIN = 0.55
FIST_TIPS_PALM_MAX = 0.45

# Pinch also accepted when free fingers (middle/ring/pinky) are clearly out.
PINCH_FREE_FINGERS_MIN = 0.70

# Camera-aimed pinch projects like a fist; tight contact + empty-space draw
# path is handled by the consumer. Well inside PINCH_RATIO_ON.
PINCH_CONTACT_RATIO = 0.22

# Fingertip-to-palm distance / hand size, per finger.
FIST_CURL_RATIO_ON = 0.75
FIST_CURL_RATIO_OFF = 1.05
FIST_FINGERS_ON = 4

# Index tip vs PIP distance from the wrist. Above this the finger is extended.
INDEX_EXTENDED_RATIO = 1.08

# --- Holding / releasing a grab ---------------------------------------------
FIST_OPEN_CURL_MIN = 0.90
FIST_OPEN_FINGERS_MIN = 3
FIST_OPEN_TIPS_PALM_MIN = 0.55
# Pinch handover while holding a grab still needs a short agreement.
FIST_RELEASE_FRAMES = 2

# --- Interaction event deadzones --------------------------------------------
MOVE_EPSILON = 5e-4  # ‖Δposition‖ in normalized screen units
RESIZE_EPSILON = 5e-4  # ‖Δhelper cursor‖ for resize.move

# Frames a held gesture survives with no hand before force-end.
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
