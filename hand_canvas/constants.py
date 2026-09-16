"""Configurable thresholds for gesture detection and interaction."""

# Pinch as fraction of hand size (wrist → middle MCP). End > start = hysteresis.
PINCH_START_RATIO = 0.50
PINCH_END_RATIO = 0.70

# Absolute floor/ceiling so extreme camera distances don't break ratios.
PINCH_START_MIN = 0.045
PINCH_START_MAX = 0.11
PINCH_END_MIN = 0.06
PINCH_END_MAX = 0.15

# Frames under the start threshold before emitting PointerDown.
PINCH_CONFIRM_FRAMES = 3

# Fist: fingertip-near-palm score (0–4). Confirm ~300–500ms before grab.
FIST_TIP_PALM_RATIO = 0.95
FIST_SCORE_START = 3
FIST_SCORE_END = 1
FIST_CONFIRM_FRAMES = 10

# Max distance (normalized) to select an existing point / corner handle.
HIT_RADIUS = 0.035
CORNER_HIT_RADIUS = 0.07

# Tiny padding for palm jitter only — palm must sit on/near the figure.
# 0.12 was ~12% of the frame per side and grabbed from well outside.
GRAB_HIT_PADDING = 0.02

# Slightly looser when helping resize a figure locked by the other hand.
HELPER_HIT_PADDING = 0.04

# EMA smoothing: smoothed = alpha * current + (1 - alpha) * previous
SMOOTH_ALPHA = 0.4

# Minimum drag distance before converting a point into a rectangle.
MIN_RECT_SIZE = 0.01

# Display window (OpenCV)
WINDOW_NAME = "Hand Geometry Canvas"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
