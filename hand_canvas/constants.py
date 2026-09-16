"""Configurable thresholds for gesture detection and interaction."""

# Pinch as fraction of hand size (wrist → middle MCP). End > start = hysteresis.
# ~0.5 ≈ tips clearly together; open hand is usually > 0.8.
PINCH_START_RATIO = 0.50
PINCH_END_RATIO = 0.70

# Absolute floor/ceiling so extreme camera distances don't break ratios.
PINCH_START_MIN = 0.045
PINCH_START_MAX = 0.11
PINCH_END_MIN = 0.06
PINCH_END_MAX = 0.15

# Frames under the start threshold before emitting PointerDown (anti-false-positive).
PINCH_CONFIRM_FRAMES = 3

# Max distance (normalized) to select an existing point / corner handle.
HIT_RADIUS = 0.035
CORNER_HIT_RADIUS = 0.07

# Extra padding when helping resize a figure locked by the other hand.
HELPER_HIT_PADDING = 0.08

# EMA smoothing: smoothed = alpha * current + (1 - alpha) * previous
SMOOTH_ALPHA = 0.4

# Minimum drag distance before converting a point into a rectangle.
MIN_RECT_SIZE = 0.01

# Fist uses middle/ring/pinky curls (index excluded — it's used for pinch).
FIST_START_CURLS = 3
FIST_END_CURLS = 1

# Display window (OpenCV)
WINDOW_NAME = "Hand Geometry Canvas"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
