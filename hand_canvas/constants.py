"""Configurable thresholds for gesture detection and interaction."""

# Pinch hysteresis (normalized landmark space). End > start avoids jitter.
PINCH_START_THRESHOLD = 0.05
PINCH_END_THRESHOLD = 0.07

# Max distance (normalized) to select an existing point / corner handle.
HIT_RADIUS = 0.05
CORNER_HIT_RADIUS = 0.055

# EMA smoothing: smoothed = alpha * current + (1 - alpha) * previous
SMOOTH_ALPHA = 0.4

# Minimum drag distance before converting a point into a rectangle.
MIN_RECT_SIZE = 0.01

# Fist detection: enter when >= START curled fingers, exit when <= END.
FIST_START_CURLS = 3
FIST_END_CURLS = 1

# Display window (OpenCV)
WINDOW_NAME = "Hand Geometry Canvas"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
