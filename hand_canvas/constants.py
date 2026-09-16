"""Configurable thresholds for gesture detection and interaction."""

# --- Gesture detection -------------------------------------------------------
# Every threshold below is a ratio of hand size, so camera distance is
# irrelevant. ON/OFF pairs leave a margin band in between: while a measure sits
# inside the band the current gesture simply holds, instead of flickering.

# Landmark noise is smoothed by taking the median of this many frames. It also
# sets the hold required to *start* a gesture (~5 frames ≈ 150–300ms).
GESTURE_WINDOW_FRAMES = 5

# Guard against a degenerate hand (all landmarks stacked) dividing by ~zero.
MIN_HAND_SCALE = 1e-3

# Thumb–index gap / hand size. Tips touching measure ~0.15; a relaxed thumb
# sits past 0.4, so 0.32 → 0.55 is a wide margin either way.
PINCH_RATIO_ON = 0.32
PINCH_RATIO_OFF = 0.55

# Where the thumb–index midpoint sits relative to the palm center. A pinch
# reaches out (0.8–1.2); a closed hand keeps the tips over the palm (~0.2).
# The gap between the two is a deliberate no-man's land: neither gesture fires.
PINCH_TIPS_PALM_MIN = 0.55
FIST_TIPS_PALM_MAX = 0.45

# Fingertip-to-palm distance / hand size, per finger. Curled reads ~0.25,
# extended reads well past 1.0.
FIST_CURL_RATIO_ON = 0.75
FIST_CURL_RATIO_OFF = 1.05
FIST_FINGERS_ON = 4
FIST_FINGERS_OFF = 2

# Index tip vs PIP distance from the wrist. Above this the finger is extended.
INDEX_EXTENDED_RATIO = 1.08

# --- Sweep to clear ---------------------------------------------------------
# Open palm dragged sideways wipes the canvas. Deliberately demanding: an open
# palm is far from both pinch and fist, and the travel has to be a real swipe.
SWEEP_TRAVEL_FRAMES = 10
SWEEP_MIN_TRAVEL = 0.35
# Horizontal dominance: |dx| must exceed this multiple of |dy|.
SWEEP_HORIZONTAL_RATIO = 2.0
# Frames to wait after a sweep before another can fire (one swipe, one wipe).
SWEEP_COOLDOWN_FRAMES = 20

# --- Delete by dropping a figure in the trash zone --------------------------
# Normalized frame coords, bottom-right corner. Roughly square at 4:3.
TRASH_ZONE_X = 0.83
TRASH_ZONE_Y = 0.76
TRASH_ZONE_W = 0.17
TRASH_ZONE_H = 0.24

# Max distance (normalized) to select an existing point / corner handle.
HIT_RADIUS = 0.035
CORNER_HIT_RADIUS = 0.07

# Palm may jitter while closing the fist — keep a small grab margin.
GRAB_HIT_PADDING = 0.03

# Slightly looser when helping resize a figure locked by the other hand.
HELPER_HIT_PADDING = 0.04

# EMA smoothing: smoothed = alpha * current + (1 - alpha) * previous
SMOOTH_ALPHA = 0.4

# Minimum drag distance before converting a point into a rectangle.
MIN_RECT_SIZE = 0.01

# Display window (OpenCV)
WINDOW_NAME = "Hand Geometry Canvas"
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768

# Hand skeleton, gesture text and drag markers. Toggle at runtime with 'd'.
SHOW_DEBUG_OVERLAY = False
