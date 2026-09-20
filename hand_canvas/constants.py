"""Configurable thresholds for gesture detection and interaction.

Shared pinch / fist / capture / smoothing knobs live in
``hand_interaction.constants`` (single source of truth). This module
re-exports them and adds canvas-only tuning (sweep, trash, fling, window).
"""

from hand_interaction.constants import (
    CAMERA_BUFFER_SIZE,
    CAMERA_FOURCC,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    FIST_CURL_RATIO_OFF,
    FIST_CURL_RATIO_ON,
    FIST_FINGERS_ON,
    FIST_OPEN_CURL_MIN,
    FIST_OPEN_FINGERS_MIN,
    FIST_OPEN_TIPS_PALM_MIN,
    FIST_RELEASE_FRAMES,
    FIST_TIPS_PALM_MAX,
    GESTURE_WINDOW_FRAMES,
    HAND_LOST_GRACE_FRAMES,
    INDEX_EXTENDED_RATIO,
    MIN_HAND_SCALE,
    PINCH_BASELINE_FRAMES,
    PINCH_CONTACT_RATIO,
    PINCH_FREE_FINGERS_MIN,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    PINCH_RELEASE_FACTOR,
    PINCH_RELEASE_MAX,
    PINCH_TIPS_PALM_MIN,
    SMOOTH_BETA,
    SMOOTH_DERIVATIVE_CUTOFF,
    TRACKING_MAX_WIDTH,
    USE_GPU_INFERENCE,
)
from hand_interaction.constants import SMOOTH_MIN_CUTOFF_XY as SMOOTH_MIN_CUTOFF

# --- Sweep to clear (canvas-only) -------------------------------------------
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

# Coming this close to the bin, a held figure settles onto the hand and shrinks
# to fit in the zone, so you can aim it. Deleting still takes opening the hand.
TRASH_PULL_RADIUS = 0.22
# Target size at the mouth: this fraction of the zone, floored so a big figure
# never turns into a speck.
TRASH_FIT_MARGIN = 0.7
TRASH_PULL_MIN_SCALE = 0.25

# Undo takes a binned figure out of the zone and parks it this far to its left,
# otherwise the bin would eat it again the moment it came back.
TRASH_EVICT_GAP = 0.02

# Max distance (normalized) to select an existing point / corner handle.
HIT_RADIUS = 0.035
CORNER_HIT_RADIUS = 0.07

# Palm may jitter while closing the fist — keep a small grab margin.
GRAB_HIT_PADDING = 0.03

# Slightly looser when helping resize a figure locked by the other hand.
HELPER_HIT_PADDING = 0.04

# --- Throwing a figure ------------------------------------------------------
FLING_DECAY_TAU = 0.26
FLING_MIN_SPEED = 0.55
FLING_MAX_SPEED = 2.8
FLING_STOP_SPEED = 0.05
FLING_BOUNCE_RESTITUTION = 0.45
FLING_SAMPLE_WINDOW = 0.3
FLING_MIN_SPAN = 0.05
FLING_TRAIL_SAMPLES = 16

# Smallest figure allowed on the canvas, required on *both* width and height.
MIN_SHAPE_SIZE = 0.04

# Display window
WINDOW_NAME = "Hand Geometry Canvas"
WINDOW_WIDTH = CAMERA_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT

# Hand skeleton, gesture text and drag markers. Toggle at runtime with 'd'.
SHOW_DEBUG_OVERLAY = False

# Print a rolling FPS figure in the window title.
SHOW_FPS = True

# Every millisecond of loop time is a millisecond the figure lags the hand, so
# when the frame rate is disappointing the question is always which stage ate
# it. Turn this on to print median per-stage times instead of guessing.
SHOW_STAGE_TIMINGS = False
