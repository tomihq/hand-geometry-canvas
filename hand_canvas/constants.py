"""Configurable thresholds for gesture detection and interaction.

Shared pinch / capture / smoothing knobs live in ``hand_interaction.constants``
(single source of truth). This module re-exports them and adds canvas-only
tuning (fist, sweep, trash, fling, hit radii, window).
"""

from hand_interaction.constants import (
    CAMERA_BUFFER_SIZE,
    CAMERA_FOURCC,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    GESTURE_WINDOW_FRAMES,
    HAND_LOST_GRACE_FRAMES,
    MIN_HAND_SCALE,
    PINCH_BASELINE_FRAMES,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    PINCH_RELEASE_FACTOR,
    PINCH_RELEASE_MAX,
    SMOOTH_BETA,
    SMOOTH_DERIVATIVE_CUTOFF,
    TRACKING_MAX_WIDTH,
    USE_GPU_INFERENCE,
)
from hand_interaction.constants import SMOOTH_MIN_CUTOFF_XY as SMOOTH_MIN_CUTOFF

# --- Pinch pose filters (canvas-only: distinguish pinch vs fist) -------------
# hand_interaction.pinch uses the shared PINCH_RATIO_* / release knobs above
# but does not apply these pose gates. Canvas gestures still need them.

# Where the thumb–index midpoint sits relative to the palm center. A pinch
# reaches out (0.8–1.2); a closed hand keeps the tips over the palm (~0.2).
# The gap between the two is a deliberate no-man's land: neither gesture fires.
PINCH_TIPS_PALM_MIN = 0.55
FIST_TIPS_PALM_MAX = 0.45

# The measure above only describes a pinch made with the index still straight,
# thumb brought up to meet its tip. Curl the index round until the tips truly
# touch — the firmest pinch there is — and the contact point ends up sitting
# over the palm, reading 0.32, right inside the fist's own band. So a pinch is
# also accepted when the fingers that take no part in it are clearly out of the
# palm, which is what a fist can never do: measured over the middle, ring and
# pinky, a fist stays under 0.51 while a pinch made that way is past 0.84.
PINCH_FREE_FINGERS_MIN = 0.70

# Neither measure above can see a pinch made with the hand pointed at the
# camera, and no threshold ever will: in projection that pose *is* a closed
# fist. Measured over a recording of it, the contact point sits 0.31 from the
# palm centre and so does a real fist's, and the three free fingers read 0.30
# against a fist's 0.21 — the distributions sit on top of each other, and
# MediaPipe's depth channel does not pull them apart either.
#
# So the pose stops being what decides. A fist only ever does anything over a
# figure; over empty space it has nothing to grab. That is the way out: a hand
# closed over empty space with the tips pressed this tightly together is taken
# as a pinch and draws. The figure is only kept if the hand then moves far
# enough to make one, so a fist resting over the canvas still does nothing.
#
# Well inside PINCH_RATIO_ON, because a relaxed fist sits at 0.30: at this
# figure a deliberate squeeze is read three times out of five, a slack fist
# only one in six — and that one costs nothing but a preview.
PINCH_CONTACT_RATIO = 0.22

# Fingertip-to-palm distance / hand size, per finger. Curled reads ~0.25,
# extended reads well past 1.0.
FIST_CURL_RATIO_ON = 0.75
FIST_CURL_RATIO_OFF = 1.05
FIST_FINGERS_ON = 4

# Index tip vs PIP distance from the wrist. Above this the finger is extended.
INDEX_EXTENDED_RATIO = 1.08

# --- Holding / releasing a grab ---------------------------------------------
# Starting a fist takes a deliberate hold (GESTURE_WINDOW_FRAMES). Letting go
# does not: once the hand is clearly open, the figure drops on that reading.
# Openness is a band of ratios (not a single cutoff and not a frame counter),
# so landmark noise and hand-size variation do not flap the decision, while a
# half-open claw still carries and a full open releases immediately.
#
# Fingertip–palm / hand size: tucked ~0.25, extended past ~1.0. Anything at or
# above the floor counts toward "open"; needing several fingers in that band
# plus the tips clear of the palm is what separates a claw from a release.
FIST_OPEN_CURL_MIN = 0.90
FIST_OPEN_FINGERS_MIN = 3
FIST_OPEN_TIPS_PALM_MIN = 0.55

# Pinch handover while holding a grab still needs a short agreement — blur can
# look briefly pinched, and that path is not the "open hand → drop" case.
FIST_RELEASE_FRAMES = 2

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
# Letting go of a moving figure hands it the hand's own velocity, in frame
# widths per second. Drag is exponential, so the travel is speed * TAU: a
# throw at 2 widths/s covers about half the frame before stopping, which makes
# distance predictable from how hard you threw.
FLING_DECAY_TAU = 0.26
# Below this a release is a drop, not a throw, and nothing flies.
FLING_MIN_SPEED = 0.55
# Ceiling, so a tracking glitch cannot fire a figure across the frame.
FLING_MAX_SPEED = 2.8
# The flight ends here rather than creeping to a halt over several seconds.
FLING_STOP_SPEED = 0.05
# Fraction of the speed kept after hitting an edge of the canvas. Well under 1
# so a thrown figure settles inside instead of rattling between the borders.
FLING_BOUNCE_RESTITUTION = 0.45
# Window of hand positions the throw speed is read from, and the shortest span
# between two of them worth dividing by.
FLING_SAMPLE_WINDOW = 0.3
FLING_MIN_SPAN = 0.05
FLING_TRAIL_SAMPLES = 16

# Smallest figure allowed on the canvas, required on *both* width and height.
# Anything under this is a sliver or a stray dot rather than a figure, so it is
# never committed: it stays a preview while the gesture runs and is dropped on
# release. Also the floor for resizing, so a figure cannot be squashed to noise.
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
