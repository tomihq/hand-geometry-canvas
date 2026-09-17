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

# Letting go is judged against the gap actually being held, not only the fixed
# figure above. A pinch held loosely enters at ~0.30, right on PINCH_RATIO_ON,
# which leaves barely two centimetres of finger travel before 0.55: relaxing
# slightly while dragging read as a release and committed the figure half-drawn.
# Scaling the exit off the held gap gives tight and loose pinches the same
# margin, and PINCH_RATIO_OFF stays the floor so a tight pinch is unaffected.
PINCH_RELEASE_FACTOR = 2.2
# Ceiling, so a pinch held very loosely can still be released by pulling apart.
PINCH_RELEASE_MAX = 0.85
# Frames of held pinch that define the reference gap.
PINCH_BASELINE_FRAMES = 20

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
FIST_FINGERS_OFF = 2

# Index tip vs PIP distance from the wrist. Above this the finger is extended.
INDEX_EXTENDED_RATIO = 1.08

# --- Holding a grab ---------------------------------------------------------
# Starting a fist takes a deliberate hold (GESTURE_WINDOW_FRAMES). Letting go
# is the opposite problem. A travelling arm smears the fingers in the camera
# for the whole length of the swing, not for one stray frame: the curl ratios
# read open, and the figure used to be dropped in the middle of the throw. So
# a release has to be confirmed over consecutive frames, and over more of them
# while the hand is moving, which is when an open reading is least believable.
#
# These stack on top of the median window: the median needs three of its five
# frames to turn before `hand_opened` is even true. A still hand therefore
# releases after ~5 frames, a travelling one after ~8.
FIST_RELEASE_FRAMES = 2
FIST_RELEASE_FRAMES_MOVING = 5

# Palm speed (frame widths per second) above which the hand counts as moving.
FIST_MOVING_SPEED = 0.6

# Window the palm speed is read over, and how many samples to keep for it.
HAND_SPEED_SPAN = 0.08
HAND_SPEED_SAMPLES = 12

# Frames a held gesture survives with no hand detected at all. Tracking drops
# out precisely when motion blur is worst, i.e. mid-throw.
HAND_LOST_GRACE_FRAMES = 3

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

# --- Landmark smoothing (one-euro) ------------------------------------------
# A fixed-alpha EMA has to pick one compromise, and pays for it either way:
# enough smoothing to kill jitter costs a fixed lag, which a fast hand turns
# into a figure trailing well behind the palm. Here the filter cutoff rises
# with the measured speed, so jitter is filtered while the hand rests and a
# throw passes through almost untouched.
#
# Cutoff (Hz) with the hand at a standstill: the jitter floor.
SMOOTH_MIN_CUTOFF = 1.5
# How much the cutoff opens up per unit of speed (frame widths per second).
SMOOTH_BETA = 2.5
# Cutoff for the speed estimate itself, which is noisier than the position.
SMOOTH_DERIVATIVE_CUTOFF = 1.0

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
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Hand skeleton, gesture text and drag markers. Toggle at runtime with 'd'.
SHOW_DEBUG_OVERLAY = False

# --- Capture -----------------------------------------------------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60

# MJPG is the only format most USB webcams can sustain at 720p/60+; the default
# uncompressed YUYV runs out of USB bandwidth first and silently drops to ~10fps.
CAMERA_FOURCC = "MJPG"

# Number of V4L2 buffers the driver maps. One is not enough: while userspace
# holds the only buffer the driver has nowhere to put the next frame, so every
# second one is dropped and a 30fps camera delivers 15. Two is the minimum that
# keeps the stream full. Staleness is not a concern at any depth here — the
# capture thread drains the queue continuously and `Camera.read` only ever
# hands back the newest frame.
CAMERA_BUFFER_SIZE = 2

# --- Tracking cost -----------------------------------------------------------
# Inference is by far the most expensive stage. TFLite's GPU delegate runs it on
# the discrete GPU instead of XNNPACK on the CPU; we fall back automatically if
# the machine cannot create a GL context for it.
USE_GPU_INFERENCE = True

# The landmark model rescales to a fixed input, so inference time is nearly flat
# above this width while the surrounding per-pixel work is not. Landmarks come
# back normalized, so shrinking first costs nothing in coordinate math.
TRACKING_MAX_WIDTH = 640

# Print a rolling FPS figure in the window title.
SHOW_FPS = True

# Every millisecond of loop time is a millisecond the figure lags the hand, so
# when the frame rate is disappointing the question is always which stage ate
# it. Turn this on to print median per-stage times instead of guessing.
SHOW_STAGE_TIMINGS = False
