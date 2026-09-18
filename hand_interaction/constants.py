"""Tunable thresholds for pose estimation and (later) interaction."""

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
