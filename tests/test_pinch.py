"""Tests for canvas-style pinch / grab gesture detection."""

from __future__ import annotations

import numpy as np
import pytest

from hand_interaction.constants import (
    GESTURE_WINDOW_FRAMES,
    INDEX_TIP,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    PINCH_RELEASE_FRAMES,
    PINCH_TO_FIST_FRAMES,
    THUMB_TIP,
    WRIST,
)
from hand_interaction.gestures import GestureDetector, GestureState
from hand_interaction.pinch import pinch_ratio_3d
from hand_interaction.types import PinchEnd, PinchMove, PinchStart


def _base_hand(ox: float = 0.5, oy: float = 0.5, scale: float = 0.12) -> np.ndarray:
    """Palm + MCPs around (ox, oy); tips filled later."""
    pts = np.zeros((21, 3), dtype=np.float64)
    pts[WRIST] = (ox, oy + 0.15 * scale, 0.0)
    pts[5] = (ox + 0.35 * scale, oy, 0.0)
    pts[9] = (ox + 0.10 * scale, oy - 0.05 * scale, 0.0)
    pts[13] = (ox - 0.15 * scale, oy, 0.0)
    pts[17] = (ox - 0.35 * scale, oy + 0.02 * scale, 0.0)
    for i in range(21):
        if np.allclose(pts[i], 0.0) and i != WRIST:
            pts[i] = pts[WRIST] + np.array([0.02 * (i % 5), -0.02 * (i % 7), 0.0])
    # Re-apply palm anchors
    pts[WRIST] = (ox, oy + 0.15 * scale, 0.0)
    pts[5] = (ox + 0.35 * scale, oy, 0.0)
    pts[9] = (ox + 0.10 * scale, oy - 0.05 * scale, 0.0)
    pts[13] = (ox - 0.15 * scale, oy, 0.0)
    pts[17] = (ox - 0.35 * scale, oy + 0.02 * scale, 0.0)
    return pts


def _pinch_hand(
    gap: float,
    ox: float = 0.5,
    oy: float = 0.5,
    scale: float = 0.12,
) -> np.ndarray:
    """Open-finger pinch with tips reaching out from the palm."""
    pts = _base_hand(ox, oy, scale)
    from hand_interaction.pose import hand_scale

    # Extended free fingers (away from palm)
    palm = pts[[0, 5, 9, 13, 17]].mean(axis=0)
    for tip, mcp in ((12, 9), (16, 13), (20, 17)):
        direction = pts[mcp] - pts[WRIST]
        pts[tip] = palm + direction / (np.linalg.norm(direction) + 1e-9) * (1.2 * scale)

    s = hand_scale(pts)
    # Contact point out in front of the palm
    mid = palm + np.array([0.0, -0.9 * scale, 0.0])
    half = 0.5 * gap * s
    pts[THUMB_TIP] = mid + np.array([-half, 0.0, 0.0])
    pts[INDEX_TIP] = mid + np.array([half, 0.0, 0.0])
    # Index PIP between wrist and tip
    pts[6] = 0.5 * (pts[WRIST] + pts[INDEX_TIP])
    return pts


def _open_hand(ox: float = 0.5, oy: float = 0.5, scale: float = 0.12) -> np.ndarray:
    """Clearly open hand (no pinch, no fist)."""
    return _pinch_hand(0.9, ox=ox, oy=oy, scale=scale)


def _fist_hand(ox: float = 0.5, oy: float = 0.5, scale: float = 0.12) -> np.ndarray:
    """Closed fist: tips over the palm, index not extended."""
    pts = _base_hand(ox, oy, scale)
    palm = pts[[0, 5, 9, 13, 17]].mean(axis=0)
    for tip in (8, 12, 16, 20):
        pts[tip] = palm + np.array([0.02 * ((tip % 5) - 2), 0.02, 0.0]) * scale
    pts[THUMB_TIP] = palm + np.array([0.15 * scale, 0.05 * scale, 0.0])
    pts[6] = 0.5 * (pts[WRIST] + pts[INDEX_TIP])
    return pts


def _feed(detector: GestureDetector, landmarks: np.ndarray, frames: int, hand_id: str = "Right"):
    events = []
    for _ in range(frames):
        events.extend(detector.update(landmarks, hand_id))
    return events


def test_pinch_ratio_tracks_gap() -> None:
    pts = _pinch_hand(0.2)
    assert pinch_ratio_3d(pts) == pytest.approx(0.2, rel=0.15)


def test_pinch_start_after_hold() -> None:
    det = GestureDetector(window=5)
    pts = _pinch_hand(0.2)
    early = _feed(det, pts, 4)
    assert early == []
    assert det.state is not GestureState.PINCHING
    started = _feed(det, pts, 1)
    assert any(isinstance(e, PinchStart) for e in started)
    assert any(isinstance(e, PinchMove) for e in started)
    assert det.state is GestureState.PINCHING


def test_pinch_end_on_release() -> None:
    det = GestureDetector(window=3)
    _feed(det, _pinch_hand(0.2), 3)
    assert det.state is GestureState.PINCHING
    # Release needs median to climb + PINCH_RELEASE_FRAMES of agreement.
    ended = _feed(det, _open_hand(), PINCH_RELEASE_FRAMES + 5)
    assert any(isinstance(e, PinchEnd) for e in ended)
    assert det.state is not GestureState.PINCHING


def test_no_duplicate_start_while_holding() -> None:
    det = GestureDetector(window=3)
    events = _feed(det, _pinch_hand(0.2), 20)
    starts = [e for e in events if isinstance(e, PinchStart)]
    assert len(starts) == 1


def test_noise_around_threshold_does_not_flap() -> None:
    det = GestureDetector(window=5)
    _feed(det, _pinch_hand(PINCH_RATIO_ON - 0.05), 5)
    assert det.state is GestureState.PINCHING

    mid = (PINCH_RATIO_ON + PINCH_RATIO_OFF) / 2
    events = []
    for i in range(30):
        ratio = mid + (0.04 if i % 2 == 0 else -0.04)
        ratio = min(ratio, PINCH_RATIO_OFF - 0.02)
        events.extend(det.update(_pinch_hand(ratio), "Right"))

    assert [e for e in events if isinstance(e, PinchEnd)] == []
    assert [e for e in events if isinstance(e, PinchStart)] == []
    assert det.state is GestureState.PINCHING


def test_brief_gap_spike_does_not_end_pinch() -> None:
    """One noisy wide-gap frame while stretching must not drop the hold."""
    det = GestureDetector(window=3)
    _feed(det, _pinch_hand(0.2), 3)
    assert det.state is GestureState.PINCHING

    spike = det.update(_open_hand(), "Right")
    assert [e for e in spike if isinstance(e, PinchEnd)] == []
    assert det.state is GestureState.PINCHING

    recovered = _feed(det, _pinch_hand(0.2), 5)
    assert [e for e in recovered if isinstance(e, PinchEnd)] == []
    assert det.state is GestureState.PINCHING


def test_brief_fist_lookalike_does_not_steal_pinch() -> None:
    """Camera-aimed pinch can look closed for a frame; keep stretching."""
    det = GestureDetector(window=3)
    _feed(det, _pinch_hand(0.2), 3)
    assert det.state is GestureState.PINCHING

    brief = PINCH_TO_FIST_FRAMES - 1
    events = _feed(det, _fist_hand(), brief)
    assert [e for e in events if isinstance(e, PinchEnd)] == []
    assert det.state is GestureState.PINCHING

    recovered = _feed(det, _pinch_hand(0.2), 5)
    assert [e for e in recovered if isinstance(e, PinchEnd)] == []
    assert det.state is GestureState.PINCHING


def test_lost_hand_grace_then_end() -> None:
    det = GestureDetector(window=GESTURE_WINDOW_FRAMES)
    _feed(det, _pinch_hand(0.2), GESTURE_WINDOW_FRAMES)
    assert det.state is GestureState.PINCHING
    for _ in range(3):
        assert det.update(None, "Right") == []
    ended = det.update(None, "Right")
    assert any(isinstance(e, PinchEnd) for e in ended)
