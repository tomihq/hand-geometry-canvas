"""Tests for 3D pinch detection with hysteresis."""

from __future__ import annotations

import numpy as np
import pytest

from hand_interaction.constants import (
    INDEX_TIP,
    PINCH_RATIO_OFF,
    PINCH_RATIO_ON,
    THUMB_TIP,
    WRIST,
)
from hand_interaction.pinch import PinchDetector, PinchPhase, pinch_ratio_3d
from hand_interaction.types import PinchEnd, PinchStart, Vector2


def _hand_with_gap(gap: float, scale: float = 0.1) -> np.ndarray:
    """Synthetic hand; pinch_ratio_3d ≈ gap (gap is the desired ratio)."""
    return _hand_at(0.5, 0.5, gap, scale=scale)


def _hand_at(
    ox: float,
    oy: float,
    gap: float,
    scale: float = 0.1,
) -> np.ndarray:
    """Synthetic hand centered near (ox, oy) with pinch ratio ≈ gap."""
    pts = np.zeros((21, 3), dtype=np.float64)
    oz = 0.0
    pts[WRIST] = (ox, oy, oz)
    # MCP spans define scale ≈ 0.1
    pts[5] = (ox + 0.4 * scale, oy - 0.3 * scale, oz)
    pts[9] = (ox + 0.1 * scale, oy - 0.35 * scale, oz)
    pts[13] = (ox - 0.15 * scale, oy - 0.3 * scale, oz)
    pts[17] = (ox - 0.4 * scale, oy - 0.25 * scale, oz)
    for i in range(21):
        if np.allclose(pts[i], 0.0) and i != WRIST:
            pts[i] = pts[WRIST] + np.array([0.01 * (i % 5), -0.01 * (i % 7), 0.0])
    # Re-apply palm
    pts[WRIST] = (ox, oy, oz)
    pts[5] = (ox + 0.4 * scale, oy - 0.3 * scale, oz)
    pts[9] = (ox + 0.1 * scale, oy - 0.35 * scale, oz)
    pts[13] = (ox - 0.15 * scale, oy - 0.3 * scale, oz)
    pts[17] = (ox - 0.4 * scale, oy - 0.25 * scale, oz)

    # Place tips so ‖thumb−index‖ / hand_scale == gap (requested ratio).
    from hand_interaction.pose import hand_scale

    # Temporary tips to measure scale
    pts[THUMB_TIP] = pts[WRIST] + np.array([0.2 * scale, -0.2 * scale, 0.0])
    pts[INDEX_TIP] = pts[5] + np.array([0.0, -0.4 * scale, 0.0])
    s = hand_scale(pts)
    # Midpoint near palm; separate along +x by gap * s
    mid = pts[WRIST] + np.array([0.15 * scale, -0.25 * scale, 0.0])
    half = 0.5 * gap * s
    pts[THUMB_TIP] = mid + np.array([-half, 0.0, 0.0])
    pts[INDEX_TIP] = mid + np.array([half, 0.0, 0.0])
    return pts


def _feed(detector: PinchDetector, gap_ratio: float, frames: int, hand_id: str = "Right"):
    events = []
    for _ in range(frames):
        events.extend(detector.update(_hand_with_gap(gap_ratio), hand_id))
    return events


def test_pinch_ratio_tracks_gap() -> None:
    pts = _hand_with_gap(0.2)
    assert pinch_ratio_3d(pts) == pytest.approx(0.2, rel=0.05)


def test_pinch_start_after_hold() -> None:
    det = PinchDetector(window=5)
    # Below ON, but need full window
    early = _feed(det, 0.2, 4)
    assert early == []
    assert det.phase is PinchPhase.OPEN
    started = _feed(det, 0.2, 1)
    assert len(started) == 1
    assert isinstance(started[0], PinchStart)
    assert started[0].type == "pinch.start"
    assert isinstance(started[0].position, Vector2)
    assert det.phase is PinchPhase.PINCHING


def test_pinch_end_on_release() -> None:
    det = PinchDetector(window=3)
    _feed(det, 0.2, 3)
    assert det.phase is PinchPhase.PINCHING
    # Well above OFF
    ended = _feed(det, 0.9, 3)
    assert any(isinstance(e, PinchEnd) for e in ended)
    assert det.phase is PinchPhase.OPEN


def test_no_duplicate_start_while_holding() -> None:
    det = PinchDetector(window=3)
    events = _feed(det, 0.2, 20)
    starts = [e for e in events if isinstance(e, PinchStart)]
    assert len(starts) == 1


def test_noise_around_threshold_does_not_flap() -> None:
    """Oscillate in the ON/OFF band: once pinching, stay until clear release."""
    det = PinchDetector(window=5)
    # Enter pinch
    _feed(det, PINCH_RATIO_ON - 0.05, 5)
    assert det.phase is PinchPhase.PINCHING

    mid = (PINCH_RATIO_ON + PINCH_RATIO_OFF) / 2
    events = []
    for i in range(30):
        # Alternate slightly around the mid band (still below OFF).
        ratio = mid + (0.04 if i % 2 == 0 else -0.04)
        # Keep below release floor
        ratio = min(ratio, PINCH_RATIO_OFF - 0.02)
        events.extend(det.update(_hand_with_gap(ratio), "Right"))

    ends = [e for e in events if isinstance(e, PinchEnd)]
    starts = [e for e in events if isinstance(e, PinchStart)]
    assert ends == []
    assert starts == []
    assert det.phase is PinchPhase.PINCHING


def test_force_end_emits_once() -> None:
    det = PinchDetector(window=3)
    _feed(det, 0.2, 3)
    first = det.force_end("Right")
    second = det.force_end("Right")
    assert len(first) == 1 and isinstance(first[0], PinchEnd)
    assert second == []
