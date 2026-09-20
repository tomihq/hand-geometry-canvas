"""Handedness correction for mirrored selfie input."""

from __future__ import annotations

from hand_interaction.landmarker import anatomical_handedness


def test_anatomical_swap_when_mirrored() -> None:
    assert anatomical_handedness("Left", mirrored_input=True) == "Right"
    assert anatomical_handedness("Right", mirrored_input=True) == "Left"
    assert anatomical_handedness("Left-1", mirrored_input=True) == "Right-1"
    assert anatomical_handedness("Right-2", mirrored_input=True) == "Left-2"


def test_anatomical_unchanged_without_mirror() -> None:
    assert anatomical_handedness("Left", mirrored_input=False) == "Left"
    assert anatomical_handedness("Right", mirrored_input=False) == "Right"
    assert anatomical_handedness("Hand0", mirrored_input=True) == "Hand0"
