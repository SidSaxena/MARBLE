"""Correctness tests for the MedleyDBMelody frame-label core.

MedleyDB ships melody annotations as per-frame f0 in Hz on a uniform grid of
hop = 256/44100 s (~172.27 fps), with unvoiced frames encoded as 0.0 Hz. The
probing task needs, per clip, a frame-level MIDI-pitch label vector at the
encoder's token rate (``label_freq``), with -1 for unvoiced/out-of-range.

These tests pin the two pure functions that do that conversion:
  * ``f0_to_midi``       : Hz array -> MIDI int array (0 Hz -> -1, clamp 0..127)
  * ``clip_frame_labels``: nearest-sample a track's per-native-frame MIDI onto
    a clip's ``label_freq`` grid; frames past the annotation -> -1.

No torch / audio / disk — pure numpy, hand-computed expectations.

Run:
    uv run pytest tests/test_medleydb_melody_labels.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from marble.tasks.MedleyDBMelody.melody_labels import (
    MEDLEYDB_NATIVE_RATE,
    clip_frame_labels,
    f0_to_midi,
    validate_native_grid,
)

# ── f0_to_midi ────────────────────────────────────────────────────────────────


def test_native_rate_is_256_over_44100():
    assert MEDLEYDB_NATIVE_RATE == 44100 / 256


def test_f0_to_midi_a440_is_69():
    out = f0_to_midi(np.array([440.0]))
    assert out.tolist() == [69]


def test_f0_to_midi_octave_up_adds_12():
    out = f0_to_midi(np.array([880.0]))
    assert out.tolist() == [81]


def test_f0_to_midi_middle_c_is_60():
    # C4 = 261.6256 Hz
    out = f0_to_midi(np.array([261.6256]))
    assert out.tolist() == [60]


def test_f0_to_midi_unvoiced_zero_is_minus_one():
    out = f0_to_midi(np.array([0.0]))
    assert out.tolist() == [-1]


def test_f0_to_midi_clamps_above_127():
    out = f0_to_midi(np.array([20000.0]))  # ~MIDI 135 -> clamp 127
    assert out.tolist() == [127]


def test_f0_to_midi_clamps_below_0_for_voiced():
    out = f0_to_midi(np.array([1.0]))  # voiced but absurdly low -> clamp 0
    assert out.tolist() == [0]


def test_f0_to_midi_mixed_array_dtype_and_values():
    out = f0_to_midi(np.array([0.0, 440.0, 0.0, 880.0]))
    assert out.dtype == np.int64
    assert out.tolist() == [-1, 69, -1, 81]


# ── clip_frame_labels ─────────────────────────────────────────────────────────


def test_clip_frame_labels_constant_pitch_from_t0():
    track = np.full(1000, 60, dtype=np.int64)
    out = clip_frame_labels(track, clip_start_time=0.0, label_freq=25, label_len=10)
    assert out.tolist() == [60] * 10


def test_clip_frame_labels_nearest_index_math():
    # track_midi[i] == i so the output reveals exactly which native index each
    # output frame sampled. Frame centers = (k+0.5)/label_freq seconds.
    # idx = round(t * 44100/256). For label_freq=25:
    #   k=0 -> 0.02 s -> round(3.445) = 3
    #   k=1 -> 0.06 s -> round(10.336) = 10
    #   k=2 -> 0.10 s -> round(17.227) = 17
    track = np.arange(1000, dtype=np.int64)
    out = clip_frame_labels(track, clip_start_time=0.0, label_freq=25, label_len=3)
    assert out.tolist() == [3, 10, 17]


def test_clip_frame_labels_start_offset_shifts_window():
    # clip_start_time=1.0 s, frame 0 center = 1.02 s -> idx round(1.02*172.265625)=176
    track = np.arange(2000, dtype=np.int64)
    out = clip_frame_labels(track, clip_start_time=1.0, label_freq=25, label_len=1)
    assert out.tolist() == [176]


def test_clip_frame_labels_past_annotation_is_minus_one():
    # track covers only indices 0..99 (~0.58 s). A 25 Hz, 25-frame clip reaches
    # ~0.98 s, so the tail frames fall past the annotation -> -1.
    track = np.full(100, 55, dtype=np.int64)
    out = clip_frame_labels(track, clip_start_time=0.0, label_freq=25, label_len=25)
    assert out[0] == 55
    assert out[-1] == -1
    assert (out == -1).any()


def test_clip_frame_labels_propagates_unvoiced_minus_one():
    track = np.full(1000, -1, dtype=np.int64)
    out = clip_frame_labels(track, clip_start_time=0.0, label_freq=25, label_len=5)
    assert out.tolist() == [-1] * 5


def test_clip_frame_labels_length_and_dtype():
    track = np.arange(5000, dtype=np.int64)
    out = clip_frame_labels(track, clip_start_time=0.0, label_freq=75, label_len=1125)
    assert out.shape == (1125,)
    assert out.dtype == np.int64


# ── validate_native_grid ──────────────────────────────────────────────────────


def _canonical_times(n: int) -> np.ndarray:
    """A clean MedleyDB time column: t_i = i * 256/44100, starting at 0."""
    return np.arange(n) / MEDLEYDB_NATIVE_RATE


def test_validate_native_grid_accepts_canonical():
    validate_native_grid(_canonical_times(10000), track="ok")  # no raise


def test_validate_native_grid_accepts_short_and_empty():
    validate_native_grid(np.zeros(0), track="empty")  # too short to validate → no raise
    validate_native_grid(np.array([0.0]), track="one")


def test_validate_native_grid_rejects_nonzero_start():
    t = _canonical_times(10000) + 0.5  # shifted start
    with pytest.raises(ValueError, match="start"):
        validate_native_grid(t, track="shifted")


def test_validate_native_grid_rejects_wrong_hop():
    t = np.arange(10000) / (MEDLEYDB_NATIVE_RATE * 2.0)  # half the expected hop
    with pytest.raises(ValueError, match="hop|grid|rate"):
        validate_native_grid(t, track="wronghop")


def test_validate_native_grid_rejects_dropped_row():
    # canonical grid but with one interior row removed → row count no longer
    # consistent with the end timestamp (end time same, one fewer row).
    t = _canonical_times(10000)
    t = np.delete(t, 5000)
    with pytest.raises(ValueError, match="row|count|grid"):
        validate_native_grid(t, track="dropped")
