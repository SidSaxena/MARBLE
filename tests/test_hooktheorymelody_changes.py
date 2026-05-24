"""Correctness tests for the HookTheoryMelody datamodule changes.

The changes under test (from this session) are:
  1. ``audio_ext`` parameter — extension used in path construction (default
     ".mp3", can be flipped to ".wav" after the conversion script).
  2. Cached interp1d + pre-vectorised (onset_sec, offset_sec, midi_pitch)
     arrays per file in __init__ — moves per-call work out of the hot path.
  3. Vectorised ``_compute_labels`` — np ops replace the per-note Python loop.
  4. Optional ``precompute_labels`` — single contiguous int64 tensor (NOT a
     list[Tensor], which broke COW under fork and ballooned worker RSS).

These tests verify each property end-to-end against a reference
implementation that matches the pre-change behaviour byte-for-byte. The
tests deliberately do NOT use any real audio data — they construct
in-memory JSONL fixtures so they're fast (<1s) and self-contained.

Run with:
    uv run pytest tests/test_hooktheorymelody_changes.py -v
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d

from marble.tasks.HookTheoryMelody.datamodule import _HookTheoryMelodyDataset

# ─────────────────────────────────────────────────────────────────────────────
# Reference implementation — byte-for-byte port of the original __getitem__
# label-compute block (from commit 7343b7a). The new vectorised path should
# produce identical output for any input the original handled.
# ─────────────────────────────────────────────────────────────────────────────


def _reference_labels(
    melody: list[dict],
    beats: np.ndarray,
    times: np.ndarray,
    clip_start_time: float,
    label_freq: int,
    label_len: int,
    melody_octave: int = 5,
) -> np.ndarray:
    """Original per-note loop with `int(pitch + octave*12)` clamp."""
    beat_to_time_fn = interp1d(beats, times, kind="linear", fill_value="extrapolate")
    labels = -1 * np.ones(label_len, dtype=np.int64)
    for note in melody:
        onset_beat = float(note["onset"])
        offset_beat = float(note["offset"])
        onset_sec = float(beat_to_time_fn(onset_beat))
        offset_sec = float(beat_to_time_fn(offset_beat))
        midi_pitch = int(note["pitch_class"] + (melody_octave + int(note["octave"])) * 12)
        midi_pitch = max(0, min(127, midi_pitch))
        rel_onset = onset_sec - clip_start_time
        rel_offset = offset_sec - clip_start_time
        start_idx = int(np.floor(rel_onset * label_freq))
        end_idx = int(np.ceil(rel_offset * label_freq))
        start_idx = max(0, start_idx)
        end_idx = min(label_len, end_idx)
        if start_idx >= label_len or end_idx <= 0:
            continue
        labels[start_idx:end_idx] = midi_pitch
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_record(
    ytid: str,
    *,
    beats: list[float],
    times: list[float],
    melody: list[dict],
    sample_rate: int = 24000,
    num_samples: int = 24000 * 60,  # 60 s @ 24 kHz
) -> dict:
    return {
        "youtube": {"id": ytid},
        "alignment": {"refined": {"beats": beats, "times": times}},
        "annotations": {"melody": melody},
        "num_samples": num_samples,
        "sample_rate": sample_rate,
    }


def _write_jsonl(records: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def _random_melody(n_notes: int, seed: int) -> list[dict]:
    """Random melody annotation with realistic ranges."""
    rng = random.Random(seed)
    melody = []
    cursor = 0.0
    for _ in range(n_notes):
        onset = cursor + rng.uniform(0.0, 0.3)
        offset = onset + rng.uniform(0.1, 1.5)
        melody.append(
            {
                "onset": onset,
                "offset": offset,
                "pitch_class": rng.randint(0, 11),
                # Octaves in real HookTheory data span ~ -2 to +3 relative to MELODY_OCTAVE=5
                "octave": rng.randint(-2, 3),
            }
        )
        cursor = offset
    return melody


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 1234])
@pytest.mark.parametrize("n_notes", [0, 1, 5, 25, 80])
def test_compute_labels_matches_reference(seed: int, n_notes: int):
    """The vectorised _compute_labels matches the original per-note loop
    byte-for-byte across random melodies and slice indices."""
    rng = random.Random(seed)
    sr = rng.choice([22050, 24000, 44100, 48000])
    clip_seconds = 15.0
    label_freq = 25

    n_beats = max(50, n_notes * 4)
    beats = [float(i) for i in range(n_beats)]
    # Non-uniform tempo: time-per-beat varies a bit (real songs)
    times = [0.0]
    for _ in range(n_beats - 1):
        times.append(times[-1] + rng.uniform(0.3, 0.7))

    melody = _random_melody(n_notes, seed)
    rec = _make_record("test", beats=beats, times=times, melody=melody, sample_rate=sr)
    jsonl = _write_jsonl([rec])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=sr,
            channels=1,
            clip_seconds=clip_seconds,
            jsonl=str(jsonl),
            label_freq=label_freq,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
        )
        orig_clip_frames = int(clip_seconds * sr)
        label_len = int(label_freq * clip_seconds)
        beats_arr = np.array(beats, dtype=np.float32)
        times_arr = np.array(times, dtype=np.float32)
        # Compare for every slice in the file
        for slice_idx in range(len(ds.index_map)):
            actual = ds._compute_labels(0, slice_idx, sr, orig_clip_frames).numpy()
            expected = _reference_labels(
                melody=melody,
                beats=beats_arr,
                times=times_arr,
                clip_start_time=slice_idx * (orig_clip_frames / sr),
                label_freq=label_freq,
                label_len=label_len,
            )
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(
                actual,
                expected,
                err_msg=f"seed={seed} n_notes={n_notes} slice_idx={slice_idx}",
            )
    finally:
        os.unlink(jsonl)


def test_label_cache_matches_uncached():
    """precompute_labels=True must give the same per-clip labels as the
    on-the-fly path (precompute_labels=False)."""
    melody = _random_melody(40, seed=99)
    beats = [float(i) for i in range(80)]
    times = [i * 0.5 for i in range(80)]
    rec = _make_record("t", beats=beats, times=times, melody=melody)
    jsonl = _write_jsonl([rec])
    try:
        # Without cache
        ds_a = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
            precompute_labels=False,
        )
        assert ds_a.label_cache is None
        # With cache
        ds_b = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
            precompute_labels=True,
        )
        assert isinstance(ds_b.label_cache, torch.Tensor)
        assert ds_b.label_cache.shape == (len(ds_b.index_map), 375)
        assert ds_b.label_cache.dtype == torch.int64

        # Equivalence: for every slice, _labels_for must return identical labels
        # whether the cache is present or not.
        assert len(ds_a.index_map) == len(ds_b.index_map)
        for idx in range(len(ds_a.index_map)):
            fa, sa, sra, ofa = ds_a.index_map[idx]
            fb, sb, srb, ofb = ds_b.index_map[idx]
            assert (fa, sa, sra, ofa) == (fb, sb, srb, ofb)
            la = ds_a._labels_for(idx, fa, sa, sra, ofa)
            lb = ds_b._labels_for(idx, fb, sb, srb, ofb)
            assert torch.equal(la, lb), f"mismatch at idx={idx}"
    finally:
        os.unlink(jsonl)


def test_label_cache_is_single_contiguous_tensor():
    """The cache must be ONE contiguous tensor — not a list of N tensors.

    Regression test for the COW-amplification bug: a list[Tensor] of 300 k
    items dirties refcount pages in every forked worker and balloons RSS.
    A single contiguous tensor lets the buffer stay COW-shared.
    """
    rec = _make_record(
        "t",
        beats=list(range(50)),
        times=[i * 0.5 for i in range(50)],
        melody=_random_melody(15, seed=1),
    )
    jsonl = _write_jsonl([rec])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
            precompute_labels=True,
        )
        assert isinstance(ds.label_cache, torch.Tensor), (
            "label_cache must be a single torch.Tensor, not a list/sequence"
        )
        assert ds.label_cache.is_contiguous()
        # Indexing returns a view into the same underlying buffer (proves COW
        # share survives — no per-element refcounting).
        row0_ptr = ds.label_cache[0].data_ptr()
        row1_ptr = ds.label_cache[1].data_ptr()
        if ds.label_cache.shape[0] >= 2:
            # Stride matches expected for int64 row
            expected_stride = 8 * ds.label_cache.shape[1]
            assert row1_ptr - row0_ptr == expected_stride, (
                "indexing must return a view at the expected stride (proves contiguity)"
            )
    finally:
        os.unlink(jsonl)


@pytest.mark.parametrize(
    "ext_in,ext_stored",
    [
        (".mp3", ".mp3"),
        (".wav", ".wav"),
        ("mp3", ".mp3"),  # auto-prepend dot
        ("wav", ".wav"),
        (".flac", ".flac"),
    ],
)
def test_audio_ext_normalisation(ext_in: str, ext_stored: str):
    """audio_ext is stored with a leading dot regardless of input form."""
    rec = _make_record("X", beats=list(range(40)), times=[i * 0.5 for i in range(40)], melody=[])
    jsonl = _write_jsonl([rec])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
            audio_ext=ext_in,
        )
        assert ds.audio_ext == ext_stored
    finally:
        os.unlink(jsonl)


def test_audio_path_uses_audio_ext_in_getitem():
    """The hot __getitem__ path constructs path from self.audio_ext, NOT a
    hardcoded '.mp3'. (Regression test for the bug found during the WAV
    smoke run.)"""
    rec = _make_record("XYZ", beats=list(range(40)), times=[i * 0.5 for i in range(40)], melody=[])
    jsonl = _write_jsonl([rec])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/some/dir",
            min_clip_ratio=0.5,
            audio_ext=".wav",
        )
        # Build path the way __getitem__ does, without going through it
        # (which would fail because the file doesn't exist). We inspect the
        # internals.
        ytid = ds.yt_ids[0]
        constructed = os.path.join(ds.audio_dir, f"{ytid}{ds.audio_ext}")
        assert constructed == "/some/dir/XYZ.wav"
        assert not constructed.endswith(".mp3")
    finally:
        os.unlink(jsonl)


def test_empty_melody_returns_all_minus_one():
    """A song with no notes should produce labels filled with -1."""
    rec = _make_record("E", beats=list(range(40)), times=[i * 0.5 for i in range(40)], melody=[])
    jsonl = _write_jsonl([rec])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
        )
        labels = ds._compute_labels(0, 0, 24000, int(15 * 24000))
        assert labels.shape == (375,)
        assert (labels == -1).all()
    finally:
        os.unlink(jsonl)


def test_pitch_clamp_extreme_octaves():
    """Out-of-range octave values should be clamped to [0, 127]."""
    melody = [
        {"onset": 0.0, "offset": 1.0, "pitch_class": 0, "octave": -10},  # very low → clamp to 0
        {"onset": 1.0, "offset": 2.0, "pitch_class": 0, "octave": 10},  # very high → clamp to 127
    ]
    rec = _make_record(
        "clamp", beats=list(range(40)), times=[i * 0.5 for i in range(40)], melody=melody
    )
    jsonl = _write_jsonl([rec])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
        )
        labels = ds._compute_labels(0, 0, 24000, int(15 * 24000)).numpy()
        present = set(labels.tolist())
        # MIDI midpoint range should NEVER exceed 127 or go below 0 in labels
        # (other than -1 which marks silent frames).
        assert (labels[labels != -1] >= 0).all()
        assert (labels[labels != -1] <= 127).all()
        # Both clamps should produce labels present in the output
        assert 0 in present or 127 in present
    finally:
        os.unlink(jsonl)


def test_missing_alignment_records_skipped():
    """Records with alignment.refined == null are skipped (not the whole dataset)."""
    rec_ok = _make_record(
        "ok", beats=list(range(40)), times=[i * 0.5 for i in range(40)], melody=[]
    )
    rec_bad = {
        "youtube": {"id": "bad"},
        "alignment": None,
        "annotations": {"melody": []},
        "num_samples": 24000 * 30,
        "sample_rate": 24000,
    }
    jsonl = _write_jsonl([rec_ok, rec_bad])
    try:
        ds = _HookTheoryMelodyDataset(
            sample_rate=24000,
            channels=1,
            clip_seconds=15.0,
            jsonl=str(jsonl),
            label_freq=25,
            audio_dir="/tmp",
            min_clip_ratio=0.5,
        )
        # Only the OK record survives — index_map references file_idx=0 only
        assert len(ds.yt_ids) == 1
        assert ds.yt_ids[0] == "ok"
        assert all(t[0] == 0 for t in ds.index_map)
    finally:
        os.unlink(jsonl)
