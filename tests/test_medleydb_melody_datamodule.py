"""Integration tests for the MedleyDBMelody dataset.

Verifies the dataset wires MedleyDB f0 CSVs into frame-level MIDI labels:
  * clip slicing from (sample_rate, num_samples) in the JSONL,
  * per-track CSV -> MIDI parsing at init,
  * get_targets() producing the right label vector for a given clip,
  * the frame-grid guard (label_freq * clip_seconds must be integer).

Self-contained: synthetic CSVs + JSONL in a tempdir, no real audio (the
audio file is never opened by __init__ or get_targets).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from marble.tasks.MedleyDBMelody.datamodule import (
    MedleyDBMelodyTrain,
    MedleyDBMelodyVal,
    _MedleyDBMelodyDataset,
)
from marble.tasks.MedleyDBMelody.melody_labels import MEDLEYDB_NATIVE_RATE


def _write_melody_csv(path: Path, freqs: list[float]) -> None:
    """Write a MedleyDB-format melody CSV: 'time_sec,freq_hz' per native frame."""
    hop = 1.0 / MEDLEYDB_NATIVE_RATE
    with open(path, "w") as fh:
        for i, fr in enumerate(freqs):
            fh.write(f"{i * hop},{fr}\n")


def _make_dataset(tmp: Path, freqs: list[float], num_samples: int, **kw):
    csv = tmp / "Track_MELODY2.csv"
    _write_melody_csv(csv, freqs)
    rec = {
        "audio_path": str(tmp / "Track_MIX.wav"),  # never opened in these tests
        "melody_csv": str(csv),
        "sample_rate": 44100,
        "num_samples": num_samples,
        "track": "Track",
    }
    # Write at the fold0/train path the template resolves to.
    template = str(tmp / "MedleyDBMelody.fold{fold}.{split}.jsonl")
    (tmp / "MedleyDBMelody.fold0.train.jsonl").write_text(json.dumps(rec) + "\n")
    params = dict(
        split="train",
        fold_idx=0,
        jsonl_template=template,
        sample_rate=44100,
        channels=1,
        clip_seconds=1.0,
        label_freq=25,
        min_clip_ratio=1.0,
    )
    params.update(kw)
    return _MedleyDBMelodyDataset(**params)


def test_index_map_slices_two_one_second_clips():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        n_native = int(round(2 * MEDLEYDB_NATIVE_RATE))
        ds = _make_dataset(tmp, [0.0] * n_native, num_samples=88200)
        assert len(ds) == 2


def test_get_targets_unvoiced_clip_is_all_minus_one():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        n_native = int(round(2 * MEDLEYDB_NATIVE_RATE))
        # first second unvoiced (0 Hz), second second voiced at 440 Hz
        cutoff = int(round(MEDLEYDB_NATIVE_RATE))
        freqs = [0.0] * cutoff + [440.0] * (n_native - cutoff)
        ds = _make_dataset(tmp, freqs, num_samples=88200)
        fi, si, osr, ocf = ds.index_map[0]
        labels = ds.get_targets(file_idx=fi, slice_idx=si, orig_sr=osr, orig_clip_frames=ocf)
        assert labels.dtype == torch.int64
        assert labels.shape == (25,)
        assert labels.tolist() == [-1] * 25


def test_get_targets_voiced_clip_is_all_midi_69():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        n_native = int(round(2 * MEDLEYDB_NATIVE_RATE))
        cutoff = int(round(MEDLEYDB_NATIVE_RATE))
        freqs = [0.0] * cutoff + [440.0] * (n_native - cutoff)
        ds = _make_dataset(tmp, freqs, num_samples=88200)
        fi, si, osr, ocf = ds.index_map[1]  # clip covering 1.0–2.0 s
        labels = ds.get_targets(file_idx=fi, slice_idx=si, orig_sr=osr, orig_clip_frames=ocf)
        assert labels.tolist() == [69] * 25


def test_frame_grid_guard_rejects_non_integer_product():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        n_native = int(round(2 * MEDLEYDB_NATIVE_RATE))
        with pytest.raises(ValueError, match="must be integer"):
            _make_dataset(tmp, [0.0] * n_native, num_samples=88200, clip_seconds=1.3)


def test_split_subclasses_hardcode_split_and_expose_fold_idx():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        n_native = int(round(2 * MEDLEYDB_NATIVE_RATE))
        csv = tmp / "Track_MELODY2.csv"
        _write_melody_csv(csv, [0.0] * n_native)
        rec = {
            "audio_path": str(tmp / "Track_MIX.wav"),
            "melody_csv": str(csv),
            "sample_rate": 44100,
            "num_samples": 88200,
            "track": "Track",
        }
        template = str(tmp / "MedleyDBMelody.fold{fold}.{split}.jsonl")
        (tmp / "MedleyDBMelody.fold2.train.jsonl").write_text(json.dumps(rec) + "\n")
        ds = MedleyDBMelodyTrain(
            fold_idx=2,
            jsonl_template=template,
            sample_rate=44100,
            channels=1,
            clip_seconds=1.0,
            label_freq=25,
            min_clip_ratio=1.0,
        )
        assert ds.split == "train"
        assert ds.fold_idx == 2
        # Val subclass resolves the {split} placeholder to "val"
        assert MedleyDBMelodyVal.__init__ is not MedleyDBMelodyTrain.__init__


def test_bad_csv_grid_is_rejected_at_construction():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # CSV with HALF the canonical hop → validate_native_grid must fire.
        csv = tmp / "Track_MELODY2.csv"
        bad_hop = 1.0 / (MEDLEYDB_NATIVE_RATE * 2.0)
        with open(csv, "w") as fh:
            for i in range(400):
                fh.write(f"{i * bad_hop},0.0\n")
        rec = {
            "audio_path": str(tmp / "Track_MIX.wav"),
            "melody_csv": str(csv),
            "sample_rate": 44100,
            "num_samples": 88200,
            "track": "Track",
        }
        template = str(tmp / "MedleyDBMelody.fold{fold}.{split}.jsonl")
        (tmp / "MedleyDBMelody.fold0.train.jsonl").write_text(json.dumps(rec) + "\n")
        with pytest.raises(ValueError, match="grid|hop"):
            MedleyDBMelodyTrain(
                fold_idx=0,
                jsonl_template=template,
                sample_rate=44100,
                channels=1,
                clip_seconds=1.0,
                label_freq=25,
                min_clip_ratio=1.0,
            )
