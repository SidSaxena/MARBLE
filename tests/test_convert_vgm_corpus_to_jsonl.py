"""
tests/test_convert_vgm_corpus_to_jsonl.py

TDD tests for scripts/data/convert_vgm_corpus_to_jsonl.py.

Builds a synthetic mini-corpus in a temp dir (3 short WAVs + a
manifest.json covering all 3 splits and all 3 loop-type labels),
invokes the converter, then asserts:
  - one JSONL file is written per split
  - row count matches the manifest
  - ``num_samples`` is read from the actual WAV (NOT from total_sec)
  - ``label`` field equals the manifest's loop_type
  - ``sample_rate`` is fixed at 24000
  - missing WAVs are skipped (warn to stderr) not aborted
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# soundfile is needed for the converter itself; guard the test if it's absent
pytest.importorskip("soundfile")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402  (after importorskip)

CONVERTER = Path(__file__).parent.parent / "scripts" / "data" / "convert_vgm_corpus_to_jsonl.py"

SAMPLE_RATE = 24000  # matches the fixed FIXED_SAMPLE_RATE in the converter


def _write_wav(path: Path, n_samples: int, sr: int = SAMPLE_RATE, channels: int = 1) -> None:
    """Write a minimal silent WAV via soundfile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((n_samples, channels) if channels > 1 else n_samples, dtype=np.float32)
    sf.write(str(path), data, samplerate=sr, subtype="PCM_16")


def _build_mini_corpus(root: Path) -> tuple[Path, Path]:
    """
    Create a synthetic corpus under ``root``:

        root/
          audio/
            train_tc.wav   (n_samples=48000)   → through_composed, split=train
            val_lfs.wav    (n_samples=72000)   → loop_from_start, split=val
            test_il.wav    (n_samples=96000)   → intro_loop, split=test
          manifest.json

    Returns (manifest_path, audio_root).
    """
    audio_dir = root / "audio"

    rows = [
        {
            "id": "tc_001",
            "split": "train",
            "loop_type": "through_composed",
            "audio_path": "audio/train_tc.wav",
            "total_sec": 1.0,  # deliberately wrong — converter must NOT use this
        },
        {
            "id": "lfs_001",
            "split": "val",
            "loop_type": "loop_from_start",
            "audio_path": "audio/val_lfs.wav",
            "total_sec": 1.5,
        },
        {
            "id": "il_001",
            "split": "test",
            "loop_type": "intro_loop",
            "audio_path": "audio/test_il.wav",
            "total_sec": 2.0,
        },
    ]

    _write_wav(root / "audio/train_tc.wav", n_samples=48000)
    _write_wav(root / "audio/val_lfs.wav", n_samples=72000)
    _write_wav(root / "audio/test_il.wav", n_samples=96000)

    manifest_path = root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(rows, f)

    return manifest_path, root


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run converter as subprocess (tests the actual CLI)
# ─────────────────────────────────────────────────────────────────────────────


def _run_converter(manifest: Path, audio_root: Path, out_dir: Path, name: str = "TestVGM") -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--manifest",
            str(manifest),
            "--audio-root",
            str(audio_root),
            "--out-dir",
            str(out_dir),
            "--name",
            name,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Converter exited with code {result.returncode}:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_per_split_files_exist(tmp_path):
    """Converter writes one JSONL file per split."""
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    for split in ("train", "val", "test"):
        assert (out_dir / f"VGMLoopStructure.{split}.wav.jsonl").exists(), f"Missing {split} JSONL"


def test_row_count_per_split(tmp_path):
    """Each split file has exactly one row (our mini-corpus has 1 row/split)."""
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGMLoopStructure.{split}.wav.jsonl")
        assert len(rows) == 1, f"Expected 1 row in {split}, got {len(rows)}"


def test_num_samples_from_file_not_total_sec(tmp_path):
    """
    num_samples must come from the WAV file, not from total_sec.
    Our mini-corpus sets total_sec to short values that do NOT match
    the actual WAV lengths (48000 / 72000 / 96000 samples at 24 kHz).
    """
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    expected = {"train": 48000, "val": 72000, "test": 96000}
    for split, expected_ns in expected.items():
        rows = _read_jsonl(out_dir / f"VGMLoopStructure.{split}.wav.jsonl")
        assert rows[0]["num_samples"] == expected_ns, (
            f"{split}: expected num_samples={expected_ns}, got {rows[0]['num_samples']}"
        )


def test_label_matches_loop_type(tmp_path):
    """label field equals the manifest's loop_type string."""
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    expected_labels = {"train": "through_composed", "val": "loop_from_start", "test": "intro_loop"}
    for split, expected_lbl in expected_labels.items():
        rows = _read_jsonl(out_dir / f"VGMLoopStructure.{split}.wav.jsonl")
        assert rows[0]["label"] == expected_lbl, (
            f"{split}: expected label={expected_lbl!r}, got {rows[0]['label']!r}"
        )


def test_fixed_sample_rate(tmp_path):
    """sample_rate is always 24000 regardless of the WAV's native rate."""
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGMLoopStructure.{split}.wav.jsonl")
        assert rows[0]["sample_rate"] == 24000, (
            f"{split}: expected sample_rate=24000, got {rows[0]['sample_rate']}"
        )


def test_duration_consistent_with_num_samples(tmp_path):
    """duration = num_samples / 24000 (no reliance on total_sec)."""
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGMLoopStructure.{split}.wav.jsonl")
        row = rows[0]
        expected_dur = row["num_samples"] / 24000
        assert abs(row["duration"] - expected_dur) < 1e-6, (
            f"{split}: duration mismatch: {row['duration']} vs {expected_dur}"
        )


def test_missing_wav_is_skipped_not_aborted(tmp_path):
    """A row whose WAV is absent is silently skipped; other rows are written."""
    corpus = tmp_path / "corpus"
    manifest, audio_root = _build_mini_corpus(corpus)

    # Delete one of the WAVs — the val split will be missing
    (corpus / "audio" / "val_lfs.wav").unlink()

    out_dir = tmp_path / "out"
    # Should NOT raise; converter must skip the missing file
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    # train and test should still have 1 row each
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.train.wav.jsonl")) == 1
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.test.wav.jsonl")) == 1
    # val should be empty (0 rows)
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.val.wav.jsonl")) == 0


def test_channels_and_bit_depth_fixed(tmp_path):
    """channels=1 and bit_depth=16 are hardcoded in every output row."""
    manifest, audio_root = _build_mini_corpus(tmp_path / "corpus")
    out_dir = tmp_path / "out"
    _run_converter(manifest, audio_root, out_dir, name="VGMLoopStructure")

    for split in ("train", "val", "test"):
        rows = _read_jsonl(out_dir / f"VGMLoopStructure.{split}.wav.jsonl")
        assert rows[0]["channels"] == 1
        assert rows[0]["bit_depth"] == 16


def test_non_24k_wav_is_skipped(tmp_path):
    """A WAV at a sample rate other than 24000 must be skipped with a warning."""
    corpus = tmp_path / "corpus"
    manifest, audio_root = _build_mini_corpus(corpus)

    # Overwrite the train WAV with a 44100 Hz file (same path, different SR)
    bad_wav = corpus / "audio" / "train_tc.wav"
    _write_wav(bad_wav, n_samples=48000, sr=44100)

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--manifest",
            str(manifest),
            "--audio-root",
            str(audio_root),
            "--out-dir",
            str(out_dir),
            "--name",
            "VGMLoopStructure",
        ],
        capture_output=True,
        text=True,
    )
    # Must not crash
    assert result.returncode == 0, f"Converter crashed:\n{result.stderr}"

    # train split must be empty (skipped)
    train_rows = _read_jsonl(out_dir / "VGMLoopStructure.train.wav.jsonl")
    assert len(train_rows) == 0, f"Expected 0 train rows (44100 Hz skipped), got {len(train_rows)}"

    # val and test must still have 1 row each
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.val.wav.jsonl")) == 1
    assert len(_read_jsonl(out_dir / "VGMLoopStructure.test.wav.jsonl")) == 1

    # Warning must appear on stderr
    assert (
        "44100" in result.stderr
        or "sample_rate" in result.stderr.lower()
        or "24000" in result.stderr
    ), f"Expected SR warning on stderr, got:\n{result.stderr}"
