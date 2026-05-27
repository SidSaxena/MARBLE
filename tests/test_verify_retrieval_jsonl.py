"""
tests/test_verify_retrieval_jsonl.py

Smoke + failure-mode tests for ``scripts/data/verify_retrieval_jsonl.py``.

We don't have real audio fixtures in the test environment, so we
generate three short FLAC files at known sample rates via
``soundfile``, write a small JSONL describing them, and run the script
as a subprocess. Tests cover:
  - PASS: every record matches disk.
  - FAIL: one record's audio missing on disk.
  - FAIL: one record's JSONL sample_rate disagrees with the file.
  - FAIL: --target-sr coverage drops below threshold.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "verify_retrieval_jsonl.py"

_FFPROBE = shutil.which("ffprobe")
ffprobe_required = pytest.mark.skipif(
    _FFPROBE is None,
    reason="ffprobe binary not available in this test environment",
)


def _write_flac(path: Path, *, sr: int, duration: float = 0.5, channels: int = 1) -> None:
    """Write a tiny FLAC at the requested rate so ffprobe returns
    deterministic metadata."""
    n = int(sr * duration)
    if channels == 1:
        arr = (np.random.RandomState(0).randn(n) * 0.1).astype("float32")
    else:
        arr = (np.random.RandomState(0).randn(n, channels) * 0.1).astype("float32")
    sf.write(str(path), arr, sr, subtype="PCM_16", format="FLAC")


def _run(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return p.returncode, (p.stdout + p.stderr)


@ffprobe_required
def test_pass_when_all_records_match(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    paths = []
    for i in range(3):
        p = audio_dir / f"clip_{i}.flac"
        _write_flac(p, sr=24000, duration=0.5)
        paths.append(p)

    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for i, p in enumerate(paths):
            n_samples = int(24000 * 0.5)
            rec = {
                "audio_path": p.as_posix(),
                "work_id": i,
                "sample_rate": 24000,
                "num_samples": n_samples,
                "channels": 1,
            }
            f.write(json.dumps(rec) + "\n")

    rc, out = _run(["--jsonl", str(jsonl), "--target-sr", "24000"])
    assert rc == 0, out
    assert "PASS" in out


@ffprobe_required
def test_fail_when_audio_missing(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    p = audio_dir / "clip_0.flac"
    _write_flac(p, sr=24000)

    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        # Record 0: file exists.
        rec_ok = {
            "audio_path": p.as_posix(),
            "work_id": 0,
            "sample_rate": 24000,
            "num_samples": int(24000 * 0.5),
            "channels": 1,
        }
        f.write(json.dumps(rec_ok) + "\n")
        # Record 1: claims a file that doesn't exist.
        rec_missing = dict(rec_ok)
        rec_missing["audio_path"] = (audio_dir / "ghost.flac").as_posix()
        rec_missing["work_id"] = 1
        f.write(json.dumps(rec_missing) + "\n")

    rc, out = _run(["--jsonl", str(jsonl)])
    assert rc == 1
    assert "missing on disk" in out


@ffprobe_required
def test_fail_when_sample_rate_disagrees(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    p = audio_dir / "clip_0.flac"
    _write_flac(p, sr=24000)

    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        # JSONL claims 44100 but the file is 24000.
        rec = {
            "audio_path": p.as_posix(),
            "work_id": 0,
            "sample_rate": 44100,
            "num_samples": int(44100 * 0.5),
            "channels": 1,
        }
        f.write(json.dumps(rec) + "\n")

    rc, out = _run(["--jsonl", str(jsonl)])
    assert rc == 1
    assert "sample_rate" in out and "mismatch" in out


@ffprobe_required
def test_target_sr_coverage_threshold(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # 1 of 2 at target rate → 50% coverage.
    p0 = audio_dir / "clip_0.flac"
    p1 = audio_dir / "clip_1.flac"
    _write_flac(p0, sr=24000)
    _write_flac(p1, sr=48000)

    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        for p, sr in [(p0, 24000), (p1, 48000)]:
            rec = {
                "audio_path": p.as_posix(),
                "work_id": 0,
                "sample_rate": sr,
                "num_samples": int(sr * 0.5),
                "channels": 1,
            }
            f.write(json.dumps(rec) + "\n")

    # Strict default (min_frac=1.0): 50% coverage → FAIL.
    rc, out = _run(["--jsonl", str(jsonl), "--target-sr", "24000"])
    assert rc == 1
    assert "target-sr coverage" in out

    # Loose (min_frac=0.4): 50% coverage → PASS on this axis (but
    # the file at 48 kHz with sample_rate=48000 still passes the
    # per-record sr-match check, so total PASS).
    rc, out = _run(["--jsonl", str(jsonl), "--target-sr", "24000", "--target-sr-min-frac", "0.4"])
    assert rc == 0, out
    assert "PASS" in out
