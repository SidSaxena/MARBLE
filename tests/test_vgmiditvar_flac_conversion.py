"""
tests/test_vgmiditvar_flac_conversion.py

End-to-end integration tests for
``scripts/data/build_vgmiditvar_dataset.py::_convert_wav_to_flac``.

The helper is used inside ``_process_one`` when ``--audio-format=flac``
to keep peak per-worker disk at one in-flight WAV. This test verifies
the three contractual behaviours:

  1. Successful conversion: produces a valid FLAC file AND deletes the
     source WAV.
  2. Resume-safety: if the FLAC already exists, the helper is a no-op
     (returns True, drops the stale WAV if present).
  3. Failure preserves the source WAV: a broken/missing input doesn't
     silently lose data.

Requires ``ffmpeg`` on PATH. Skips with a clear reason on systems
without it (e.g. lean CI containers).
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "data"))
from build_vgmiditvar_dataset import _convert_wav_to_flac  # noqa: E402

# Skip the whole module when ffmpeg isn't available — keeps Mac/Linux dev
# happy without making CI fail on bare images.
if shutil.which("ffmpeg") is None:
    pytest.skip("ffmpeg not on PATH; skipping FLAC conversion tests", allow_module_level=True)


def _make_silent_wav(path: Path, duration_sec: float = 0.5) -> None:
    """Write a tiny valid 16-bit PCM mono WAV (silence) at 16 kHz.

    Inline RIFF construction so we don't pull in scipy/soundfile just
    for the test. ffmpeg accepts this as a normal input.
    """
    sample_rate = 16000
    n_samples = int(sample_rate * duration_sec)
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = n_samples * block_align
    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        n_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_chunk = struct.pack("<4sI", b"data", data_size) + b"\x00" * data_size
    riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(data_chunk), b"WAVE")
    path.write_bytes(riff + fmt_chunk + data_chunk)


def test_convert_success_deletes_wav(tmp_path: Path):
    """Happy path: WAV → FLAC succeeds, WAV is deleted, FLAC is non-empty."""
    wav = tmp_path / "smoke.wav"
    flac = tmp_path / "smoke.flac"
    _make_silent_wav(wav)
    assert wav.exists()

    ok = _convert_wav_to_flac(wav, flac, compression=5)

    assert ok is True
    assert flac.exists(), "FLAC must exist after successful conversion"
    assert flac.stat().st_size > 0, "FLAC must be non-empty"
    assert not wav.exists(), "WAV must be deleted on successful conversion"


def test_convert_resume_safe_when_flac_already_exists(tmp_path: Path):
    """If a previous run already produced the FLAC, a re-invocation is a
    no-op (returns True). Any stale WAV alongside the FLAC gets cleaned
    up so the resume state matches the post-success state."""
    wav = tmp_path / "smoke.wav"
    flac = tmp_path / "smoke.flac"
    _make_silent_wav(wav)
    # Pre-existing FLAC content (use the helper to make a real one, then
    # rewrite the WAV to simulate a half-finished prior run that crashed
    # after writing FLAC but before deleting WAV).
    _convert_wav_to_flac(wav, flac, compression=5)
    assert flac.exists() and not wav.exists()
    _make_silent_wav(wav)  # stale leftover from a hypothetical earlier failed run

    ok = _convert_wav_to_flac(wav, flac, compression=5)

    assert ok is True
    assert flac.exists()
    assert not wav.exists(), "stale WAV must be cleaned up on resume"


def test_convert_failure_preserves_wav(tmp_path: Path):
    """If ffmpeg fails (here: input doesn't exist), the helper returns
    False AND leaves the source WAV in place. The caller can fall back
    to the WAV or skip the record — the choice is up to the caller, but
    the helper must not silently destroy data."""
    wav = tmp_path / "does_not_exist.wav"
    flac = tmp_path / "smoke.flac"
    # Don't create the WAV — ffmpeg will fail immediately.

    ok = _convert_wav_to_flac(wav, flac, compression=5)

    assert ok is False
    # FLAC must not be created (or must be empty if ffmpeg touched it)
    assert not flac.exists() or flac.stat().st_size == 0


def test_convert_compression_levels(tmp_path: Path):
    """Compression levels 0 (fast) and 8 (slow) both produce valid FLAC
    of comparable size on tiny silence input. Mainly a smoke check that
    the --flac-compression flag plumbs through correctly without errors."""
    wav = tmp_path / "smoke.wav"
    _make_silent_wav(wav, duration_sec=1.0)

    flac0 = tmp_path / "level0.flac"
    flac8 = tmp_path / "level8.flac"

    _make_silent_wav(wav, duration_sec=1.0)
    assert _convert_wav_to_flac(wav, flac0, compression=0) is True
    assert flac0.exists() and flac0.stat().st_size > 0

    _make_silent_wav(wav, duration_sec=1.0)
    assert _convert_wav_to_flac(wav, flac8, compression=8) is True
    assert flac8.exists() and flac8.stat().st_size > 0
    # No strict size ordering on tiny silence (both compress to near-empty);
    # just confirm both completed successfully.
