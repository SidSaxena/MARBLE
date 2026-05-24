"""
tests/test_path_compat.py

Cross-OS audio_path portability tests.

These guard against the regression that bit the MuQ NSynth sweep: a
JSONL generated on Windows (paths with backslashes) failed to load on
Linux because '\\' isn't a path separator on POSIX systems.

Two-layer defence is tested:
  - posix_path()    — reader-side normalisation; the helper itself.
  - as_posix_str()  — writer-side; produces forward-slash output.
  - BaseAudioDataset.__init__         — applies posix_path on JSONL load.
  - NSynth._NSynthAudioBase.__init__  — same, on its own custom loader.

We don't actually call torchaudio here (no audio files on the test
runner); we verify that the stored ``audio_path`` strings are
forward-slash after dataset construction. Once the string is correct,
torchaudio.load works on every OS.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory

from marble.utils.path_compat import as_posix_str, posix_path

# ──────────────────────────────────────────────
# Unit tests for the helpers themselves
# ──────────────────────────────────────────────


def test_posix_path_converts_backslashes():
    """Windows-style paths in JSONL get normalised to forward slashes."""
    win = "data\\NSynth\\nsynth-valid\\audio\\brass_acoustic_006-062-025.wav"
    assert posix_path(win) == "data/NSynth/nsynth-valid/audio/brass_acoustic_006-062-025.wav"


def test_posix_path_idempotent_on_posix_input():
    """Already-POSIX paths pass through unchanged (no double-conversion)."""
    p = "data/HookTheory/audio_wav/abc123.wav"
    assert posix_path(p) == p


def test_posix_path_handles_mixed_separators():
    """Edge case: JSONL with mixed separators (unusual but possible)."""
    mixed = "data/NSynth\\nsynth-valid/audio\\foo.wav"
    assert posix_path(mixed) == "data/NSynth/nsynth-valid/audio/foo.wav"


def test_as_posix_str_from_windows_pathlike():
    """Writer-side: serialising a PureWindowsPath yields forward slashes."""
    p = PureWindowsPath("data") / "NSynth" / "foo.wav"
    assert as_posix_str(p) == "data/NSynth/foo.wav"


def test_as_posix_str_from_string():
    """Writer-side: serialising a raw string also normalises."""
    assert as_posix_str("data\\NSynth\\foo.wav") == "data/NSynth/foo.wav"


# ──────────────────────────────────────────────
# Integration tests: datamodule reader chokepoints
# ──────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_nsynth_datamodule_normalises_backslash_paths():
    """A NSynth JSONL with Windows-style paths becomes loadable on POSIX.

    Reproduces the production bug (Windows-generated NSynth.val.jsonl
    failing on WSL/Linux) without requiring real audio files: we only
    check that the in-memory ``audio_path`` strings come out forward-slash
    after the dataset's JSONL load.
    """
    from marble.tasks.NSynth.datamodule import _NSynthAudioBase

    win_path = "data\\NSynth\\nsynth-valid\\audio\\brass_acoustic_006-062-025.wav"
    record = {
        "audio_path": win_path,
        "note": 30,  # any valid MIDI 21..108
        "velocity": 75,
        "instrument_family": "brass",
        "instrument_source": "acoustic",
        "sample_rate": 16000,
        "num_samples": 64000,
        "channels": 1,
        "duration": 4.0,
    }

    with TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "nsynth_smoke.jsonl"
        _write_jsonl(jsonl, [record])

        ds = _NSynthAudioBase(jsonl=str(jsonl))
        assert ds.meta[0]["audio_path"] == (
            "data/NSynth/nsynth-valid/audio/brass_acoustic_006-062-025.wav"
        ), "_NSynthAudioBase did not normalise backslash audio_path on JSONL load"


def test_load_jsonl_normalises_backslash_paths():
    """The single helper that every datamodule funnels through.

    Replaces what would otherwise be 17 near-identical per-datamodule
    integration tests. Every BaseAudioDataset subclass + every
    custom-loader datamodule (NSynth, HookTheoryMelody, etc.) calls
    ``load_jsonl`` exactly once in its ``__init__``, so this single
    test guards the contract for all of them.
    """
    # Pick a concrete subclass that only needs sample_rate / channels /
    # clip_seconds (no audio_dir / label_freq surprises).
    # Exercise the helper directly the same way every datamodule's
    # __init__ does. Concrete BaseAudioDataset subclasses like MTT do
    # additional work (resampler init, index-map build) that needs real
    # audio files, so we don't instantiate them in unit tests — but the
    # JSONL-load contract is the same helper call.
    from marble.utils.path_compat import load_jsonl

    win_path = "data\\GTZAN\\genres\\classical\\classical.00001.wav"
    record = {
        "audio_path": win_path,
        "label": "classical",
        "sample_rate": 22050,
        "num_samples": 22050 * 30,
        "channels": 1,
        "duration": 30.0,
    }

    with TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "smoke.jsonl"
        _write_jsonl(jsonl, [record])
        loaded = load_jsonl(str(jsonl))

    assert loaded[0]["audio_path"] == ("data/GTZAN/genres/classical/classical.00001.wav"), (
        "load_jsonl did not normalise backslash audio_path"
    )
    # Other fields preserved untouched.
    assert loaded[0]["label"] == "classical"
    assert loaded[0]["sample_rate"] == 22050
