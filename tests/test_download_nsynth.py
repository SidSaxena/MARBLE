"""
tests/test_download_nsynth.py

Guards two bugs in ``scripts/data/download_nsynth.py``:

1. **The "note vs pitch" schema bug** (caused 0% test accuracy on every
   NSynth probe pre-fix). NSynth's ``examples.json`` uses ``note`` for a
   globally unique sample ID and ``pitch`` for the MIDI pitch — the
   reverse of what the field name suggests. Writing ``meta["note"]`` to
   the JSONL meant ~76 records survived the 21–108 range filter (only
   IDs that coincidentally landed in that range), and the downstream
   datamodule treated unique sample IDs as MIDI pitches. Result: tiny
   train set + garbage labels = uniformly 0% val/test acc.

2. **Partial-extraction safety check** — hard-fail when <10% of input
   records survive, with an opt-out (``--allow-partial``) for smoke
   subsets. Catches the failure mode in seconds instead of waiting
   for a full training run to crash on missing data.

Tests build a tiny fake ``nsynth-train/`` directory tree (examples.json
+ matching .wav files) and exercise ``generate_jsonl`` directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Add scripts/ to sys.path so we can import the script as a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "data"))
import download_nsynth  # noqa: E402


def _fake_split_dir(tmp: Path, examples: dict, audio_files: list[str]) -> Path:
    """Create a fake nsynth-<split>/ dir with examples.json + audio/ contents."""
    split_dir = tmp / "nsynth-train"
    (split_dir / "audio").mkdir(parents=True)
    with open(split_dir / "examples.json", "w") as f:
        json.dump(examples, f)
    # Empty placeholder .wav files — generate_jsonl only does .exists(),
    # not audio decode, so file content is irrelevant here.
    for name in audio_files:
        (split_dir / "audio" / name).touch()
    return split_dir


def test_jsonl_field_is_pitch_not_note():
    """Critical: written JSONL must use ``pitch`` (the MIDI pitch) — not
    ``note`` (the unique sample ID). This is the actual regression that
    caused 0% test accuracy on every NSynth probe."""
    examples = {
        "guitar_acoustic_001-082-050": {
            "note": 16629,  # unique sample ID — NOT MIDI pitch
            "pitch": 82,  # MIDI pitch — what label = pitch - 21 needs
            "velocity": 50,
            "instrument_family_str": "guitar",
            "instrument_source_str": "acoustic",
        },
    }
    with TemporaryDirectory() as t:
        tmp = Path(t)
        split_dir = _fake_split_dir(tmp, examples, ["guitar_acoustic_001-082-050.wav"])
        out = tmp / "NSynth.train.jsonl"

        n = download_nsynth.generate_jsonl(split_dir, out)
        assert n == 1
        rec = json.loads(out.read_text().strip())
        assert rec["pitch"] == 82, "JSONL must store MIDI pitch (82), not unique sample ID (16629)"
        assert "note" not in rec, (
            "JSONL must NOT have a 'note' field — it was renamed to 'pitch' "
            "because the source field of the same name is a unique ID, not MIDI pitch."
        )


def test_pitch_range_filter_uses_pitch_not_note():
    """Pre-fix the filter ``21 <= meta['note'] <= 108`` accidentally passed
    records whose unique sample ID happened to land in 21..108 (irrelevant
    to MIDI pitch). The fix filters on ``meta['pitch']``."""
    examples = {
        # Pitch 50 (in range) but note ID 999 (out of range) — must KEEP
        "in_range_keep": {
            "note": 999,
            "pitch": 50,
            "velocity": 50,
            "instrument_family_str": "bass",
            "instrument_source_str": "acoustic",
        },
        # Pitch 10 (out of range) but note ID 75 (in range pre-fix) — must DROP
        "out_of_range_drop": {
            "note": 75,
            "pitch": 10,
            "velocity": 50,
            "instrument_family_str": "bass",
            "instrument_source_str": "acoustic",
        },
        # Pitch 109 (just over) — must DROP
        "just_over_drop": {
            "note": 50,
            "pitch": 109,
            "velocity": 50,
            "instrument_family_str": "bass",
            "instrument_source_str": "acoustic",
        },
    }
    with TemporaryDirectory() as t:
        tmp = Path(t)
        split_dir = _fake_split_dir(
            tmp,
            examples,
            [f"{k}.wav" for k in examples],
        )
        out = tmp / "NSynth.train.jsonl"

        n = download_nsynth.generate_jsonl(split_dir, out, allow_partial=True)
        assert n == 1, f"expected exactly 1 in-range record, wrote {n}"
        rec = json.loads(out.read_text().strip())
        assert rec["pitch"] == 50


def test_emits_posix_paths_on_any_os():
    """``Path.as_posix()`` guarantees forward slashes regardless of host OS,
    so JSONLs generated on Windows are readable on Linux/Modal verbatim.
    Tested in tandem with the path_compat reader fix."""
    examples = {
        "guitar_acoustic_001-082-050": {
            "note": 1,
            "pitch": 82,
            "velocity": 50,
            "instrument_family_str": "guitar",
            "instrument_source_str": "acoustic",
        },
    }
    with TemporaryDirectory() as t:
        tmp = Path(t)
        split_dir = _fake_split_dir(tmp, examples, ["guitar_acoustic_001-082-050.wav"])
        out = tmp / "out.jsonl"
        download_nsynth.generate_jsonl(split_dir, out)
        rec = json.loads(out.read_text().strip())
        assert "\\" not in rec["audio_path"], f"audio_path must be POSIX, got {rec['audio_path']!r}"


def test_safety_hard_fails_when_too_many_skipped():
    """If <10% of examples.json records survive the filters, hard-fail with a
    clear error pointing at the most likely causes. This is the safety net
    that would have surfaced the note-vs-pitch bug in seconds."""
    # 100 records, only 1 has audio on disk — 99% skip rate → hard fail
    examples = {
        f"key_{i:03d}": {
            "note": i,
            "pitch": 60,
            "velocity": 50,
            "instrument_family_str": "x",
            "instrument_source_str": "y",
        }
        for i in range(100)
    }
    with TemporaryDirectory() as t:
        tmp = Path(t)
        split_dir = _fake_split_dir(tmp, examples, ["key_000.wav"])  # only 1 audio file
        out = tmp / "out.jsonl"

        with pytest.raises(RuntimeError, match=r"wrote only 1 of 100 records"):
            download_nsynth.generate_jsonl(split_dir, out)


def test_safety_bypassable_with_allow_partial():
    """``--allow-partial`` opts out of the safety check so smoke tests can
    legitimately produce tiny JSONLs from partial audio."""
    examples = {
        f"key_{i:03d}": {
            "note": i,
            "pitch": 60,
            "velocity": 50,
            "instrument_family_str": "x",
            "instrument_source_str": "y",
        }
        for i in range(100)
    }
    with TemporaryDirectory() as t:
        tmp = Path(t)
        split_dir = _fake_split_dir(tmp, examples, ["key_000.wav"])
        out = tmp / "out.jsonl"

        # Should NOT raise with allow_partial=True
        n = download_nsynth.generate_jsonl(split_dir, out, allow_partial=True)
        assert n == 1
