"""Tests for scripts/data/rewrite_jsonl_audio_paths.py.

The script is used to flip task JSONLs from MP3-pointing to WAV-pointing
when their datamodules store the audio path as a full string (rather than
constructing it from id+ext like HookTheoryMelody). HookTheoryKey and
HookTheoryStructure use this; NSynth's setup script does it inline.

Tests below verify the rewrite is correct, atomic (no half-written files
on failure), schema-preserving (all other JSONL fields untouched), and
idempotent (re-running produces the same output).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path("scripts/data/rewrite_jsonl_audio_paths.py").resolve()


def _write_jsonl(records: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in records:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _run_rewrite(jsonl: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--jsonl", str(jsonl), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_rewrites_dir_and_ext():
    """Basic case: HookTheoryKey-style flip from clip MP3 paths to WAV."""
    recs = [
        {"audio_path": "data/HookTheory/hooktheory_clips/abc.mp3", "label": "C:maj"},
        {"audio_path": "data/HookTheory/hooktheory_clips/def.mp3", "label": "G:min"},
    ]
    jsonl = _write_jsonl(recs)
    _run_rewrite(
        jsonl,
        "--from-dir",
        "hooktheory_clips",
        "--to-dir",
        "hooktheory_clips_wav",
        "--from-ext",
        ".mp3",
        "--to-ext",
        ".wav",
    )
    out = jsonl.with_name(jsonl.stem + ".wav.jsonl")
    assert out.exists(), f"output file not written: {out}"
    new_recs = _read_jsonl(out)
    assert len(new_recs) == 2
    assert new_recs[0]["audio_path"] == "data/HookTheory/hooktheory_clips_wav/abc.wav"
    assert new_recs[1]["audio_path"] == "data/HookTheory/hooktheory_clips_wav/def.wav"
    # Labels (and any other fields) MUST be preserved byte-for-byte
    assert new_recs[0]["label"] == "C:maj"
    assert new_recs[1]["label"] == "G:min"
    jsonl.unlink()
    out.unlink()


def test_preserves_other_fields():
    """Records with many fields: only audio_path changes; everything else
    is identical (including nested dicts and numeric/null types)."""
    rec = {
        "audio_path": "data/foo/bar.mp3",
        "label": "x",
        "metadata": {"bpm": 120, "key": None, "tags": ["a", "b"]},
        "num_samples": 64000,
        "sample_rate": 16000,
    }
    jsonl = _write_jsonl([rec])
    _run_rewrite(jsonl, "--from-ext", ".mp3", "--to-ext", ".wav")
    out = jsonl.with_name(jsonl.stem + ".wav.jsonl")
    new = _read_jsonl(out)[0]
    assert new["audio_path"] == "data/foo/bar.wav"
    # Every other field byte-identical
    for k in ("label", "metadata", "num_samples", "sample_rate"):
        assert new[k] == rec[k], f"field {k!r} changed"
    jsonl.unlink()
    out.unlink()


def test_idempotent_when_rerun():
    """Re-running on the same input produces the same output bytes."""
    rec = {"audio_path": "data/foo/bar.mp3", "label": "x"}
    jsonl = _write_jsonl([rec])
    _run_rewrite(jsonl, "--from-ext", ".mp3", "--to-ext", ".wav")
    out = jsonl.with_name(jsonl.stem + ".wav.jsonl")
    first = out.read_bytes()
    _run_rewrite(jsonl, "--from-ext", ".mp3", "--to-ext", ".wav")
    second = out.read_bytes()
    assert first == second, "rewrite is not idempotent"
    jsonl.unlink()
    out.unlink()


def test_skips_records_with_missing_key():
    """Records that lack audio_path are passed through unmodified (no crash)."""
    recs = [
        {"audio_path": "data/foo.mp3", "label": "x"},
        {"label": "y"},  # missing audio_path
        {"audio_path": "data/bar.mp3", "label": "z"},
    ]
    jsonl = _write_jsonl(recs)
    _run_rewrite(jsonl, "--from-ext", ".mp3", "--to-ext", ".wav")
    out = jsonl.with_name(jsonl.stem + ".wav.jsonl")
    new_recs = _read_jsonl(out)
    assert len(new_recs) == 3
    assert new_recs[0]["audio_path"] == "data/foo.wav"
    assert "audio_path" not in new_recs[1]  # passthrough
    assert new_recs[1]["label"] == "y"
    assert new_recs[2]["audio_path"] == "data/bar.wav"
    jsonl.unlink()
    out.unlink()


def test_dir_swap_only_first_occurrence():
    """If from_dir appears multiple times in a path, only the first instance
    is swapped (predictable, avoids accidental over-rewrites)."""
    recs = [
        {"audio_path": "data/clips/sub/clips/a.mp3", "label": "x"},
    ]
    jsonl = _write_jsonl(recs)
    _run_rewrite(
        jsonl,
        "--from-dir",
        "clips",
        "--to-dir",
        "clips_wav",
        "--from-ext",
        ".mp3",
        "--to-ext",
        ".wav",
    )
    out = jsonl.with_name(jsonl.stem + ".wav.jsonl")
    new = _read_jsonl(out)[0]
    assert new["audio_path"] == "data/clips_wav/sub/clips/a.wav"
    jsonl.unlink()
    out.unlink()


def test_dry_run_does_not_write():
    """--dry-run prints previews but produces no output file."""
    rec = {"audio_path": "data/foo.mp3", "label": "x"}
    jsonl = _write_jsonl([rec])
    _run_rewrite(jsonl, "--from-ext", ".mp3", "--to-ext", ".wav", "--dry-run")
    out = jsonl.with_name(jsonl.stem + ".wav.jsonl")
    assert not out.exists(), "dry-run should not write the output file"
    jsonl.unlink()


def test_output_path_multi_dot_name():
    """Multi-dot filename (HookTheoryKey.train.jsonl) → HookTheoryKey.train.wav.jsonl
    not HookTheoryKey.wav.jsonl. Regression test against the with_suffix trap."""
    # Build a fake multi-dot path
    tmpdir = Path(tempfile.mkdtemp())
    jsonl = tmpdir / "HookTheoryKey.train.jsonl"
    jsonl.write_text(json.dumps({"audio_path": "data/x.mp3", "label": "y"}) + "\n")
    _run_rewrite(jsonl, "--from-ext", ".mp3", "--to-ext", ".wav")
    expected = tmpdir / "HookTheoryKey.train.wav.jsonl"
    assert expected.exists(), f"expected {expected}, got dir contents: {list(tmpdir.iterdir())}"
    # Cleanup
    for p in tmpdir.iterdir():
        p.unlink()
    tmpdir.rmdir()
