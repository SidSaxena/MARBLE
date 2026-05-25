"""Tests for scripts/data/convert_audio_format.py.

Most tests work on the pure-Python helpers (ffmpeg command builder, dst
path resolution, JSONL rewrite, JSONL-driven discovery) so they don't
need ffmpeg installed. The integration test that actually runs ffmpeg
is skipped when ffmpeg isn't on PATH.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import wave
from pathlib import Path

import pytest

# Load the script as a module (it lives outside marble/ so it's not on
# the import path by default).
_SCRIPT = Path(__file__).parent.parent / "scripts" / "data" / "convert_audio_format.py"
_spec = importlib.util.spec_from_file_location("convert_audio_format", _SCRIPT)
caf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(caf)


# ── ffmpeg command builder ──────────────────────────────────────────────


def test_build_ffmpeg_cmd_wav_default():
    cmd = caf._build_ffmpeg_cmd(
        Path("a.mp3"),
        Path("a.wav"),
        to_fmt="wav",
        sample_rate=None,
        channels=None,
        bit_depth=16,
        flac_compression=5,
        mp3_bitrate=192,
        ogg_quality=6,
    )
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and "a.mp3" in cmd
    assert "pcm_s16le" in cmd
    assert cmd[-1] == "a.wav"
    # No resample/downmix flags since sample_rate=None, channels=None
    assert "-ar" not in cmd
    assert "-ac" not in cmd


def test_build_ffmpeg_cmd_resample_and_downmix():
    cmd = caf._build_ffmpeg_cmd(
        Path("a.mp3"),
        Path("a.wav"),
        to_fmt="wav",
        sample_rate=24000,
        channels=1,
        bit_depth=16,
        flac_compression=5,
        mp3_bitrate=192,
        ogg_quality=6,
    )
    assert "-ar" in cmd and "24000" in cmd
    assert "-ac" in cmd and "1" in cmd


def test_build_ffmpeg_cmd_flac_compression_flag():
    cmd = caf._build_ffmpeg_cmd(
        Path("a.wav"),
        Path("a.flac"),
        to_fmt="flac",
        sample_rate=None,
        channels=None,
        bit_depth=16,
        flac_compression=8,
        mp3_bitrate=192,
        ogg_quality=6,
    )
    assert "flac" in cmd
    assert "-compression_level" in cmd
    i = cmd.index("-compression_level")
    assert cmd[i + 1] == "8"


def test_build_ffmpeg_cmd_mp3_bitrate():
    cmd = caf._build_ffmpeg_cmd(
        Path("a.wav"),
        Path("a.mp3"),
        to_fmt="mp3",
        sample_rate=None,
        channels=None,
        bit_depth=16,
        flac_compression=5,
        mp3_bitrate=320,
        ogg_quality=6,
    )
    assert "libmp3lame" in cmd
    assert "320k" in cmd


def test_build_ffmpeg_cmd_unknown_fmt_raises():
    with pytest.raises(ValueError, match="unsupported"):
        caf._build_ffmpeg_cmd(
            Path("a.wav"),
            Path("a.aiff"),
            to_fmt="aiff",
            sample_rate=None,
            channels=None,
            bit_depth=16,
            flac_compression=5,
            mp3_bitrate=192,
            ogg_quality=6,
        )


def test_build_ffmpeg_cmd_bit_depth_24():
    cmd = caf._build_ffmpeg_cmd(
        Path("a.wav"),
        Path("a.wav"),
        to_fmt="wav",
        sample_rate=None,
        channels=None,
        bit_depth=24,
        flac_compression=5,
        mp3_bitrate=192,
        ogg_quality=6,
    )
    assert "pcm_s24le" in cmd


# ── _resolve_dst_path ──────────────────────────────────────────────────


def test_resolve_dst_path_in_place_changes_extension(tmp_path):
    src = tmp_path / "x.mp3"
    dst = caf._resolve_dst_path(src, tmp_path, tmp_path, ".wav", in_place=True)
    assert dst == tmp_path / "x.wav"


def test_resolve_dst_path_separate_dst_mirrors_layout(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src = src_root / "sub" / "x.mp3"
    dst = caf._resolve_dst_path(src, src_root, dst_root, ".wav", in_place=False)
    assert dst == dst_root / "sub" / "x.wav"


# ── _discover_files ────────────────────────────────────────────────────


def test_discover_files_filters_by_input_ext(tmp_path):
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.wav").touch()
    (tmp_path / "c.flac").touch()
    files = caf._discover_files(tmp_path, ".mp3")
    assert [f.name for f in files] == ["a.mp3"]


def test_discover_files_autodetect_audio(tmp_path):
    (tmp_path / "a.mp3").touch()
    (tmp_path / "b.txt").touch()
    (tmp_path / "c.flac").touch()
    files = caf._discover_files(tmp_path, None)
    assert sorted(f.name for f in files) == ["a.mp3", "c.flac"]


def test_discover_files_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.wav").touch()
    files = caf._discover_files(tmp_path, ".wav")
    assert [f.name for f in files] == ["deep.wav"]


def test_discover_files_missing_src_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        caf._discover_files(tmp_path / "nope", None)


# ── JSONL-driven discovery ─────────────────────────────────────────────


def test_get_nested_dotted():
    d = {"youtube": {"id": "abc123"}}
    assert caf._get_nested(d, "youtube.id") == "abc123"
    assert caf._get_nested(d, "youtube.missing") is None
    assert caf._get_nested(d, "foo.bar") is None


def test_discover_files_from_jsonl(tmp_path):
    jsonl = tmp_path / "smoke.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps({"youtube": {"id": "abc"}}) + "\n")
        f.write(json.dumps({"youtube": {"id": "def"}}) + "\n")
        f.write(json.dumps({"other": "field"}) + "\n")  # no id -> skipped
        f.write("\n")  # empty -> skipped
    src = tmp_path / "audio"
    src.mkdir()
    files = caf._discover_files_from_jsonl([jsonl], src, ".mp3", "youtube.id")
    assert [f.name for f in files] == ["abc.mp3", "def.mp3"]
    # All resolved to src dir
    assert all(f.parent == src for f in files)


def test_discover_files_from_jsonl_dedupes_across_files(tmp_path):
    j1 = tmp_path / "a.jsonl"
    j2 = tmp_path / "b.jsonl"
    j1.write_text(json.dumps({"id": "x"}) + "\n")
    j2.write_text(json.dumps({"id": "x"}) + "\n" + json.dumps({"id": "y"}) + "\n")
    src = tmp_path / "audio"
    src.mkdir()
    files = caf._discover_files_from_jsonl([j1, j2], src, ".wav", "id")
    assert [f.name for f in files] == ["x.wav", "y.wav"]


def test_discover_files_from_jsonl_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        caf._discover_files_from_jsonl([tmp_path / "nope.jsonl"], tmp_path, ".wav", "id")


# ── _rewrite_jsonl ─────────────────────────────────────────────────────


def test_rewrite_jsonl_swaps_extension(tmp_path):
    jp = tmp_path / "x.jsonl"
    records = [
        {"audio_path": "data/foo/a.m4a", "label": 1},
        {"audio_path": "data/foo/b.m4a", "label": 2},
        {"audio_path": "data/foo/c.flac", "label": 3},  # already flac -> unchanged
        {"label": 4},  # no audio_path -> unchanged
    ]
    with jp.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_rewritten, n_total = caf._rewrite_jsonl(jp, ".m4a", ".flac")
    assert n_rewritten == 2
    assert n_total == 4

    out = [json.loads(line) for line in jp.read_text().splitlines() if line.strip()]
    assert out[0]["audio_path"] == "data/foo/a.flac"
    assert out[1]["audio_path"] == "data/foo/b.flac"
    assert out[2]["audio_path"] == "data/foo/c.flac"  # unchanged
    assert "audio_path" not in out[3]
    assert out[0]["label"] == 1  # other fields preserved


def test_rewrite_jsonl_idempotent(tmp_path):
    jp = tmp_path / "x.jsonl"
    jp.write_text(json.dumps({"audio_path": "a.flac"}) + "\n")
    n_rewritten, _ = caf._rewrite_jsonl(jp, ".m4a", ".flac")
    assert n_rewritten == 0  # nothing to do


def test_rewrite_jsonl_atomic_via_tmp(tmp_path, monkeypatch):
    """Confirm the tmp-then-replace pattern: tmp file shouldn't linger."""
    jp = tmp_path / "x.jsonl"
    jp.write_text(json.dumps({"audio_path": "a.m4a"}) + "\n")
    caf._rewrite_jsonl(jp, ".m4a", ".flac")
    assert not (tmp_path / "x.jsonl.tmp").exists()


# ── integration: end-to-end conversion (needs ffmpeg) ──────────────────


def _make_wav(path: Path, sr=8000, duration_s=0.1):
    """Write a tiny silent mono int16 WAV."""
    n_frames = int(sr * duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_convert_one_wav_to_flac_in_place(tmp_path):
    src = tmp_path / "x.wav"
    _make_wav(src)
    dst = src.with_suffix(".flac")
    res = caf._convert_one(
        src,
        dst,
        to_fmt="flac",
        sample_rate=None,
        channels=None,
        bit_depth=16,
        flac_compression=5,
        mp3_bitrate=192,
        ogg_quality=6,
        delete_source=True,
        force=False,
    )
    assert res[1] is True, res
    assert dst.exists() and dst.stat().st_size > 0
    assert not src.exists(), "source should be deleted with delete_source=True"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_convert_one_skips_existing_dst(tmp_path):
    src = tmp_path / "x.wav"
    _make_wav(src)
    dst = tmp_path / "x.flac"
    dst.write_bytes(b"existing-non-empty")  # pretend already converted

    res = caf._convert_one(
        src,
        dst,
        to_fmt="flac",
        sample_rate=None,
        channels=None,
        bit_depth=16,
        flac_compression=5,
        mp3_bitrate=192,
        ogg_quality=6,
        delete_source=False,
        force=False,
    )
    assert res == (src, True, "already")
    # dst untouched
    assert dst.read_bytes() == b"existing-non-empty"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_convert_one_missing_src(tmp_path):
    src = tmp_path / "nope.wav"
    dst = tmp_path / "nope.flac"
    res = caf._convert_one(
        src,
        dst,
        to_fmt="flac",
        sample_rate=None,
        channels=None,
        bit_depth=16,
        flac_compression=5,
        mp3_bitrate=192,
        ogg_quality=6,
        delete_source=False,
        force=False,
    )
    assert res == (src, False, "src missing")
