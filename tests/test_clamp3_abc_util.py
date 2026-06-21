"""Tests for the shared **kern → MusicXML → interleaved-ABC converter.

Covers ``marble/encoders/CLaMP3/abc_util.py`` — the score-native symbolic
input path reused by the CLaMP3 symbolic benchmarks (MTC-ANN, JKUPDD,
BPS-Motif). Tests the deterministic conversion parts only; the end-to-end
embedding verification (ABC-path vs MTF-path) runs on the PC with CLaMP3
weights (see the module docstring / PR notes).

Each test that needs the ``symbolic-abc`` extra (converter21 / music21 /
abctoolkit) or local kern data SKIPS rather than fails when prerequisites are
missing, so this file is safe to check in on a CI runner without the corpus.

Run manually:
    uv run --extra symbolic-abc python -m pytest tests/test_clamp3_abc_util.py -v
or:
    uv run --extra symbolic-abc python tests/test_clamp3_abc_util.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A real BPS-Motif **kern (Beethoven sonata) ships in the repo's kern_sources.
# MTC-ANN's kern lives on the PC only; BPS is the local stand-in for the shared
# converter (the converter is dataset-agnostic — same code path for all three).
_KERN_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "kern_sources"
    / "BPS-Motif"
    / "beethoven_sonatas_kern"
)


def _require_symbolic_abc():
    """Skip if the symbolic-abc extra isn't installed."""
    try:
        import abctoolkit  # noqa: F401
        import converter21  # noqa: F401
        import music21  # noqa: F401
    except ImportError as e:
        pytest.skip(f"symbolic-abc extra not installed: {e}")


def _a_real_kern() -> Path:
    """Return one real .krn, or skip if none are present locally."""
    if not _KERN_DIR.is_dir():
        pytest.skip(f"{_KERN_DIR} not found — no local kern to convert")
    krns = sorted(_KERN_DIR.glob("*.krn"))
    if not krns:
        pytest.skip(f"no .krn files under {_KERN_DIR}")
    return krns[0]


# Header prefixes M3's patchiliser recognises (the interleave step strips
# X:/T:/C:/Z:/W:/w:/%%MIDI, so the first header may be %%score / L: / M: / K:).
_ABC_HEADER_PREFIXES = (
    "X:",
    "T:",
    "C:",
    "Z:",
    "%%",
    "L:",
    "M:",
    "K:",
    "Q:",
    "V:",
    "I:",
    "W:",
    "w:",
)
# These must NOT survive the interleave step (non-musical training metadata).
_FORBIDDEN_PREFIXES = ("X:", "T:", "C:", "Z:", "W:", "w:", "%%MIDI")


# ──────────────────────────────────────────────────────────────────────────
# 1. kern → non-empty interleaved ABC with barlines + correct headers
# ──────────────────────────────────────────────────────────────────────────


def test_kern_to_abc_produces_interleaved_abc_with_barlines():
    _require_symbolic_abc()
    from marble.encoders.CLaMP3.abc_util import kern_to_abc

    krn = _a_real_kern()
    abc = kern_to_abc(krn)

    assert isinstance(abc, str) and abc.strip(), "empty ABC output"
    assert "|" in abc, "interleaved ABC must contain barlines"
    lines = abc.splitlines()
    # First non-empty line must be a recognised ABC header (not MTF's
    # `ticks_per_beat ...`, which would silently route the patchiliser to
    # MTF mode and invalidate the score-native experiment).
    first = lines[0]
    assert not first.startswith("ticks_per_beat"), f"output looks like MTF, not ABC: {first!r}"
    assert any(first.startswith(h) for h in _ABC_HEADER_PREFIXES), (
        f"first line {first!r} is not a recognised ABC header"
    )
    # Musically load-bearing headers preserved.
    assert any(line.startswith("K:") for line in lines), "missing K: (key) header"
    assert any(line.startswith("M:") for line in lines), "missing M: (meter) header"


def test_kern_to_abc_strips_training_metadata_fields():
    """The interleave step must strip non-musical metadata (X:, T:, C:, …) and
    xml2abc's `%N` bar-number comments — exactly CLaMP3's training preprocessing.
    Leaking these is off-distribution for the M3 patchiliser."""
    _require_symbolic_abc()
    import re

    from marble.encoders.CLaMP3.abc_util import kern_to_abc

    abc = kern_to_abc(_a_real_kern())
    lines = abc.splitlines()
    for prefix in _FORBIDDEN_PREFIXES:
        offenders = [line for line in lines if line.startswith(prefix)]
        assert not offenders, f"forbidden metadata field {prefix!r} survived: {offenders[:2]}"
    # No trailing `%N` bar-number comments (preceded by whitespace; `%%score`
    # directives start the line and are fine).
    assert not re.search(r"\s%\d+\s*$", abc, flags=re.MULTILINE), (
        "xml2abc `%N` bar-number comments were not stripped"
    )


def test_kern_to_abc_patchilizer_enters_abc_mode():
    """The output must tokenise as ABC, not MTF, through the real M3Patchilizer
    — the same tokeniser the CLaMP3 symbolic encoder consumes."""
    _require_symbolic_abc()
    from marble.encoders.CLaMP3.abc_util import kern_to_abc
    from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer

    abc = kern_to_abc(_a_real_kern())
    patchilizer = M3Patchilizer()
    patches = patchilizer.encode(abc, add_special_patches=True)
    # BOS + content + EOS.
    assert len(patches) > 2, f"too few patches ({len(patches)})"
    first_real = bytes(b for b in patches[1] if b > 3).decode("ascii", errors="replace")
    assert any(first_real.startswith(h) for h in _ABC_HEADER_PREFIXES), (
        f"first real patch {first_real!r} is not an ABC header — MTF mode would "
        "emit 'ticks_per_beat ...' here"
    )
    # A mid-piece patch must not look like packed MIDI events.
    mid = bytes(b for b in patches[len(patches) // 2] if b > 3).decode("ascii", errors="replace")
    assert "note_on" not in mid and "set_tempo" not in mid, (
        f"mid patch looks like MTF events: {mid[:80]!r}"
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. _abc_to_interleaved — the deterministic interleave/strip primitive
# ──────────────────────────────────────────────────────────────────────────


def test_abc_to_interleaved_strips_metadata_and_keeps_music():
    """Known standard-ABC input → metadata fields gone, musical headers + body
    kept. Mirrors the leitmotifs/SuperMario regression for this primitive."""
    _require_symbolic_abc()
    from marble.encoders.CLaMP3.abc_util import _abc_to_interleaved

    abc_input = (
        "X:1\n"
        "T:Test Theme\n"
        "C:Anonymous\n"
        "L:1/4\n"
        "M:4/4\n"
        "K:C\n"
        "%%MIDI program 0\n"
        "V:1\n"
        "|C D E F|G A B c|\n"
        "|C D E F|G A B c|\n"
    )
    out = _abc_to_interleaved(abc_input)
    assert isinstance(out, str) and out.strip()
    assert "X:1" not in out
    assert "T:Test Theme" not in out
    assert "C:Anonymous" not in out
    assert "%%MIDI" not in out
    # Musical content survives.
    assert "K:C" in out
    assert "|" in out


# ──────────────────────────────────────────────────────────────────────────
# 3. score_to_abc — fragment slicing (partial bars + carried key/clef context)
# ──────────────────────────────────────────────────────────────────────────


def test_score_to_abc_fragment_carries_key_context_and_partial_bar():
    """A synthetic single-voice fragment with an explicit key signature and a
    note count that doesn't fill the final bar must (a) carry the key into the
    ABC `K:` header (not default to C major) and (b) produce a partial final
    bar without crashing."""
    _require_symbolic_abc()
    import music21

    from marble.encoders.CLaMP3.abc_util import score_to_abc

    part = music21.stream.Part()
    part.append(music21.key.KeySignature(-4))  # 4 flats → Ab major / F minor
    part.append(music21.meter.TimeSignature("2/2"))
    # 5 quarter notes in 2/2 = one full bar (4 quarters) + a partial bar.
    for pitch in ["C5", "E-5", "G5", "C6", "B-5"]:
        part.append(music21.note.Note(pitch, quarterLength=1.0))

    abc = score_to_abc(part)
    lines = abc.splitlines()
    assert "|" in abc, "fragment ABC must contain barlines"
    k_lines = [line for line in lines if line.startswith("K:")]
    assert k_lines, "fragment lost its K: (key) header"
    # 4 flats → music21/xml2abc spell as Ab (major) — NOT C major. The exact
    # spelling can be Ab; assert it is not the C-major default.
    assert not any(line.strip() in ("K:C", "K:Cmaj", "K:C major") for line in k_lines), (
        f"key context not carried into fragment: {k_lines}"
    )


def test_score_to_abc_round_trips_a_real_kern_slice():
    """Slicing a real score by measure range and re-emitting must yield valid
    interleaved ABC whose bar count matches the requested window."""
    _require_symbolic_abc()
    import music21

    from marble.encoders.CLaMP3.abc_util import _register_converter21, score_to_abc

    _register_converter21()
    score = music21.converter.parse(str(_a_real_kern()))
    frag = score.measures(5, 8)  # 4-bar window
    abc = score_to_abc(frag)
    lines = abc.splitlines()

    assert isinstance(abc, str) and abc.strip()
    assert "|" in abc
    assert any(line.startswith("K:") for line in lines), "slice lost key header"
    assert any(line.startswith("M:") for line in lines), "slice lost meter header"
    # The body lines (those carrying inline [V:N] markers) correspond to bars;
    # for a 4-measure window there should be roughly that many body lines
    # (interleaved ABC = one line per bar). Allow slack for anacrusis handling.
    body = [line for line in lines if "[V:" in line]
    assert 1 <= len(body) <= 8, f"4-bar slice produced {len(body)} body lines: {body[:2]}"


# ──────────────────────────────────────────────────────────────────────────
# 4. Error surfaces
# ──────────────────────────────────────────────────────────────────────────


def test_kern_to_abc_missing_file_raises():
    from marble.encoders.CLaMP3.abc_util import kern_to_abc

    with pytest.raises(FileNotFoundError):
        kern_to_abc("/nonexistent/does_not_exist.krn")


def test_musicxml_to_interleaved_abc_missing_file_raises():
    from marble.encoders.CLaMP3.abc_util import musicxml_to_interleaved_abc

    with pytest.raises(FileNotFoundError):
        musicxml_to_interleaved_abc("/nonexistent/does_not_exist.musicxml")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
