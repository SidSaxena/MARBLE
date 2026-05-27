"""
tests/test_rewrite_vgmidi_programs.py

Unit tests for the pure logic in
``scripts/data/rewrite_vgmidi_programs.py`` — primarily the new
cross-product-mode helpers (``parse_programs_arg``, ``GM_NAMES``) added
for the ``VGMIDITVar-timbre`` variant.

The schedule-mode logic (``target_program_for_idx``, ``schedule_hash``)
is already exercised by the existing in-script verifier (``--verify``)
and the offline analysis script — covered by integration, not unit
tests here.

The rewrite function itself (``rewrite_midi``) needs a real MIDI file
to exercise mido's parser — not tested in this unit suite. Manual
verification via ``rewrite_vgmidi_programs.py --verify`` after a
pilot rewrite covers it.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# Stub ``mido`` so importing the rewriter script doesn't require the
# real package (which isn't a hard dep of this Mac-side venv). The
# rewriter only references mido inside ``rewrite_midi``, which we
# don't exercise here.
class _Stub:
    def __getattr__(self, _name):
        return _Stub()

    def __call__(self, *a, **kw):
        return _Stub()


sys.modules.setdefault("mido", _Stub())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "data"))
from rewrite_vgmidi_programs import (  # noqa: E402
    GM_NAMES,
    parse_programs_arg,
    target_program_for_idx,
)

# ──────────────────────────────────────────────────────────────────────
# parse_programs_arg — the new cross-product CLI parser
# ──────────────────────────────────────────────────────────────────────


def test_parse_set_c():
    """The chosen Set C for the VGMIDITVar-timbre variant: 8 programs
    spanning keys, plucked, strings, vocal, brass, wind, electronic."""
    out = parse_programs_arg("0,24,48,52,60,73,80,89")
    assert out == [0, 24, 48, 52, 60, 73, 80, 89]


def test_parse_returns_sorted():
    """Output must be sorted regardless of input order — used as a
    stable key for the programs.json idempotency guard."""
    assert parse_programs_arg("73,0,48,24") == [0, 24, 48, 73]


def test_parse_dedupes():
    """Duplicates collapse silently. Users may type ``0,48,0,73`` and
    expect ``[0, 48, 73]`` — that's the intent."""
    assert parse_programs_arg("0,48,0,48,73") == [0, 48, 73]


def test_parse_tolerates_whitespace():
    assert parse_programs_arg("  0 , 48 , 73  ") == [0, 48, 73]


def test_parse_tolerates_trailing_comma():
    assert parse_programs_arg("0,48,73,") == [0, 48, 73]


def test_parse_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        parse_programs_arg("")


def test_parse_rejects_whitespace_only():
    with pytest.raises(ValueError, match="non-empty"):
        parse_programs_arg("  ,  , ")


def test_parse_rejects_non_int():
    with pytest.raises(ValueError, match="not an integer"):
        parse_programs_arg("0,abc,48")


def test_parse_rejects_out_of_range_high():
    """GM programs are 0..127."""
    with pytest.raises(ValueError, match="out of GM range"):
        parse_programs_arg("0,128")


def test_parse_rejects_out_of_range_negative():
    with pytest.raises(ValueError, match="out of GM range"):
        parse_programs_arg("0,-1")


def test_parse_boundary_values():
    """0 and 127 are the boundaries of GM range and must be accepted."""
    assert parse_programs_arg("0") == [0]
    assert parse_programs_arg("127") == [127]
    assert parse_programs_arg("0,127") == [0, 127]


# ──────────────────────────────────────────────────────────────────────
# GM_NAMES — log clarity helper
# ──────────────────────────────────────────────────────────────────────


def test_gm_names_covers_set_c():
    """Every program in the recommended Set C should have a human name
    in the GM_NAMES dict so logs print 'Piano' rather than 'GM0'."""
    set_c = [0, 24, 48, 52, 60, 73, 80, 89]
    for p in set_c:
        assert p in GM_NAMES, f"missing GM name for {p}"
        assert isinstance(GM_NAMES[p], str)
        assert len(GM_NAMES[p]) > 0


# ──────────────────────────────────────────────────────────────────────
# Legacy schedule-mode logic — still works
# ──────────────────────────────────────────────────────────────────────


def test_target_program_for_idx_cycles():
    """``idx ≥ len(SCHEDULE)`` cycles modulo len. Verify on idx 0..9."""
    # Schedule has 5 entries (idx 0..4)
    program_for = [target_program_for_idx(i) for i in range(10)]
    # First 5 match the schedule
    assert program_for[:5] == [0, 48, 60, 73, 56]
    # idx 5..9 wraps back to idx 0..4
    assert program_for[5:10] == program_for[:5]


# ──────────────────────────────────────────────────────────────────────
# Cross-product filename construction ↔ verify-side parsing.
#
# These two halves of the timbre-variant pipeline have to agree: the
# rewriter writes ``<piece>_<section>_<idx>_p<prog>.mid`` (probe.py
# line 485) and ``verify_dir`` recovers ``program`` from the suffix
# via ``_parse_filename`` (rewrite_vgmidi_programs.py line 297, where
# the audit-2 #8 fix lives). If either side drifts, ``--verify``
# silently mis-validates the dataset. Running mido is out of scope
# for this Mac-side venv (no real package installed), so this test
# pins the string contract directly.
# ──────────────────────────────────────────────────────────────────────

import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "data"))
from build_vgmiditvar_dataset import _parse_filename  # noqa: E402


def _cross_product_stem(src_stem: str, program: int) -> str:
    """Mirror the rewriter's stem construction at line 485:
    ``f"{src.stem}_p{p}.mid"`` — sans the .mid suffix."""
    return f"{src_stem}_p{program}"


def test_cross_product_stems_parseable_by_verify_dir():
    """For every (src, program) pair the rewriter would emit, the
    recovered ``_parse_filename(stem)["program"]`` must equal the program
    we wrote into the filename. Three source-stem styles cover the
    POP909-TVar, VGMIDI-TVar long-name, and recommended Set C cases."""
    src_stems = [
        "052_A_0",  # POP909-TVar
        "e0_real_Other games_NES_Monster Party_Title Screen_A_1",  # VGMIDI-TVar
        "999_Z_42",  # boundary
    ]
    programs = [0, 24, 48, 52, 60, 73, 80, 89]
    for src in src_stems:
        for p in programs:
            stem = _cross_product_stem(src, p)
            parsed = _parse_filename(stem)
            assert parsed is not None, f"parse failed on {stem!r}"
            assert parsed.get("program") == p, (
                f"program mismatch on {stem!r}: got {parsed.get('program')}, want {p}"
            )


def test_cross_product_filenames_have_no_collisions():
    """No two (src, program) pairs may produce the same on-disk
    filename. Pins the basic invariant that the rewriter doesn't
    accidentally drop or overwrite outputs."""
    src_stems = ["052_A_0", "052_B_0", "999_A_0", "999_A_1"]
    programs = [0, 24, 48, 52, 60, 73, 80, 89]
    stems = [_cross_product_stem(s, p) for s in src_stems for p in programs]
    assert len(set(stems)) == len(stems), "filename collision in cross-product"
    # Every output is still parseable and reports the expected piece+program.
    for src in src_stems:
        for p in programs:
            stem = _cross_product_stem(src, p)
            parsed = _parse_filename(stem)
            assert parsed["program"] == p


def test_schedule_mode_stems_have_no_program_field():
    """Schedule-mode stems (no ``_p<N>`` suffix) must NOT populate the
    ``program`` field. The audit-2 #8 fix in verify_dir falls back to
    ``target_program_for_idx`` when ``program`` is None — confirm we
    keep that distinction."""
    parsed = _parse_filename("052_A_0")
    assert parsed is not None
    assert "program" not in parsed
