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
