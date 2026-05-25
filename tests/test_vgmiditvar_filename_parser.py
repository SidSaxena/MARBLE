"""
tests/test_vgmiditvar_filename_parser.py

Guards the ``_parse_filename`` regex in
``scripts/data/build_vgmiditvar_dataset.py``. That regex parses stems
from three filename conventions:

  POP909-TVar:                 ``052_A_0``
  VGMIDI-TVar:                 ``e0_real_NES_Title_A_3``
  VGMIDI-TVar timbre variant:  ``052_A_0_p48``  (cross-product, NEW)

Backward compatibility is critical — every prior dataset's filenames
must still parse correctly OR the corresponding sweep / analysis script
silently breaks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "data"))
from build_vgmiditvar_dataset import _parse_filename  # noqa: E402

# ──────────────────────────────────────────────────────────────────────
# Backward compatibility — pre-timbre filenames
# ──────────────────────────────────────────────────────────────────────


def test_pop909_tvar_stem():
    assert _parse_filename("052_A_0") == {"piece": "052", "section": "A", "idx": 0}
    assert _parse_filename("052_A_3") == {"piece": "052", "section": "A", "idx": 3}


def test_vgmidi_tvar_stem_with_spaces_and_underscores_in_piece():
    out = _parse_filename("e0_real_Other games_NES_Monster Party_Title Screen_A_1")
    assert out == {
        "piece": "e0_real_Other games_NES_Monster Party_Title Screen",
        "section": "A",
        "idx": 1,
    }


def test_section_can_be_multi_char():
    # Section uses [A-Z]+ — verify 'AA' or 'BB' work
    assert _parse_filename("piece_AA_2") == {"piece": "piece", "section": "AA", "idx": 2}


def test_high_idx():
    assert _parse_filename("piece_A_15") == {"piece": "piece", "section": "A", "idx": 15}


# ──────────────────────────────────────────────────────────────────────
# New: cross-product timbre stems
# ──────────────────────────────────────────────────────────────────────


def test_timbre_stem_basic():
    assert _parse_filename("052_A_0_p48") == {
        "piece": "052",
        "section": "A",
        "idx": 0,
        "program": 48,
    }


def test_timbre_stem_program_0():
    """Program 0 (Piano) is a valid GM value; the regex must not collapse
    it as falsy / empty."""
    assert _parse_filename("052_A_2_p0") == {
        "piece": "052",
        "section": "A",
        "idx": 2,
        "program": 0,
    }


def test_timbre_stem_large_program():
    """GM allows programs 0..127."""
    assert _parse_filename("piece_A_3_p127") == {
        "piece": "piece",
        "section": "A",
        "idx": 3,
        "program": 127,
    }


def test_timbre_stem_with_complex_piece():
    """Piece IDs with underscores + spaces (VGMIDI-TVar convention) plus
    the new program suffix."""
    out = _parse_filename("e0_real_NES_Monster Party_Title_A_4_p89")
    assert out == {
        "piece": "e0_real_NES_Monster Party_Title",
        "section": "A",
        "idx": 4,
        "program": 89,
    }


# ──────────────────────────────────────────────────────────────────────
# Rejects (malformed input → None)
# ──────────────────────────────────────────────────────────────────────


def test_rejects_no_idx():
    assert _parse_filename("piece_A") is None


def test_rejects_lowercase_section():
    """Section must be uppercase [A-Z]+ — lowercase 'a' is part of the
    piece name, not the section. The regex requires `[A-Z]+` so this
    correctly fails to parse."""
    assert _parse_filename("piece_a_0") is None


def test_rejects_empty():
    assert _parse_filename("") is None


def test_rejects_malformed_program_suffix():
    """Program suffix must be ``_p<digits>``. Anything else (no digits,
    extra characters) falls back to the legacy non-suffixed regex —
    which won't match because the program suffix isn't structurally
    part of <piece>_<section>_<idx>."""
    # "_pX" with non-digit — should fail
    assert _parse_filename("piece_A_0_pX") is None
    # "_p" with nothing after — should fail
    assert _parse_filename("piece_A_0_p") is None


# ──────────────────────────────────────────────────────────────────────
# Round-trip: legacy stem should still NOT have a 'program' key
# ──────────────────────────────────────────────────────────────────────


def test_legacy_stems_have_no_program_key():
    """When the program suffix is absent, the parsed dict must NOT contain
    a 'program' key (so downstream code can safely use 'program' in dict
    membership checks)."""
    out = _parse_filename("052_A_0")
    assert "program" not in out
    out = _parse_filename("e0_real_NES_Title_A_3")
    assert "program" not in out
