"""Unit tests for the JKUPDD ABC build + ABC-vs-MTF relevance parity.

The ABC retrieval task must score relevance *identically* to the MTF task so the
per-layer MAP is a clean A/B (same 78/20 dedup'd occurrences, same
``(piece, annotator, pattern)`` → ``work_id`` mapping). We also test the build
script's pure-Python alignment helpers (point-set parse + (ontime, pitch) join)
without needing CLaMP3 / music21 / converter21 installed.
"""

from pathlib import Path

# _work_id is the relevance key shared by the MTF and ABC datasets, imported from
# the datamodule (no torch-heavy CLaMP3 import is triggered at module load).
from marble.tasks.JKUPDDRetrieval.datamodule import _work_id
from scripts.data.build_jkupdd_abc import (
    PIECE_KERN_STEM,
    _bridge_kern_to_pointset,
    _match_occurrence,
    _read_occ_pointset,
)


def test_work_id_relevance_parity():
    """Two occurrences of the same (piece, annotator, pattern) group share a
    work_id; different groups get different ids — identical to the MTF task."""
    g1 = "bachBWV889Fg|bruhn|A"
    g2 = "bachBWV889Fg|bruhn|B"
    assert _work_id(g1) == _work_id(g1)  # deterministic
    assert _work_id(g1) != _work_id(g2)  # group-distinct


def test_all_pieces_have_kern_stem():
    """Every JKUPDD piece id used in the MTF JSONL maps to a kern stem."""
    expected = {
        "bachBWV889Fg",
        "beethovenOp2No1Mvt3",
        "chopinOp24No4",
        "gibbonsSilverSwan1612",
        "mozartK282Mvt2",
    }
    assert set(PIECE_KERN_STEM) == expected


def test_read_occ_pointset(tmp_path: Path):
    """Point-set CSV → (ontime, midi, morphetic) rows; floats rounded to int."""
    csv = tmp_path / "occ1.csv"
    csv.write_text(
        "1.0000000000, 64.0000000000, 62.0000000000, 1.0000000000, 1.0000000000\n"
        "2.0000000000, 60.0000000000, 60.0000000000, 1.0000000000, 1.0000000000\n"
    )
    rows = _read_occ_pointset(csv)
    assert rows == [(1.0, 64, 62), (2.0, 60, 60)]


def test_bridge_calibrates_origin_shift():
    """The kern and point-set share a beat grid up to a constant origin shift.
    With a +1 shift (kern offset 0 ↔ point-set ontime 1) the bridge keys every
    kern note at the point-set ontime and reports 100% piece-level match."""
    kern = [
        {"offset": 0.0, "midi": 64, "note": "n1"},
        {"offset": 1.0, "midi": 60, "note": "n2"},
        {"offset": 2.0, "midi": 65, "note": "n3"},
    ]
    full_ps = [(1.0, 64, 62), (2.0, 60, 60), (3.0, 65, 63)]  # ontime = offset + 1
    bridge, n_matched, n_ps = _bridge_kern_to_pointset(kern, full_ps)
    assert n_ps == 3
    assert n_matched == 3  # every point-set note has a kern note at its ontime
    # the bridge is keyed by point-set ontime → correct kern note
    assert bridge[(1.0, 64)]["note"] == "n1"
    assert bridge[(3.0, 65)]["note"] == "n3"


def test_bridge_calibrates_negative_pickup_shift():
    """A pickup puts the point-set origin at -1.0 (kern offset 0 ↔ ontime -1).
    The shift calibration must find -1.0, not assume +1."""
    kern = [
        {"offset": 0.0, "midi": 74, "note": "p1"},
        {"offset": 1.0, "midi": 72, "note": "p2"},
        {"offset": 2.0, "midi": 62, "note": "p3"},
    ]
    full_ps = [(-1.0, 74, 70), (0.0, 72, 68), (1.0, 62, 61)]  # ontime = offset - 1
    bridge, n_matched, n_ps = _bridge_kern_to_pointset(kern, full_ps)
    assert n_matched == 3
    assert bridge[(-1.0, 74)]["note"] == "p1"


def test_bridge_unmatched_pointset_lowers_rate():
    """A point-set note with no kern counterpart at its (ontime, midi) is not
    matched — the piece-level rate reflects the real disagreement."""
    kern = [
        {"offset": 0.0, "midi": 64, "note": "n1"},
        {"offset": 1.0, "midi": 60, "note": "n2"},
    ]
    full_ps = [(1.0, 64, 62), (2.0, 60, 60), (3.0, 99, 80)]  # midi 99 absent in kern
    _bridge, n_matched, n_ps = _bridge_kern_to_pointset(kern, full_ps)
    assert n_ps == 3
    assert n_matched == 2


def test_match_occurrence_via_bridge():
    """An occurrence sub-sequence resolves to its kern notes through the bridge
    (keyed by the same quantised (ontime, midi))."""
    kern = [
        {"offset": 0.0, "midi": 64, "note": "n1"},
        {"offset": 1.0, "midi": 60, "note": "n2"},
        {"offset": 2.0, "midi": 65, "note": "n3"},
    ]
    full_ps = [(1.0, 64, 62), (2.0, 60, 60), (3.0, 65, 63)]
    bridge, _n, _np = _bridge_kern_to_pointset(kern, full_ps)
    occ = [(1.0, 64, 62), (2.0, 60, 60)]
    matched, n = _match_occurrence(occ, bridge)
    assert n == 2
    assert [m["note"] for m in matched] == ["n1", "n2"]


def test_match_occurrence_misses_absent_key():
    """A row whose (ontime, midi) key is absent is unmatched (lowers rate)."""
    bridge = {(1.0, 64): {"note": "n1"}}
    occ = [(1.0, 99, 80)]  # midi 99 not in bridge
    matched, n = _match_occurrence(occ, bridge)
    assert n == 0
    assert matched == []
