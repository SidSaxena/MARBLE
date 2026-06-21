#!/usr/bin/env python3
"""Build the JKUPDD motif-retrieval dataset as score-native **interleaved ABC**.

This is the ABC counterpart of ``scripts/data/build_jkupdd_retrieval.py`` (which
emits one lossy MIDI window per occurrence for the MIDI→MTF path). Instead of a
MIDI window, each occurrence here gets an **interleaved-ABC** string sliced from
the piece's ``**kern`` — preserving key / pitch-spelling / meter / bar structure
that the ``**kern → MIDI → MTF`` round-trip discards. The goal is a clean A/B at
fixed identity: does notation-preserving ABC beat lossy MTF for cross-piece motif
retrieval with CLaMP3? (See ``docs/jkupdd_abc_vs_mtf.md``.)

**Canonical identity (apples-to-apples).** We do NOT re-enumerate occurrences.
We read the *already-built* dedup'd MTF JSONL
(``data/JKUPDD/JKUPDDRetrieval.test.jsonl`` — 78 byte-distinct occurrences / 20
groups) and produce one ABC record per MTF record, with the **same**
``group`` / ``occurrence_id`` / ``piece_id`` / ``annotator`` / ``pattern``. So the
ABC task's ``work_id`` and relevance are byte-identical to the MTF task and the
per-layer MAP is directly comparable.

**Alignment (the validation gate).** JKUPDD ships, per pattern occurrence, a
point-set CSV ``occurrences/csv/occN.csv`` whose rows are
``(ontime_beats, MIDI, morphetic, duration, voice)`` — a literal contiguous
sub-sequence of the piece's **full-piece point-set** CSV (which ships beside the
``**kern`` as ``<stem>.csv``). We use that full point-set as a *positional
bridge* between the occurrence and the kern:

  1. Parse the piece ``**kern`` with converter21 (registered into music21),
     flatten to a per-note ``(offset_ql, midi)`` list.
  2. Parse the full-piece point-set CSV → ``(ontime, midi)`` list. The kern
     offsets and point-set ontimes share a beat grid up to a constant per-piece
     **origin shift** (the ontime origin varies — Bach +1.0, the pickup pieces
     −1.0); we calibrate that shift as the modal ``ontime − offset`` over
     same-MIDI candidate pairs, then key every kern note by its calibrated
     ``(ontime, midi)``. We report a **piece-level match-rate** (point-set notes
     that find a kern note at the same calibrated ontime+pitch — the real
     edition-match signal).
  3. For each occurrence, look up its ``(ontime, midi)`` rows in that bridge →
     kern notes → measure span.

We report the **per-occurrence note match-rate**; an occurrence below
``--min-match-rate`` (default 0.9 → tolerate a few ornament/grace-note mismatches
whose absence doesn't move the measure span) is dropped/flagged, and the summary
says how many. The matched kern notes define the occurrence's measure span; we
slice that range from the score and run ``score_to_abc``.

**Alignment is NOT uniformly clean (a real finding).** The
``docs/kern_sourcing_bps_jkupdd.md`` assessment called JKUPDD "clean / low-risk
— a deterministic join". In practice converter21 + the repeat structure of the
Menuetto pieces break that:

  * **Beethoven Op.2/1 mvt-3** parses into a score whose every note collapses to
    a single ``offset-in-hierarchy`` (a converter21 quirk on this written-out-
    repeat Menuetto). The ontime bridge is then meaningless; we DETECT this
    degeneracy (a large fraction of notes sharing one offset) and **drop the
    piece** with a clear reason rather than mis-slice. All 8 Beethoven
    occurrences are lost.
  * The point-set is the **unfolded performance** (repeats played) while the
    kern is **folded notation**; for the repeat-heavy pieces the ontime↔kern map
    is therefore non-monotone past the first section. Bach (no repeats) and most
    Chopin/Gibbons/Mozart occurrences still align ≥0.9; a handful that straddle a
    repeat boundary fall below threshold and are dropped.

So this build yields a clean ABC set for **4 of 5 composers**; the A/B against
MTF restricts both sides to the surviving occurrences (apples-to-apples on the
aligned subset). See ``docs/jkupdd_abc_vs_mtf.md`` for the honest accounting.

Usage::

    uv run python scripts/data/build_jkupdd_abc.py \
        --jkupdd-root data/kern_sources/jkupdd_sparse \
        --kern-dir data/kern_sources/JKUPDD \
        [--mtf-jsonl data/JKUPDD/JKUPDDRetrieval.test.jsonl] \
        [--min-match-rate 0.9]

Outputs ``data/JKUPDD/JKUPDDRetrievalABC.test.jsonl`` (one record per surviving
occurrence, each with an inline ``abc`` field) and prints a stats JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "data" / "JKUPDD"
JSONL_OUT = OUT_DIR / "JKUPDDRetrievalABC.test.jsonl"
DEFAULT_MTF_JSONL = OUT_DIR / "JKUPDDRetrieval.test.jsonl"

# Map a piece_id (as it appears in the MTF JSONL / JKUPDD groundTruth) to its
# **kern stem under --kern-dir/<piece_id>/<stem>.krn.
PIECE_KERN_STEM = {
    "bachBWV889Fg": "wtc2f20",
    "beethovenOp2No1Mvt3": "sonata01-3",
    "chopinOp24No4": "mazurka24-4",
    "gibbonsSilverSwan1612": "silverswan",
    "mozartK282Mvt2": "sonata04-2",
}


def _read_occ_pointset(csv_path: Path) -> list[tuple[float, int, int]]:
    """Read an occurrence point-set CSV → ``[(ontime, midi, morphetic), ...]``.

    Columns are ``ontime, MIDI, morphetic, duration, voice`` (comma-separated,
    fixed-point). Only the first three are needed for the join.
    """
    rows: list[tuple[float, int, int]] = []
    for line in csv_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        ontime = float(parts[0])
        midi = int(round(float(parts[1])))
        morph = int(round(float(parts[2]))) if len(parts) > 2 and parts[2] else -1
        rows.append((ontime, midi, morph))
    return rows


def _kern_note_list(score) -> list[dict]:
    """Flatten a parsed kern score → per-note records sorted by (offset_ql, midi).

    Each record: ``{offset, midi, note}`` where ``offset`` is the note's
    quarter-length position from the start of the score (used only to define the
    sort order, which must match the point-set's ontime order) and ``note`` is
    the music21 element (kept so the caller can find the measure span for
    slicing). Chords expand to one record per constituent pitch (the JKUPDD
    point-set is per-note).
    """
    import music21

    notes: list[dict] = []
    for el in score.flatten().notes:
        pitches = el.pitches if isinstance(el, music21.chord.Chord) else [el.pitch]
        try:
            off_ql = float(el.getOffsetInHierarchy(score))
        except Exception:  # noqa: BLE001 — fall back to flat offset
            off_ql = float(el.offset)
        for p in pitches:
            notes.append({"offset": off_ql, "midi": int(p.midi), "note": el})
    notes.sort(key=lambda r: (r["offset"], r["midi"]))
    return notes


def _read_full_pointset(csv_path: Path) -> list[tuple[float, int, int]]:
    """Full-piece point-set CSV → ordered ``[(ontime, midi, morphetic), ...]``
    sorted by ``(ontime, midi)`` (same order key as ``_kern_note_list``)."""
    rows = _read_occ_pointset(csv_path)
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def _ontime_key(ontime: float) -> float:
    """Quantise an ontime to a stable dict key (kill fixed-point/float dust)."""
    return round(ontime, 3)


def _calibrate_shift(kern_notes: list[dict], full_ps: list[tuple[float, int, int]]) -> float:
    """Find the constant ``ontime = offset_ql + shift`` between the kern note
    offsets (music21 quarterLengths from 0) and the point-set ontimes (beats,
    piece-specific origin — e.g. a pickup puts the origin at -1.0).

    For these JKUPDD pieces the beat == quarter note, so the kern→point-set map
    is a pure *translation*: we take the modal value of ``ontime - offset`` over
    same-MIDI (kern, point-set) candidate pairs in the opening of the piece (the
    dominant peak is unambiguous — see the diagnostic in the module docstring).
    """
    from collections import Counter, defaultdict

    ps_by_midi: dict[int, list[float]] = defaultdict(list)
    for o, m, _morph in full_ps:
        ps_by_midi[m].append(o)
    shifts: Counter = Counter()
    for kn in kern_notes[:120]:
        for o in ps_by_midi.get(kn["midi"], []):
            shifts[round(o - kn["offset"], 4)] += 1
    if not shifts:
        return 0.0
    return float(shifts.most_common(1)[0][0])


def _bridge_kern_to_pointset(
    kern_notes: list[dict], full_ps: list[tuple[float, int, int]]
) -> tuple[dict[tuple[float, int], dict], int, int]:
    """Build a direct ``(ontime, midi) -> kern_note`` map between the kern and
    the full-piece point-set.

    The kern and the point-set share a beat grid up to a constant per-piece
    origin shift (calibrated by :func:`_calibrate_shift`): a kern note at
    music21 offset ``off_ql`` has point-set ontime ``off_ql + shift``. We key
    every kern note by ``(_ontime_key(off_ql + shift), midi)``. Returns
    ``(bridge, n_pointset_matched, n_pointset)`` where ``n_pointset_matched`` is
    how many of the full point-set's notes have a kern note at the same
    (calibrated ontime, midi) — the **true piece-level alignment rate** (the
    edition-match signal). Notes the kern has but the point-set doesn't (grace
    notes, trill expansions) are harmless; they just sit unused in the bridge.
    """
    shift = _calibrate_shift(kern_notes, full_ps)
    bridge: dict[tuple[float, int], dict] = {}
    for kn in kern_notes:
        key = (_ontime_key(kn["offset"] + shift), kn["midi"])
        # First writer wins on exact-duplicate keys (chord unisons); the slice
        # is by measure span so which of two identical-pitch notes we keep does
        # not matter.
        bridge.setdefault(key, kn)
    n_matched = 0
    for ontime, midi, _morph in full_ps:
        if (_ontime_key(ontime), midi) in bridge:
            n_matched += 1
    return bridge, n_matched, len(full_ps)


def _match_occurrence(
    occ_rows: list[tuple[float, int, int]],
    ps_to_kern: dict[tuple[float, int], dict],
) -> tuple[list[dict], int]:
    """Resolve each occurrence (ontime, midi) row to its kern note via the
    point-set bridge.

    Returns ``(matched_notes, n_matched)``. An occurrence row resolves iff its
    exact ``(ontime, midi)`` key is in the bridge (occurrences are literal
    sub-sequences of the full point-set, so exact keys are expected). Rows that
    don't resolve are skipped and lower the reported match-rate. A small float
    tolerance on ontime guards against fixed-point dust in the CSVs.
    """
    matched: list[dict] = []
    n_matched = 0
    for ontime, midi, _morph in occ_rows:
        kn = ps_to_kern.get((_ontime_key(ontime), midi))
        if kn is not None:
            matched.append(kn)
            n_matched += 1
    return matched, n_matched


def _slice_abc(score, matched_notes: list[dict]):
    """Slice the measure span covering ``matched_notes`` and convert to ABC.

    Returns the interleaved-ABC string. We slice by *measure* range (the lowest
    to highest measure number touched by the matched notes) so the fragment is
    bar-aligned and carries its key/clef/meter context — exactly what makes the
    ABC path notation-faithful. ``score.measures(a, b)`` copies the governing
    KeySignature / Clef / TimeSignature into the first measure of the slice.
    """
    from marble.encoders.CLaMP3.abc_util import score_to_abc

    measures = set()
    for kn in matched_notes:
        m = kn["note"].getContextByClass("Measure")
        if m is not None and m.number is not None:
            measures.add(int(m.number))
    if not measures:
        raise RuntimeError("no measure numbers on matched notes")
    lo, hi = min(measures), max(measures)
    fragment = score.measures(lo, hi)
    try:
        abc = score_to_abc(fragment)
    except Exception as e:  # noqa: BLE001
        # music21's MusicXML export rejects an over-long ("duplex-maxima")
        # duration that a tie-continuation note inherits when the slice cuts a
        # tied chain. Resolving the ties to real durations fixes the export.
        if "duplex-maxima" not in str(e) and "too long" not in str(e):
            raise
        fragment = fragment.stripTies(retainContainers=True)
        abc = score_to_abc(fragment)
    return abc, (lo, hi)


def _renumber_duplicate_staves(kern_text: str) -> str:
    """Give every ``**kern`` spine a unique ``*staffN`` number.

    converter21's strict Humdrum reader raises ``*staffN interpretations have
    duplicated staff numbers`` when two spines share a staff (e.g. the JKUPDD
    Bach fugue ``wtc2f20.krn`` has ``*staff2  *staff1  *staff1`` — two right-hand
    voices notated on one staff). We rewrite the ``*staff`` interpretation line so
    each ``*staff`` token gets a distinct number (left→right, descending), which
    lets converter21 parse the file. This only changes the *staff grouping* in
    the rendered layout (two voices become two staves); every note's pitch /
    rhythm / measure is preserved, so the ABC motif content is unchanged.
    """
    lines = kern_text.splitlines()
    for i, ln in enumerate(lines):
        toks = ln.split("\t")
        staff_toks = [t for t in toks if t.startswith("*staff")]
        if len(staff_toks) < 2:
            continue
        nums = [t[len("*staff") :] for t in staff_toks]
        if len(set(nums)) == len(nums):
            continue  # already unique
        n = len(staff_toks)
        cnt = 0
        new = []
        for t in toks:
            if t.startswith("*staff"):
                new.append(f"*staff{n - cnt}")
                cnt += 1
            else:
                new.append(t)
        lines[i] = "\t".join(new)
        break
    return "\n".join(lines) + "\n"


def _parse_kern_robust(krn: Path):
    """Parse a ``**kern`` with converter21, repairing duplicate ``*staff``
    numbers (the only converter21 failure mode hit on the JKUPDD pieces).

    Returns ``(score, repaired: bool)``. ``repaired`` is True when the staff
    renumbering workaround was applied, so the caller can flag it.
    """
    import tempfile

    import music21

    try:
        return music21.converter.parse(str(krn)), False
    except Exception as e:  # noqa: BLE001
        if "duplicated staff numbers" not in str(e):
            raise
    fixed = _renumber_duplicate_staves(krn.read_text())
    tmp = tempfile.NamedTemporaryFile(suffix=".krn", delete=False, mode="w")  # noqa: SIM115
    tmp.write(fixed)
    tmp.close()
    try:
        return music21.converter.parse(tmp.name), True
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def build(jkupdd_root: Path, kern_dir: Path, mtf_jsonl: Path, min_match_rate: float) -> dict:
    from marble.encoders.CLaMP3.abc_util import _register_converter21

    _register_converter21()

    gt = jkupdd_root / "groundTruth"
    if not gt.is_dir():
        raise FileNotFoundError(f"groundTruth/ not found under {jkupdd_root}")
    if not mtf_jsonl.exists():
        raise FileNotFoundError(
            f"canonical MTF JSONL not found: {mtf_jsonl}\n"
            "  Build it first: uv run python scripts/data/build_jkupdd_retrieval.py "
            "--jkupdd-root <path>"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mtf_records = [json.loads(line) for line in mtf_jsonl.read_text().splitlines() if line.strip()]

    # Cache parsed score + (point-set→kern) bridge per piece (parsing kern is
    # the slow step). ``piece_agreement`` records the piece-level kern↔point-set
    # pitch-agreement (the edition-match signal) for the summary.
    score_cache: dict[str, object] = {}
    bridge_cache: dict[str, dict] = {}
    piece_agreement: dict[str, dict] = {}
    piece_usable: dict[str, bool] = {}

    def _piece_assets(piece_id: str):
        if piece_id not in score_cache:
            stem = PIECE_KERN_STEM[piece_id]
            krn = kern_dir / piece_id / f"{stem}.krn"
            ps_csv = kern_dir / piece_id / f"{stem}.csv"
            if not krn.exists():
                raise FileNotFoundError(f"**kern not found for {piece_id}: {krn}")
            if not ps_csv.exists():
                raise FileNotFoundError(f"full-piece point-set not found for {piece_id}: {ps_csv}")
            score, repaired = _parse_kern_robust(krn)
            kern_notes = _kern_note_list(score)
            full_ps = _read_full_pointset(ps_csv)
            # Degeneracy guard: converter21 sometimes parses a multi-voice score
            # (e.g. the Beethoven Op.2/1 mvt-3 Menuetto with written-out repeats)
            # into a structure where every note's offset-in-hierarchy collapses to
            # a single value — the ontime bridge is then meaningless. Detect it
            # (a large fraction of notes sharing one offset) and mark the piece
            # unusable so its occurrences are dropped with a clear reason rather
            # than silently mis-sliced.
            from collections import Counter as _C

            offs = [n["offset"] for n in kern_notes]
            top_frac = (_C(offs).most_common(1)[0][1] / len(offs)) if offs else 1.0
            degenerate = top_frac > 0.3
            bridge, n_agree, n_aligned = _bridge_kern_to_pointset(kern_notes, full_ps)
            score_cache[piece_id] = score
            bridge_cache[piece_id] = bridge
            piece_usable[piece_id] = not degenerate
            piece_agreement[piece_id] = {
                "n_kern_notes": len(kern_notes),
                "n_pointset_notes": len(full_ps),
                "n_aligned": n_aligned,
                "n_pitch_agree": n_agree,
                "pitch_agree_rate": round(n_agree / n_aligned, 4) if n_aligned else None,
                "staff_renumber_repaired": repaired,
                "offset_degenerate": degenerate,
                "usable": not degenerate,
            }
        return score_cache[piece_id], bridge_cache[piece_id]

    records: list[dict] = []
    match_rates: list[float] = []
    dropped: list[dict] = []

    for r in mtf_records:
        piece = r["piece_id"]
        annot = r["annotator"]
        letter = r["pattern"]
        occ_id = r["occurrence_id"]  # e.g. bachBWV889Fg__bruhn__A__occ1
        occ_name = occ_id.split("__")[-1]  # occ1

        # Locate the occurrence point-set CSV under groundTruth.
        # Layout: <piece>/<texture>/repeatedPatterns/<annot>/<letter>/occurrences/csv/<occN>.csv
        occ_csv = None
        for texture in ("polyphonic", "monophonic"):
            cand = (
                gt
                / piece
                / texture
                / "repeatedPatterns"
                / annot
                / letter
                / "occurrences"
                / "csv"
                / f"{occ_name}.csv"
            )
            if cand.exists():
                occ_csv = cand
                break
        if occ_csv is None:
            dropped.append({"occurrence_id": occ_id, "reason": "occurrence CSV not found"})
            continue

        occ_rows = _read_occ_pointset(occ_csv)
        if not occ_rows:
            dropped.append({"occurrence_id": occ_id, "reason": "empty occurrence CSV"})
            continue

        score, bridge = _piece_assets(piece)
        if not piece_usable.get(piece, True):
            dropped.append(
                {
                    "occurrence_id": occ_id,
                    "reason": "piece offsets degenerate (converter21 collapsed "
                    "offset-in-hierarchy) — ontime bridge unreliable",
                }
            )
            continue
        matched, n_matched = _match_occurrence(occ_rows, bridge)
        rate = n_matched / len(occ_rows)
        match_rates.append(rate)

        if rate < min_match_rate or not matched:
            dropped.append(
                {
                    "occurrence_id": occ_id,
                    "reason": f"match_rate {rate:.3f} < {min_match_rate}",
                    "n_occ_notes": len(occ_rows),
                    "n_matched": n_matched,
                }
            )
            continue

        try:
            abc, (lo, hi) = _slice_abc(score, matched)
        except Exception as e:  # noqa: BLE001
            dropped.append(
                {"occurrence_id": occ_id, "reason": f"abc-slice failed: {type(e).__name__}: {e}"}
            )
            continue

        records.append(
            {
                "abc": abc,
                "piece_id": piece,
                "annotator": annot,
                "pattern": letter,
                "group": r["group"],
                "occurrence_id": occ_id,
                "n_occ_notes": len(occ_rows),
                "n_matched": n_matched,
                "match_rate": round(rate, 4),
                "measure_span": [lo, hi],
                "split": "test",
            }
        )

    with open(JSONL_OUT, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # Also emit a MATCHED MTF subset — the original (lossy-MIDI) MTF records for
    # exactly the occurrences that survived ABC alignment. Running the MTF task
    # on this subset makes the ABC-vs-MTF A/B perfectly apples-to-apples (same
    # occurrence pool, same groups), instead of comparing 66 ABC windows against
    # the full-78 MTF leaderboard.
    surviving_ids = {rec["occurrence_id"] for rec in records}
    matched_mtf = [r for r in mtf_records if r["occurrence_id"] in surviving_ids]
    matched_path = OUT_DIR / "JKUPDDRetrieval.matched.test.jsonl"
    with open(matched_path, "w") as fh:
        for r in matched_mtf:
            fh.write(json.dumps(r) + "\n")

    # Group coverage on the surviving ABC set (a valid query pool needs >= 2).
    from collections import Counter

    gc = Counter(rec["group"] for rec in records)
    singleton_groups = [g for g, c in gc.items() if c < 2]

    return {
        "mtf_records": len(mtf_records),
        "abc_records": len(records),
        "dropped": len(dropped),
        "dropped_detail": dropped,
        "groups": len(gc),
        "singleton_groups_after_drop": singleton_groups,
        "mean_match_rate": round(sum(match_rates) / len(match_rates), 4) if match_rates else None,
        "min_match_rate_seen": round(min(match_rates), 4) if match_rates else None,
        "n_perfect_match": sum(1 for x in match_rates if x >= 1.0),
        "n_total_aligned": len(match_rates),
        "piece_kern_pointset_agreement": piece_agreement,
        "jsonl": str(JSONL_OUT),
        "matched_mtf_jsonl": str(matched_path),
        "matched_mtf_records": len(matched_mtf),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jkupdd-root",
        type=Path,
        required=True,
        help="path to the JKUPDD distribution dir that contains groundTruth/ "
        "(e.g. data/kern_sources/jkupdd_dataset/JKUPDD-Aug2013)",
    )
    ap.add_argument(
        "--kern-dir",
        type=Path,
        required=True,
        help="dir with <piece_id>/<stem>.krn (e.g. data/kern_sources/JKUPDD)",
    )
    ap.add_argument(
        "--mtf-jsonl",
        type=Path,
        default=DEFAULT_MTF_JSONL,
        help="canonical dedup'd MTF JSONL to mirror identity from "
        "(default: data/JKUPDD/JKUPDDRetrieval.test.jsonl)",
    )
    ap.add_argument(
        "--min-match-rate",
        type=float,
        default=0.9,
        help="drop an occurrence whose (ontime,pitch) join to the kern is below "
        "this rate (default 0.9 — keep near-perfect occurrences whose only "
        "unmatched rows are a handful of ornament/grace notes the kern spells "
        "differently; the measure span is still correct). Pass 1.0 to require a "
        "perfect join.",
    )
    args = ap.parse_args()
    stats = build(args.jkupdd_root, args.kern_dir, args.mtf_jsonl, args.min_match_rate)
    # Print stats without the (potentially long) per-drop detail inline; keep it
    # accessible but compact.
    summary = {k: v for k, v in stats.items() if k != "dropped_detail"}
    print(json.dumps(summary, indent=2))
    if stats["dropped_detail"]:
        print("\n-- dropped occurrences --", file=sys.stderr)
        for d in stats["dropped_detail"]:
            print(json.dumps(d), file=sys.stderr)


if __name__ == "__main__":
    main()
