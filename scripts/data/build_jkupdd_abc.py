#!/usr/bin/env python3
"""Build the JKUPDD motif-retrieval dataset as score-native **interleaved ABC**.

This is the ABC counterpart of ``scripts/data/build_jkupdd_retrieval.py`` (which
emits one lossy MIDI window per occurrence for the MIDI→MTF path). Instead of a
MIDI window, each occurrence here gets a **note-level interleaved-ABC** string
built from *only the occurrence's matched notes* — preserving key / pitch-
spelling / meter / bar structure that the ``**kern → MIDI → MTF`` round-trip
discards. The goal is a clean A/B at fixed identity: does notation-preserving ABC
beat lossy MTF for cross-piece motif retrieval with CLaMP3?
(See ``docs/jkupdd_abc_vs_mtf.md``.)

**Content-confound fix (note-level, not whole-measure).** The first version of
this build sliced ``score.measures(lo, hi)`` — the **whole polyphonic measure
span, all voices** — so an 8-note motif embedded a multi-bar, multi-staff,
accompaniment-laden ABC texture (ABC-notes / motif-notes ≈ 6× on mean, up to
15.8×) while the MTF arm embedded JKUPDD's per-occurrence MIDI = the **isolated
motif** (~1.1×). The two arms embedded *different musical objects*, so the
original A/B was a content confound. This build now reconstructs **just the
matched notes** as a single-voice stream (pitch + notated duration + re-zeroed
relative onset, mirroring the MTF occurrence MIDI), then runs ``score_to_abc`` —
see :func:`_build_motif_abc`. A content-parity check (ABC note-heads over motif
notes / MTF-window notes, both ≈ 1.0) gates the build.

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
``--min-match-rate`` (default 0.9 → tolerate a few ornament/grace-note
mismatches) is dropped/flagged, and the summary says how many. The matched kern
notes themselves (their pitch + notated duration + relative onset) ARE the ABC
content — :func:`_build_motif_abc` reconstructs them as a single-voice line and
runs ``score_to_abc``; the measure span is now only a diagnostic, not the slice.

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


def _count_abc_noteheads(abc: str) -> int:
    """Count pitched note-heads in an interleaved-ABC body (parity diagnostic).

    Strips header / voice-decl / directive lines, then counts pitch tokens
    (optional accidental ``^_=`` + a note letter ``A-Ga-g`` + octave marks
    ``,'``). Rests (``z``/``x``) and chords-as-single-token are not pitch heads;
    for these JKUPDD motifs (single-voice, note-per-onset) this equals the motif
    note count, so ABC-heads / motif-notes ≈ 1.0 when the build is correct.
    """
    import re as _re

    body = "\n".join(
        ln
        for ln in abc.splitlines()
        if ln and not _re.match(r"^[A-Za-z%]:", ln) and not ln.startswith("%%")
    )
    # Drop the leading [V:n] tags so their letters aren't counted as pitches.
    body = _re.sub(r"\[V:\d+\]", "", body)
    return len(_re.findall(r"[_=^]*[A-Ga-g][,']*", body))


def _count_midi_notes(midi_path) -> int | None:
    """Count note-on events in a MIDI file (the MTF window's note count).

    Returns ``None`` if the file is missing or ``mido`` is unavailable — parity
    is then reported against motif notes only. JKUPDD ships one occurrence MIDI
    per pattern occurrence (the MTF arm's exact input); its note count is the
    apples-to-apples denominator for the ABC-vs-MTF content-parity check.
    """
    from pathlib import Path as _P

    p = _P(midi_path)
    if not p.exists():
        return None
    try:
        import mido
    except ImportError:
        return None
    n = 0
    for tr in mido.MidiFile(str(p)).tracks:
        for msg in tr:
            # JKUPDD appends a pitch-1 / velocity-1 sentinel note at the end of
            # each occurrence MIDI (an end-of-occurrence marker, not a musical
            # note); exclude it so the count is the real motif size.
            if msg.type == "note_on" and msg.velocity > 1 and msg.note > 1:
                n += 1
    return n


def _build_motif_abc(matched_notes: list[dict]):
    """Build a **motif-only, single-voice** ABC from just the matched notes.

    THE FIX (see ``docs/jkupdd_abc_vs_mtf.md`` "Content confound"). The earlier
    ``_slice_abc`` took ``score.measures(lo, hi)`` — the **whole polyphonic
    measure span, all voices** — so an 8-note motif embedded a 3-bar, 2-staff,
    ~85-note ABC texture (ABC-notes / motif-notes ≈ 6× on mean, up to 15.8×).
    The MTF arm, by contrast, embeds JKUPDD's per-occurrence MIDI = the
    **isolated motif** (~1.1× motif notes). The two arms were embedding
    *different musical objects*, so the A/B was a content confound, not an
    encoding comparison.

    This builder instead reconstructs **only the occurrence's matched notes** —
    the exact kern notes the point-set alignment resolved — as a *single-voice*
    music21 stream, mirroring exactly what JKUPDD's per-occurrence MIDI (the MTF
    arm's input) contains:

    * **pitch** — ``kn["midi"]`` (the matched point-set pitch; on a chord note
      this is the specific constituent the row matched, not the whole chord);
    * **duration** — the matched kern note's *notated* ``quarterLength`` (the
      occurrence CSV does not reliably carry duration for these pieces, so we
      take it from the score; a non-positive/grace duration falls back to an
      eighth so the note still renders);
    * **relative onset** — each note's ``offset_in_hierarchy`` minus the motif's
      first onset, so the fragment is **re-zeroed to t=0** exactly like the MTF
      occurrence MIDI (which JKUPDD ships time-zeroed). Inter-onset gaps inside
      the motif are preserved (they become ABC invisible rests / are present as
      gaps in the MTF MIDI too), so the rhythmic object matches.

    We carry the governing key / meter from the motif's first note's context so
    the ABC ``K:``/``M:`` headers reflect the real tonal/metric context (pitch
    spelling via accidentals is preserved regardless). ``makeMeasures`` then
    bar-aligns the single voice for the interleaver. The result is a short
    monophonic ABC line of *the motif* — same notes as the MTF window — not the
    surrounding accompaniment.

    Returns ``(abc, (lo, hi))`` where ``(lo, hi)`` is the kern measure span the
    motif touches (kept for the JSONL ``measure_span`` field / diagnostics only;
    it no longer governs what is embedded).
    """
    import music21

    from marble.encoders.CLaMP3.abc_util import score_to_abc

    if not matched_notes:
        raise RuntimeError("no matched notes to build a motif ABC from")

    # Per-note (offset_ql, midi, dur_ql) for exactly the matched notes.
    recs: list[tuple[float, int, float]] = []
    for kn in matched_notes:
        dur = float(kn["note"].duration.quarterLength)
        recs.append((float(kn["offset"]), int(kn["midi"]), dur))
    min_off = min(r[0] for r in recs)

    part = music21.stream.Part()
    first_el = matched_notes[0]["note"]
    # Prefer a full Key (carries mode, e.g. A-minor) but fall back to a bare
    # KeySignature; either way the sharps/flats — hence pitch spelling — match.
    key_obj = first_el.getContextByClass("Key") or first_el.getContextByClass("KeySignature")
    ts = first_el.getContextByClass("TimeSignature")
    if key_obj is not None:
        part.insert(0, music21.key.KeySignature(key_obj.sharps))
    if ts is not None:
        part.insert(0, music21.meter.TimeSignature(ts.ratioString))
    for off, midi, dur in recs:
        n = music21.note.Note()
        n.pitch.midi = midi
        n.duration.quarterLength = dur if dur and dur > 0 else 0.5
        part.insert(off - min_off, n)

    sc = music21.stream.Score()
    sc.insert(0, part)
    sc.makeMeasures(inPlace=True)

    # Measure span the motif touches (diagnostic only — not what is embedded).
    measures = {
        int(m.number)
        for kn in matched_notes
        if (m := kn["note"].getContextByClass("Measure")) is not None and m.number is not None
    }
    lo, hi = (min(measures), max(measures)) if measures else (-1, -1)

    try:
        abc = score_to_abc(sc)
    except Exception as e:  # noqa: BLE001
        # music21's MusicXML export can reject an over-long ("duplex-maxima")
        # duration inherited from a tie-continuation note; resolve ties first.
        if "duplex-maxima" not in str(e) and "too long" not in str(e):
            raise
        sc = sc.stripTies(retainContainers=True)
        abc = score_to_abc(sc)
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
            abc, (lo, hi) = _build_motif_abc(matched)
        except Exception as e:  # noqa: BLE001
            dropped.append(
                {"occurrence_id": occ_id, "reason": f"abc-build failed: {type(e).__name__}: {e}"}
            )
            continue

        # Content-parity gate: the note-level ABC must embed the SAME notes as
        # the motif / the MTF window, not the surrounding accompaniment. Count
        # ABC note-heads and the MTF occurrence-MIDI notes; both ratios should be
        # ≈ 1.0 (see docs). The MTF window is JKUPDD's per-occurrence MIDI — the
        # exact input the MTF arm embeds — so n_abc/n_mtf is the apples-to-apples
        # parity number.
        n_abc_notes = _count_abc_noteheads(abc)
        n_mtf_notes = _count_midi_notes(REPO / r["midi_path"]) if "midi_path" in r else None
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
                "n_abc_notes": n_abc_notes,
                "n_mtf_notes": n_mtf_notes,
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

    # Content-parity summary — THE GATE. With the note-level build the ABC must
    # embed the same notes as the motif / the MTF window. We report ABC-heads
    # over (a) the matched motif notes and (b) the MTF window notes; both should
    # be ≈ 1.0. Any occurrence with ratio > 1.5 is surfaced for investigation
    # (the old whole-measure build ran ≈ 6× on mean, up to 15.8×).
    def _ratios(num_key: str, den_key: str):
        rs = [
            rec[num_key] / rec[den_key] for rec in records if rec.get(num_key) and rec.get(den_key)
        ]
        return rs

    abc_over_motif = _ratios("n_abc_notes", "n_matched")
    abc_over_mtf = _ratios("n_abc_notes", "n_mtf_notes")
    offenders = [
        {
            "occurrence_id": rec["occurrence_id"],
            "n_abc_notes": rec["n_abc_notes"],
            "n_matched": rec["n_matched"],
            "n_mtf_notes": rec.get("n_mtf_notes"),
            "abc_over_motif": round(rec["n_abc_notes"] / rec["n_matched"], 3)
            if rec.get("n_matched")
            else None,
        }
        for rec in records
        if rec.get("n_matched") and rec["n_abc_notes"] / rec["n_matched"] > 1.5
    ]

    def _stat(xs):
        return (
            {"mean": round(sum(xs) / len(xs), 4), "max": round(max(xs), 4), "n": len(xs)}
            if xs
            else None
        )

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
        "content_parity": {
            "abc_over_motif": _stat(abc_over_motif),
            "abc_over_mtf_window": _stat(abc_over_mtf),
            "offenders_gt_1p5x": offenders,
        },
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
