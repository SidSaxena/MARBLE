#!/usr/bin/env python3
"""Build the BPS-Motif dataset as score-native **interleaved ABC** (Option B).

This is the ABC counterpart of ``scripts/data/build_bps_motif_dataset.py`` (which
synthesises a 60-QPM movement MIDI from ``csv_notes`` and slices per-window
sub-MIDIs for the MIDI -> MTF path). Instead of a MIDI window, each window here
gets a **single-voice interleaved-ABC** string built from the *exact same notes*
the MTF path embeds -- so the ABC-vs-MTF A/B is a clean encoding comparison at
fixed content, not a content confound.

Why "Option B (csv_notes-direct)"
---------------------------------
There is no kern / score-XML edition for BPS-Motif; the only symbolic source is
the upstream point-set CSVs. So we do NOT round-trip through a score and we do
NOT do any alignment. We reconstruct the window's notes **directly from
``csv_notes``**, mirroring the MTF window byte-for-byte in note content:

  * **Same windows.** We read the *already-built* MTF JSONLs
    (``data/BPS-Motif/BPSMotif.{MNID,Retrieval}.fold{F}.{split}.jsonl``) and emit
    one ABC record per MTF record, carrying the SAME ``piece_id`` / ``fold`` /
    ``split`` / ``is_motif`` / ``motif_letter`` / ``occurrence_id`` /
    ``start_sec`` / ``end_sec``. So the ABC task's labels / ``work_id`` /
    relevance are byte-identical to the MTF task and the per-layer numbers are
    directly comparable.

  * **Same notes.** The MTF builder synthesises a movement MIDI at 60 QPM
    (1 beat = 1 second; a negative pickup onset is clamped to t=0 with its
    duration preserved) then slices each window by *time* across **all staves**,
    clipping notes to the window edge and re-zeroing to the window start
    (:func:`build_bps_motif_dataset._slice_midi_window`). We replicate that exact
    slicing here -- same clamp, same all-staff overlap test, same clip+rezero --
    but carry each note's ``morphetic_number`` so we can spell pitches
    diatonically. NOTE: the MTF slice does NOT filter by ``track``; it takes
    every note overlapping ``[start, end)`` regardless of staff. We do the same,
    so the ABC note-set == the MTF window note-set.

  * **Single voice.** The matched notes are inserted into one music21 Part at
    their re-zeroed onsets (so simultaneous notes from the two staves become a
    bar-local cluster, exactly as they coexist in the MTF window), carrying the
    movement's key signature + time signature for correct ABC ``K:`` / ``M:``
    headers and pitch spelling, then ``makeMeasures`` + ``score_to_abc``.

Parity gate (run BEFORE any sweep)
----------------------------------
For every window we count ABC note-heads and the MTF window's note count
(from the sliced MIDI). With Option B these must be ~1.0 (ABC embeds the same
notes as the MTF window). We report mean/max of ABC-heads / MTF-notes and flag
any window > 1.3x for investigation. The build still writes the JSONLs, but the
printed stats are the gate -- do not run the sweep if parity is off.

Usage::

    uv run python scripts/data/build_bps_motif_abc.py \
        [--out-dir data/BPS-Motif] \
        [--upstream-dir data/BPS-Motif/_upstream/Beethoven_motif] \
        [--folds 0 1 2 3 4]

Outputs (one per existing MTF JSONL)::

    data/BPS-Motif/BPSMotifABC.MNID.fold{F}.{train,val,test}.jsonl
    data/BPS-Motif/BPSMotifABC.Retrieval.fold{F}.{train,val,test}.jsonl

and prints a stats JSON (parity + coverage).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# 60 QPM: 1 crotchet beat == 1 second, matching the MTF synth. csv onsets/
# durations are in crotchet beats; the MTF window times (start_sec/end_sec in
# the JSONL) are therefore the same numbers as the beat positions.
SCORE_TEMPO_QPM = 60.0

STEP_NAMES = ["C", "D", "E", "F", "G", "A", "B"]

# Notation grid for re-zeroed onsets + durations. The MTF arm tolerates any
# float (MIDI ticks); MusicXML/ABC cannot notate an arbitrary fraction and
# music21 raises "Cannot convert inexpressible durations to MusicXML". The
# NEGATIVE windows draw their [start,end) from rng.uniform, so their clipped
# durations land on un-notatable fractions (e.g. 0.2207 ql). Snapping onset +
# duration to a 1/12-quarter grid makes every value notatable while preserving
# the note COUNT exactly (verified: ABC note-heads == sliced notes) and keeping
# both duple (1/2,1/4,1/8) and triplet (1/3,1/6) subdivisions representable.
NOTE_GRID_QL = 1.0 / 12.0


# ── pitch spelling (midi + morphetic -> diatonic) ────────────────────────────


def _morphetic_to_diatonic(morph: int) -> tuple[str, int]:
    """Map a morphetic (diatonic staff-step) number to (step_letter, octave).

    Anchored on C4: upstream uses ``morphetic_number == 60`` for C4 (verified
    against ``csv_notes`` rows where ``midi_number == 60`` carries
    ``morphetic_number == 60``). 7 diatonic steps per octave.
    """
    rel = morph - 60
    octave = 4 + (rel // 7)
    step = STEP_NAMES[rel % 7]
    return step, octave


def _make_pitch(midi: int, morph: int):
    """Build a correctly-spelled music21 Pitch from (midi_number, morphetic).

    The morphetic number fixes the diatonic letter+octave (so F#/Gb don't get
    confused); the accidental is whatever makes the sounding pitch == ``midi``.
    Falls back to music21's default spelling if the morphetic-derived step is
    pathological (accidental magnitude > 2).
    """
    import music21

    step, octave = _morphetic_to_diatonic(morph)
    natural = music21.pitch.Pitch(step=step, octave=octave)
    alter = midi - natural.midi
    if abs(alter) > 2:
        # Morphetic spelling would need a triple+ accidental -- almost certainly
        # an out-of-range morphetic value; fall back to plain MIDI spelling.
        p = music21.pitch.Pitch()
        p.midi = midi
        return p
    p = music21.pitch.Pitch()
    p.step = step
    p.octave = octave
    if alter != 0:
        p.accidental = music21.pitch.Accidental(alter)
    return p


# ── csv parsing (mirrors build_bps_motif_dataset) ────────────────────────────


def _parse_notes_csv(path: Path) -> list[dict]:
    """csv_notes/<id>.csv -> note dicts (onset/dur in beats, midi+morphetic)."""
    notes: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                notes.append(
                    {
                        "onset_beats": float(row["onset"]),
                        "pitch": int(row["midi_number"]),
                        "morph": int(row["morphetic_number"]),
                        "duration_beats": float(row["duration"]),
                        "staff": int(row["staff_number"]),
                    }
                )
            except (ValueError, KeyError):
                continue
    return notes


def _synth_note_times(notes: list[dict]) -> list[dict]:
    """Apply the MTF synth's negative-onset clamp (start>=0, duration preserved).

    Mirrors :func:`build_bps_motif_dataset._synthesise_movement_midi`: a note with
    ``onset < 0`` (pickup) is moved to ``start = 0`` with the part of its
    duration that falls at/after 0 retained. Zero/negative-duration notes are
    dropped. Returns notes with absolute ``start``/``end`` (in seconds==beats).
    """
    out: list[dict] = []
    for n in notes:
        if n["duration_beats"] <= 0:
            continue
        start = n["onset_beats"]
        end = start + n["duration_beats"]
        if start < 0:
            duration = end - max(0.0, start)
            start = 0.0
            if duration <= 0:
                continue
            end = start + duration
        out.append({**n, "start": start, "end": end})
    return out


def _slice_window_notes(synth_notes: list[dict], start_sec: float, end_sec: float) -> list[dict]:
    """Replicate :func:`build_bps_motif_dataset._slice_midi_window` exactly.

    Take every note (any staff) overlapping ``[start_sec, end_sec)``; clip its
    onset/offset to the window; re-zero to the window start. A note clipped to
    zero/negative length is dropped (matches the MTF builder's guard).
    """
    sliced: list[dict] = []
    for n in synth_notes:
        if n["end"] <= start_sec or n["start"] >= end_sec:
            continue
        clipped_start = max(n["start"], start_sec) - start_sec
        clipped_end = min(n["end"], end_sec) - start_sec
        if clipped_end <= clipped_start:
            continue
        sliced.append(
            {
                "pitch": n["pitch"],
                "morph": n["morph"],
                "rel_start": clipped_start,
                "rel_dur": clipped_end - clipped_start,
                "staff": n["staff"],
            }
        )
    return sliced


# ── ABC build ────────────────────────────────────────────────────────────────


def _count_abc_noteheads(abc: str) -> int:
    """Count pitched note-heads in an interleaved-ABC body (parity diagnostic).

    Strips header / voice-decl / directive lines, drops ``[V:n]`` tags, then
    counts pitch tokens (optional ``^_=`` accidental + ``A-Ga-g`` + octave
    marks). Rests (z/x) are not heads. Mirrors the JKUPDD build's counter.
    """
    import re

    body = "\n".join(
        ln
        for ln in abc.splitlines()
        if ln and not re.match(r"^[A-Za-z%]:", ln) and not ln.startswith("%%")
    )
    body = re.sub(r"\[V:\d+\]", "", body)
    return len(re.findall(r"[_=^]*[A-Ga-g][,']*", body))


def _movement_key_meter(label_path: Path) -> tuple[int | None, str | None]:
    """Read the movement's time signature from csv_label (its ``TS`` column).

    BPS-Motif has no global key annotation; the upstream point-set is
    key-agnostic. We carry the time signature (so the ABC ``M:`` header is
    correct) and leave the key signature unset (C/a default) -- pitch SPELLING
    is preserved via the morphetic-derived accidentals regardless of ``K:``, so
    this does not affect the embedded notes, only the (cosmetic) key header.
    Returns ``(sharps, ts_ratio_string)``; sharps is always None here.
    """
    ts = None
    try:
        with label_path.open(newline="") as f:
            for row in csv.DictReader(f):
                ts = (row.get("TS") or "").strip() or None
                if ts:
                    break
    except OSError:
        pass
    return None, ts


def _build_window_abc(sliced: list[dict], ts_ratio: str | None) -> str:
    """Build a single-voice interleaved-ABC string from the sliced window notes.

    Notes are inserted into one music21 Part at their re-zeroed onsets (so
    cross-staff simultaneities sit at the same offset, exactly as they coexist
    in the MTF window). The movement time signature is carried for a correct
    ``M:`` header; ``makeMeasures`` bar-aligns; ``score_to_abc`` does
    xml2abc + interleave (the same converter the JKUPDD/SuperMario ABC paths use).
    """
    import music21

    from marble.encoders.CLaMP3.abc_util import score_to_abc

    if not sliced:
        raise RuntimeError("empty window -- no notes to build ABC from")

    part = music21.stream.Part()
    if ts_ratio:
        try:
            part.insert(0, music21.meter.TimeSignature(ts_ratio))
        except Exception:  # noqa: BLE001 -- exotic/blank TS string; skip the header
            pass
    for n in sliced:
        note = music21.note.Note()
        note.pitch = _make_pitch(n["pitch"], n["morph"])
        # Snap onset + duration to the notation grid so MusicXML/ABC can express
        # them (the negative windows' rng.uniform boundaries otherwise yield
        # inexpressible fractions). Count is preserved; a duration that rounds to
        # 0 is floored to one grid step so the note still renders.
        rel_start = round(n["rel_start"] / NOTE_GRID_QL) * NOTE_GRID_QL
        rel_dur = round(n["rel_dur"] / NOTE_GRID_QL) * NOTE_GRID_QL
        if rel_dur <= 0:
            rel_dur = NOTE_GRID_QL
        note.duration.quarterLength = rel_dur
        part.insert(rel_start, note)

    sc = music21.stream.Score()
    sc.insert(0, part)
    sc.makeMeasures(inPlace=True)

    try:
        return score_to_abc(sc)
    except Exception as e:  # noqa: BLE001
        # music21 MusicXML export can reject an over-long duration on a clipped
        # tie-continuation; resolve ties and retry (mirrors the JKUPDD build).
        if "duplex-maxima" not in str(e) and "too long" not in str(e):
            raise
        sc = sc.stripTies(retainContainers=True)
        return score_to_abc(sc)


# ── parallel worker ──────────────────────────────────────────────────────────
#
# The bottleneck is per-occurrence ``score_to_abc`` (music21 MusicXML write +
# an xml2abc subprocess + abctoolkit interleave) — each occurrence is fully
# independent, so we farm them out to a process pool. The parent does the cheap
# work (parse CSV, slice the window, count the MTF MIDI) and hands each worker a
# small ``(occ_id, sliced_notes, ts_ratio, n_mtf)`` job; the worker returns
# ``(occ_id, abc_or_None, n_abc_heads, n_mtf, error)``. converter21 is registered
# once per worker process.

_WORKER_READY = False


def _worker_init() -> None:
    global _WORKER_READY
    if not _WORKER_READY:
        from marble.encoders.CLaMP3.abc_util import _register_converter21

        _register_converter21()
        _WORKER_READY = True


def _build_one(job: tuple) -> tuple:
    occ_id, sliced, ts_ratio, n_mtf = job
    _worker_init()
    try:
        abc = _build_window_abc(sliced, ts_ratio)
    except Exception as e:  # noqa: BLE001
        return (occ_id, None, 0, n_mtf, f"{type(e).__name__}: {e}")
    return (occ_id, abc, _count_abc_noteheads(abc), n_mtf, None)


# ── driver ───────────────────────────────────────────────────────────────────


def build(out_dir: Path, upstream: Path, folds: list[int], workers: int) -> dict:
    import mido

    tasks = ("MNID", "Retrieval")
    splits = ("train", "val", "test")

    # Cache per movement: synthesised note set + time signature.
    synth_cache: dict[str, list[dict]] = {}
    ts_cache: dict[str, str | None] = {}

    def _movement_assets(piece_id: str) -> tuple[list[dict], str | None]:
        if piece_id not in synth_cache:
            notes = _parse_notes_csv(upstream / "csv_notes" / f"{piece_id}.csv")
            synth_cache[piece_id] = _synth_note_times(notes)
            _, ts = _movement_key_meter(upstream / "csv_label" / f"{piece_id}.csv")
            ts_cache[piece_id] = ts
        return synth_cache[piece_id], ts_cache[piece_id]

    def _mtf_note_count(midi_path: Path) -> int | None:
        if not midi_path.exists():
            return None
        try:
            mf = mido.MidiFile(str(midi_path))
        except Exception:  # noqa: BLE001
            return None
        return sum(
            1 for tr in mf.tracks for msg in tr if msg.type == "note_on" and msg.velocity > 0
        )

    # ── Pass 1: scan all MTF JSONLs → unique occurrence jobs ────────────────
    # An occurrence's window is identical across its task/fold/split copies, so
    # we build each ABC ONCE. Collect the per-occurrence record (for metadata)
    # plus the slice job.
    occ_record: dict[str, dict] = {}  # occ_id → one representative MTF record
    jobs: list[tuple] = []
    nsliced_by_occ: dict[str, int] = {}
    nmtf_by_occ: dict[str, int | None] = {}
    for task in tasks:
        for fold in folds:
            for split in splits:
                src = out_dir / f"BPSMotif.{task}.fold{fold}.{split}.jsonl"
                if not src.exists():
                    raise FileNotFoundError(
                        f"MTF JSONL not found: {src}\n  Build the MTF dataset first:\n"
                        "    uv run python scripts/data/build_bps_motif_dataset.py"
                    )
                for line in src.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    occ = r["occurrence_id"]
                    if occ in occ_record:
                        continue
                    occ_record[occ] = r
                    synth_notes, ts = _movement_assets(r["piece_id"])
                    sliced = _slice_window_notes(synth_notes, r["start_sec"], r["end_sec"])
                    nsliced_by_occ[occ] = len(sliced)
                    nmtf = _mtf_note_count(REPO / r["midi_path"])
                    nmtf_by_occ[occ] = nmtf
                    jobs.append((occ, sliced, ts, nmtf))

    print(
        f"[build] {len(jobs)} unique occurrences to build across "
        f"{len(tasks)} tasks x {len(folds)} folds; workers={workers}",
        file=sys.stderr,
        flush=True,
    )

    # ── Pass 2: build ABCs in parallel ──────────────────────────────────────
    abc_by_occ: dict[str, str | None] = {}
    nabc_by_occ: dict[str, int] = {}
    failures: list[dict] = []
    done = 0
    if workers <= 1:
        _worker_init()
        results_iter = (_build_one(j) for j in jobs)
    else:
        from concurrent.futures import ProcessPoolExecutor

        ex = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
        results_iter = ex.map(_build_one, jobs, chunksize=8)
    try:
        for occ_id, abc, n_abc, n_mtf, err in results_iter:
            abc_by_occ[occ_id] = abc
            nabc_by_occ[occ_id] = n_abc
            if err is not None:
                failures.append({"occurrence_id": occ_id, "error": err})
            done += 1
            if done % 500 == 0:
                print(f"[build]   ... {done}/{len(jobs)} ABCs built", file=sys.stderr, flush=True)
    finally:
        if workers > 1:
            ex.shutdown()

    # ── parity stats over unique occurrences ────────────────────────────────
    ratios: list[float] = []
    offenders: list[dict] = []
    for occ, n_mtf in nmtf_by_occ.items():
        if abc_by_occ.get(occ) is None or not n_mtf:
            continue
        ratio = nabc_by_occ[occ] / n_mtf
        ratios.append(ratio)
        if ratio > 1.3:
            offenders.append(
                {
                    "occurrence_id": occ,
                    "n_abc": nabc_by_occ[occ],
                    "n_mtf": n_mtf,
                    "n_sliced": nsliced_by_occ[occ],
                    "ratio": round(ratio, 3),
                }
            )

    # ── Pass 3: emit ABC JSONLs (re-scan so per-fold/split membership holds) ─
    counts: dict[str, int] = defaultdict(int)
    for task in tasks:
        for fold in folds:
            for split in splits:
                src = out_dir / f"BPSMotif.{task}.fold{fold}.{split}.jsonl"
                dst = out_dir / f"BPSMotifABC.{task}.fold{fold}.{split}.jsonl"
                recs_out: list[dict] = []
                for line in src.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    occ = r["occurrence_id"]
                    abc = abc_by_occ.get(occ)
                    if abc is None:
                        continue  # build failed; skip this record everywhere
                    out_rec = dict(r)
                    out_rec["abc"] = abc
                    out_rec["n_abc_notes"] = nabc_by_occ[occ]
                    out_rec["n_mtf_notes"] = nmtf_by_occ[occ]
                    recs_out.append(out_rec)
                with dst.open("w") as fh:
                    for rec in recs_out:
                        fh.write(json.dumps(rec) + "\n")
                counts[f"{task}.fold{fold}.{split}"] = len(recs_out)

    def _stat(xs):
        return (
            {
                "mean": round(sum(xs) / len(xs), 4),
                "max": round(max(xs), 4),
                "min": round(min(xs), 4),
                "n": len(xs),
            }
            if xs
            else None
        )

    n_perfect = sum(1 for r in ratios if abs(r - 1.0) < 1e-9)
    return {
        "unique_occurrences": len(jobs),
        "abc_build_failures": len(failures),
        "abc_build_failure_detail": failures[:20],
        "jsonl_record_counts": dict(counts),
        "content_parity_abc_over_mtf": _stat(ratios),
        "n_perfect_parity": n_perfect,
        "n_parity_pairs": len(ratios),
        "offenders_gt_1p3x": offenders[:30],
        "n_offenders_gt_1p3x": len(offenders),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir", default="data/BPS-Motif", help="MTF JSONL dir (default: data/BPS-Motif)"
    )
    ap.add_argument(
        "--upstream-dir",
        default=None,
        help="Beethoven_motif clone (default: <out-dir>/_upstream/Beethoven_motif)",
    )
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel ABC-build worker processes (default: os.cpu_count() − 1). "
        "Each occurrence's score_to_abc is independent; the xml2abc subprocess is "
        "the bottleneck, so this scales near-linearly. Pass 1 to run serially.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    upstream = (
        Path(args.upstream_dir) if args.upstream_dir else out_dir / "_upstream" / "Beethoven_motif"
    )
    if not (upstream / "csv_notes").is_dir():
        print(f"ERROR: csv_notes/ not found under {upstream}", file=sys.stderr)
        sys.exit(1)

    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) - 1)

    stats = build(out_dir, upstream, args.folds, workers)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
