#!/usr/bin/env python3
"""Build the MTC-ANN CLaMP3 retrieval datasets — two tasks × two encodings.

MTC-ANN-2.0.1 (Meertens Tune Collection, Annotated subset) ships 360 monophonic
Dutch-folksong melodies as Humdrum ``**kern`` (+ ``**text`` lyrics) with two
ground-truth annotation layers we turn into zero-shot CLaMP3 retrieval tasks:

* **Tune-family** (``--task tunefamily``) — one fragment per WHOLE melody;
  relevance = same tune family (``MTC-ANN-tune-family-labels.csv``). 360
  fragments / 26 families.
* **Motif-class** (``--task motif``) — one fragment per annotated motif
  OCCURRENCE (a contiguous note span inside a melody); relevance = same motif
  class. Spans come from ``MTC-ANN-motifs.csv`` (``startindex``/``endindex``,
  0-based inclusive over the melody's flattened notes — see ALIGNMENT below).

Each task is emitted in **two arms** that feed the *same* CLaMP3 M3 patchiliser
so the per-layer MAP is an apples-to-apples ABC-vs-MTF comparison:

* **ABC arm** — score-native interleaved ABC (key / pitch-spelling / meter / bar
  structure preserved), built with the shared converter
  ``marble/encoders/CLaMP3/abc_util.py`` (``kern_to_abc`` for whole melodies,
  ``score_to_abc`` for re-zeroed single-voice motif fragments).
* **MTF arm** — a matched MIDI of the *same notes* (built here from the same
  music21 fragment, written to ``data/MTC-ANN/midi/`` and tokenised downstream
  via ``midi_to_mtf`` exactly like JKUPDD/BPS-Motif).

The JSONL schema matches the JKUPDD retrieval datasets so the harness datamodule
reads them unchanged:

  * ABC arm record: ``{abc, group, occurrence_id, work_id_src, split, ...}``
    (read by ``_JKUPDDRetrievalABCDataset`` → ``abc`` / ``group`` /
    ``occurrence_id``; ``work_id = sha1(group)``).
  * MTF arm record: ``{midi_path, group, occurrence_id, work_id_src, split, ...}``
    (read by ``_JKUPDDRetrievalDataset`` → ``midi_path`` / ``group`` /
    ``occurrence_id``).

``group`` is the relevance key (CoverRetrievalTask counts two clips relevant iff
their ``work_id = sha1(group)`` match):
  * tune-family task → ``group = family``
  * motif task       → ``group = "<family>|<motifclass>"`` (see GROUPING below).

ALIGNMENT (verified, not assumed)
---------------------------------
Motif ``startindex``/``endindex`` are **0-based, end-inclusive** indices into the
melody's ``score.flatten().notes`` list (offset-sorted; MTC-ANN is monophonic so
each element is one Note, no chords). Evidence: ``endindex - startindex + 1 ==
numberofnotes`` for **all 1657 rows**, and the slice's start-offset equals the
annotated ``begintime`` for 84% of rows (the other 16% are songs with an
anacrusis/pickup measure, where ``begintime`` counts beats from the first
downbeat while music21 offsets count from the pickup — a constant per-song origin
shift that does NOT move the note-index slice). The ``motifclass`` mnemonic
(e.g. ``1:bag``) is a tune-family-level class *name*, NOT a literal pitch slice of
each melody (a motif realises transposed across melodies — e.g. ``1:bag`` is F#-E-D
in a D-major melody), so it is the relevance label, not an alignment oracle; the
note-index slice is the authoritative content.

GROUPING (annotator + class)
----------------------------
The motif CSV has 3 annotators (ann1: 699 occ / 163 melodies / 36 family-classes;
ann2: 656 / 128 / 50; ann3: 302 / 79 / 17). Every ``motifclass`` spans multiple
melodies, so cross-melody positives exist; but the ``1:``/``2:`` class *names*
recur across DIFFERENT tune families (33 global vs 36 family-scoped classes for
ann1), so we key relevance on ``family|motifclass`` to avoid conflating same-named
classes in different families. To avoid the **annotator confound** (3 partly-
overlapping analyses inflating MAP with near-duplicate spans) we use a SINGLE
annotator for v1; ``--annotator ann1`` (the broadest: most occurrences AND most
melodies) is the default. No two rows share an identical ``(song, startindex,
endindex)`` span across the whole CSV, so there are no literal duplicates to drop.

Usage (run on the PC where converter21/music21/abctoolkit + the data live)::

    .venv/bin/python scripts/data/build_mtc_ann_dataset.py \
        --mtc-root data/MTC-ANN-2.0.1/MTC-ANN-2.0.1 \
        --task tunefamily --workers 14
    .venv/bin/python scripts/data/build_mtc_ann_dataset.py \
        --mtc-root data/MTC-ANN-2.0.1/MTC-ANN-2.0.1 \
        --task motif --annotator ann1 --workers 14

Outputs (``data/MTC-ANN/``):
  MTCANN.TuneFamily.ABC.jsonl  MTCANN.TuneFamily.MTF.jsonl
  MTCANN.Motif.ABC.jsonl       MTCANN.Motif.MTF.jsonl
plus the matched MIDIs under ``data/MTC-ANN/midi/{tunefamily,motif}/``.
Prints a stats JSON (counts, parity, alignment evidence).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "data" / "MTC-ANN"
MIDI_DIR = OUT_DIR / "midi"


# ─── kern parsing (music21 / converter21 — the same parser the ABC path uses) ──


def _flatten_notes(score):
    """Offset-sorted list of the melody's Notes (monophonic → no chords, no rests).

    The motif ``startindex``/``endindex`` are indices into THIS list (0-based,
    end-inclusive), so the parser must match what the annotations counted. We use
    ``score.flatten().notes`` (excludes rests) sorted by hierarchy offset.
    """
    notes = [n for n in score.flatten().notes if n.isNote]
    notes.sort(key=lambda n: (float(n.getOffsetInHierarchy(score)), int(n.pitch.midi)))
    return notes


def _parse_kern(krn_path: Path):
    import music21

    return music21.converter.parse(str(krn_path))


# ─── fragment → MIDI (the MTF arm's matched input) ─────────────────────────────


def _notes_to_part(recs, key_sharps, ts_ratio):
    """Build a re-zeroed single-voice music21 Part from ``[(offset, midi, dur)]``.

    Mirrors ``build_jkupdd_abc._build_motif_abc``: re-zero to the fragment's first
    onset, carry the governing key/meter so the ABC headers reflect real context,
    bar-align with ``makeMeasures``. Returns a ``Score`` ready for both
    ``score_to_abc`` (ABC arm) and ``.write('midi')`` (MTF arm) — guaranteeing the
    two arms embed byte-for-byte the same notes.
    """
    import music21

    min_off = min(r[0] for r in recs)
    part = music21.stream.Part()
    if key_sharps is not None:
        part.insert(0, music21.key.KeySignature(key_sharps))
    if ts_ratio is not None:
        part.insert(0, music21.meter.TimeSignature(ts_ratio))
    for off, midi, dur in recs:
        n = music21.note.Note()
        n.pitch.midi = midi
        n.duration.quarterLength = dur if dur and dur > 0 else 0.5
        part.insert(off - min_off, n)
    sc = music21.stream.Score()
    sc.insert(0, part)
    sc.makeMeasures(inPlace=True)
    return sc


def _fragment_to_abc(sc):
    from marble.encoders.CLaMP3.abc_util import score_to_abc

    try:
        return score_to_abc(sc)
    except Exception as e:  # noqa: BLE001 — over-long tie-continuation durations
        if "duplex-maxima" not in str(e) and "too long" not in str(e):
            raise
        return score_to_abc(sc.stripTies(retainContainers=True))


def _write_midi(sc, midi_path: Path) -> None:
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    sc.write("midi", fp=str(midi_path))


def _count_abc_noteheads(abc: str) -> int:
    """Pitched note-heads in interleaved-ABC (parity diagnostic).

    Extends ``build_jkupdd_abc._count_abc_noteheads`` to also strip body
    decorations that leak letters into a naive pitch-token regex — whole MTC-ANN
    melodies (unlike the short JKUPDD motif slices) carry ``!fermata!`` /
    ``!decoration!`` ornaments (the letters f-e-r-m-a... look like pitches) and
    inline ``[M:..]`` / ``[K:..]`` field changes. We drop header lines, ``[V:n]``
    voice tags, ``!...!`` decorations, quoted text, and ``[X:..]`` inline fields
    before counting pitch tokens, so the count equals the real note count.
    """
    import re as _re

    body = "\n".join(
        ln
        for ln in abc.splitlines()
        if ln and not _re.match(r"^[A-Za-z%]:", ln) and not ln.startswith("%%")
    )
    body = _re.sub(r"\[V:\d+\]", "", body)  # voice tags
    body = _re.sub(r"\[[A-Za-z]:[^\]]*\]", "", body)  # inline fields [M:4/4] [K:G] ...
    body = _re.sub(r"![^!]*!", "", body)  # !fermata! !decoration! ...
    body = _re.sub(r'"[^"]*"', "", body)  # "chord"/annotation text
    return len(_re.findall(r"[_=^]*[A-Ga-g][,']*", body))


def _count_midi_notes(midi_path: Path) -> int | None:
    try:
        import mido
    except ImportError:
        return None
    if not midi_path.exists():
        return None
    n = 0
    for tr in mido.MidiFile(str(midi_path)).tracks:
        for msg in tr:
            if msg.type == "note_on" and msg.velocity > 0:
                n += 1
    return n


# ─── per-fragment worker ───────────────────────────────────────────────────────


def _build_one(job: dict) -> dict:
    """Build one fragment's ABC + MTF arms. Runs in a worker process.

    ``job`` carries the krn path, the note-index span (or None = whole melody),
    the ``group``/``occurrence_id``/provenance, and the MIDI output path. Returns
    a record dict with both arms' fields + parity counts, or an ``error`` dict.
    """
    from marble.encoders.CLaMP3.abc_util import _register_converter21, kern_to_abc

    _register_converter21()
    try:
        krn = Path(job["krn"])
        span = job["span"]  # None or (si, ei)
        midi_path = Path(job["midi_path"])

        if span is None:
            # Whole-melody: ABC straight from the kern (full notation fidelity).
            abc = kern_to_abc(krn)
            score = _parse_kern(krn)
            notes = _flatten_notes(score)
            recs = [
                (
                    float(n.getOffsetInHierarchy(score)),
                    int(n.pitch.midi),
                    float(n.duration.quarterLength),
                )
                for n in notes
            ]
            first = notes[0]
            key_obj = first.getContextByClass("Key") or first.getContextByClass("KeySignature")
            ts = first.getContextByClass("TimeSignature")
            sc = _notes_to_part(
                recs,
                key_obj.sharps if key_obj is not None else None,
                ts.ratioString if ts is not None else None,
            )
            n_src = len(notes)
        else:
            si, ei = span
            score = _parse_kern(krn)
            notes = _flatten_notes(score)
            if ei >= len(notes):
                return {"error": "endindex out of range", "occurrence_id": job["occurrence_id"]}
            sl = notes[si : ei + 1]
            recs = [
                (
                    float(n.getOffsetInHierarchy(score)),
                    int(n.pitch.midi),
                    float(n.duration.quarterLength),
                )
                for n in sl
            ]
            first = sl[0]
            key_obj = first.getContextByClass("Key") or first.getContextByClass("KeySignature")
            ts = first.getContextByClass("TimeSignature")
            sc = _notes_to_part(
                recs,
                key_obj.sharps if key_obj is not None else None,
                ts.ratioString if ts is not None else None,
            )
            abc = _fragment_to_abc(sc)
            n_src = len(sl)

        # MTF arm: write the SAME fragment to MIDI (same notes by construction).
        _write_midi(sc, midi_path)

        n_abc = _count_abc_noteheads(abc)
        n_mtf = _count_midi_notes(midi_path)
        rec = {
            "abc": abc,
            "midi_path": str(midi_path.relative_to(REPO)),
            "group": job["group"],
            "occurrence_id": job["occurrence_id"],
            "work_id_src": job["group"],
            "n_src_notes": n_src,
            "n_abc_notes": n_abc,
            "n_mtf_notes": n_mtf,
            "split": "test",
        }
        rec.update(job["provenance"])
        return rec
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "occurrence_id": job.get("occurrence_id")}


# ─── job enumeration ───────────────────────────────────────────────────────────


def _tunefamily_jobs(mtc_root: Path) -> list[dict]:
    krn_dir = mtc_root / "krn"
    fam_csv = mtc_root / "metadata" / "MTC-ANN-tune-family-labels.csv"
    fam: dict[str, str] = {}
    with open(fam_csv) as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            fam[row[0].strip()] = row[1].strip()
    jobs = []
    for songid, family in sorted(fam.items()):
        krn = krn_dir / f"{songid}.krn"
        if not krn.exists():
            continue
        jobs.append(
            {
                "krn": str(krn),
                "span": None,
                "group": family,
                "occurrence_id": songid,
                "midi_path": str(MIDI_DIR / "tunefamily" / f"{songid}.mid"),
                "provenance": {"songid": songid, "family": family},
            }
        )
    return jobs


def _motif_jobs(mtc_root: Path, annotator: str) -> list[dict]:
    krn_dir = mtc_root / "krn"
    motif_csv = mtc_root / "metadata" / "MTC-ANN-motifs.csv"
    jobs = []
    with open(motif_csv) as fh:
        for row in csv.reader(fh):
            if len(row) < 12:
                continue
            ann = row[11].strip().strip('"')
            if ann != annotator:
                continue
            family = row[0].strip()
            songid = row[1].strip()
            motifid = row[2].strip()
            try:
                si, ei = int(row[6]), int(row[7])
            except ValueError:
                continue
            motifclass = row[9].strip().strip('"')
            krn = krn_dir / f"{songid}.krn"
            if not krn.exists():
                continue
            group = f"{family}|{motifclass}"
            occ_id = motifid  # e.g. NLB015569_01_00 — already unique per occurrence
            jobs.append(
                {
                    "krn": str(krn),
                    "span": (si, ei),
                    "group": group,
                    "occurrence_id": occ_id,
                    "midi_path": str(MIDI_DIR / "motif" / f"{occ_id}.mid"),
                    "provenance": {
                        "songid": songid,
                        "family": family,
                        "motifclass": motifclass,
                        "annotator": ann,
                        "startindex": si,
                        "endindex": ei,
                    },
                }
            )
    return jobs


# ─── orchestration ─────────────────────────────────────────────────────────────


def _run(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [_build_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_build_one, jobs))


def _summarise(records: list[dict], errors: list[dict], task: str) -> dict:
    from collections import Counter

    gc = Counter(r["group"] for r in records)
    singletons = [g for g, c in gc.items() if c < 2]

    def _ratios(num, den):
        return [r[num] / r[den] for r in records if r.get(num) and r.get(den)]

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

    abc_over_src = _ratios("n_abc_notes", "n_src_notes")
    abc_over_mtf = _ratios("n_abc_notes", "n_mtf_notes")
    mtf_over_src = _ratios("n_mtf_notes", "n_src_notes")
    offenders = [
        {
            "occurrence_id": r["occurrence_id"],
            "n_src": r["n_src_notes"],
            "n_abc": r["n_abc_notes"],
            "n_mtf": r.get("n_mtf_notes"),
        }
        for r in records
        if r.get("n_src_notes")
        and (
            r["n_abc_notes"] / r["n_src_notes"] > 1.5
            or (r.get("n_mtf_notes") and r["n_mtf_notes"] / r["n_src_notes"] > 1.5)
        )
    ]
    return {
        "task": task,
        "fragments": len(records),
        "errors": len(errors),
        "error_detail": errors[:20],
        "groups": len(gc),
        "singleton_groups": singletons,
        "group_size_min": min(gc.values()) if gc else None,
        "group_size_max": max(gc.values()) if gc else None,
        "content_parity": {
            "abc_over_src": _stat(abc_over_src),
            "mtf_over_src": _stat(mtf_over_src),
            "abc_over_mtf": _stat(abc_over_mtf),
            "offenders_gt_1p5x": offenders,
        },
    }


def _write_arm(records: list[dict], abc_path: Path, mtf_path: Path) -> None:
    """Emit the two arm JSONLs. ABC arm carries ``abc`` (no midi); MTF arm carries
    ``midi_path`` (no abc) — each arm's datamodule reads only its own field."""
    abc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(abc_path, "w") as fh:
        for r in records:
            rec = {k: v for k, v in r.items() if k != "midi_path"}
            fh.write(json.dumps(rec) + "\n")
    with open(mtf_path, "w") as fh:
        for r in records:
            rec = {k: v for k, v in r.items() if k != "abc"}
            fh.write(json.dumps(rec) + "\n")


def build(mtc_root: Path, task: str, annotator: str, workers: int, smoke: bool) -> dict:
    if task == "tunefamily":
        jobs = _tunefamily_jobs(mtc_root)
        abc_out = OUT_DIR / "MTCANN.TuneFamily.ABC.jsonl"
        mtf_out = OUT_DIR / "MTCANN.TuneFamily.MTF.jsonl"
    elif task == "motif":
        jobs = _motif_jobs(mtc_root, annotator)
        abc_out = OUT_DIR / "MTCANN.Motif.ABC.jsonl"
        mtf_out = OUT_DIR / "MTCANN.Motif.MTF.jsonl"
    else:
        raise ValueError(task)

    if smoke:
        # Serial-vs-parallel byte-identity smoke test on a 16-job prefix.
        sample = jobs[:16]
        serial = _run(sample, workers=1)
        parallel = _run(sample, workers=min(8, len(sample)))

        def _key(recs):
            # Compare on the deterministic ABC + parity fields (midi_path is a
            # stable name; the MIDI bytes are re-derived deterministically too,
            # but we compare the in-memory record to isolate compute determinism).
            return [
                (r.get("occurrence_id"), r.get("abc"), r.get("n_abc_notes"), r.get("n_mtf_notes"))
                for r in recs
            ]

        identical = _key(serial) == _key(parallel)
        return {"smoke_serial_vs_parallel_identical": identical, "n_smoke_jobs": len(sample)}

    records_raw = _run(jobs, workers)
    records = [r for r in records_raw if "error" not in r]
    errors = [r for r in records_raw if "error" in r]
    _write_arm(records, abc_out, mtf_out)
    stats = _summarise(records, errors, task)
    stats["annotator"] = annotator if task == "motif" else None
    stats["abc_jsonl"] = str(abc_out)
    stats["mtf_jsonl"] = str(mtf_out)
    return stats


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mtc-root", type=Path, required=True, help="MTC-ANN-2.0.1/MTC-ANN-2.0.1 dir")
    ap.add_argument("--task", choices=["tunefamily", "motif"], required=True)
    ap.add_argument(
        "--annotator",
        default="ann1",
        help="motif task: single annotator (default ann1, broadest coverage)",
    )
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="run only the serial-vs-parallel byte-identity check on a 16-job prefix",
    )
    args = ap.parse_args()
    stats = build(args.mtc_root, args.task, args.annotator, args.workers, args.smoke)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
