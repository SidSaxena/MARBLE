#!/usr/bin/env python3
"""
scripts/data/build_bps_motif_dataset.py
────────────────────────────────────────
Build the BPS-Motif dataset for MARBLE.

Source: https://github.com/Wiilly07/Beethoven_motif
  Hsiao, Hung, Chen, Su (ISMIR 2023) — leitmotif-style annotations on the
  first movements of all 32 Beethoven piano sonatas. ~127k notes, 263
  distinct motifs, 4,944 occurrences. Symbolic only (CSV + the original
  MIDIs are NOT shipped — only motif-only MIDIs at 60 QPM). License
  CC-BY-4.0 per the Zenodo record (zenodo.org/records/10265277).

Pipeline
--------
  1. Clone the upstream annotation repo into data/BPS-Motif/_upstream/.
  2. For each of the 32 movements:
     a. Parse csv_notes/<id>-1.csv → full note table.
     b. Parse csv_label/<id>-1.csv → motif occurrence windows.
     c. Synthesise a full-movement MIDI from the note table at 60 QPM
        (so csv_label.start_midi / end_midi seconds align by definition).
     d. Slice the full MIDI into per-occurrence MIDIs (positives) AND
        sample equal-sized random non-motif windows (negatives). Each
        sliced MIDI is saved under data/BPS-Motif/midi_windows/.
  3. Apply movement-level 5-fold CV (seed 1234) — sonata-level splits
     so motif occurrences never leak across folds.
  4. Emit
       data/BPS-Motif/BPSMotif.MNID.fold{0..4}.{train,val,test}.jsonl
       data/BPS-Motif/BPSMotif.Retrieval.fold{0..4}.{train,val,test}.jsonl

Why window-level MNID (and not per-note like Hsiao TISMIR'24)
-------------------------------------------------------------
CLaMP3-symbolic outputs per-PATCH embeddings, where one patch ≈ one bar
of M3-tokenised MTF. Per-note labels from csv_notes don't map cleanly to
patches without recovering time-signature-aware bar boundaries through
midi_to_mtf (which is fragile across Beethoven's varying time sigs:
2/2, 2/4, 4/4, 6/8, 3/4, 12/8, 3/8). v1 ships window-level binary
classification (motif span vs. equal-sized random non-motif window) —
simpler, robust, and uses MARBLE's existing TimeAvgPool + BaseTask
pipeline without modification. Numbers reported here are NOT directly
comparable to Hsiao's per-note F1=0.721; they answer a related but
distinct question ("does this clip contain a motif?"). A per-note v2
can layer on top once we have results from v1.

Output JSONL schemas
--------------------
MNID — one row per window (positive motif spans + sampled negatives):
  {
    "midi_path":     "data/BPS-Motif/midi_windows/01-1__a__0.mid",
    "piece_id":      "01-1",
    "fold":          0,
    "split":         "train" | "val" | "test",
    "is_motif":      1,                       # 1 = motif span, 0 = sampled non-motif
    "motif_letter":  "a",                     # for positives; "neg" for negatives
    "occurrence_id": "01-1__a__0",            # for positives; "01-1__neg__0" etc.
    "start_sec":     3.0,
    "end_sec":       8.0
  }

Retrieval — same as MNID but ONLY positive (motif) rows. Probe queries
each motif window against the others in the SAME movement and scores
on motif-letter match.

Prerequisites
-------------
  git              — for cloning the upstream repo
  pretty_midi      — synthesising full-movement MIDIs (project dep)

Usage
-----
  # Full build (~30 s total at default settings, ~230 MB upstream + ~5 MB output)
  uv run python scripts/data/build_bps_motif_dataset.py

  # Pilot on a few movements
  uv run python scripts/data/build_bps_motif_dataset.py --max-movements 4

  # Re-emit JSONLs from already-cloned upstream + already-synthesised MIDIs
  uv run python scripts/data/build_bps_motif_dataset.py --skip-clone --skip-midi

  # Override the upstream location (if you cloned it manually)
  uv run python scripts/data/build_bps_motif_dataset.py \\
      --upstream-dir /path/to/Beethoven_motif

Audio variant
-------------
This script handles the symbolic side only. The audio variant — for probing
MERT/MuQ/OMARRQ on real Beethoven performances — needs:
  - Real recordings (the user sources separately; do NOT synthesise from
    these MIDIs, the score-time mapping won't survive interpretive tempo)
  - DTW alignment of recordings to the score-time MIDIs
  - A separate `audio_path` field on the JSONL
Out of scope here; see docs/data/bps_motif_setup.md once it exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # pretty_midi is imported lazily inside _synthesise_movement_midi to
    # keep the --help flow snappy; this stub makes type annotations resolvable
    # to the static linter without paying the runtime import cost.
    import pretty_midi  # noqa: F401

log = logging.getLogger(__name__)

_UPSTREAM_REPO = "https://github.com/Wiilly07/Beethoven_motif.git"

# Score-time tempo for the synthesised full-movement MIDIs.
# csv_label.start_midi/end_midi are emitted by upstream at 60 QPM, so we
# render at 60 QPM too — every beat = exactly 1 second. The CLaMP3 M3
# patchiliser is event-based and tempo-agnostic, so this only matters for
# the seconds↔beats correspondence in the retrieval probe metadata.
SCORE_TEMPO_QPM = 60.0


# ── Upstream repo fetch ──────────────────────────────────────────────────────


def _clone_or_update_upstream(dest: Path) -> Path:
    """Clone the BPS-Motif annotation repo or pull if present."""
    repo_dir = dest / "Beethoven_motif"
    if repo_dir.exists() and (repo_dir / ".git").exists():
        log.info(f"  found existing clone: {repo_dir} (pulling)")
        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "pull", "--ff-only", "--quiet"],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            log.warning("  git pull timed out — using existing checkout as-is")
    else:
        log.info(f"  cloning {_UPSTREAM_REPO} → {repo_dir}")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", _UPSTREAM_REPO, str(repo_dir)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.error(
                "git clone failed: %s. Install git and verify network access, "
                "or clone manually and pass --upstream-dir.",
                e,
            )
            sys.exit(1)
    return repo_dir


def _discover_movements(upstream: Path) -> list[str]:
    """Return sorted list of sonata IDs (e.g. ['01-1', '02-1', ..., '32-1'])."""
    csv_notes_dir = upstream / "csv_notes"
    if not csv_notes_dir.is_dir():
        log.error(f"Expected csv_notes/ at {csv_notes_dir} — wrong upstream layout?")
        sys.exit(1)
    ids = sorted(p.stem for p in csv_notes_dir.glob("*.csv"))
    if not ids:
        log.error(f"No CSVs found under {csv_notes_dir}")
        sys.exit(1)
    return ids


# ── Note + label parsing ─────────────────────────────────────────────────────


def _parse_notes_csv(path: Path) -> list[dict]:
    """Parse csv_notes/<id>-1.csv → ordered list of note dicts.

    Columns: onset, midi_number, morphetic_number, duration, staff_number,
    measure, type. The `type` column is the motif letter (lowercase) or
    empty for non-motivic notes.
    """
    notes: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                notes.append(
                    {
                        "onset_beats": float(row["onset"]),
                        "pitch": int(row["midi_number"]),
                        "duration_beats": float(row["duration"]),
                        "staff": int(row["staff_number"]),
                        "measure": int(row["measure"]),
                        "motif_letter": row["type"].strip() or None,
                    }
                )
            except (ValueError, KeyError) as e:
                log.warning(f"  skipping malformed note row in {path.name}: {e}")
    return notes


def _parse_label_csv(path: Path) -> list[dict]:
    """Parse csv_label/<id>-1.csv → list of motif occurrences.

    Columns (relevant ones): type, start_beat, duration, track, start_midi,
    end_midi. start_midi/end_midi are seconds at 60 QPM.
    """
    out: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out.append(
                    {
                        "motif_letter": row["type"].strip(),
                        "start_beat": float(row["start"]),
                        "end_beat": float(row["end"]),
                        "start_sec": float(row["start_midi"]),
                        "end_sec": float(row["end_midi"]),
                        "track": int(row["track"]),
                    }
                )
            except (ValueError, KeyError) as e:
                log.warning(f"  skipping malformed label row in {path.name}: {e}")
    return out


# ── MIDI synthesis from notes ────────────────────────────────────────────────


def _synthesise_movement_midi(notes: list[dict], out_path: Path) -> pretty_midi.PrettyMIDI:
    """Write a full-movement MIDI at 60 QPM containing all notes.

    Onset/duration are in crotchet beats; at 60 QPM each beat = 1 second
    so the resulting MIDI's timing matches csv_label.start_midi/end_midi.
    Returns the in-memory PrettyMIDI so the caller can slice it without
    re-loading from disk.
    """
    import pretty_midi  # heavy import; lazy

    pm = pretty_midi.PrettyMIDI(initial_tempo=SCORE_TEMPO_QPM)
    # One instrument per distinct staff so the M3 patchiliser sees them as
    # separate tracks (matches how Beethoven's two-staff piano part is
    # usually voiced in MIDI). Acoustic Grand Piano = program 0.
    by_staff: dict[int, pretty_midi.Instrument] = {}
    for n in notes:
        if n["duration_beats"] <= 0:
            continue
        start = n["onset_beats"]
        end = start + n["duration_beats"]
        # Negative onset (pickup measure) — clamp to t=0 for the MIDI
        # write since pretty_midi handles negative times poorly, but the
        # csv_label seconds use the same upstream convention so motif
        # spans we slice later already account for this.
        if start < 0:
            duration = end - max(0.0, start)
            start = 0.0
            if duration <= 0:
                continue
            end = start + duration
        inst = by_staff.setdefault(
            n["staff"],
            pretty_midi.Instrument(program=0, name=f"staff_{n['staff']}"),
        )
        inst.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=n["pitch"],
                start=start,
                end=end,
            )
        )
    pm.instruments.extend(sorted(by_staff.values(), key=lambda i: i.name))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    return pm


def _slice_midi_window(
    pm: pretty_midi.PrettyMIDI,
    start_sec: float,
    end_sec: float,
    out_path: Path,
) -> bool:
    """Write a new MIDI containing only the notes that fall in [start, end).

    Note times are re-zeroed (the new MIDI starts at t=0). Returns False
    if the window ends up empty — the caller skips empty windows.
    """
    import pretty_midi

    new_pm = pretty_midi.PrettyMIDI(initial_tempo=SCORE_TEMPO_QPM)
    has_notes = False
    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, name=inst.name)
        for note in inst.notes:
            # Include note if it overlaps [start, end) at all.
            if note.end <= start_sec or note.start >= end_sec:
                continue
            # Clip to the window and re-zero.
            clipped_start = max(note.start, start_sec) - start_sec
            clipped_end = min(note.end, end_sec) - start_sec
            if clipped_end <= clipped_start:
                continue
            new_inst.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=clipped_start,
                    end=clipped_end,
                )
            )
            has_notes = True
        if new_inst.notes:
            new_pm.instruments.append(new_inst)
    if not has_notes:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_pm.write(str(out_path))
    return True


def _sample_negative_windows(
    occurrences: list[dict],
    movement_end_sec: float,
    n_negatives: int,
    rng: random.Random,
    max_attempts: int = 100,
) -> list[tuple[float, float]]:
    """Sample non-overlapping random windows that don't overlap any motif.

    Window durations are drawn from the positive distribution (the same
    set of motif durations), so positives and negatives have matched
    length statistics and the model can't trivially classify on length.
    """
    if not occurrences:
        return []
    durations = [occ["end_sec"] - occ["start_sec"] for occ in occurrences]
    motif_spans = sorted((occ["start_sec"], occ["end_sec"]) for occ in occurrences)

    def overlaps_any(start: float, end: float) -> bool:
        return any(not (end <= ms or start >= me) for ms, me in motif_spans)

    negatives: list[tuple[float, float]] = []
    for _ in range(n_negatives):
        for _attempt in range(max_attempts):
            dur = rng.choice(durations)
            if movement_end_sec <= dur:
                break  # movement too short for any window of this duration
            start = rng.uniform(0.0, movement_end_sec - dur)
            end = start + dur
            if not overlaps_any(start, end):
                negatives.append((start, end))
                break
    return negatives


# ── Build records per movement ───────────────────────────────────────────────


def _build_window_records(
    piece_id: str,
    pm: pretty_midi.PrettyMIDI,
    occurrences: list[dict],
    windows_dir: Path,
    rng: random.Random,
    negative_ratio: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    """Slice positive (motif) windows + sample negative windows.

    Returns ``(positive_records, negative_records)`` — both are list[dict]
    with the same JSONL schema. Retrieval uses only positives; MNID uses
    both. Each record's ``midi_path`` points at a fresh per-window MIDI
    sliced from the movement.
    """
    positives: list[dict] = []
    counters: dict[str, int] = defaultdict(int)

    # Stable per-letter occurrence index so occurrence_ids are reproducible.
    for occ in occurrences:
        letter = occ["motif_letter"]
        idx = counters[letter]
        counters[letter] += 1
        occ_id = f"{piece_id}__{letter}__{idx}"
        window_path = windows_dir / f"{occ_id}.mid"
        # csv_label.start_midi can be slightly negative for pickup-driven
        # occurrences; clip the slice range to >= 0.
        start = max(0.0, occ["start_sec"])
        end = occ["end_sec"]
        if end <= start:
            log.warning(f"  ! {occ_id}: empty span ({start} → {end}), skipping")
            continue
        ok = _slice_midi_window(pm, start, end, window_path)
        if not ok:
            log.warning(f"  ! {occ_id}: no notes in window, skipping")
            continue
        positives.append(
            {
                "midi_path": str(window_path),
                "piece_id": piece_id,
                "is_motif": 1,
                "motif_letter": letter,
                "occurrence_id": occ_id,
                "start_sec": start,
                "end_sec": end,
            }
        )

    # Sample negatives. Use the movement's actual end time from the
    # synthesised MIDI rather than relying on csv_label which only covers
    # motif spans.
    movement_end = pm.get_end_time() if pm.instruments else 0.0
    n_neg = max(1, int(round(len(positives) * negative_ratio)))
    neg_spans = _sample_negative_windows(occurrences, movement_end, n_neg, rng)
    negatives: list[dict] = []
    for neg_idx, (start, end) in enumerate(neg_spans):
        occ_id = f"{piece_id}__neg__{neg_idx}"
        window_path = windows_dir / f"{occ_id}.mid"
        if not _slice_midi_window(pm, start, end, window_path):
            continue
        negatives.append(
            {
                "midi_path": str(window_path),
                "piece_id": piece_id,
                "is_motif": 0,
                "motif_letter": "neg",
                "occurrence_id": occ_id,
                "start_sec": start,
                "end_sec": end,
            }
        )
    return positives, negatives


# ── 5-fold CV split ──────────────────────────────────────────────────────────


def _five_fold_splits(
    movements: list[str],
    seed: int = 1234,
) -> dict[int, dict[str, list[str]]]:
    """Movement-level 5-fold CV with a small val carve-out.

    For each fold k: test = ~6 movements (1/5 of 32), val = ~4 movements
    from the remaining train pool, train = the rest. Sonata-level splits
    so motif occurrences never leak across folds.

    Returns {fold_idx: {"train": [...], "val": [...], "test": [...]}}.
    """
    rng = random.Random(seed)
    shuffled = sorted(movements)  # deterministic baseline
    rng.shuffle(shuffled)
    n = len(shuffled)
    folds: dict[int, dict[str, list[str]]] = {}
    for k in range(5):
        test = shuffled[k::5]  # ~6 movements
        trainval = [m for m in shuffled if m not in set(test)]
        # Carve val from the END of trainval so train indices are stable
        # across folds (cleaner reproducibility).
        val_size = max(2, len(trainval) // 6)  # ~4 movements
        val = trainval[-val_size:]
        train = trainval[:-val_size]
        folds[k] = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    return folds


# ── JSONL emit ───────────────────────────────────────────────────────────────


def _emit_split_jsonls(
    out_dir: Path,
    probe: str,
    fold: int,
    records_by_piece: dict[str, list[dict]],
    splits: dict[str, list[str]],
) -> dict[str, int]:
    """Write BPSMotif.<probe>.fold<fold>.{train,val,test}.jsonl."""
    counts = {}
    for split_name in ("train", "val", "test"):
        piece_ids = splits[split_name]
        path = out_dir / f"BPSMotif.{probe}.fold{fold}.{split_name}.jsonl"
        n_records = 0
        with path.open("w") as f:
            for pid in piece_ids:
                for rec in records_by_piece.get(pid, []):
                    rec_out = dict(rec)
                    rec_out["fold"] = fold
                    rec_out["split"] = split_name
                    f.write(json.dumps(rec_out) + "\n")
                    n_records += 1
        counts[split_name] = n_records
        log.info(f"    {path.name}: {n_records} records")
    return counts


# ── Main pipeline ────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out-dir", default="data/BPS-Motif", help="Output directory (default: data/BPS-Motif)"
    )
    ap.add_argument(
        "--upstream-dir",
        default=None,
        help="Path to a pre-cloned Beethoven_motif repo (skips clone)",
    )
    ap.add_argument(
        "--skip-clone",
        action="store_true",
        help="Don't clone; expect upstream at <out-dir>/_upstream/Beethoven_motif",
    )
    ap.add_argument(
        "--skip-midi",
        action="store_true",
        help="Don't synthesise per-movement MIDIs (assume they exist)",
    )
    ap.add_argument(
        "--max-movements", type=int, default=None, help="Pilot: process only the first N movements"
    )
    ap.add_argument(
        "--seed", type=int, default=1234, help="Random seed for 5-fold CV split (default: 1234)"
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = out_dir / "midi"
    windows_dir = out_dir / "midi_windows"
    rng = random.Random(args.seed)

    # ── Step 1: resolve upstream ────────────────────────────────────────────
    if args.upstream_dir:
        upstream = Path(args.upstream_dir)
        if not upstream.is_dir():
            log.error(f"--upstream-dir {upstream} does not exist")
            sys.exit(1)
    elif args.skip_clone:
        upstream = out_dir / "_upstream" / "Beethoven_motif"
        if not upstream.is_dir():
            log.error(f"--skip-clone but {upstream} not found; clone first")
            sys.exit(1)
    else:
        upstream = _clone_or_update_upstream(out_dir / "_upstream")

    # ── Step 2: discover movements ──────────────────────────────────────────
    movements = _discover_movements(upstream)
    if args.max_movements:
        movements = movements[: args.max_movements]
    log.info(f"Found {len(movements)} movements: {movements[:3]}…{movements[-1]}")

    # ── Step 3: parse + synthesise + slice per movement ─────────────────────
    # MNID gets both positives + negatives; Retrieval gets positives only.
    mnid_records: dict[str, list[dict]] = defaultdict(list)
    retrieval_records: dict[str, list[dict]] = defaultdict(list)
    total_positives = 0
    total_negatives = 0
    motif_letter_counter: Counter[str] = Counter()

    import pretty_midi as _pm_mod

    for piece_id in movements:
        notes_path = upstream / "csv_notes" / f"{piece_id}.csv"
        label_path = upstream / "csv_label" / f"{piece_id}.csv"
        if not notes_path.exists() or not label_path.exists():
            log.warning(f"  ! {piece_id}: missing CSVs, skipping")
            continue

        notes = _parse_notes_csv(notes_path)
        occurrences = _parse_label_csv(label_path)
        midi_path = midi_dir / f"{piece_id}.mid"

        if args.skip_midi:
            # Load the existing full-movement MIDI rather than re-synthesising.
            if not midi_path.exists():
                log.error(f"  ! --skip-midi but {midi_path} not found")
                sys.exit(1)
            pm = _pm_mod.PrettyMIDI(str(midi_path))
        else:
            pm = _synthesise_movement_midi(notes, midi_path)

        positives, negatives = _build_window_records(
            piece_id,
            pm,
            occurrences,
            windows_dir,
            rng,
        )
        # Retrieval: positives only.
        retrieval_records[piece_id].extend(positives)
        # MNID: positives AND negatives, in stable order so the dataloader's
        # determinism doesn't depend on dict iteration order.
        mnid_records[piece_id].extend(positives)
        mnid_records[piece_id].extend(negatives)

        total_positives += len(positives)
        total_negatives += len(negatives)
        motif_letter_counter.update(p["motif_letter"] for p in positives)
        log.info(
            f"  {piece_id}: {len(notes)} notes, {len(occurrences)} occurrences → "
            f"{len(positives)} pos + {len(negatives)} neg windows "
            f"({len({p['motif_letter'] for p in positives})} distinct letters)"
        )

    log.info(
        f"\nTotal: {len(movements)} movements, "
        f"{total_positives} positive windows, {total_negatives} negative windows. "
        f"Top motif letters: {motif_letter_counter.most_common(5)}"
    )

    # ── Step 4: 5-fold CV split + emit JSONLs ───────────────────────────────
    folds = _five_fold_splits(movements, seed=args.seed)
    for fold, splits in folds.items():
        log.info(
            f"\nFold {fold}: train={len(splits['train'])} "
            f"val={len(splits['val'])} test={len(splits['test'])}"
        )
        log.info("  MNID:")
        _emit_split_jsonls(out_dir, "MNID", fold, mnid_records, splits)
        log.info("  Retrieval:")
        _emit_split_jsonls(out_dir, "Retrieval", fold, retrieval_records, splits)

    log.info(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
