#!/usr/bin/env python3
"""Build the BPS-Motif **within-piece phrase-window** dataset (2-voice ABC).

The within-piece counterpart of ``scripts/data/build_bps_motif_abc.py``. Where
that builder slices one ABC fragment per labelled motif OCCURRENCE (for the
cross-piece within-letter Retrieval task), THIS builder slides ``N``-bar phrase
windows (stride 1) over each WHOLE movement and emits one ABC fragment per
window. The downstream task (``BPSMotifWithinPieceTask``) measures, per movement,
whether two phrase windows that share a motif letter retrieve each other —
genuine within-movement recurrence — via
``compute_within_group_multilabel_map``. The semantics are the
shuffle-control-validated leitmotifs prototype
(``leitmotifs-symbolic/scripts/eval/bps_within_piece_metric.py`` +
``bps_motif_assemble.py``); this script ports that assembler into MARBLE.

Four ported corrections (a–d) — see the prototype's ``bps_motif_assemble.py``:

  (a) **Physical bars from music21's own measure structure.** After
      ``makeMeasures()`` we read each note's physical ``measureNumber`` (1-based,
      sequential along the score), NEVER ``csv_notes.measure`` (which RESETS on
      repeats). Occurrence spans (csv_label ``start``/``end`` in absolute crotchet
      beats) map to physical measures by intersecting ``[start, end)`` with each
      measure's actual ``[offset, offset+barDuration)`` — meter-agnostic, no
      onset→bar arithmetic.
  (b) **Snap onsets + durations to a 1/12-quarter grid** before MusicXML export,
      so music21 never raises "Cannot convert inexpressible durations" (~11/32
      movements die otherwise).
  (c) **2-voice ABC** — split notes by ``staff_number`` into two music21 Parts
      (lowest staff → V1, rest → V2); ``makeMeasures`` then interleaved ABC
      yields ``[V:1]``/``[V:2]`` (NOT melody-only).
  (d) **Parse float-formatted int columns via ``int(float(s))``** (21-1 writes
      ``midi_number``/``morphetic_number``/``measure`` as ``46.0`` etc.).

Per window we emit a JSONL row::

    {movement_id, window_id, bar_start, bar_end, abc, letters:[...],
     occurrence_ids:[...], n_bars, split:"test"}

``abc`` = ``score_to_abc`` (``marble.encoders.CLaMP3.abc_util``) on the
measure-sliced fragment (``[V:1]``/``[V:2]`` asserted). ``letters`` = the union of
motif letters over the window's physical bars. ``occurrence_ids`` = occurrences
whose physical bar-span intersects the window's bars (occ_id =
``f"{mov}:{letter}:{start_beat}"``).

Usage::

    uv run python scripts/data/build_bps_motif_within_piece.py \
        [--upstream-dir data/BPS-Motif/_upstream/Beethoven_motif] \
        [--out-dir data/BPS-Motif] --window 4 [--stride 1] [--workers N]

Outputs::

    data/BPS-Motif/BPSMotifWithinPiece.N{N}.ABC.jsonl
    data/BPS-Motif/BPSMotifWithinPiece.N{N}.ABC.stats.json   (provenance)

The ``--upstream-dir`` accepts BOTH layouts (it only needs
``<dir>/csv_notes``, ``<dir>/csv_label``, ``<dir>/pickup.csv``):

  * the PC ``_upstream/Beethoven_motif`` clone, and
  * the Mac ``data/kern_sources/BPS-Motif/beethoven_motif_csv`` checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Notation grid for onsets + durations (correction b). Mirrors MARBLE's
# build_bps_motif_abc NOTE_GRID_QL: 1/12 quarter keeps both duple (1/2,1/4,1/8)
# and triplet (1/3,1/6) subdivisions notatable so MusicXML export never raises.
NOTE_GRID_QL = 1.0 / 12.0

STEP_NAMES = ["C", "D", "E", "F", "G", "A", "B"]


# ── CSV parsing (correction d: int(float(s))) ────────────────────────────────


def _to_int(s: str) -> int:
    """Parse an int that may be formatted as a float string (e.g. ``46.0``)."""
    return int(float(s))


def parse_notes_csv(path: Path) -> list[dict]:
    """csv_notes/<id>.csv → note dicts (onset/dur in crotchet beats)."""
    rows: list[dict] = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "onset": float(row["onset"]),
                        "midi": _to_int(row["midi_number"]),
                        "morphetic": _to_int(row["morphetic_number"]),
                        "dur": float(row["duration"]),
                        "staff": _to_int(row["staff_number"]),
                        "letter": (row.get("type") or "").strip(),
                    }
                )
            except (ValueError, KeyError):
                continue
    return rows


def parse_label_csv(path: Path) -> list[dict]:
    """csv_label/<id>.csv → occurrence rows.

    ``start``/``end`` are absolute crotchet beats (the quarterLength axis the
    music21 score is built on); ``type`` is the motif letter; ``TS`` is the
    movement time signature (M: header). ``measure`` is the SCORE measure (kept
    for diagnostics only, never used as a bar key — correction a).
    """
    rows: list[dict] = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "letter": (row.get("type") or "").strip(),
                        "start_beat": float(row["start"]),
                        "end_beat": float(row["end"]),
                        "ts": (row.get("TS") or "").strip(),
                    }
                )
            except (ValueError, KeyError):
                continue
    return rows


def load_pickup_ts(pickup_csv: Path, movement_id: str) -> str | None:
    """Movement time signature from pickup.csv (keyed by the leading ``no``)."""
    try:
        no = int(movement_id.split("-")[0])
    except (ValueError, IndexError):
        return None
    try:
        with Path(pickup_csv).open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    if int(row["no"]) == no:
                        return (row.get("TS") or "").strip() or None
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return None


def _movement_ts(label_rows: list[dict]) -> str:
    for r in label_rows:
        if r.get("ts"):
            return r["ts"]
    return "4/4"


# ── pitch spelling (midi + morphetic → diatonic) ─────────────────────────────


def _morphetic_to_diatonic(morph: int) -> tuple[str, int]:
    """Morphetic (diatonic staff-step) → (step_letter, octave). Anchored C4=60."""
    rel = morph - 60
    octave = 4 + (rel // 7)
    step = STEP_NAMES[rel % 7]
    return step, octave


def _make_pitch(midi: int, morph: int):
    """Correctly-spelled music21 Pitch from (midi, morphetic). Ported from MARBLE."""
    import music21

    step, octave = _morphetic_to_diatonic(morph)
    natural = music21.pitch.Pitch(step=step, octave=octave)
    alter = midi - natural.midi
    if abs(alter) > 2:
        p = music21.pitch.Pitch()
        p.midi = midi
        return p
    p = music21.pitch.Pitch()
    p.step = step
    p.octave = octave
    if alter != 0:
        p.accidental = music21.pitch.Accidental(alter)
    return p


def _snap(ql: float) -> float:
    return round(ql / NOTE_GRID_QL) * NOTE_GRID_QL


# ── score build (correction a + b + c) ───────────────────────────────────────


def build_movement_score(notes: list[dict], ts: str | None):
    """Build the 2-Part music21 Score AND tag motif letters on each note.

    Notes split by ``staff_number`` into two Parts (lowest staff → V1, rest →
    V2) at their absolute onsets (snapped to the grid; a negative pickup onset
    is clamped to 0 with its post-zero duration kept). Each note carries its
    csv motif ``letter`` in ``note.editorial.comment`` so the physical-bar→letter
    map can be read off the SAME score that builds the ABC (correction a). The
    shorter voice is padded with a trailing rest to the global end so both parts
    span identical measures (the abctoolkit interleave alignment gate).

    Returns ``(score, part0)`` — ``part0`` is the V1 measure axis.
    """
    import music21

    by_staff: dict[int, list[dict]] = {}
    for n in notes:
        if n["dur"] <= 0:
            continue
        by_staff.setdefault(n["staff"], []).append(n)

    staves = sorted(by_staff)
    if not staves:
        raise RuntimeError("movement has no notes")
    upper = staves[0]
    voice_notes: dict[int, list[dict]] = {0: [], 1: []}
    for s in staves:
        target = 0 if s == upper else 1
        voice_notes[target].extend(by_staff[s])

    snapped: dict[int, list[dict]] = {0: [], 1: []}
    global_end = 0.0
    for v in (0, 1):
        for n in voice_notes[v]:
            start = n["onset"]
            dur = n["dur"]
            if start < 0:
                dur = (start + dur) - max(0.0, start)
                start = 0.0
                if dur <= 0:
                    continue
            start = _snap(start)
            dur = _snap(dur)
            if dur <= 0:
                dur = NOTE_GRID_QL
            snapped[v].append({**n, "_start": start, "_dur": dur})
            global_end = max(global_end, start + dur)

    score = music21.stream.Score()
    parts: list = []
    for v in (0, 1):
        part = music21.stream.Part()
        if ts:
            try:
                part.insert(0, music21.meter.TimeSignature(ts))
            except Exception:  # noqa: BLE001
                pass
        voice_end = 0.0
        for n in snapped[v]:
            note = music21.note.Note()
            note.pitch = _make_pitch(n["midi"], n["morphetic"])
            note.duration.quarterLength = n["_dur"]
            note.editorial.comment = n["letter"] or ""
            part.insert(n["_start"], note)
            voice_end = max(voice_end, n["_start"] + n["_dur"])
        if voice_end < global_end:
            rest = music21.note.Rest()
            rest.duration.quarterLength = global_end - voice_end
            part.insert(voice_end, rest)
        part.makeMeasures(inPlace=True)
        score.insert(0, part)
        parts.append(part)

    return score, parts[0]


def bar_letters_from_part(part) -> dict[int, set[str]]:
    """``physical measureNumber → set(motif letters)`` from a built part (a)."""
    bar_letters: dict[int, set[str]] = {}
    for n in part.recurse().notes:
        letter = (getattr(n.editorial, "comment", "") or "").strip()
        if not letter:
            continue
        bar_letters.setdefault(n.measureNumber, set()).add(letter)
    return bar_letters


def occurrence_bars_from_part(part, start: float, end: float) -> set[int]:
    """Physical measure numbers whose ``[offset, offset+len) ∩ [start, end)`` (a)."""
    bars: set[int] = set()
    for m in part.getElementsByClass("Measure"):
        m_start = float(m.offset)
        m_end = m_start + float(m.barDuration.quarterLength)
        if m_start < end and m_end > start:
            bars.add(m.measureNumber)
    return bars


# ── ABC build: convert the WHOLE movement once, slice windows by bar-line ─────
#
# We deliberately do NOT call ``score.measures(a, b)`` + ``score_to_abc`` per
# window. That re-runs xml2abc per window and, worse, music21's ``.measures()``
# slice on a 2-Part score duplicates a spillover measure object (StreamException
# on write) and xml2abc can emit unequal V1/V2 bar counts on a fragment
# (abctoolkit ``strip_empty_bars`` then returns None) — together that silently
# dropped ~15% of windows in the first cut. Instead we convert each movement to
# interleaved ABC ONCE (where padding guarantees equal-length parts) and observe
# the invariant (verified across all 32 movements): the interleaved-ABC body is
# EXACTLY one line per physical bar, each line ``[V:1]...|[V:2]...|``. So a window
# [bar_start, bar_end] is the header + body lines [bar_start-1 : bar_end]. This
# guarantees [V:1]/[V:2] parity on every window and never re-runs the converter.


def _split_movement_abc(abc: str) -> tuple[list[str], list[str]]:
    """Split a whole-movement interleaved ABC into (header_lines, bar_lines).

    ``bar_lines[i]`` is the interleaved body line for physical bar ``i+1`` (each
    is ``[V:1]...|[V:2]...|``). ``header_lines`` are the leading non-body lines
    (``%%score``, ``L:``, ``M:``, ``K:``, ``V:`` decls) reused verbatim per window.
    """
    header: list[str] = []
    bars: list[str] = []
    for ln in abc.splitlines():
        if ln.startswith("[V:"):
            bars.append(ln)
        elif not bars:
            header.append(ln)
        # A non-body line appearing AFTER bars have started (rare trailing
        # directive) is dropped — it is not part of any single bar.
    return header, bars


def _window_abc_from_bars(
    header: list[str], bar_lines: list[str], bar_start: int, bar_end: int
) -> str:
    """Reassemble the interleaved ABC for physical bars [bar_start, bar_end].

    1-based inclusive bar indices into ``bar_lines`` (``bar_lines[0]`` == bar 1).
    """
    window_bars = bar_lines[bar_start - 1 : bar_end]
    return "\n".join(header + window_bars) + "\n"


# ── per-movement assembly ────────────────────────────────────────────────────


def _window_spans(max_bar: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Inclusive physical-bar spans ``[bar_start, bar_end]`` per window.

    Mirrors the prototype's ``_window_starts`` geometry on the physical-bar axis
    (1-based, inclusive): ``M = 1 + (max_bar - window)//stride`` windows when
    ``max_bar >= window``; a movement shorter than one window collapses to a
    single full-span window.
    """
    if max_bar <= 0:
        return []
    if window <= 1:
        return [(b, b) for b in range(1, max_bar + 1)]
    if max_bar < window:
        return [(1, max_bar)]
    return [(s, s + window - 1) for s in range(1, max_bar - window + 2, stride)]


def assemble_movement(
    notes_csv: Path,
    label_csv: Path,
    pickup_csv: Path,
    movement_id: str,
    window: int,
    stride: int,
) -> tuple[list[dict], dict]:
    """Assemble one movement → list of window rows + a per-movement stat dict.

    Returns ``(rows, stat)``. ``rows`` is a list of JSONL dicts (one per window);
    ``stat`` carries the per-movement provenance (max bar, n_windows, V1/V2
    parity, exclusion reason if the movement was dropped).
    """
    notes = parse_notes_csv(notes_csv)
    labels = parse_label_csv(label_csv)
    ts = load_pickup_ts(pickup_csv, movement_id) or _movement_ts(labels)

    try:
        score, part0 = build_movement_score(notes, ts)
    except Exception as e:  # noqa: BLE001
        return [], {
            "movement_id": movement_id,
            "built": False,
            "reason": f"score build failed: {type(e).__name__}: {e}",
        }

    bar_letters = bar_letters_from_part(part0)
    max_bar = max((m.measureNumber for m in part0.getElementsByClass("Measure")), default=0)
    if max_bar <= 0:
        return [], {
            "movement_id": movement_id,
            "built": False,
            "reason": "no physical measures after makeMeasures",
        }

    # Convert the WHOLE movement to interleaved ABC ONCE, then slice windows by
    # bar-line. Guarantees [V:1]/[V:2] parity + bar/line alignment (see
    # _split_movement_abc above). A whole-movement conversion failure drops the
    # movement (reported, never silent).
    from marble.encoders.CLaMP3.abc_util import score_to_abc

    try:
        movement_abc = score_to_abc(score)
    except Exception as e:  # noqa: BLE001
        return [], {
            "movement_id": movement_id,
            "built": False,
            "reason": f"whole-movement score_to_abc failed: {type(e).__name__}: {e}",
        }
    header, bar_lines = _split_movement_abc(movement_abc)
    # The bar-line count MUST equal the physical measure count (the invariant the
    # whole-movement approach relies on). If it ever diverges, drop the movement
    # rather than silently misalign labels to bars.
    if len(bar_lines) != max_bar:
        return [], {
            "movement_id": movement_id,
            "built": False,
            "reason": (
                f"bar-line/measure mismatch: {len(bar_lines)} ABC bar lines vs "
                f"{max_bar} physical measures (would misalign labels)"
            ),
        }

    occurrences: list[dict] = []
    for r in labels:
        bars = occurrence_bars_from_part(part0, r["start_beat"], r["end_beat"])
        if not bars:
            continue
        occ_id = f"{movement_id}:{r['letter']}:{r['start_beat']}"
        occurrences.append({"letter": r["letter"], "bars": bars, "occ_id": occ_id})

    spans = _window_spans(max_bar, window, stride)
    rows: list[dict] = []
    n_v1 = 0
    n_v2 = 0
    abc_failures: list[str] = []
    for bar_start, bar_end in spans:
        abc = _window_abc_from_bars(header, bar_lines, bar_start, bar_end)
        has_v1 = "[V:1]" in abc
        has_v2 = "[V:2]" in abc
        n_v1 += int(has_v1)
        n_v2 += int(has_v2)
        win_bars = set(range(bar_start, bar_end + 1))
        letters: set[str] = set()
        for b in win_bars:
            letters |= bar_letters.get(b, set())
        occ_ids = sorted(o["occ_id"] for o in occurrences if o["bars"] & win_bars)
        window_id = f"{movement_id}:w{bar_start:04d}-{bar_end:04d}"
        rows.append(
            {
                "movement_id": movement_id,
                "window_id": window_id,
                "bar_start": bar_start,
                "bar_end": bar_end,
                "abc": abc,
                "letters": sorted(letters),
                "occurrence_ids": occ_ids,
                "n_bars": bar_end - bar_start + 1,
                "split": "test",
            }
        )

    stat = {
        "movement_id": movement_id,
        "built": True,
        "ts": ts,
        "max_bar": max_bar,
        "n_windows": len(rows),
        "n_windows_with_v1": n_v1,
        "n_windows_with_v2": n_v2,
        "n_occurrences": len(occurrences),
        "n_labelled_bars": len(bar_letters),
        "abc_failures": abc_failures,
    }
    return rows, stat


# ── parallel worker ──────────────────────────────────────────────────────────

_WORKER_READY = False


def _worker_init() -> None:
    global _WORKER_READY
    if not _WORKER_READY:
        from marble.encoders.CLaMP3.abc_util import _register_converter21

        _register_converter21()
        _WORKER_READY = True


def _build_one(job: tuple) -> tuple:
    upstream_s, movement_id, window, stride = job
    _worker_init()
    upstream = Path(upstream_s)
    rows, stat = assemble_movement(
        upstream / "csv_notes" / f"{movement_id}.csv",
        upstream / "csv_label" / f"{movement_id}.csv",
        upstream / "pickup.csv",
        movement_id,
        window,
        stride,
    )
    return movement_id, rows, stat


# ── driver ───────────────────────────────────────────────────────────────────


def _list_movements(upstream: Path) -> list[str]:
    notes_dir = upstream / "csv_notes"
    return sorted(p.stem for p in notes_dir.glob("*.csv"))


def build(out_dir: Path, upstream: Path, window: int, stride: int, workers: int) -> dict:
    movements = _list_movements(upstream)
    jobs = [(str(upstream), mov, window, stride) for mov in movements]

    print(
        f"[build] {len(movements)} movements; window={window} stride={stride} workers={workers}",
        file=sys.stderr,
        flush=True,
    )

    results: list[tuple] = []
    if workers <= 1:
        _worker_init()
        for j in jobs:
            results.append(_build_one(j))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
            for res in ex.map(_build_one, jobs):
                results.append(res)

    results.sort(key=lambda r: r[0])
    all_rows: list[dict] = []
    per_movement: list[dict] = []
    built = 0
    excluded: list[dict] = []
    total_v1 = 0
    total_v2 = 0
    for _mov, rows, stat in results:
        per_movement.append(stat)
        if stat.get("built"):
            built += 1
            all_rows.extend(rows)
            total_v1 += stat["n_windows_with_v1"]
            total_v2 += stat["n_windows_with_v2"]
            if stat.get("abc_failures"):
                excluded.append(
                    {
                        "movement_id": stat["movement_id"],
                        "reason": "partial: some windows failed ABC build",
                        "detail": stat["abc_failures"][:5],
                    }
                )
        else:
            excluded.append({"movement_id": stat["movement_id"], "reason": stat.get("reason")})

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"BPSMotifWithinPiece.N{window}.ABC.jsonl"
    with jsonl_path.open("w") as fh:
        for rec in all_rows:
            fh.write(json.dumps(rec) + "\n")

    stats = {
        "window": window,
        "stride": stride,
        "movements_total": len(movements),
        "movements_built": built,
        "movements_excluded": len(movements) - built,
        "total_windows": len(all_rows),
        "windows_with_v1": total_v1,
        "windows_with_v2": total_v2,
        "v1_v2_parity_ok": total_v1 == total_v2 == len(all_rows),
        "jsonl_path": str(jsonl_path),
        "excluded_detail": excluded,
        "per_movement": per_movement,
    }
    stats_path = out_dir / f"BPSMotifWithinPiece.N{window}.ABC.stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--upstream-dir",
        default="data/BPS-Motif/_upstream/Beethoven_motif",
        help="Dir with csv_notes/ csv_label/ pickup.csv. Accepts the PC "
        "_upstream/Beethoven_motif clone OR the Mac "
        "data/kern_sources/BPS-Motif/beethoven_motif_csv checkout.",
    )
    ap.add_argument("--out-dir", default="data/BPS-Motif")
    ap.add_argument("--window", type=int, required=True, help="phrase-window bars (N)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: os.cpu_count() − 1).",
    )
    args = ap.parse_args()

    upstream = Path(args.upstream_dir)
    if not (upstream / "csv_notes").is_dir():
        print(f"ERROR: csv_notes/ not found under {upstream}", file=sys.stderr)
        sys.exit(1)

    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 2) - 1)
    stats = build(Path(args.out_dir), upstream, args.window, args.stride, workers)
    # Print a compact provenance summary (full per-movement detail is in the
    # stats.json next to the JSONL).
    summary = {k: v for k, v in stats.items() if k not in ("per_movement", "excluded_detail")}
    summary["excluded_detail"] = stats["excluded_detail"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
