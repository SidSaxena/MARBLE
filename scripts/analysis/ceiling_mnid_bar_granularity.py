#!/usr/bin/env python3
"""Per-note F1 CEILING of bar-granularity motif labeling on BPS-Motif.

Question: if a classifier can only emit ONE label per bar (≈ a CLaMP3 ABC patch),
broadcast to every note in that bar, what is the best per-note micro-F1 it could
reach against the per-note ground truth? That is the ceiling for a per-patch
CLaMP3 MNID. We bracket it with three per-bar labelings:
  - any   : bar=motif if it contains >=1 motif note  (max recall)
  - maj   : bar=motif if >=50% notes are motif        (accuracy-optimal)
  - best  : sweep the motif-fraction threshold, take the F1-maximizing one (the ceiling)
Two granularities: per-measure (coarse, ~CLaMP3 interleaved patch) and
per-(measure,staff) (finer, voices kept separate).

GROUND TRUTH: csv_notes 'type' column non-empty == motif note (it carries the
motif letter for notes in any occurrence). 'measure'/'staff_number' give the bar.

Serial vs parallel are byte-identical on stdout: stdout = the result JSON only;
mode + timing go to stderr. Micro counts are sum-aggregated, so worker order is
irrelevant.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _movement_bars(task):
    """Return per-movement bar tallies for both granularities.

    bars_* are lists of [n_motif, n_total] per bar.
    """
    piece_id, csv_path = task
    by_meas: dict[int, list[int]] = {}
    by_meas_staff: dict[tuple[int, int], list[int]] = {}
    n_notes = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                meas = int(row["measure"])
                staff = int(row["staff_number"])
                is_motif = 1 if row["type"].strip() else 0
            except (ValueError, KeyError):
                continue
            n_notes += 1
            b = by_meas.setdefault(meas, [0, 0])
            b[0] += is_motif
            b[1] += 1
            bs = by_meas_staff.setdefault((meas, staff), [0, 0])
            bs[0] += is_motif
            bs[1] += 1
    return {
        "piece": piece_id,
        "n_notes": n_notes,
        "bars_measure": list(by_meas.values()),
        "bars_measure_staff": list(by_meas_staff.values()),
    }


def _prf_at(all_bars, t):
    tp = fp = fn = 0
    for nm, nt in all_bars:
        if nt == 0:
            continue
        if (nm / nt) >= t:
            tp += nm
            fp += nt - nm
        else:
            fn += nm
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "threshold": round(t, 4),
        "f1": round(f1, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
    }


def _ceiling(all_bars):
    """any / majority / best-threshold per-note F1 for a per-bar labeler."""
    fracs = sorted({nm / nt for nm, nt in all_bars if nt > 0})
    best = None
    for t in [1e-9] + fracs:
        m = _prf_at(all_bars, t)
        if best is None or m["f1"] > best["f1"]:
            best = m
    n_mixed = sum(1 for nm, nt in all_bars if 0 < nm < nt)
    n_motif = sum(nm for nm, nt in all_bars)
    n_total = sum(nt for nm, nt in all_bars)
    return {
        "any": _prf_at(all_bars, 1e-9),
        "majority": _prf_at(all_bars, 0.5),
        "best": best,
        "n_bars": len(all_bars),
        "mixed_bar_frac": round(n_mixed / len(all_bars), 4) if all_bars else 0.0,
        "motif_note_frac": round(n_motif / n_total, 4) if n_total else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", default="data/BPS-Motif/_upstream/Beethoven_motif/csv_notes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--serial", action="store_true")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csvs:
        print(json.dumps({"error": "no csvs", "dir": args.csv_dir}))
        sys.exit(1)
    tasks = [(Path(c).stem, c) for c in csvs]

    t0 = time.perf_counter()
    if args.serial:
        results = [_movement_bars(t) for t in tasks]
        mode = "serial"
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_movement_bars, tasks))
        mode = f"parallel(w={args.workers})"
    elapsed = time.perf_counter() - t0

    results.sort(key=lambda r: r["piece"])
    bars_m = [b for r in results for b in r["bars_measure"]]
    bars_ms = [b for r in results for b in r["bars_measure_staff"]]
    n_notes = sum(r["n_notes"] for r in results)

    out = {
        "n_movements": len(results),
        "n_notes": n_notes,
        "ceiling_per_measure": _ceiling(bars_m),
        "ceiling_per_measure_staff": _ceiling(bars_ms),
        "sota_target_note_f1": 0.721,
    }
    # stdout: result only (identical serial vs parallel). stderr: mode + timing.
    print(json.dumps(out, indent=2, sort_keys=True))
    print(
        f"[{mode}] {len(results)} movements, {n_notes} notes, compute {elapsed * 1000:.1f} ms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
