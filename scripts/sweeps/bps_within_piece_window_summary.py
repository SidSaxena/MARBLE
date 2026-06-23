"""Aggregate the fold-style BPS-Motif within-piece WINDOW-SIZE sweep — with the
prevalence permutation null.

For BOTH encoding arms (clip-isolated ``BPSMotifWithinPiece`` and whole-piece
``BPSMotifWithinPieceWhole``) this reads every ``wandb-summary.json`` under the
per-N sweep output dirs and, per (arm, window N, layer), pulls:

  * ``test/map``                 — raw within-movement same-motif MAP (inflates
                                   with N via letter prevalence — the artifact),
  * ``test/map_centered``        — per-movement-centered real MAP,
  * ``test/map_centered_null``   — the within-group label-permutation null,
  * ``test/map_centered_lift``   — real - null (the HONEST, prevalence-controlled
                                   signal; the breaking-point is where it peaks),
  * ``test/map_centered_p``      — empirical p (real vs null distribution).

Output per arm:

  1. a layer x N table (raw MAP) AND a layer x N table (lift),
  2. the **raw breaking-point curve** (best-layer raw MAP vs N) — shows the
     monotone prevalence inflation,
  3. the **controlled breaking-point curve** (best-layer LIFT vs N, with the
     per-N best layer + p annotated) — the real motif-scale peak,
  4. a layer x N LIFT heatmap (diverging, centered at 0),

saved under ``docs/`` (default). The two arms are overlaid on the controlled
curve for a clean clip-vs-whole comparison.

Output dir convention (per-N save_dir + gen_sweep_configs' ``.layer{L}`` patch):

  output/probe.BPSMotifWithinPiece[Whole]N{N}.CLaMP3-symbolic-abc-layers.layer{L}/...
  output/probe.BPSMotifWithinPiece[Whole]N{N}.CLaMP3-symbolic-abc-meanall/...

Usage::

    python scripts/sweeps/bps_within_piece_window_summary.py \
        --windows 1 2 3 4 6 8 12 16 24 32 [--arms clip whole] \
        [--output-root output] [--docs-dir docs]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_LAYER_RX = re.compile(r"\.CLaMP3-symbolic-abc-layers\.layer(\d+)(?:\b|/)")

ARM_INFIX = {"clip": "BPSMotifWithinPiece", "whole": "BPSMotifWithinPieceWhole"}

# Metrics we read from each layer run's wandb-summary.json.
RAW = "test/map"
REAL = "test/map_centered"
NULL = "test/map_centered_null"
LIFT = "test/map_centered_lift"
PVAL = "test/map_centered_p"


def _load(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def collect_layer_runs(output_root: Path, infix: str, n: int) -> dict[int, dict]:
    """layer -> summary dict for one (arm, window N). Latest run per layer wins."""
    pat = str(
        output_root
        / f"probe.{infix}N{n}.CLaMP3-symbolic-abc-layers.layer*"
        / "wandb"
        / "run-*"
        / "files"
        / "wandb-summary.json"
    )
    out: dict[int, dict] = {}
    for p in sorted(glob.glob(pat)):
        m = _LAYER_RX.search(p)
        if not m:
            continue
        d = _load(p)
        if d is None or RAW not in d:
            continue
        out[int(m.group(1))] = d
    return out


def collect_meanall(output_root: Path, infix: str, n: int) -> dict | None:
    pat = str(
        output_root
        / f"probe.{infix}N{n}.CLaMP3-symbolic-abc-meanall"
        / "wandb"
        / "run-*"
        / "files"
        / "wandb-summary.json"
    )
    best = None
    for p in sorted(glob.glob(pat)):
        d = _load(p)
        if d is not None and RAW in d:
            best = d
    return best


def _g(d: dict, key: str) -> float | None:
    v = d.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def build_arm(output_root: Path, arm: str, windows: list[int]) -> dict:
    """Per (window, layer) metric grids + per-window best-layer-by-lift picks."""
    infix = ARM_INFIX[arm]
    per_window: dict[int, dict] = {}
    for n in windows:
        runs = collect_layer_runs(output_root, infix, n)
        if not runs:
            continue
        layers = sorted(runs)
        cells = {
            li: {
                "raw": _g(runs[li], RAW),
                "real": _g(runs[li], REAL),
                "null": _g(runs[li], NULL),
                "lift": _g(runs[li], LIFT),
                "p": _g(runs[li], PVAL),
            }
            for li in layers
        }
        # best layer by LIFT (the honest signal), ignoring missing lifts.
        scored = [(li, c["lift"]) for li, c in cells.items() if c["lift"] is not None]
        best_layer = max(scored, key=lambda t: t[1])[0] if scored else None
        per_window[n] = {
            "cells": cells,
            "best_layer": best_layer,
            "meanall": collect_meanall(output_root, infix, n),
        }
    return per_window


def print_tables(arm: str, per_window: dict) -> None:
    windows = sorted(per_window)
    if not windows:
        print(f"[{arm}] no runs found.")
        return
    layers = sorted({li for n in windows for li in per_window[n]["cells"]})
    for title, key, fmt in (("raw MAP", "raw", "{:.3f}"), ("LIFT", "lift", "{:+.3f}")):
        print(f"\n[{arm}] {title}  (layer x window)")
        print("layer\\N " + "  ".join(f"N{n:>6}" for n in windows))
        for li in layers:
            row = []
            for n in windows:
                c = per_window[n]["cells"].get(li, {})
                v = c.get(key)
                row.append(fmt.format(v) if v is not None else "   --  ")
            print(f"L{li:>5} " + "  ".join(f"{x:>8}" for x in row))
    print(f"\n[{arm}] per-window best layer (by lift) + p:")
    for n in windows:
        bl = per_window[n]["best_layer"]
        c = per_window[n]["cells"].get(bl, {}) if bl is not None else {}
        print(f"  N={n:>2}: best L{bl}  lift={c.get('lift')}  raw={c.get('raw')}  p={c.get('p')}")


def plot_controlled(arms: dict[str, dict], windows: list[int], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots")
        return

    # Controlled curve: best-layer lift vs N, raw best-layer MAP vs N (overlay).
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    for arm, pw in arms.items():
        ns = [n for n in windows if n in pw]
        lifts, raws = [], []
        for n in ns:
            bl = pw[n]["best_layer"]
            c = pw[n]["cells"].get(bl, {}) if bl is not None else {}
            lifts.append(c.get("lift"))
            # raw best layer = argmax raw (the inflated view's own best)
            raw_scored = [
                (li, cc["raw"]) for li, cc in pw[n]["cells"].items() if cc["raw"] is not None
            ]
            raws.append(max(raw_scored, key=lambda t: t[1])[1] if raw_scored else None)
        axL.plot(ns, lifts, "-o", label=arm)
        axR.plot(ns, raws, "-o", label=arm)
    axL.set_title("Prevalence-CONTROLLED breaking point\n(best-layer lift = real - null vs window)")
    axL.set_xlabel("phrase window N (bars)")
    axL.set_ylabel("lift (real - null MAP)")
    axL.axhline(0, color="k", lw=0.5)
    axL.grid(alpha=0.3)
    axL.legend()
    axR.set_title("RAW MAP vs window (prevalence-inflated — the artifact)")
    axR.set_xlabel("phrase window N (bars)")
    axR.set_ylabel("best-layer raw MAP")
    axR.grid(alpha=0.3)
    axR.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def plot_lift_heatmap(arm: str, per_window: dict, windows: list[int], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    ns = [n for n in windows if n in per_window]
    if not ns:
        return
    layers = sorted({li for n in ns for li in per_window[n]["cells"]})
    grid = np.full((len(layers), len(ns)), np.nan)
    for j, n in enumerate(ns):
        for i, li in enumerate(layers):
            v = per_window[n]["cells"].get(li, {}).get("lift")
            if v is not None:
                grid[i, j] = v
    vmax = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels([f"N{n}" for n in ns])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{li}" for li in layers])
    ax.set_xlabel("phrase window N (bars)")
    ax.set_ylabel("CLaMP3 hidden layer")
    ax.set_title(f"{arm}: prevalence-controlled LIFT (real - null) over layer x window")
    for i in range(len(layers)):
        for j in range(len(ns)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="k", fontsize=6)
    fig.colorbar(im, ax=ax, label="lift")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--arms", nargs="+", choices=["clip", "whole"], default=["clip", "whole"])
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--docs-dir", default="docs")
    args = ap.parse_args()

    output_root = (
        (REPO / args.output_root)
        if not Path(args.output_root).is_absolute()
        else Path(args.output_root)
    )
    docs = (REPO / args.docs_dir) if not Path(args.docs_dir).is_absolute() else Path(args.docs_dir)
    windows = sorted(set(args.windows))

    arms: dict[str, dict] = {}
    for arm in args.arms:
        pw = build_arm(output_root, arm, windows)
        arms[arm] = pw
        print_tables(arm, pw)
        plot_lift_heatmap(
            arm, pw, windows, docs / "plots" / f"bps_wp_window_lift_heatmap_{arm}.png"
        )

    plot_controlled(arms, windows, docs / "plots" / "bps_wp_window_breaking_point.png")


if __name__ == "__main__":
    main()
