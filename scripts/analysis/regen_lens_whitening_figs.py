#!/usr/bin/env python3
"""
scripts/analysis/regen_lens_whitening_figs.py
──────────────────────────────────────────────
Regenerate the two VGMIDITVar-timbre line figures — ``vgm_two_lenses_<enc>``
and ``vgm_depth_whitening_<enc>`` — per encoder, from the *committed* per-layer
``summary_table.csv`` slices, WITHOUT re-running any sweep.

Context
-------
``vgm_timbre_report.py`` (cmd_thesis) draws these two figures for the LEAD
encoder only (MuQ), from the raw per-layer results tree. That tree isn't always
present, but the derived per-layer numbers are committed in
``docs/figures/vgmiditvar_timbre_<enc>_varctl/summary_table.csv`` for every
encoder — enough to reproduce both figures for MERT and OMAR-RQ (appendix
symmetry). Styling is imported directly from ``vgm_timbre_report`` so the
per-encoder versions match the canonical MuQ ones exactly. The whitening title
is humanised ("buys"->"improves") and carries the encoder name for the appendix.

Usage
-----
  uv run python scripts/analysis/regen_lens_whitening_figs.py \
      --thesis-dir ~/Developer/UPF/SMC/Thesis/smc-msc-thesis --encoders mert omarrq

Writes figures/chapters/vgm_{two_lenses,depth_whitening}_<slug>.pdf + defense PNGs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import matplotlib.pyplot as plt  # noqa: E402
import vgm_timbre_report as V  # noqa: E402  (reuse exact palette + _th_axes + _th_save)

MARBLE_ROOT = _HERE.parents[1]
FIGDIR = MARBLE_ROOT / "docs" / "figures"

# display name -> filename slug (matches the committed dir + thesis figure slugs)
SPECS = {"muq": "MuQ", "mert": "MERT-95M", "omarrq": "OMAR-RQ", "clamp3": "CLaMP3"}


def load_summary(slug: str):
    """Read summary_table.csv → (layers, {col: array}); drop the meanall row."""
    path = FIGDIR / f"vgmiditvar_timbre_{slug}_varctl" / "summary_table.csv"
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["layer"].lstrip("-").isdigit()]
    L = np.array([int(r["layer"]) for r in rows])
    cols = ("within", "cross_conf", "cross_varctl", "map_raw", "map_centered", "map_whitened")
    C = {k: np.array([float(r[k]) for r in rows]) for k in cols}
    return L, C


def two_lenses(name: str, slug: str, L, C, thesis_dir: Path):
    fig, ax = plt.subplots(figsize=(9.75, 5.25))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.97)
    ax.plot(
        L,
        C["within"],
        "-o",
        color="#7F7F7F",
        lw=2.2,
        ms=6,
        label="within-instrument (same under both lenses)",
    )
    ax.plot(
        L,
        C["cross_conf"],
        "-s",
        color=V.TH_BLUE,
        lw=2.2,
        ms=6,
        label="cross-instrument, work-level lens\n(includes the same variation re-rendered)",
    )
    ax.plot(
        L,
        C["cross_varctl"],
        "-^",
        color=V.TH_CRIMSON,
        lw=2.2,
        ms=6,
        label="cross-instrument, variation-controlled lens\n(a different variation, different instrument)",
    )
    ax.set_ylabel("MAP", fontsize=12.5)
    V._th_axes(ax, list(L), f"{name} layer")
    ax.set_title(
        f"Two relevance lenses on the rendered benchmark ({name}, centered)", fontsize=13.5, pad=12
    )
    ax.legend(loc="upper left", fontsize=10.3, labelspacing=0.6, **V.TH_LEG)
    V._th_save(fig, thesis_dir, f"vgm_two_lenses_{slug}")


def depth_whitening(name: str, slug: str, L, C, thesis_dir: Path):
    fig, ax = plt.subplots(figsize=(9.75, 5.25))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.97)
    cc = C["cross_varctl"]
    plateau = np.where(cc >= 0.97 * cc.max())[0]
    if len(plateau) > 1:
        ax.axvspan(
            float(L[plateau[0]]) - 0.5, float(L[plateau[-1]]) + 0.5, color=V.TH_RED, alpha=0.06
        )
    ax.plot(
        L,
        cc,
        "-^",
        color=V.TH_CRIMSON,
        lw=2.4,
        ms=6,
        label="variation-controlled cross-instrument MAP",
    )
    ax.plot(
        L,
        C["map_centered"],
        "-o",
        color=V.TH_FOREST,
        lw=2.4,
        ms=6,
        label="overall retrieval MAP (centered)",
    )
    ax.plot(
        L,
        C["map_whitened"],
        "--o",
        color=V.TH_GOLD,
        lw=2.4,
        ms=6,
        label="overall retrieval MAP (whitened, transductive)",
    )
    ax.set_ylabel("MAP", fontsize=12.5)
    V._th_axes(ax, list(L), f"{name} layer")
    ax.set_title(
        f"{name}: depth improves cross-timbre generalisation; "
        "whitening lifts retrieval at every layer",
        fontsize=12.5,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=10.5, **V.TH_LEG)
    V._th_save(fig, thesis_dir, f"vgm_depth_whitening_{slug}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--thesis-dir", required=True, type=Path)
    ap.add_argument(
        "--encoders",
        nargs="+",
        default=["mert", "omarrq"],
        help="filename slugs (default: mert omarrq)",
    )
    args = ap.parse_args()
    for slug in args.encoders:
        name = SPECS.get(slug, slug)
        L, C = load_summary(slug)
        two_lenses(name, slug, L, C, args.thesis_dir)
        depth_whitening(name, slug, L, C, args.thesis_dir)
        print(f"[lens/whiten] {name}: wrote vgm_two_lenses_{slug} + vgm_depth_whitening_{slug}")


if __name__ == "__main__":
    main()
