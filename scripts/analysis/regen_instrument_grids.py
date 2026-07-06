#!/usr/bin/env python3
"""
scripts/analysis/regen_instrument_grids.py
───────────────────────────────────────────
Regenerate the per-instrument MAP grid figures (``vgm_instrument_grids_<enc>``)
that appear in the thesis chapters + defense deck, using the *committed*
best-layer condition-grid CSV slices — WITHOUT re-running any sweep.

Why this exists
---------------
The thesis-mode grid step in ``vgm_timbre_report.py`` (``cmd_thesis`` →
``vgm_instrument_grids_<enc>``) reads a raw per-layer sweep tree
(``<results>/layer{BEST}/condition_grid.csv``). That tree is bulky and not
always present on the machine doing the figure export, and the raw imshow it
drew left thin **white seams between cells** (the page colour bleeding between
the 8×8 blocks in the PDF). This script:

  1. reads the portable, committed 8×8 slices
     ``docs/figures/vgmiditvar_timbre_<enc>_varctl/best_layer_condition_grid{,_varctl}.csv``
     (long format: ``query_program,target_program,map,n_queries``) — the same
     numbers as the thesis, guaranteed matching, self-contained; and
  2. draws each grid with the **seam-free** heatmap style
     (``imshow(interpolation='nearest')`` + ``set_rasterized(True)`` so the
     cells flatten to a single raster, plus an explicit ``grid(False)``), then
  3. writes ``figures/chapters/vgm_instrument_grids_<slug>.pdf`` +
     ``defense/vgm_instrument_grids_<slug>.png`` — identical layout/paths to the
     canonical generator, so all encoders stay styled consistently.

Extra (non-best) deployed layers — e.g. MuQ L11, MERT L7 — can be rendered by
passing ``--extra "<Display>:<layer>:<work_csv>:<varctl_csv>"``; the two 8×8 CSVs
must be in the same long format.

Usage
-----
  # all four best-layer figures (default):
  uv run python scripts/analysis/regen_instrument_grids.py \
      --thesis-dir ~/Developer/UPF/SMC/Thesis/smc-msc-thesis

  # a subset, or an extra deployed-layer figure:
  uv run python scripts/analysis/regen_instrument_grids.py \
      --thesis-dir <thesis> --only muq mert \
      --extra "MuQ:11:/path/work.csv:/path/varctl.csv"

Source of truth: the committed CSVs are the flattened best-layer slices written
alongside the per-encoder ``docs/figures/vgmiditvar_timbre_<enc>_varctl/`` sets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ── styling (kept in sync with vgm_timbre_report.py cmd_thesis grid step) ────
GM = {
    0: "piano",
    24: "guitar",
    48: "strings",
    52: "choir",
    60: "horn",
    73: "flute",
    80: "square",
    89: "pad",
}

MARBLE_ROOT = Path(__file__).resolve().parents[2]
FIGDIR = MARBLE_ROOT / "docs" / "figures"

# (display name, best layer, committed-CSV dir slug, filename slug)
BEST_LAYER_SPECS = [
    ("CLaMP3", 4, "clamp3", "clamp3"),
    ("MERT-v1-95M", 11, "mert", "mert"),
    ("MuQ", 8, "muq", "muq"),
    ("OMAR-RQ", 15, "omarrq", "omarrq"),
]


def load_grid_csv(path: Path):
    """Long-format (query_program,target_program,map,...) → (8×8 array, conds)."""
    grid, conds = {}, []
    with open(path) as f:
        for r in csv.DictReader(f):
            q, t = int(r["query_program"]), int(r["target_program"])
            grid[(q, t)] = float(r["map"])
            if q not in conds:
                conds.append(q)
    conds = sorted(conds)
    A = np.array([[grid[(q, t)] for t in conds] for q in conds])
    return A, conds


def draw_grid(name: str, layer: int, work_csv: Path, varctl_csv: Path, thesis_dir: Path, slug: str):
    A, conds = load_grid_csv(work_csv)
    B, _ = load_grid_csv(varctl_csv)
    inst = [GM.get(c, str(c)) for c in conds]
    vmax = max(A.max(), B.max())
    n = len(inst)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    fig.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.90, wspace=0.28)
    for ax, Mx, title in (
        (axes[0], A, "work-level lens (twin in relevance)"),
        (axes[1], B, "variation-controlled lens (twin masked)"),
    ):
        im = ax.imshow(Mx, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        im.set_rasterized(True)  # flatten cells to raster: no white seams between imshow blocks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(inst, rotation=40, ha="right", fontsize=10.5)
        ax.set_yticklabels(inst, fontsize=10.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("target instrument", fontsize=11)
        ax.grid(False)  # global axes.grid default would otherwise stroke lines over the heatmap
        for i in range(n):
            for j in range(n):
                v = Mx[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}"[1:] if v < 1 else f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="white" if v > 0.55 * vmax else "#222222",
                )
    axes[0].set_ylabel("query instrument", fontsize=11)
    cax = fig.add_axes([0.92, 0.17, 0.018, 0.69])
    fig.colorbar(im, cax=cax, label="MAP")
    fig.text(0.5, 0.945, f"Per-instrument MAP grid, {name} layer {layer}", ha="center", fontsize=13)

    chapters = thesis_dir / "figures" / "chapters"
    defense = thesis_dir / "defense"
    chapters.mkdir(parents=True, exist_ok=True)
    stem = f"vgm_instrument_grids_{slug}"
    fig.savefig(chapters / f"{stem}.pdf")
    if defense.is_dir():
        fig.savefig(defense / f"{stem}.png", dpi=160)
    plt.close(fig)
    print(f"[grids] wrote {chapters / (stem + '.pdf')}  (layer {layer})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--thesis-dir",
        required=True,
        type=Path,
        help="thesis repo root (writes figures/chapters + defense)",
    )
    ap.add_argument(
        "--only", nargs="*", default=None, help="subset of filename slugs to render (e.g. muq mert)"
    )
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="extra figures: 'Display:layer:work_csv:varctl_csv' "
        "(slug auto-suffixed with _l<layer>)",
    )
    args = ap.parse_args()

    for name, layer, dirslug, slug in BEST_LAYER_SPECS:
        if args.only and slug not in args.only:
            continue
        d = FIGDIR / f"vgmiditvar_timbre_{dirslug}_varctl"
        draw_grid(
            name,
            layer,
            d / "best_layer_condition_grid.csv",
            d / "best_layer_condition_grid_varctl.csv",
            args.thesis_dir,
            slug,
        )

    for spec in args.extra:
        name, layer_s, work_csv, varctl_csv = spec.split(":", 3)
        layer = int(layer_s)
        # filename slug: reuse best-layer slug base + _l<layer> so it never
        # clobbers the canonical best-layer figure.
        base = {"MERT-v1-95M": "mert", "MuQ": "muq", "OMAR-RQ": "omarrq", "CLaMP3": "clamp3"}.get(
            name, name.lower()
        )
        draw_grid(
            name, layer, Path(work_csv), Path(varctl_csv), args.thesis_dir, f"{base}_l{layer}"
        )


if __name__ == "__main__":
    main()
