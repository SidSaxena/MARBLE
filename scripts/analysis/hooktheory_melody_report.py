#!/usr/bin/env python3
"""Thesis figures for the HookTheoryMelody multi-head+weighted campaign.

Sibling of ``medleydb_melody_report.py`` for the larger, single-split (no CV
folds) HookTheory corpus. Reads the committed CSVs in
``docs/figures/hooktheory_melody_multihead/`` (written by
``hooktheory_melody_export.py`` from the *fixed* test runs — post the
compute_groups correctness fix 4e24707) and renders two thesis-styled figures
into ``<thesis>/figures/chapters/`` (+ 160 dpi PNGs into ``<thesis>/defense/``):

  * ``hooktheory_layer_curves`` — per-layer test RPA (solid) AND RCA (dashed)
    per encoder, best-RPA-layer markers. The persistent RPA↔RCA gap (octave
    confusion, ~0.18 on this wide-register pop corpus) is the headline HTM
    story, so both lines are drawn.
  * ``hooktheory_layer_gates`` — learned SUPERB softmax gates (weighted head)
    per encoder, accuracy-argmax marked; the gate-vs-accuracy exhibit.

Renders whatever encoders are present in the CSVs (1–3), so it can be run for
a MuQ-only preview and re-run once MERT / OMAR-RQ retests land.

Usage:
    uv run python scripts/analysis/hooktheory_melody_report.py \\
        --thesis-dir ~/Developer/UPF/SMC/Thesis/smc-msc-thesis
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("docs/figures/hooktheory_melody_multihead")
TH_VIOLET, TH_ORANGE, TH_GREEN = "#5B4FC4", "#E8752E", "#1BAF7A"
TH_LEG = dict(frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
ENC_STYLE = {  # CSV encoder key -> (display name, colour)
    "MuQ": ("MuQ", TH_VIOLET),
    "MERT-v1-95M": ("MERT-v1-95M", TH_ORANGE),
    "OMARRQ-multifeature-25hz": ("OMAR-RQ", TH_GREEN),
}


def _th_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(labelsize=10.5)
    ax.grid(alpha=0.28, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _th_save(fig, thesis_dir: Path, stem: str):
    chapters = thesis_dir / "figures" / "chapters"
    defense = thesis_dir / "defense"
    chapters.mkdir(parents=True, exist_ok=True)
    fig.savefig(chapters / f"{stem}.pdf")
    if defense.is_dir():
        fig.savefig(defense / f"{stem}.png", dpi=160)
    plt.close(fig)
    print(f"[hooktheory] wrote {chapters / (stem + '.pdf')}")


def _present(rows_by_enc):
    """ENC_STYLE order, restricted to encoders actually present in the CSV."""
    return [(k, v) for k, v in ENC_STYLE.items() if k in rows_by_enc]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thesis-dir", required=True)
    args = ap.parse_args()
    thesis_dir = Path(args.thesis_dir).expanduser()

    # ── load curves (RPA + RCA per layer; meanall/weighted separate) ──
    rpa = defaultdict(dict)
    rca = defaultdict(dict)
    extra = defaultdict(dict)  # enc -> {"meanall":(rpa,rca), "weighted":(rpa,rca)}
    for r in csv.DictReader((DATA / "layer_curves.csv").open()):
        enc, lay = r["encoder"], r["layer"]
        if lay in ("meanall", "weighted"):
            extra[enc][lay] = (float(r["rpa"]), float(r["rca"]))
        else:
            rpa[enc][int(lay)] = float(r["rpa"])
            rca[enc][int(lay)] = float(r["rca"])
    present = _present(rpa)

    # ── figure 1: per-layer RPA (solid) + RCA (dashed) ──
    fig, ax = plt.subplots(figsize=(9.75, 5.6))
    fig.subplots_adjust(top=0.85, bottom=0.11, left=0.09, right=0.97)
    for key, (name, c) in present:
        X = np.array(sorted(rpa[key]))
        mp = np.array([rpa[key][L] for L in X])
        mc = np.array([rca[key][L] for L in X])
        ax.plot(X, mp, "-o", color=c, lw=2.4, ms=5.0, label=f"{name} RPA", zorder=3)
        ax.plot(X, mc, "--", color=c, lw=1.6, alpha=0.75, label=f"{name} RCA", zorder=2)
        b = int(X[np.argmax(mp)])
        ax.scatter([b], [mp.max()], s=110, facecolor="none", edgecolor=c, lw=1.5, zorder=4)
        # MERT's below-the-marker label collides with OMAR's curve; lift it above
        dy = +0.014 if key == "MERT-v1-95M" else -0.028
        ax.annotate(
            f"{name} best L{b} = {mp.max():.3f}",
            xy=(b, mp.max()),
            xytext=(b + 0.4, mp.max() + dy),
            fontsize=9.5,
            color=c,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )
    maxL = max(len(rpa[k]) for k, _ in present)
    ax.set_xticks(range(0, maxL, 2))
    _th_axes(ax, "encoder layer", "test accuracy")
    ax.legend(loc="center right", fontsize=9.0, ncol=len(present), **TH_LEG)
    fig.text(0.53, 0.955, "Melody probing by layer — HookTheory", ha="center", fontsize=13.5)
    fig.text(
        0.53,
        0.905,
        "solid = RPA (raw pitch), dashed = RCA (chroma); the persistent "
        "~0.18 gap is octave confusion",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    _th_save(fig, thesis_dir, "hooktheory_layer_curves")

    # ── figure 2: learned gates per encoder ──
    gates = defaultdict(dict)
    for r in csv.DictReader((DATA / "layer_gates.csv").open()):
        gates[r["encoder"]][int(r["layer"])] = float(r["gate"])
    gpres = _present(gates)
    # width floor so the suptitle never crops when only 1-2 encoders are present
    fig_w = max(5.8, 3.9 * len(gpres))
    fig, axes = plt.subplots(1, len(gpres), figsize=(fig_w, 4.2), squeeze=False)
    axes = axes[0]
    left = 0.10 if len(gpres) == 1 else 0.06
    fig.subplots_adjust(top=0.80, bottom=0.15, left=left, right=0.985, wspace=0.24)
    for ax, (key, (name, c)) in zip(axes, gpres, strict=True):
        g = np.array([gates[key][L] for L in sorted(gates[key])])
        nl = len(g)
        ax.bar(range(nl), g, color=c, alpha=0.85, width=0.8)
        ax.axhline(1.0 / nl, color="#666666", lw=1.1, ls="--")
        acc_best = int(max(rpa[key], key=lambda L: rpa[key][L]))
        ax.annotate(
            "accuracy\nargmax",
            xy=(acc_best, g[acc_best]),
            xytext=(acc_best, max(g) * 1.12),
            ha="center",
            fontsize=9.0,
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
        )
        ax.set_title(name, fontsize=12, color=c, fontweight="bold")
        ax.set_ylim(0, max(g) * 1.3)
        ax.set_xticks(range(0, nl, 2 if nl > 15 else 1))
        _th_axes(ax, "layer", "softmax gate" if ax is axes[0] else "")
    axes[0].text(
        0.985,
        0.94,
        "dashed = uniform",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=9.0,
        color="#666666",
        style="italic",
    )
    fig.text(0.5, 0.955, "Learned layer gates — HookTheory", ha="center", fontsize=13)
    _th_save(fig, thesis_dir, "hooktheory_layer_gates")


if __name__ == "__main__":
    main()
