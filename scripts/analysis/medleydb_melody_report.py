#!/usr/bin/env python3
"""Thesis figures for the MedleyDB melody multi-head campaign.

Reads the committed CSVs in ``docs/figures/medleydb_melody_multihead/``
(written by the campaign aggregation, wandb-sourced) and renders two
thesis-styled figures into ``<thesis>/figures/chapters/`` (+ 160 dpi PNGs
into ``<thesis>/defense/``):

  * ``medleydb_layer_curves``  — per-layer test RPA, 5-fold mean ± sd bands,
    all encoders, best-layer markers.
  * ``medleydb_layer_gates``   — the learned SUPERB softmax gates (weighted
    head, fold 0) per encoder, with the accuracy-argmax layer marked: the
    gates-vs-accuracy dissociation exhibit (Feng et al., TASLP 2024).

Usage:
    uv run python scripts/analysis/medleydb_melody_report.py \\
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

DATA = Path("docs/figures/medleydb_melody_multihead")
TH_VIOLET, TH_ORANGE, TH_GREEN = "#5B4FC4", "#E8752E", "#1BAF7A"
TH_LEG = dict(frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
ENC_STYLE = {  # display name, colour  (CSV encoder key -> presentation)
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
    print(f"[medleydb] wrote {chapters / (stem + '.pdf')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thesis-dir", required=True)
    args = ap.parse_args()
    thesis_dir = Path(args.thesis_dir).expanduser()

    # ── load curves ──
    curves = defaultdict(dict)  # enc -> layer(int) -> (mean, sd)
    meanall = {}
    for r in csv.DictReader((DATA / "layer_curves_5fold.csv").open()):
        if r["layer"] == "meanall":
            meanall[r["encoder"]] = float(r["rpa_mean"])
        else:
            curves[r["encoder"]][int(r["layer"])] = (float(r["rpa_mean"]), float(r["rpa_sd"]))

    # ── figure 1: layer curves with sd bands ──
    fig, ax = plt.subplots(figsize=(9.75, 5.4))
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.97)
    for key, (name, c) in ENC_STYLE.items():
        d = curves[key]
        X = np.array(sorted(d))
        m = np.array([d[L][0] for L in X])
        sd = np.array([d[L][1] for L in X])
        ax.plot(X, m, "-o", color=c, lw=2.4, ms=5.5, label=name, zorder=3)
        ax.fill_between(X, m - sd, m + sd, color=c, alpha=0.10, zorder=1)
        b = int(X[np.argmax(m)])
        ax.scatter([b], [m.max()], s=110, facecolor="none", edgecolor=c, lw=1.5, zorder=4)
        ax.annotate(
            f"{name} best L{b} = {m.max():.3f}",
            xy=(b, m.max()),
            xytext=(b + 0.5, m.max() + 0.011),
            fontsize=10.0,
            color=c,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )
    ax.set_xticks(range(0, 24, 2))
    _th_axes(ax, "encoder layer", "test RPA (5-fold mean ± sd)")
    ax.legend(loc="lower right", fontsize=10.5, **TH_LEG)
    fig.text(
        0.53,
        0.955,
        "Melody probing by layer: three encoders, one protocol",
        ha="center",
        fontsize=13.5,
    )
    fig.text(
        0.53,
        0.905,
        "the melody peak sits below the instrument-invariance peak in every encoder "
        "(MuQ 1<8 · OMAR-RQ 5<15 · MERT 9<11)",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    _th_save(fig, thesis_dir, "medleydb_layer_curves")

    # ── figure 2: learned gates per encoder ──
    gates = defaultdict(dict)
    for r in csv.DictReader((DATA / "layer_gates_fold0.csv").open()):
        gates[r["encoder"]][int(r["layer"])] = float(r["gate"])
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.2))
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.06, right=0.985, wspace=0.24)
    for ax, (key, (name, c)) in zip(axes, ENC_STYLE.items(), strict=False):
        g = np.array([gates[key][L] for L in sorted(gates[key])])
        nl = len(g)
        ax.bar(range(nl), g, color=c, alpha=0.85, width=0.8)
        ax.axhline(1.0 / nl, color="#666666", lw=1.1, ls="--")
        acc_best = int(max(curves[key], key=lambda L: curves[key][L][0]))
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
        _th_axes(ax, "layer", "softmax gate" if key == "MuQ" else "")
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
    fig.text(
        0.525,
        0.955,
        "Learned layer gates of the weighted-sum head (fold 0)",
        ha="center",
        fontsize=13,
    )
    fig.text(
        0.525,
        0.895,
        "gates ≠ importance: aligned for MERT (ρ=0.97), loose for MuQ (ρ=0.69), "
        "bimodal and dissociated for OMAR-RQ (ρ=0.15)",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    _th_save(fig, thesis_dir, "medleydb_layer_gates")


if __name__ == "__main__":
    main()
