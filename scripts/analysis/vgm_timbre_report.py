#!/usr/bin/env python3
"""Reproducible figure/CSV report for a VGMIDITVar-timbre from-cache sweep.

Consolidates the one-off per-encoder analysis scripts (MuQ/MERT July 2026)
into one parameterized, committed tool. Two modes:

  # Per-encoder report: fig1-6 + figS1-3 (PNG+PDF) + summary_table.csv +
  # pool_means_by_layer.csv from a sweep output dir (layer{N}/ +
  # layermeanall/ as written by vgm_timbre_sweep_from_cache.py):
  uv run python scripts/analysis/vgm_timbre_report.py report \\
      --results output/vgm_timbre_from_cache_omarrq \\
      --encoder "OMAR-RQ" --num-layers 24 \\
      --out docs/figures/vgmiditvar_timbre_omarrq_varctl

  # Cross-encoder comparison (capability + mechanism panels):
  uv run python scripts/analysis/vgm_timbre_report.py compare \\
      --results output/vgm_timbre_from_cache_muq:MuQ-large \\
                output/vgm_timbre_from_cache_mert:MERT-v1-95M \\
                output/vgm_timbre_from_cache_omarrq:OMAR-RQ \\
      --out docs/figures/vgmiditvar_timbre_comparison

Annotations are data-driven (negative-gap ranges, best layers, converge/
re-widen verdicts computed from the numbers), so the script is
encoder-agnostic and re-runnable. Figures follow the repo dataviz
conventions (validated palette, one axis, thin marks, direct labels,
recessive grid). Precision rules baked in: "converge" only when the final
within-vs-twin gap < 0.05; the twin is always labelled a re-orchestration,
never a duplicate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── dataviz palette (validated) ────────────────────────────────────────────
BLUE, RED, AQUA, VIOLET, ORANGE = "#2a78d6", "#e34948", "#1baf7a", "#4a3aa7", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e4"
SEQ = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
ENCODER_COLORS = [VIOLET, ORANGE, AQUA, BLUE, RED]  # fixed assignment order

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "legend.frameon": False,
        "figure.dpi": 110,
    }
)

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


# ── loaders ────────────────────────────────────────────────────────────────


def load_metrics(results: Path, num_layers: int):
    layers = list(range(num_layers))
    M = {L: json.loads((results / f"layer{L}" / "metrics.json").read_text()) for L in layers}
    mean_p = results / "layermeanall" / "metrics.json"
    MEAN = json.loads(mean_p.read_text()) if mean_p.exists() else None
    return layers, M, MEAN


def load_dumps(results: Path, num_layers: int):
    return {
        L: json.loads((results / f"layer{L}" / "retrieval_score_distributions.json").read_text())
        for L in range(num_layers)
    }


def wmean(dump, pool, diag):
    """Count-weighted mean of a score pool over within- (diag) or
    cross-timbre (off-diag) cells."""
    ns = sm = 0.0
    for k, v in dump["cells"].items():
        q, t = k.split("_to_")
        if (q == t) == diag:
            ns += v[pool]["n"]
            sm += v[pool]["n"] * v[pool]["mean"]
    return sm / ns if ns else float("nan")


def twin_mean(dump):
    """Mean cosine of the same-composition twin pool: relevant minus the
    different-variation subset, cross-timbre cells only."""
    off = [v for k, v in dump["cells"].items() if k.split("_to_")[0] != k.split("_to_")[1]]
    rn = sum(v["relevant"]["n"] for v in off)
    vn = sum(v["relevant_diffvar"]["n"] for v in off)
    rm = wmean(dump, "relevant", False)
    vm = wmean(dump, "relevant_diffvar", False)
    return (rm * rn - vm * vn) / (rn - vn)


def pool_means(dumps, layers):
    within = np.array([wmean(dumps[L], "relevant", True) for L in layers])
    twin = np.array([twin_mean(dumps[L]) for L in layers])
    honest = np.array([wmean(dumps[L], "relevant_diffvar", False) for L in layers])
    distr = np.array([wmean(dumps[L], "distractor", False) for L in layers])
    return within, twin, honest, distr


def endlabel(ax, x, y, text, color, dy=0):
    ax.annotate(
        text,
        (x[-1], y[-1]),
        xytext=(6, dy),
        textcoords="offset points",
        color=color,
        fontsize=10.5,
        fontweight="bold",
        va="center",
    )


def _xticks(ax, X):
    ax.set_xticks(X[::2] if len(X) > 15 else X)


def _save(fig, out: Path, name: str, tight=True, **kw):
    if tight:
        fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=200, **kw)
    fig.savefig(out / f"{name}.pdf", **kw)
    plt.close(fig)


# ── per-encoder report ─────────────────────────────────────────────────────


def cmd_report(args):
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc, NL = args.encoder, args.num_layers
    layers, M, MEAN = load_metrics(results, NL)
    X = np.array(layers)

    def col(key):
        return np.array([M[L][key] for L in layers])

    gap_conf, gap_ctl = col("test/condition_gap"), col("test/condition_gap_varctl")
    within = col("test/map_same_condition")
    cross_conf, cross_ctl = col("test/map_cross_condition"), col("test/map_cross_condition_varctl")
    map_raw, map_cen, map_wht = col("test/map"), col("test/map_centered"), col("test/map_whitened")
    sep_cross, sep_ctl = col("test/score_sep_cross"), col("test/score_sep_cross_varctl")
    eff = col("test/anisotropy/effective_rank")
    best = int(np.argmax(cross_ctl))
    neg = np.where(gap_conf < 0)[0]

    # ── summary_table.csv ──
    hdr = [
        "layer",
        "map_raw",
        "map_centered",
        "map_whitened",
        "within",
        "cross_conf",
        "gap_conf",
        "cross_varctl",
        "gap_varctl",
        "twin_inflation",
        "sep_cross",
        "sep_cross_varctl",
        "eff_rank",
    ]
    rows = [
        [
            L,
            map_raw[i],
            map_cen[i],
            map_wht[i],
            within[i],
            cross_conf[i],
            gap_conf[i],
            cross_ctl[i],
            gap_ctl[i],
            cross_conf[i] - cross_ctl[i],
            sep_cross[i],
            sep_ctl[i],
            eff[i],
        ]
        for i, L in enumerate(layers)
    ]
    if MEAN:
        rows.append(
            [
                "meanall",
                MEAN["test/map"],
                MEAN["test/map_centered"],
                MEAN["test/map_whitened"],
                MEAN["test/map_same_condition"],
                MEAN["test/map_cross_condition"],
                MEAN["test/condition_gap"],
                MEAN["test/map_cross_condition_varctl"],
                MEAN["test/condition_gap_varctl"],
                MEAN["test/map_cross_condition"] - MEAN["test/map_cross_condition_varctl"],
                MEAN["test/score_sep_cross"],
                MEAN["test/score_sep_cross_varctl"],
                MEAN["test/anisotropy/effective_rank"],
            ]
        )
    with open(out / "summary_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow([r[0]] + [f"{v:.4f}" for v in r[1:]])

    # ── fig1: confounded vs controlled gap ──
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.axhline(0, color=INK2, lw=1.0, zorder=1)
    ax.fill_between(X, gap_conf, gap_ctl, color=RED, alpha=0.07, zorder=1)
    ax.plot(
        X,
        gap_conf,
        color=RED,
        lw=2.2,
        marker="o",
        ms=5,
        zorder=3,
        label="confounded (twin in relevance)",
    )
    ax.plot(
        X,
        gap_ctl,
        color=BLUE,
        lw=2.2,
        marker="o",
        ms=5,
        zorder=3,
        label="variation-controlled (twin masked)",
    )
    endlabel(ax, X, gap_conf, "confounded", RED, dy=-2)
    endlabel(ax, X, gap_ctl, "controlled", BLUE, dy=2)
    if len(neg):
        story = (
            f"confounded flips negative at L{neg[0]}–L{neg[-1]}"
            if len(neg) > 1
            else f"confounded flips negative at L{neg[0]}"
        )
        title1 = f"{enc}: the twin flips the gap negative ({story.split('at ')[1]})"
    else:
        story = "confounded narrows but never flips negative"
        title1 = f"{enc}: the twin narrows the gap but never flips it negative"
    ax.annotate(
        f"controlled: within > cross at all {NL} layers\n{story} (twin artifact)",
        (X.mean(), float(np.median(gap_ctl)) * 0.62),
        color=INK2,
        fontsize=9.4,
        ha="center",
    )
    _xticks(ax, X)
    ax.set_xlabel(f"{enc} layer")
    ax.set_ylabel("timbre gap  (within − cross MAP)")
    ax.set_title(
        f"{title1}\n{enc} · VGMIDITVar-timbre · N=102,960 · 5,040 works × 8 GM programs",
        fontsize=11.5,
    )
    ax.legend(loc="best", fontsize=9.2)
    ax.set_xlim(-0.4, NL * 1.12)
    _save(fig, out, "fig1_gap_confound_vs_controlled")

    # ── fig2: within vs cross panels ──
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
    for ax, cr, ttl, mark in [
        (axes[0], cross_conf, "Confounded relevance (twin included)", False),
        (axes[1], cross_ctl, "Variation-controlled (twin masked)", True),
    ]:
        ax.plot(X, within, color=BLUE, lw=2.2, marker="o", ms=4.5, label="within-timbre")
        ax.plot(X, cr, color=RED, lw=2.2, marker="o", ms=4.5, label="cross-timbre")
        if mark:
            endlabel(ax, X, within, "within", BLUE, dy=3)
            endlabel(ax, X, cr, "cross", RED, dy=-3)
        _xticks(ax, X)
        ax.set_xlabel(f"{enc} layer")
        ax.set_title(ttl, fontsize=11.5)
        ax.set_xlim(-0.4, NL * 1.14)
    axes[0].set_ylabel("MAP (centered cosine)")
    axes[0].legend(loc="lower right", fontsize=9.5)
    fig.suptitle(
        f"{enc}: variation control restores within > cross at every layer", fontsize=12.2, y=1.0
    )
    _save(fig, out, "fig2_within_vs_cross_panels")

    # ── fig3: honest cross-timbre curve ──
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(X, cross_ctl, color=BLUE, lw=2.2, marker="o", ms=5, zorder=3)
    ax.scatter([best], [cross_ctl[best]], s=110, facecolor="none", edgecolor=INK, lw=1.4, zorder=4)
    ax.annotate(
        f"best: layer {best} ({cross_ctl[best]:.3f})",
        (best, cross_ctl[best]),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=10,
        color=INK,
    )
    if MEAN:
        mv = MEAN["test/map_cross_condition_varctl"]
        ax.axhline(mv, color=VIOLET, lw=1.6, ls="--")
        ax.annotate(
            f"meanall ({mv:.3f})",
            (X[-1] * 0.97, mv),
            xytext=(0, 6),
            textcoords="offset points",
            color=VIOLET,
            fontsize=9.5,
            ha="right",
        )
    _xticks(ax, X)
    ax.set_xlabel(f"{enc} layer")
    ax.set_ylabel("cross-timbre MAP, different variation")
    ax.set_title(
        f"Honest cross-orchestration retrieval by depth — {enc}\n"
        "(retrieve a different variation of the theme in a different instrument)"
    )
    _save(fig, out, "fig3_cross_timbre_varctl_by_layer")

    # ── fig4: geometry treatments ──
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for y, c, lbl, dy in [
        (map_raw, BLUE, "raw cosine", -4),
        (map_cen, AQUA, "centered", 6),
        (map_wht, VIOLET, "ZCA-whitened (α=1)", 2),
    ]:
        ax.plot(X, y, color=c, lw=2.2, marker="o", ms=4.5, label=lbl)
        endlabel(ax, X, y, lbl.split(" ")[0], c, dy=dy)
    _xticks(ax, X)
    ax.set_xlabel(f"{enc} layer")
    ax.set_ylabel("MAP (all-variation relevance)")
    ax.set_title(f"Transductive whitening lifts retrieval MAP at every layer — {enc}")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim(-0.4, NL * 1.18)
    _save(fig, out, "fig4_geometry_treatments")

    # ── fig5: 8×8 grids at the best layer ──
    def load_grid(name):
        grid, conds = {}, []
        with open(results / f"layer{best}" / name) as f:
            for r in csv.DictReader(f):
                q, t = int(r["query_program"]), int(r["target_program"])
                grid[(q, t)] = float(r["map"])
                if q not in conds:
                    conds.append(q)
        conds = sorted(conds)
        return np.array([[grid[(q, t)] for t in conds] for q in conds]), conds

    from matplotlib.colors import LinearSegmentedColormap

    Ac, conds = load_grid("condition_grid.csv")
    Av, _ = load_grid("condition_grid_varctl.csv")
    labels = [GM.get(c, str(c)) for c in conds]
    vmax = max(Ac.max(), Av.max())
    cmap = LinearSegmentedColormap.from_list("seqblue", [SURFACE] + SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
    for ax, A, ttl in [
        (axes[0], Ac, "confounded (twin in relevance)"),
        (axes[1], Av, "variation-controlled (twin masked)"),
    ]:
        im = ax.imshow(A, cmap=cmap, vmin=0, vmax=vmax)
        n = len(labels)
        ax.set_xticks(range(n), labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n), labels, fontsize=9)
        ax.set_xlabel("target instrument", fontsize=10)
        ax.set_title(ttl, fontsize=11.5)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                v = A[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}"[1:] if v < 1 else f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    color=SURFACE if v > 0.55 * vmax else INK,
                )
    axes[0].set_ylabel("query instrument", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.85, label="MAP")
    fig.suptitle(
        f"Per-instrument MAP grid, {enc} layer {best} — "
        "off-diagonal dominance disappears once the twin is masked",
        fontsize=12.2,
    )
    _save(fig, out, "fig5_grids_best_layer", tight=False, bbox_inches="tight")

    # ── fig6: score-separation fingerprint ──
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(
        X, sep_cross, color=RED, lw=2.2, marker="o", ms=4.5, label="cross-timbre, twin included"
    )
    ax.plot(X, sep_ctl, color=BLUE, lw=2.2, marker="o", ms=4.5, label="cross-timbre, twin masked")
    endlabel(ax, X, sep_cross, "with twin", RED, dy=3)
    endlabel(ax, X, sep_ctl, "twin masked", BLUE, dy=-3)
    _xticks(ax, X)
    ax.set_xlabel(f"{enc} layer")
    ax.set_ylabel("relevant − distractor mean cosine")
    ax.set_title(
        f"The twin's fingerprint in the raw score geometry — {enc}\n"
        "(masking the twin lowers cross-timbre separation at every layer)"
    )
    ax.legend(loc="best", fontsize=9.5)
    ax.set_xlim(-0.4, NL * 1.17)
    _save(fig, out, "fig6_score_separation")

    # ── score-geometry figures (S1–S3) + pool_means CSV ──
    dumps = load_dumps(results, NL)
    within_m, twin_m, honest_m, distr_m = pool_means(dumps, layers)
    with open(out / "pool_means_by_layer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "layer",
                "within_timbre_diffvar",
                "twin_same_composition_crosstimbre",
                "honest_crosstimbre_diffvar",
                "distractor",
            ]
        )
        for i, L in enumerate(layers):
            w.writerow(
                [
                    L,
                    f"{within_m[i]:.4f}",
                    f"{twin_m[i]:.4f}",
                    f"{honest_m[i]:.4f}",
                    f"{distr_m[i]:.4f}",
                ]
            )

    L_ill = best
    D = dumps[L_ill]
    edges = np.array(D["overall"]["relevant"]["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    BW = edges[1] - edges[0]

    def agg_hist(pool, diag):
        h = np.zeros(len(centers))
        for k, v in D["cells"].items():
            q, t = k.split("_to_")
            if (q == t) == diag:
                h += np.array(v[pool]["hist"], dtype=float)
        return h

    def density(h):
        s = h.sum()
        return h / (s * BW) if s > 0 else h

    # figS1 — cross-timbre cosine distributions
    h_rel, h_diff = agg_hist("relevant", False), agg_hist("relevant_diffvar", False)
    tot = h_rel.sum()
    d_distr = density(agg_hist("distractor", False))
    d_diff, d_twin = h_diff / (tot * BW), (h_rel - h_diff) / (tot * BW)
    m_distr, m_diff = wmean(D, "distractor", False), wmean(D, "relevant_diffvar", False)
    m_twin = twin_mean(D)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for d, c, a, lbl in [
        (d_distr, INK2, 0.15, "distractor  (different work)"),
        (d_diff, BLUE, 0.18, "honest relevant  (same theme, different variation)"),
        (d_twin, RED, 0.26, "twin  (same composition, re-orchestrated)"),
    ]:
        ax.fill_between(centers, d, color=c, alpha=a)
        ax.plot(centers, d, color=c, lw=2.0, label=lbl)
    for m, c in [(m_distr, INK2), (m_diff, BLUE), (m_twin, RED)]:
        ax.annotate(
            f"μ={m:.2f}",
            (m, 0),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            color=c,
            fontsize=9.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=c, lw=1.0, alpha=0.5),
        )
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(bottom=-0.18)
    ax.set_xlabel("cosine similarity (centered embeddings)")
    ax.set_ylabel("density")
    ax.set_title(
        f"Score geometry of cross-timbre retrieval, {enc} layer {L_ill}\n"
        "the confound is a same-composition re-render (not a cos≈1 duplicate)"
    )
    ax.legend(loc="upper left", fontsize=9.2)
    _save(fig, out, "figS1_score_distributions_crosstimbre")

    # figS2 — within vs cross relevant distributions
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True, sharex=True)
    for ax, diag, ttl, rc, rlabel in [
        (
            axes[0],
            True,
            "within-timbre cells (query instrument = target)",
            BLUE,
            "relevant — same theme, different variation",
        ),
        (
            axes[1],
            False,
            "cross-timbre cells (query ≠ target)",
            RED,
            "relevant — same theme (incl. re-orchestrated twin)",
        ),
    ]:
        dd, dr = density(agg_hist("distractor", diag)), density(agg_hist("relevant", diag))
        ax.fill_between(centers, dd, color=INK2, alpha=0.15)
        ax.plot(centers, dd, color=INK2, lw=1.6, label="distractor (different work)")
        ax.fill_between(centers, dr, color=rc, alpha=0.18)
        ax.plot(centers, dr, color=rc, lw=2.0, label=rlabel)
        ax.set_xlabel("cosine similarity")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlim(-0.5, 1.0)
        ax.legend(loc="upper left", fontsize=8.8)
    axes[0].set_ylabel("density")
    fig.suptitle(
        "The same-composition twin adds high-score mass ONLY in cross-timbre cells "
        f"— {enc} layer {L_ill}",
        fontsize=12.0,
        y=1.0,
    )
    _save(fig, out, "figS2_within_vs_cross_distributions")

    # figS3 — the four pool means by depth, data-driven verdict
    g0, gN = within_m[0] - twin_m[0], within_m[-1] - twin_m[-1]
    g_min_i = int(np.argmin(within_m - twin_m))
    g_min = float((within_m - twin_m)[g_min_i])
    if gN < 0.05:
        verdict = f"converge (gap {gN:+.2f})"
        sub = "same-timbre falls · same-notes-across-timbre rises · they converge deep"
    elif g_min < gN - 0.03:
        verdict = f"gap narrows to {g_min:+.2f} at L{layers[g_min_i]}\nthen RE-WIDENS to {gN:+.2f}"
        sub = "composition similarity peaks mid-network, then late layers re-specialize"
    else:
        verdict = f"gap stays wide {gN:+.2f}\n(does NOT converge)"
        sub = "same-timbre barely falls · same-notes-across-timbre rises modestly · gap stays wide"
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    for y, c, lbl, ms in [
        (within_m, AQUA, "same timbre, different variation  (within-timbre positive)", 5),
        (twin_m, RED, "same composition, different timbre  (the twin)", 5),
        (honest_m, BLUE, "different variation, different timbre  (honest cross)", 5),
        (distr_m, INK2, "different work  (distractor)", 4),
    ]:
        ax.plot(X, y, color=c, lw=2.2, marker="o", ms=ms, label=lbl)
    ax.annotate(
        f"gap {g0:+.2f}",
        (0, (within_m[0] + twin_m[0]) / 2),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        color=INK2,
        fontsize=9.0,
    )
    ax.annotate(
        verdict,
        (X[-1], (within_m[-1] + twin_m[-1]) / 2),
        xytext=(-16, -6),
        textcoords="offset points",
        ha="right",
        va="center",
        color=INK,
        fontsize=9.0,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    _xticks(ax, X)
    ax.set_xlabel(f"{enc} layer")
    ax.set_ylabel("mean cosine similarity")
    ax.set_ylim(-0.12, 0.92)
    ax.set_xlim(-0.4, NL * 1.02)
    ax.set_title(f"{enc}: timbre vs composition across depth\n{sub}")
    ax.legend(loc="center left", fontsize=8.4)
    _save(fig, out, "figS3_timbre_composition_shift")

    print(
        f"[report] {enc}: {NL} layers  best honest-cross L{best}={cross_ctl[best]:.4f}  "
        f"neg-gap layers={list(neg)}  within-twin gap L0 {g0:+.2f} -> L{layers[-1]} {gN:+.2f}"
    )
    print(f"[report] wrote fig1-6 + figS1-3 + summary_table.csv + pool_means_by_layer.csv -> {out}")


# ── cross-encoder comparison ───────────────────────────────────────────────


def cmd_compare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    encs = []  # (name, layers, cross_ctl, gap_series)
    for spec in args.results:
        path, name = spec.rsplit(":", 1)
        results = Path(path)
        nl = 0
        while (results / f"layer{nl}" / "metrics.json").exists():
            nl += 1
        layers, M, _ = load_metrics(results, nl)
        cross_ctl = np.array([M[L]["test/map_cross_condition_varctl"] for L in layers])
        dumps = load_dumps(results, nl)
        within_m, twin_m, _, _ = pool_means(dumps, layers)
        encs.append((name, np.array(layers), cross_ctl, within_m - twin_m))
        print(
            f"[compare] {name}: {nl} layers, best L{int(np.argmax(cross_ctl))}={cross_ctl.max():.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
    ax = axes[0]
    for i, (name, X, cc, _g) in enumerate(encs):
        c = ENCODER_COLORS[i % len(ENCODER_COLORS)]
        ax.plot(X, cc, color=c, lw=2.3, marker="o", ms=4.5, label=name)
        b = int(np.argmax(cc))
        ax.scatter([b], [cc[b]], s=90, facecolor="none", edgecolor=c, lw=1.4, zorder=4)
        ax.annotate(
            f"{name} best L{b} = {cc[b]:.3f}",
            (b, cc[b]),
            xytext=(6, 8),
            textcoords="offset points",
            color=c,
            fontsize=8.8,
            fontweight="bold",
        )
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("cross-timbre MAP, different variation")
    ax.set_title("Capability: honest cross-orchestration retrieval", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.axhline(0, color=INK2, lw=1.0)
    for i, (name, X, _cc, g) in enumerate(encs):
        c = ENCODER_COLORS[i % len(ENCODER_COLORS)]
        ax.plot(X, g, color=c, lw=2.3, marker="o", ms=4.5, label=f"{name} ({g[-1]:+.2f} @ last)")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("within-timbre − twin  (mean cosine)")
    ax.set_title("Mechanism: does same-timbre still beat same-notes deep?", fontsize=11.5)
    ax.legend(loc="lower left", fontsize=9.0)

    fig.suptitle(
        "Cross-orchestration theme identity across encoders — "
        "the depth shift is general in direction, encoder-specific in magnitude",
        fontsize=12.3,
        y=1.005,
    )
    _save(fig, out, "fig_encoder_comparison", tight=True, bbox_inches="tight")
    print(f"[compare] wrote fig_encoder_comparison -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="per-encoder figures + CSVs")
    r.add_argument("--results", required=True, help="sweep output dir (layer{N}/ + layermeanall/)")
    r.add_argument("--encoder", required=True, help="display name, e.g. 'OMAR-RQ'")
    r.add_argument("--num-layers", type=int, required=True)
    r.add_argument("--out", required=True, help="output dir for figures + CSVs")
    c = sub.add_parser("compare", help="multi-encoder comparison figure")
    c.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="one or more '<results_dir>:<display name>' specs",
    )
    c.add_argument("--out", required=True)
    t = sub.add_parser(
        "thesis",
        help="thesis-styled figure set: two_lenses, timbre_composition_shift, "
        "depth_whitening (lead encoder = FIRST spec), encoder_comparison (all), "
        "instrument_grids (per encoder). Writes figures/chapters/*.pdf + defense/*.png",
    )
    t.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="'<results_dir>:<display name>' specs; the FIRST is the lead encoder",
    )
    t.add_argument("--thesis-dir", required=True, help="thesis repo root (contains figures/)")
    args = ap.parse_args()
    if args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        cmd_thesis(args)


# ── thesis-figure mode ─────────────────────────────────────────────────────
# Reproduces (and supersedes) the thesis repo's figures/src/vgm_*.py one-off
# scripts: exact thesis styling — "instrument" terminology, framed legends,
# fig.text title + gray subtitle, white-boxed annotations, italic
# direction hints — generated straight from sweep outputs / committed CSVs.
# Adding an encoder (e.g. CLaMP3) = one more --results spec, zero edits.

TH_GREEN, TH_RED, TH_BLUE, TH_GRAY = "#1BAF7A", "#E34948", "#2A78D6", "#52514E"
TH_VIOLET, TH_ORANGE = "#5B4FC4", "#E8752E"
TH_CRIMSON, TH_FOREST, TH_GOLD = "#B02637", "#2B8A57", "#D9A521"
TH_COMPARE = [TH_VIOLET, TH_ORANGE, TH_GREEN, TH_CRIMSON, TH_GOLD]
TH_BOX = dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85)
TH_LEG = dict(frameon=True, framealpha=0.95, edgecolor="#CCCCCC")


def _th_axes(ax, layers, xlabel):
    ax.set_xlabel(xlabel, fontsize=11.5)
    ax.set_xticks(layers[:: 2 if len(layers) > 15 else 1])
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
    print(f"[thesis] wrote {chapters / (stem + '.pdf')}")


def _slug(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _enc_slug(name: str) -> str:
    """Filename slug for an encoder display name: 'MERT-v1-95M' -> 'mert',
    'MuQ' -> 'muq', 'OMAR-RQ' -> 'omarrq', 'CLaMP3' -> 'clamp3'."""
    s = _slug(name)
    for junk in ("large", "v195m"):
        s = s.replace(junk, "")
    return s or _slug(name)


def cmd_thesis(args):
    thesis_dir = Path(args.thesis_dir)
    encs = []  # (display, layers, summary-cols dict, results Path)
    for spec in args.results:
        path, name = spec.rsplit(":", 1)
        results = Path(path)
        nl = 0
        while (results / f"layer{nl}" / "metrics.json").exists():
            nl += 1
        layers, M, _ = load_metrics(results, nl)
        cols = {
            "within": np.array([M[L]["test/map_same_condition"] for L in layers]),
            "cross_conf": np.array([M[L]["test/map_cross_condition"] for L in layers]),
            "cross_ctl": np.array([M[L]["test/map_cross_condition_varctl"] for L in layers]),
            "map_cen": np.array([M[L]["test/map_centered"] for L in layers]),
            "map_wht": np.array([M[L]["test/map_whitened"] for L in layers]),
        }
        dumps = load_dumps(results, nl)
        within_m, twin_m, _, distr_m = pool_means(dumps, layers)
        honest_m = np.array([wmean(dumps[L], "relevant_diffvar", False) for L in layers])
        cols.update(within_m=within_m, twin_m=twin_m, honest_m=honest_m, distr_m=distr_m)
        encs.append((name, np.array(layers), cols, results))
        print(f"[thesis] loaded {name}: {nl} layers")

    lead_name, LX, LC, lead_res = encs[0]

    # ── 1. vgm_two_lenses (lead encoder) ──
    fig, ax = plt.subplots(figsize=(9.75, 5.25))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.97)
    ax.plot(
        LX,
        LC["within"],
        "-o",
        color="#7F7F7F",
        lw=2.2,
        ms=6,
        label="within-instrument (same under both lenses)",
    )
    ax.plot(
        LX,
        LC["cross_conf"],
        "-s",
        color=TH_BLUE,
        lw=2.2,
        ms=6,
        label="cross-instrument, work-level lens\n(includes the same variation re-rendered)",
    )
    ax.plot(
        LX,
        LC["cross_ctl"],
        "-^",
        color=TH_CRIMSON,
        lw=2.2,
        ms=6,
        label="cross-instrument, variation-controlled lens\n(a different variation, different instrument)",
    )
    ax.set_ylabel("MAP", fontsize=12.5)
    _th_axes(ax, list(LX), f"{lead_name} layer")
    ax.set_title(
        f"Two relevance lenses on the rendered benchmark ({lead_name}, centered)",
        fontsize=13.5,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=10.3, labelspacing=0.6, **TH_LEG)
    _th_save(fig, thesis_dir, "vgm_two_lenses")

    # ── 2. vgm_timbre_composition_shift (ALL encoders; lead gets the
    #      canonical unsuffixed name the .tex includes) ──
    for name, EX, EC, _res in encs:
        within_m, twin_m = EC["within_m"], EC["twin_m"]
        g0, gN = within_m[0] - twin_m[0], within_m[-1] - twin_m[-1]
        fig, ax = plt.subplots(figsize=(9.75, 5.875))
        fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.97)
        ax.plot(
            EX,
            within_m,
            "-o",
            color=TH_GREEN,
            lw=2.4,
            ms=6,
            zorder=3,
            label="same instrument, different variation",
        )
        ax.plot(
            EX,
            twin_m,
            "-o",
            color=TH_RED,
            lw=2.4,
            ms=6,
            zorder=3,
            label="same composition, different instrument (the twin)",
        )
        ax.plot(
            EX,
            EC["honest_m"],
            "-o",
            color=TH_BLUE,
            lw=2.4,
            ms=6,
            zorder=3,
            label="different variation, different instrument (controlled positive)",
        )
        ax.plot(
            EX,
            EC["distr_m"],
            "-o",
            color=TH_GRAY,
            lw=2.0,
            ms=5,
            zorder=3,
            label="different work (distractor)",
        )
        ax.set_ylabel("mean cosine similarity", fontsize=12.5)
        _th_axes(ax, list(EX), f"{name} layer")
        ax.set_ylim(-0.12, 0.90)
        if gN < 0.05:
            head = f"{name} shifts from timbre-dominated to composition-dominated with depth"
            sub = (
                "same-instrument similarity falls · same-notes-across-instruments rises · "
                "they converge at depth"
            )
            verdict = f"converge\n(gap {gN:+.2f})"
            red_label = "composition rising"
        else:
            g_min_i = int(np.argmin(within_m - twin_m))
            g_min = float((within_m - twin_m)[g_min_i])
            if g_min < gN - 0.03:
                head = f"{name}: composition similarity peaks mid-network, then late layers re-specialize"
                sub = (
                    f"the instrument-vs-notes gap narrows to {g_min:+.2f} (layer {EX[g_min_i]}) "
                    f"then re-widens to {gN:+.2f}"
                )
                verdict = f"re-widens\n(gap {gN:+.2f})"
                red_label = "composition peaks mid-network"
            else:
                head = f"{name} stays instrument-biased at depth"
                sub = f"the instrument-vs-notes gap only narrows to {gN:+.2f} — no convergence"
                verdict = f"stays wide\n(gap {gN:+.2f})"
                red_label = "composition rises weakly"
        fig.text(0.53, 0.955, head, ha="center", fontsize=13.5)
        fig.text(0.53, 0.905, sub, ha="center", fontsize=11, color="#555555")
        ax.text(
            float(EX[0]) + 0.1 * len(EX),
            0.845,
            "timbre-dominated",
            color=TH_GREEN,
            fontsize=11.5,
            fontweight="bold",
        )
        ax.text(
            float(EX[0]) + 0.5 * len(EX),
            0.585,
            red_label,
            color=TH_RED,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
        )
        ax.annotate(
            "",
            xy=(0.14, within_m[0] - 0.012),
            xytext=(0.14, twin_m[0] + 0.015),
            arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.2),
        )
        ax.text(
            0.34, (within_m[0] + twin_m[0]) / 2, f"gap {g0:+.2f}", color="#666666", fontsize=10.5
        )
        end_mid = (within_m[-1] + twin_m[-1]) / 2
        if gN < 0.05:
            # curves nearly touch at the end: annotate from below the red endpoint
            v_xy = (float(EX[-1]) - 0.03 * len(EX), twin_m[-1] - 0.015)
            v_xytext = (float(EX[-1]) - 0.09 * len(EX), end_mid - 0.14)
        else:
            # a wide green-red endpoint gap: annotate inside it, arrow down to red
            v_xy = (float(EX[-1]) - 0.02 * len(EX), twin_m[-1] + 0.02)
            v_xytext = (float(EX[-1]) - 0.09 * len(EX), end_mid + 0.03)
        ax.annotate(
            verdict,
            xy=v_xy,
            xytext=v_xytext,
            ha="center",
            fontsize=10.5,
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
        )
        leg = ax.legend(
            loc="center",
            bbox_to_anchor=(0.505, 0.205),
            fontsize=10.3,
            borderpad=0.7,
            labelspacing=0.45,
            **TH_LEG,
        )
        leg.set_zorder(5)
        # Save PDF+PNG once, then hard-link/copy the canonical name for the lead.
        _th_save_multi = [f"vgm_timbre_composition_shift_{_enc_slug(name)}"]
        if name == lead_name:
            _th_save_multi.append("vgm_timbre_composition_shift")
        chapters = thesis_dir / "figures" / "chapters"
        defense = thesis_dir / "defense"
        chapters.mkdir(parents=True, exist_ok=True)
        for stem in _th_save_multi:
            fig.savefig(chapters / f"{stem}.pdf")
            if defense.is_dir():
                fig.savefig(defense / f"{stem}.png", dpi=160)
            print(f"[thesis] wrote {chapters / (stem + '.pdf')}")
        plt.close(fig)

    # ── 3. vgm_depth_whitening (lead encoder) ──
    fig, ax = plt.subplots(figsize=(9.75, 5.25))
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.09, right=0.97)
    cc = LC["cross_ctl"]
    plateau = np.where(cc >= 0.97 * cc.max())[0]
    if len(plateau) > 1:
        ax.axvspan(float(plateau[0]) - 0.5, float(plateau[-1]) + 0.5, color=TH_RED, alpha=0.06)
    ax.plot(
        LX,
        cc,
        "-^",
        color=TH_CRIMSON,
        lw=2.4,
        ms=6,
        label="variation-controlled cross-instrument MAP",
    )
    ax.plot(
        LX,
        LC["map_cen"],
        "-o",
        color=TH_FOREST,
        lw=2.4,
        ms=6,
        label="overall retrieval MAP (centered)",
    )
    ax.plot(
        LX,
        LC["map_wht"],
        "--o",
        color=TH_GOLD,
        lw=2.4,
        ms=6,
        label="overall retrieval MAP (whitened, transductive)",
    )
    ax.set_ylabel("MAP", fontsize=12.5)
    _th_axes(ax, list(LX), f"{lead_name} layer")
    ax.set_title(
        "Depth buys cross-timbre generalization; whitening lifts retrieval at every layer",
        fontsize=13.5,
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=10.5, **TH_LEG)
    _th_save(fig, thesis_dir, "vgm_depth_whitening")

    # ── 4. vgm_encoder_comparison (ALL encoders) ──
    max_nl = max(len(X) for _, X, _, _ in encs)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.075, right=0.975, wspace=0.26)
    ax = axes[0]
    ymax = 0.0
    for i, (name, X, C, _r) in enumerate(encs):
        c = TH_COMPARE[i % len(TH_COMPARE)]
        cc = C["cross_ctl"]
        ymax = max(ymax, float(cc.max()))
        ax.plot(X, cc, "-o", color=c, lw=2.4, ms=5.5, label=name)
        b = int(np.argmax(cc))
        dy = 0.017 if i % 2 == 0 else -0.022
        # Label right of the peak (clears the upper-left legend); flip to the
        # left when the peak sits in the right quarter of the axis.
        if b < 0.72 * max_nl:
            xt, ha = b + 0.03 * max_nl, "left"
        else:
            xt, ha = b - 0.03 * max_nl, "right"
        ax.annotate(
            f"{name} best L{b} = {cc[b]:.3f}",
            xy=(b, cc[b]),
            xytext=(xt, cc[b] + dy),
            ha=ha,
            fontsize=10.0,
            color=c,
            fontweight="bold",
            bbox=TH_BOX,
        )
    ax.set_title(
        "variation-controlled cross-instrument MAP\n(a different variation, a different instrument)",
        fontsize=12,
    )
    ax.set_ylabel("MAP", fontsize=11.5)
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(loc="upper left", fontsize=10.5, **TH_LEG)
    ax.text(
        0.98,
        0.03,
        "higher is better",
        transform=ax.transAxes,
        ha="right",
        fontsize=9.5,
        color="#666666",
        style="italic",
    )
    _th_axes(ax, list(range(max_nl)), "encoder layer")

    ax = axes[1]
    ax.axhline(0, color="#444444", lw=1.0)
    # Stack the verdict annotations in a fixed bottom-right column (an area
    # no curve occupies for any encoder measured so far) with arrows to each
    # curve's endpoint. Slots are assigned by endpoint rank (highest gap →
    # highest slot) so arrows never cross — deterministic for any encoder count.
    slots = [0.035, 0.095, 0.155, 0.215, 0.275]
    ends = []
    for i, (name, X, C, _r) in enumerate(encs):
        c = TH_COMPARE[i % len(TH_COMPARE)]
        g = C["within_m"] - C["twin_m"]
        ax.plot(X, g, "-o", color=c, lw=2.4, ms=5.5, label=name)
        gN = float(g[-1])
        word = (
            "converges to"
            if gN < 0.05
            else ("re-widens to" if float(g.min()) < gN - 0.03 else "narrows to")
        )
        ends.append((gN, name, c, float(X[-1]), word))
    for rank, (gN, name, c, xend, word) in enumerate(sorted(ends)):
        ax.annotate(
            f"{name}: {word} {gN:+.2f}",
            xy=(xend, gN),
            xytext=(0.985 * max_nl, slots[rank % len(slots)]),
            ha="right",
            fontsize=10.0,
            color=c,
            bbox=TH_BOX,
            arrowprops=dict(arrowstyle="->", color=c, lw=1.1, connectionstyle="arc3,rad=0.15"),
        )
    ax.set_title(
        "within-instrument minus twin, mean cosine\n(how much the instrument still dominates the notes)",
        fontsize=12,
    )
    ax.set_ylabel("cosine gap", fontsize=11.5)
    ax.set_ylim(-0.02, 0.50)
    ax.legend(loc="lower left", fontsize=10.5, **TH_LEG)
    ax.text(
        0.98,
        0.945,
        "lower is better (0 = instrument-invariant)",
        transform=ax.transAxes,
        ha="right",
        fontsize=9.5,
        color="#666666",
        style="italic",
    )
    _th_axes(ax, list(range(max_nl)), "encoder layer")
    names = [n for n, _x, _c, _r in encs]
    head = (
        " against ".join(names[:2])
        if len(names) == 2
        else ", ".join(names[:-1]) + f" and {names[-1]}"
    )
    fig.text(0.525, 0.955, f"{head} on the rendered benchmark", ha="center", fontsize=13)
    fig.text(
        0.525,
        0.905,
        "the depth shift is general in direction, encoder-specific in magnitude",
        ha="center",
        fontsize=11,
        color="#555555",
    )
    _th_save(fig, thesis_dir, "vgm_encoder_comparison")

    # ── 5. vgm_instrument_grids_<enc> (per encoder, best layer, lens framing) ──
    for name, _X, C, results in encs:
        best = int(np.argmax(C["cross_ctl"]))
        grids = {}
        for kind, fn in (("work", "condition_grid.csv"), ("ctl", "condition_grid_varctl.csv")):
            grid, conds = {}, []
            with open(results / f"layer{best}" / fn) as f:
                for r in csv.DictReader(f):
                    q, t = int(r["query_program"]), int(r["target_program"])
                    grid[(q, t)] = float(r["map"])
                    if q not in conds:
                        conds.append(q)
            conds = sorted(conds)
            grids[kind] = np.array([[grid[(q, t)] for t in conds] for q in conds])
        inst = [GM.get(c, str(c)) for c in conds]
        A, B = grids["work"], grids["ctl"]
        vmax = max(A.max(), B.max())
        n = len(inst)
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
        fig.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.90, wspace=0.28)
        for ax, Mx, title in (
            (axes[0], A, "work-level lens (twin in relevance)"),
            (axes[1], B, "variation-controlled lens (twin masked)"),
        ):
            im = ax.imshow(Mx, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(inst, rotation=40, ha="right", fontsize=10.5)
            ax.set_yticklabels(inst, fontsize=10.5)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("target instrument", fontsize=11)
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
        fig.text(
            0.5, 0.945, f"Per-instrument MAP grid, {name} layer {best}", ha="center", fontsize=13
        )
        _th_save(
            fig,
            thesis_dir,
            f"vgm_instrument_grids_{_enc_slug(name)}",
        )


if __name__ == "__main__":
    main()
