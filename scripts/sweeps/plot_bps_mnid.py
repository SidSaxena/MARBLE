"""Comprehensive plots for the BPS-Motif MNID per-layer probe sweep.

Reads the JSON written by ``bps_mnid_summary.py`` and renders a 2x2 figure:

  (a) Per-layer metric profiles  — auc_roc / auc_pr / acc / f1, each as a
      5-fold mean with a ±std band, plus the meanall baseline as a horizontal
      reference band and the best layer starred. Shows WHERE the motif signal
      peaks across CLaMP3's depth.
  (b) auc_roc distribution by layer — boxplot of the 5 per-fold values per
      layer. Shows the SPREAD, not just the mean: which layers are reliably
      good vs which only look good on easy folds.
  (c) layer x fold heatmap (auc_roc) — every cell annotated. Surfaces the
      hard fold (column) and confirms the peak band (rows) is consistent.
  (d) Per-fold trajectories — each fold's auc_roc-vs-layer curve, the bold
      cross-fold mean, and the meanall band. Shows the peak is robust, not an
      artifact of averaging.

A second, optional figure overlays the MNID profile against the zero-shot
Retrieval MAP profile (``--retrieval-json``) to test whether supervised
separability and unsupervised retrieval peak at the same depth.

Usage:
  python3 scripts/sweeps/plot_bps_mnid.py --json bps_mnid_summary.json \
      --out fig.png [--retrieval-json retr.json --out-cmp cmp.png]
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {
    "auc_roc": "#1f77b4",
    "auc_pr": "#2ca02c",
    "acc": "#d62728",
    "f1": "#ff7f0e",
}


def _matrix(d, metric):
    """(n_layers, n_folds) array of per-cell values for `metric`."""
    layers, folds = d["layers"], d["folds"]
    return np.array(
        [[d["per_cell"][f"{f}.{l}"][metric] for f in folds] for l in layers],
        dtype=float,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/tmp/bps_mnid_summary.json")
    ap.add_argument("--out", default="/tmp/bps_mnid_layers.png")
    ap.add_argument("--retrieval-json", default=None)
    ap.add_argument("--out-cmp", default="/tmp/bps_mnid_vs_retrieval.png")
    args = ap.parse_args()

    with open(args.json) as fh:
        d = json.load(fh)
    layers = d["layers"]
    folds = d["folds"]
    best = d["best_layer"]

    def mean_std(metric):
        m = [d["layer_agg"][str(l)][metric]["mean"] for l in layers]
        s = [d["layer_agg"][str(l)][metric]["std"] or 0.0 for l in layers]
        return np.array(m, float), np.array(s, float)

    fig, ax = plt.subplots(2, 2, figsize=(15.5, 11.5))

    # ---- (a) metric profiles ----
    a = ax[0, 0]
    for metric in ["auc_roc", "auc_pr", "acc", "f1"]:
        m, s = mean_std(metric)
        a.plot(layers, m, "-o", color=COLORS[metric], label=metric, lw=1.9, ms=4)
        a.fill_between(layers, m - s, m + s, color=COLORS[metric], alpha=0.12)
    ma = d["meanall_agg"].get("auc_roc", {})
    if ma.get("mean") is not None:
        mu, sd = ma["mean"], (ma["std"] or 0.0)
        a.axhspan(mu - sd, mu + sd, color="gray", alpha=0.12)
        a.axhline(mu, color="gray", ls="--", lw=1.2)
        a.text(
            layers[0],
            mu,
            f" meanall auc_roc {mu:.3f}±{sd:.3f}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="dimgray",
        )
    bm = d["layer_agg"][str(best)]["auc_roc"]["mean"]
    a.plot(
        [best],
        [bm],
        "*",
        color="gold",
        ms=20,
        mec="black",
        mew=0.7,
        zorder=5,
        label=f"best layer = {best}",
    )
    a.set_title("(a) Per-layer metric profile (5-fold mean ± std)")
    a.set_xlabel("CLaMP3-symbolic layer")
    a.set_ylabel("test metric")
    a.set_xticks(layers)
    a.grid(alpha=0.3)
    a.legend(loc="lower left", ncol=2, fontsize=8)

    # ---- (b) auc_roc distribution by layer ----
    b = ax[0, 1]
    M = _matrix(d, "auc_roc")
    bp = b.boxplot(
        [M[i] for i in range(len(layers))],
        positions=layers,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
    )
    for patch in bp["boxes"]:
        patch.set(facecolor=COLORS["auc_roc"], alpha=0.30)
    for i, l in enumerate(layers):
        b.scatter([l] * M.shape[1], M[i], s=12, color=COLORS["auc_roc"], alpha=0.7, zorder=3)
    if ma.get("mean") is not None:
        b.axhspan(
            ma["mean"] - (ma["std"] or 0), ma["mean"] + (ma["std"] or 0), color="gray", alpha=0.12
        )
        b.axhline(ma["mean"], color="gray", ls="--", lw=1.2, label="meanall")
        b.legend(loc="lower left", fontsize=8)
    b.set_title("(b) auc_roc distribution across folds, by layer")
    b.set_xlabel("CLaMP3-symbolic layer")
    b.set_ylabel("test/auc_roc (per fold)")
    b.set_xticks(layers)
    b.grid(alpha=0.3, axis="y")

    # ---- (c) layer x fold heatmap ----
    c = ax[1, 0]
    im = c.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    c.set_xticks(range(len(folds)))
    c.set_xticklabels([f"fold{f}" for f in folds])
    c.set_yticks(range(len(layers)))
    c.set_yticklabels(layers)
    c.set_xlabel("CV fold")
    c.set_ylabel("CLaMP3-symbolic layer")
    c.set_title("(c) test/auc_roc — layer × fold")
    for i in range(len(layers)):
        for j in range(len(folds)):
            c.text(
                j,
                i,
                f"{M[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if M[i, j] < M.mean() else "black",
            )
    fig.colorbar(im, ax=c, fraction=0.046, pad=0.04, label="auc_roc")

    # ---- (d) per-fold trajectories ----
    e = ax[1, 1]
    for j, f in enumerate(folds):
        e.plot(layers, M[:, j], "-", lw=1.1, alpha=0.6, label=f"fold{f}")
    e.plot(
        layers, M.mean(axis=1), "-o", color="black", lw=2.4, ms=4, label="cross-fold mean", zorder=5
    )
    if ma.get("mean") is not None:
        e.axhspan(
            ma["mean"] - (ma["std"] or 0), ma["mean"] + (ma["std"] or 0), color="gray", alpha=0.12
        )
        e.axhline(ma["mean"], color="gray", ls="--", lw=1.2)
    e.axvline(best, color="gold", lw=2, alpha=0.6)
    e.set_title("(d) Per-fold auc_roc trajectories")
    e.set_xlabel("CLaMP3-symbolic layer")
    e.set_ylabel("test/auc_roc")
    e.set_xticks(layers)
    e.grid(alpha=0.3)
    e.legend(loc="lower left", ncol=2, fontsize=8)

    fig.suptitle(
        "BPS-Motif MNID — CLaMP3-symbolic per-layer probe (5-fold movement-level CV)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)

    # ---- optional: MNID vs Retrieval overlay ----
    if args.retrieval_json:
        try:
            with open(args.retrieval_json) as fh:
                r = json.load(fh)
        except Exception as ex:
            print("retrieval json unreadable:", ex)
            return
        fig2, axc = plt.subplots(figsize=(9.5, 5.8))
        m, s = mean_std("auc_roc")
        axc.plot(
            layers, m, "-o", color=COLORS["auc_roc"], label="MNID auc_roc (supervised)", lw=1.9
        )
        axc.fill_between(layers, m - s, m + s, color=COLORS["auc_roc"], alpha=0.12)
        axc.set_xlabel("CLaMP3-symbolic layer")
        axc.set_ylabel("MNID test/auc_roc", color=COLORS["auc_roc"])
        axc.tick_params(axis="y", labelcolor=COLORS["auc_roc"])
        axc.set_xticks(layers)
        axc.grid(alpha=0.3)
        # retrieval on twin axis (expects r["layers"], r["map_mean"] or similar)
        rl = r.get("layers", layers)
        rmap = r.get("map_mean") or r.get("map")
        if rmap:
            axr = axc.twinx()
            axr.plot(rl, rmap, "-s", color="#9467bd", label="Retrieval MAP (zero-shot)", lw=1.9)
            axr.set_ylabel("Retrieval MAP", color="#9467bd")
            axr.tick_params(axis="y", labelcolor="#9467bd")
        axc.set_title("CLaMP3-symbolic: supervised MNID vs zero-shot Retrieval — layer profiles")
        fig2.tight_layout()
        fig2.savefig(args.out_cmp, dpi=150)
        print("wrote", args.out_cmp)


if __name__ == "__main__":
    main()
