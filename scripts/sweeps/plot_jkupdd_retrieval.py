"""Per-layer MAP figure for the JKUPDD within-piece motif retrieval sweep.

Reads the leaderboard CSV written by ``jkupdd_retrieval_summary.py`` (rows are
ranked best-first; one row per layer plus a final ``meanall`` row) and renders a
single line plot of raw / centered / whitened ``test/map`` vs layer index, with
the meanall baseline as a horizontal reference band and the best layer starred.

The point of the figure is the cross-composer *shape*: does JKUPDD reproduce the
BPS-Motif mid-layer peak (L6/L7) and last-layer (L12) collapse? See
``docs/jkupdd_retrieval_clamp3_layersweep.md``.

Usage:
  python3 scripts/sweeps/plot_jkupdd_retrieval.py --csv LEADERBOARD.csv --out FIG.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VARIANTS = [
    ("map", "raw", "#1f77b4", "o"),
    ("map_centered", "centered", "#2ca02c", "s"),
    ("map_whitened", "whitened", "#d62728", "^"),
]


def load(csv_path):
    """Return (per_layer: {col: {layer: val}}, meanall: {col: val}, best_layer)."""
    per_layer = {col: {} for col, *_ in VARIANTS}
    meanall = {}
    best_layer = None
    with open(csv_path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if row["layer"] == "meanall":
                for col, *_ in VARIANTS:
                    meanall[col] = float(row[col])
                continue
            layer = int(row["layer"])
            if i == 0:  # first data row is rank-1 = best layer
                best_layer = layer
            for col, *_ in VARIANTS:
                per_layer[col][layer] = float(row[col])
    return per_layer, meanall, best_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_layer, meanall, best_layer = load(args.csv)
    layers = sorted(per_layer["map"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label, color, marker in VARIANTS:
        ys = [per_layer[col][l] for l in layers]
        ax.plot(layers, ys, marker=marker, color=color, label=f"{label} MAP", lw=1.8)
        if col in meanall:
            ax.axhline(
                meanall[col],
                color=color,
                ls="--",
                lw=1.0,
                alpha=0.5,
            )

    # Star the best (raw) layer.
    if best_layer is not None:
        ax.scatter(
            [best_layer],
            [per_layer["map"][best_layer]],
            s=260,
            marker="*",
            color="#1f77b4",
            edgecolor="black",
            zorder=5,
            label=f"best layer {best_layer}",
        )

    ax.set_xlabel("CLaMP3-symbolic transformer layer")
    ax.set_ylabel("test/MAP")
    ax.set_title(
        "JKUPDD within-piece motif retrieval — CLaMP3-symbolic per-layer MAP\n"
        "(5 cross-composer pieces, 165 occurrence windows, no folds; "
        "dashed = meanall baseline)"
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
