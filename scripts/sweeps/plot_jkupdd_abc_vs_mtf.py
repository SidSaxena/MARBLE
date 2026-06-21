"""Per-layer MAP figure for the JKUPDD ABC-vs-MTF A/B.

Reads the leaderboard CSV written by ``jkupdd_abc_vs_mtf_summary.py`` (one row
per layer + a final ``meanall`` row, columns ``abc_map`` / ``mtf_map``) and
overlays the two arms' raw ``test/map`` vs layer, with each arm's meanall as a
horizontal reference and each arm's best layer starred. The question the figure
answers: does notation-preserving ABC beat lossy MTF, and does the mid-stack
peak land at the same layer? See ``docs/jkupdd_abc_vs_mtf.md``.

Usage:
  python3 scripts/sweeps/plot_jkupdd_abc_vs_mtf.py --csv LEADERBOARD.csv --out FIG.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_path):
    per = {"abc_map": {}, "mtf_map": {}}
    meanall = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["layer"] == "meanall":
                for c in ("abc_map", "mtf_map"):
                    meanall[c] = float(row[c]) if row[c] else None
                continue
            layer = int(row["layer"])
            for c in ("abc_map", "mtf_map"):
                if row[c]:
                    per[c][layer] = float(row[c])
    return per, meanall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per, meanall = load(args.csv)
    layers = sorted(set(per["abc_map"]) | set(per["mtf_map"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        ("abc_map", "ABC (score-native)", "#d62728", "o"),
        ("mtf_map", "MTF (lossy MIDI)", "#1f77b4", "s"),
    ]
    for col, label, color, marker in series:
        ys = [per[col].get(l) for l in layers]
        ax.plot(layers, ys, marker=marker, color=color, label=label, lw=1.9)
        if meanall.get(col) is not None:
            ax.axhline(meanall[col], color=color, ls="--", lw=1.0, alpha=0.5)
        # star the best layer for this arm
        valid = {l: per[col][l] for l in layers if l in per[col]}
        if valid:
            best = max(valid, key=valid.__getitem__)
            ax.scatter(
                [best],
                [valid[best]],
                s=240,
                marker="*",
                color=color,
                edgecolor="black",
                zorder=5,
            )

    ax.set_xlabel("CLaMP3-symbolic transformer layer")
    ax.set_ylabel("test/MAP (raw)")
    ax.set_title(
        "JKUPDD motif retrieval — score-native ABC vs lossy MTF\n"
        "(identical 66 occurrences / 15 groups, 4 composers, no folds; "
        "dashed = meanall; ★ = best layer)"
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
