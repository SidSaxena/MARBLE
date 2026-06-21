"""Per-layer MAP figure for the MTC-ANN ABC-vs-MTF A/B (both tasks).

Reads the leaderboard CSV written by ``mtc_ann_abc_vs_mtf_summary.py`` (rows
``task,layer,abc_map,mtf_map,delta_abc_minus_mtf``; one ``meanall`` row per task)
and overlays the two arms' raw ``test/map`` vs layer for each task in its own
subplot, with each arm's meanall as a horizontal reference and each arm's best
layer starred. The question the figure answers: does notation-preserving ABC
beat lossy MTF on folk-melody retrieval, and does the peak land at the same
depth as the JKUPDD/BPS sweeps?

Usage:
  python3 scripts/sweeps/plot_mtc_ann_abc_vs_mtf.py --csv LEADERBOARD.csv --out FIG.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _f(x):
    return float(x) if x not in (None, "", "  --  ") else None


def load(csv_path):
    """Return ``{task: ({'abc_map': {layer: v}, 'mtf_map': {...}}, meanall)}``."""
    tasks: dict[str, tuple[dict, dict]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            task = row["task"]
            per, meanall = tasks.setdefault(task, ({"abc_map": {}, "mtf_map": {}}, {}))
            if row["layer"] == "meanall":
                for c in ("abc_map", "mtf_map"):
                    meanall[c] = _f(row[c])
                continue
            layer = int(row["layer"])
            for c in ("abc_map", "mtf_map"):
                v = _f(row[c])
                if v is not None:
                    per[c][layer] = v
    return tasks


def _plot_one(ax, per, meanall, title):
    layers = sorted(set(per["abc_map"]) | set(per["mtf_map"]))
    series = [
        ("abc_map", "ABC (score-native)", "#d62728", "o"),
        ("mtf_map", "MTF (lossy MIDI)", "#1f77b4", "s"),
    ]
    for col, label, color, marker in series:
        ys = [per[col].get(l) for l in layers]
        ax.plot(layers, ys, marker=marker, color=color, label=label, lw=1.9)
        if meanall.get(col) is not None:
            ax.axhline(meanall[col], color=color, ls="--", lw=1.0, alpha=0.5)
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
    ax.set_title(title)
    if layers:
        ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = load(args.csv)
    names = [t for t in ("TuneFamily", "Motif") if t in tasks] or list(tasks)
    n = len(names) or 1
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), squeeze=False)
    for ax, name in zip(axes[0], names, strict=False):
        per, meanall = tasks[name]
        _plot_one(
            ax,
            per,
            meanall,
            f"MTC-ANN {name} — score-native ABC vs lossy MTF\n"
            "(identical pool, no folds; dashed = meanall; ★ = best layer)",
        )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
