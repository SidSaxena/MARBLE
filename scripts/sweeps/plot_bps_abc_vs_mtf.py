"""Per-layer figure for the BPS-Motif ABC-vs-MTF A/B (both tasks).

Reads the leaderboard CSV written by ``bps_abc_vs_mtf_summary.py`` (columns
``task, layer, abc, mtf, delta_abc_minus_mtf``; a ``meanall`` row per task) and
draws one panel per task — MNID (test/auc_roc) and Retrieval (test/map) — each
overlaying the ABC and MTF arms vs layer, with each arm's meanall as a dashed
reference and each arm's best layer starred. The question: does notation-
preserving ABC match/beat the lossy MIDI→MTF path on BPS-Motif, and where does
the depth peak land? See ``docs/bps_motif_abc_vs_mtf.md``.

Usage:
  python3 scripts/sweeps/plot_bps_abc_vs_mtf.py --csv LEADERBOARD.csv --out FIG.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TASK_METRIC = {"MNID": "test/auc_roc", "Retrieval": "test/MAP"}


def load(csv_path):
    per = {}  # task -> {"abc": {layer: v}, "mtf": {...}}
    meanall = {}  # task -> {"abc": v, "mtf": v}

    def _f(s):
        s = (s or "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            t = row["task"]
            per.setdefault(t, {"abc": {}, "mtf": {}})
            meanall.setdefault(t, {})
            if row["layer"] == "meanall":
                meanall[t]["abc"] = _f(row["abc"])
                meanall[t]["mtf"] = _f(row["mtf"])
                continue
            layer = int(row["layer"])
            a, m = _f(row["abc"]), _f(row["mtf"])
            if a is not None:
                per[t]["abc"][layer] = a
            if m is not None:
                per[t]["mtf"][layer] = m
    return per, meanall


def _panel(ax, task, per_t, mean_t):
    layers = sorted(set(per_t["abc"]) | set(per_t["mtf"]))
    series = [
        ("abc", "ABC (score-native)", "#d62728", "o"),
        ("mtf", "MTF (lossy MIDI)", "#1f77b4", "s"),
    ]
    for key, label, color, marker in series:
        ys = [per_t[key].get(l) for l in layers]
        ax.plot(layers, ys, marker=marker, color=color, label=label, lw=1.9)
        if mean_t.get(key) is not None:
            ax.axhline(mean_t[key], color=color, ls="--", lw=1.0, alpha=0.5)
        valid = {l: per_t[key][l] for l in layers if l in per_t[key]}
        if valid:
            best = max(valid, key=valid.__getitem__)
            ax.scatter(
                [best], [valid[best]], s=240, marker="*", color=color, edgecolor="black", zorder=5
            )
    ax.set_xlabel("CLaMP3-symbolic transformer layer")
    ax.set_ylabel(TASK_METRIC.get(task, task))
    ax.set_title(f"BPSMotif {task}")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per, meanall = load(args.csv)
    tasks = [t for t in ("MNID", "Retrieval") if t in per]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks, strict=False):
        _panel(ax, task, per[task], meanall.get(task, {}))
    fig.suptitle(
        "BPS-Motif — score-native ABC vs lossy MTF (CLaMP3-symbolic, 5-fold CV; "
        "identical windows; dashed = meanall; ★ = best layer)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150)
    print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
