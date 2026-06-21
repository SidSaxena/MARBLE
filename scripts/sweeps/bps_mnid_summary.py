"""Aggregate the BPS-Motif MNID layer sweep (+ meanall baseline) across folds.

MNID = supervised motif-window vs non-motif-window classification. The sweep
trains one MLP probe per (fold, layer) on a frozen CLaMP3 layer, plus a
``meanall`` baseline per fold that probes the mean over all 13 layers.

This reads every ``wandb-summary.json`` under
  output/probe.BPSMotifMNID.CLaMP3-symbolic-layers.fold{F}.layer{L}/...   (per-layer)
  output/probe.BPSMotifMNID.CLaMP3-symbolic-meanall.fold{F}/...           (meanall)
and reports, for each test metric (acc, auc_pr, auc_roc, f1, precision,
recall):
  1. per-layer cross-fold mean +/- std, with the best layer flagged
  2. the per-fold breakdown for the primary metric (layer x fold)
  3. best-layer vs meanall delta
  4. coverage / duplicate sanity check

A 5-fold movement-level CV means "the score" for a layer is the mean over
folds; std across folds is the spread we report as the error bar. When a
(fold, layer) dir holds multiple completed test runs (e.g. a killed run + a
re-run), the *latest* by run-dir timestamp wins so all folds reflect the same
code/conditions; disagreement among duplicates is surfaced as a warning.

Writes a JSON (per-fold raw values + aggregates) for downstream plotting and
prints a markdown report. Pure stdlib so it runs anywhere python3 does.

Usage:
  python3 scripts/sweeps/bps_mnid_summary.py [--base DIR] [--metric auc_roc]
      [--out-json PATH] [--out-md PATH]
"""

import argparse
import glob
import json
import os
import re
import statistics

METRICS = ["acc", "auc_pr", "auc_roc", "f1", "precision", "recall"]
MEANALL_RX = re.compile(r"-meanall\.fold(\d+)")
_LAYER_PRIMARY_RX = re.compile(r"\.layer(\d+)\.fold(\d+)")
_FOLD_PRIMARY_RX = re.compile(r"\.fold(\d+)\.layer(\d+)")


def parse_fold_layer(name: str) -> tuple[int, int] | None:
    """Return (fold, layer) from a sweep dir name, accepting both orderings:
    layer-primary ``...-layers.layer6.fold3`` (current) and fold-primary
    ``...-layers.fold3.layer6`` (legacy). None if neither matches."""
    m = _LAYER_PRIMARY_RX.search(name)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = _FOLD_PRIMARY_RX.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _latest_test_summary(run_glob: str) -> dict | None:
    """Return the test-metric summary from the latest run dir matching
    ``run_glob`` (ascending sort -> last wins), or None. Also returns the
    list of all completed test summaries so callers can sanity-check dupes."""
    picked = None
    for p in sorted(glob.glob(run_glob)):  # ascending timestamp -> last wins
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if any(k.startswith("test/") for k in d):
            picked = d
    return picked


def _all_test_summaries(run_glob: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(run_glob)):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if any(k.startswith("test/") for k in d):
            out.append(d)
    return out


def load_layers(base: str) -> tuple[dict, dict]:
    """Return ({(fold,layer): summary}, {(fold,layer): n_completed_runs})."""
    cells: dict[tuple[int, int], dict] = {}
    dupes: dict[tuple[int, int], int] = {}
    root = os.path.join(base, "output", "probe.BPSMotifMNID.CLaMP3-symbolic-layers.*")
    for d in sorted(glob.glob(root)):
        key = parse_fold_layer(d)
        if key is None:
            continue
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            cells[key] = summ
            dupes[key] = len(_all_test_summaries(g))
    return cells, dupes


def load_meanall(base: str) -> dict:
    cells: dict[int, dict] = {}
    root = os.path.join(base, "output", "probe.BPSMotifMNID.CLaMP3-symbolic-meanall.fold*")
    for d in sorted(glob.glob(root)):
        m = MEANALL_RX.search(d)
        if not m:
            continue
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            cells[int(m.group(1))] = summ
    return cells


def _agg(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0, "vals": []}
    return {
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "vals": vals,
    }


def _f(x, w=6, p=4):
    return f"{x:.{p}f}".rjust(w) if x is not None else "  --  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/developer/marble"))
    ap.add_argument("--metric", default="auc_roc", choices=METRICS)
    ap.add_argument("--out-json", default="/tmp/bps_mnid_summary.json")
    ap.add_argument("--out-md", default="/tmp/bps_mnid_summary.md")
    args = ap.parse_args()

    layers, dupes = load_layers(args.base)
    meanall = load_meanall(args.base)
    if not layers:
        print("no layer runs found under", args.base)
        return

    folds = sorted({f for f, _ in layers})
    layer_ids = sorted({l for _, l in layers})

    def cell(f, l, metric):
        d = layers.get((f, l))
        return d.get(f"test/{metric}") if d else None

    def ma_cell(f, metric):
        d = meanall.get(f)
        return d.get(f"test/{metric}") if d else None

    # ---- aggregates ----
    layer_agg = {l: {m: _agg([cell(f, l, m) for f in folds]) for m in METRICS} for l in layer_ids}
    meanall_agg = {m: _agg([ma_cell(f, m) for f in folds]) for m in METRICS}

    pm = args.metric
    ranked = sorted(layer_ids, key=lambda l: (layer_agg[l][pm]["mean"] or -1), reverse=True)
    best = ranked[0]

    out = []
    out.append(f"# BPS-Motif MNID — CLaMP3-symbolic layer sweep ({len(folds)}-fold CV)")
    out.append("")
    out.append(f"Primary metric: **test/{pm}** (cross-fold mean ± std over folds {folds}).")
    out.append("")

    # 1. per-layer mean ± std, all metrics
    out.append("## Per-layer cross-fold mean ± std")
    out.append("")
    hdr = "| layer | " + " | ".join(METRICS) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (len(METRICS) + 1))
    for l in layer_ids:
        cells_txt = []
        for m in METRICS:
            a = layer_agg[l][m]
            cells_txt.append(
                f"{_f(a['mean'])}±{_f(a['std'], 5, 3).strip()}" if a["mean"] is not None else "--"
            )
        flag = " ⭐" if l == best else ""
        out.append(f"| {l}{flag} | " + " | ".join(cells_txt) + " |")
    # meanall row
    if any(meanall_agg[m]["n"] for m in METRICS):
        cells_txt = []
        for m in METRICS:
            a = meanall_agg[m]
            cells_txt.append(
                f"{_f(a['mean'])}±{_f(a['std'], 5, 3).strip()}" if a["mean"] is not None else "--"
            )
        out.append("| **meanall** | " + " | ".join(cells_txt) + " |")
    out.append("")

    # 2. per-fold breakdown for primary metric
    out.append(f"## test/{pm} by layer × fold")
    out.append("")
    out.append("| layer | " + " | ".join(f"fold{f}" for f in folds) + " | mean | std |")
    out.append("|" + "---|" * (len(folds) + 3))
    for l in layer_ids:
        a = layer_agg[l][pm]
        label = f"{l} ⭐" if l == best else f"{l}"
        row = [label] + [_f(cell(f, l, pm)) for f in folds]
        row += [_f(a["mean"]), _f(a["std"], 6, 4)]
        out.append("| " + " | ".join(row) + " |")
    if meanall:
        a = meanall_agg[pm]
        row = (
            ["meanall"] + [_f(ma_cell(f, pm)) for f in folds] + [_f(a["mean"]), _f(a["std"], 6, 4)]
        )
        out.append("| " + " | ".join(row) + " |")
    out.append("")

    # 3. best vs meanall
    out.append("## Best layer vs meanall")
    out.append("")
    out.append(f"| metric | best layer ({best}) | meanall | Δ (best−meanall) |")
    out.append("|---|---|---|---|")
    for m in METRICS:
        bl = layer_agg[best][m]["mean"]
        ma = meanall_agg[m]["mean"]
        delta = (bl - ma) if (bl is not None and ma is not None) else None
        out.append(f"| {m} | {_f(bl)} | {_f(ma)} | {_f(delta)} |")
    out.append("")

    # 4. coverage + dupes
    expected = len(folds) * len(layer_ids)
    got = len(layers)
    dup_cells = {k: n for k, n in dupes.items() if n > 1}
    out.append("## Coverage")
    out.append("")
    out.append(f"- layer cells: {got}/{expected} ({len(folds)} folds × {len(layer_ids)} layers)")
    out.append(f"- meanall cells: {len(meanall)}/{len(folds)} folds")
    if dup_cells:
        out.append(f"- ⚠ cells with >1 completed test run (kept latest): {sorted(dup_cells)}")
    else:
        out.append("- no duplicate test runs (one per cell)")
    out.append("")

    md = "\n".join(out)
    print(md)
    with open(args.out_md, "w") as fh:
        fh.write(md)

    payload = {
        "metric_primary": pm,
        "metrics": METRICS,
        "folds": folds,
        "layers": layer_ids,
        "best_layer": best,
        "ranking": ranked,
        "layer_agg": {str(l): {m: layer_agg[l][m] for m in METRICS} for l in layer_ids},
        "meanall_agg": meanall_agg,
        "per_cell": {
            f"{f}.{l}": {m: cell(f, l, m) for m in METRICS} for f in folds for l in layer_ids
        },
        "meanall_per_cell": {str(f): {m: ma_cell(f, m) for m in METRICS} for f in folds},
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[wrote {args.out_json} and {args.out_md}]")


if __name__ == "__main__":
    main()
