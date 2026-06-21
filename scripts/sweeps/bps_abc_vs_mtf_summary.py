"""Aggregate the BPS-Motif **ABC-vs-MTF** A/B layer sweeps, side by side.

Compares two CLaMP3-symbolic layer sweeps run on the *identical* windows
(same 5-fold movement-level CV, same labels / work_id) but different symbolic
encodings:

  * **MTF** — model tag ``CLaMP3-symbolic``: each window tokenised from the
    lossy ``csv_notes → 60-QPM MIDI → MTF`` slice (the existing pipeline).
  * **ABC** — model tag ``CLaMP3-symbolic-abc``: the SAME notes reconstructed
    directly from ``csv_notes`` as a single-voice interleaved-ABC string
    (Option B, ``scripts/data/build_bps_motif_abc.py``) — notation-preserving.

Both tasks:

  * **MNID** (supervised, 5-fold): primary metric ``test/auc_roc`` (+ ap / f1 /
    acc / precision / recall). Best layer by cross-fold mean. meanall baseline.
  * **Retrieval** (zero-shot, 5-fold): primary metric ``test/map`` (+ recall@K).

Reads every ``wandb-summary.json`` under (layer-primary per-fold dirs)::

    output/probe.BPSMotif{MNID,Retrieval}.CLaMP3-symbolic(-abc)-layers.layer{L}.fold{F}/...
    output/probe.BPSMotif{MNID,Retrieval}.CLaMP3-symbolic(-abc)-meanall.fold{F}/...

Per layer (mean across folds) prints ABC | MTF | Δ tables, flags each arm's best
layer + the depth peak, and writes a leaderboard CSV + a JSON. Pure stdlib.

Usage::

  python3 scripts/sweeps/bps_abc_vs_mtf_summary.py \
      [--base DIR] [--task MNID|Retrieval] [--out-csv PATH] [--out-md PATH]
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics

# Layer-primary per-fold dirs: ...-layers.layer6.fold3
_LAYER_FOLD_RX = re.compile(r"-layers\.layer(\d+)\.fold(\d+)")
_MEANALL_FOLD_RX = re.compile(r"-meanall\.fold(\d+)")

MNID_METRICS = ["acc", "auc_pr", "auc_roc", "f1", "precision", "recall"]
RETR_METRICS = ["map", "map_centered", "recall@1", "recall@5", "recall@10", "mrr"]
RETR_RECALL_KS = (1, 5, 10, 50)


def _latest_test_summary(run_glob: str) -> dict | None:
    picked = None
    for p in sorted(glob.glob(run_glob)):  # ascending timestamp → last wins
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if any(k.startswith("test/") for k in d):
            picked = d
    return picked


def load_arm(base: str, task: str, model_tag: str) -> tuple[dict, dict]:
    """Return ({(fold,layer): summary}, {fold: meanall_summary}) for one arm."""
    cells: dict[tuple[int, int], dict] = {}
    root = os.path.join(base, "output", f"probe.BPSMotif{task}.{model_tag}-layers.*")
    for d in sorted(glob.glob(root)):
        m = _LAYER_FOLD_RX.search(d)
        if not m:
            continue
        layer, fold = int(m.group(1)), int(m.group(2))
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            cells[(fold, layer)] = summ
    meanall: dict[int, dict] = {}
    mroot = os.path.join(base, "output", f"probe.BPSMotif{task}.{model_tag}-meanall.fold*")
    for d in sorted(glob.glob(mroot)):
        m = _MEANALL_FOLD_RX.search(d)
        if not m:
            continue
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            meanall[int(m.group(1))] = summ
    return cells, meanall


def load_arm_wandb(task: str, model_tag: str, entity: str, project: str) -> tuple[dict, dict]:
    """Same as :func:`load_arm` but pulls from W&B instead of local ``output/``.

    Used when an arm's local sweep dirs were cleaned up but the runs persist on
    W&B (e.g. the MTF Retrieval baseline — group ``CLaMP3-symbolic /
    BPSMotifRetrieval``). Reads the per-(fold,layer) test runs + the meanall
    folds from the group ``{model_tag} / BPSMotif{task}`` via the
    ``sweep/{layer,fold,repr,stage}`` coords stamped by LogSweepCoordsCallback.
    Returns the same ``({(fold,layer): summary}, {fold: meanall_summary})``.
    """
    import wandb

    api = wandb.Api()
    group = f"{model_tag} / BPSMotif{task}"
    runs = api.runs(f"{entity}/{project}", filters={"group": group})
    cells: dict[tuple[int, int], dict] = {}
    meanall: dict[int, dict] = {}
    for r in runs:
        cfg = r.config
        if cfg.get("sweep/stage") != "test":
            continue
        summ = {k: v for k, v in r.summary.items() if isinstance(k, str) and k.startswith("test/")}
        if not summ:
            continue
        fold = cfg.get("sweep/fold")
        layer = cfg.get("sweep/layer")
        if cfg.get("sweep/repr") == "meanall" or layer == -1:
            if fold is not None:
                meanall[int(fold)] = summ
        elif fold is not None and layer is not None:
            cells[(int(fold), int(layer))] = summ
    return cells, meanall


def _mean(cells: dict, folds: list[int], layer: int, key: str):
    vs = [
        cells[(f, layer)][f"test/{key}"]
        for f in folds
        if (f, layer) in cells and f"test/{key}" in cells[(f, layer)]
    ]
    return statistics.mean(vs) if vs else None


def _mean_meanall(meanall: dict, folds: list[int], key: str):
    vs = [meanall[f][f"test/{key}"] for f in folds if f in meanall and f"test/{key}" in meanall[f]]
    return statistics.mean(vs) if vs else None


def _f(x, p=4):
    return f"{x:.{p}f}" if x is not None else "  --  "


def _delta(a, b):
    return (a - b) if (a is not None and b is not None) else None


def _best_layer(cells: dict, folds: list[int], layers: list[int], key: str):
    scored = {l: _mean(cells, folds, l, key) for l in layers}
    scored = {l: v for l, v in scored.items() if v is not None}
    return max(scored, key=scored.__getitem__) if scored else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/developer/python/marble"))
    ap.add_argument("--task", default="both", choices=["MNID", "Retrieval", "both"])
    ap.add_argument("--out-json", default="/tmp/bps_abc_vs_mtf.json")
    ap.add_argument("--out-md", default="/tmp/bps_abc_vs_mtf.md")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument(
        "--wandb-fallback",
        action="store_true",
        help="If an arm has no local output dirs, pull it from W&B by group "
        "(needed for the MTF Retrieval baseline, whose local dirs were cleaned "
        "up but whose runs persist in `CLaMP3-symbolic / BPSMotifRetrieval`).",
    )
    ap.add_argument("--wandb-entity", default="sidsaxena-universitat-pompeu-fabra")
    ap.add_argument("--wandb-project", default="marble")
    args = ap.parse_args()

    tasks = ["MNID", "Retrieval"] if args.task == "both" else [args.task]
    out: list[str] = []
    payload: dict = {}
    csv_rows: list[list] = []

    for task in tasks:
        primary = "auc_roc" if task == "MNID" else "map"
        metrics = MNID_METRICS if task == "MNID" else RETR_METRICS
        abc, abc_mean = load_arm(args.base, task, "CLaMP3-symbolic-abc")
        mtf, mtf_mean = load_arm(args.base, task, "CLaMP3-symbolic")
        # MTF baseline may live only on W&B (local dirs cleaned up). Pull it.
        if not mtf and args.wandb_fallback:
            mtf, mtf_mean = load_arm_wandb(
                task, "CLaMP3-symbolic", args.wandb_entity, args.wandb_project
            )
        if not abc and args.wandb_fallback:
            abc, abc_mean = load_arm_wandb(
                task, "CLaMP3-symbolic-abc", args.wandb_entity, args.wandb_project
            )
        if not abc or not mtf:
            out.append(
                f"## BPSMotif{task}: missing runs (abc cells={len(abc)} mtf cells={len(mtf)})"
            )
            out.append("")
            continue

        folds = sorted({f for f, _ in abc} | {f for f, _ in mtf})
        layers = sorted({l for _, l in abc} | {l for _, l in mtf})
        abc_best = _best_layer(abc, folds, layers, primary)
        mtf_best = _best_layer(mtf, folds, layers, primary)

        out.append(f"# BPS-Motif {task} — ABC vs MTF (CLaMP3-symbolic, {len(folds)}-fold CV)")
        out.append("")
        out.append(
            f"Identical windows (same notes, same labels). Primary metric "
            f"**test/{primary}**, cross-fold mean over folds {folds}. "
            f"Δ = ABC − MTF (positive ⇒ notation-preserving ABC helps)."
        )
        out.append("")
        out.append(f"## Per-layer mean test/{primary}")
        out.append("")
        out.append("| layer | ABC | MTF | Δ (ABC−MTF) |")
        out.append("|------:|----:|----:|------------:|")
        for l in layers:
            a, m = _mean(abc, folds, l, primary), _mean(mtf, folds, l, primary)
            flags = ""
            if l == abc_best:
                flags += " ⭐ABC"
            if l == mtf_best:
                flags += " ⭐MTF"
            out.append(f"| {l}{flags} | {_f(a)} | {_f(m)} | {_f(_delta(a, m))} |")
            csv_rows.append([task, l, _f(a), _f(m), _f(_delta(a, m))])
        am = _mean_meanall(abc_mean, folds, primary)
        mm = _mean_meanall(mtf_mean, folds, primary)
        out.append(f"| **meanall** | {_f(am)} | {_f(mm)} | {_f(_delta(am, mm))} |")
        csv_rows.append([task, "meanall", _f(am), _f(mm), _f(_delta(am, mm))])
        out.append("")

        a_peak = _mean(abc, folds, abc_best, primary) if abc_best is not None else None
        m_peak = _mean(mtf, folds, mtf_best, primary) if mtf_best is not None else None
        out.append("## Verdict")
        out.append("")
        out.append(f"- **ABC best layer = {abc_best}** ({primary} {_f(a_peak)})")
        out.append(f"- **MTF best layer = {mtf_best}** ({primary} {_f(m_peak)})")
        out.append(f"- peak Δ (ABC_best − MTF_best) = {_f(_delta(a_peak, m_peak))}")
        if abc_best is not None:
            out.append(
                f"- same-layer Δ at ABC's peak (L{abc_best}): "
                f"{_f(_delta(_mean(abc, folds, abc_best, primary), _mean(mtf, folds, abc_best, primary)))}"
            )
        out.append("")

        # full per-metric per-layer mean for both arms (best layer of each)
        out.append("## All metrics at each arm's best layer (cross-fold mean)")
        out.append("")
        out.append("| metric | ABC@L" + str(abc_best) + " | MTF@L" + str(mtf_best) + " |")
        out.append("|---|---:|---:|")
        for mk in metrics:
            out.append(
                f"| {mk} | {_f(_mean(abc, folds, abc_best, mk))} | "
                f"{_f(_mean(mtf, folds, mtf_best, mk))} |"
            )
        if task == "Retrieval":
            for k in RETR_RECALL_KS:
                out.append(
                    f"| recall@{k} | {_f(_mean(abc, folds, abc_best, f'recall@{k}'))} | "
                    f"{_f(_mean(mtf, folds, mtf_best, f'recall@{k}'))} |"
                )
        out.append("")

        expected = len(folds) * len(layers)
        out.append("## Coverage")
        out.append("")
        out.append(
            f"- ABC layer cells: {len(abc)}/{expected}; MTF layer cells: {len(mtf)}/{expected}"
        )
        out.append(
            f"- ABC meanall folds: {len(abc_mean)}/{len(folds)}; MTF meanall folds: {len(mtf_mean)}/{len(folds)}"
        )
        out.append("")

        payload[task] = {
            "primary": primary,
            "folds": folds,
            "layers": layers,
            "abc_best_layer": abc_best,
            "mtf_best_layer": mtf_best,
            "abc_per_layer": {
                str(l): {mk: _mean(abc, folds, l, mk) for mk in metrics} for l in layers
            },
            "mtf_per_layer": {
                str(l): {mk: _mean(mtf, folds, l, mk) for mk in metrics} for l in layers
            },
            "abc_meanall": {mk: _mean_meanall(abc_mean, folds, mk) for mk in metrics},
            "mtf_meanall": {mk: _mean_meanall(mtf_mean, folds, mk) for mk in metrics},
            "abc_layer_cells": len(abc),
            "mtf_layer_cells": len(mtf),
        }

    md = "\n".join(out)
    print(md)
    with open(args.out_md, "w") as fh:
        fh.write(md)
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[wrote {args.out_json} and {args.out_md}]")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["task", "layer", "abc", "mtf", "delta_abc_minus_mtf"])
            for r in csv_rows:
                w.writerow(r)
        print(f"[wrote {args.out_csv}]")


if __name__ == "__main__":
    main()
