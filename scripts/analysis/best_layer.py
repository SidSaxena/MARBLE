#!/usr/bin/env python3
"""scripts/analysis/best_layer.py
─────────────────────────
Query the WandB project for completed layer-sweep + meanall runs and
report layer / encoder analyses across them.

A "completed" run is one whose summary contains at least one `test/*`
key (the same signal `_layer_done` uses to mark sweeps as done).

Views
-----
    --view best          (default) Best layer per group (single-layer sweep groups)
    --view cross-encoder Best (layer, metric) per task, encoders as columns
    --view summary       Per-(encoder, task) summary stats (min/median/max/std/...)
    --view meanall-gap   Best-layer-MAP vs meanall-MAP gap (one row per task/encoder)
    --view consistency   Per encoder: which layers tend to win across tasks?

Examples
--------
    uv run python scripts/analysis/best_layer.py
    uv run python scripts/analysis/best_layer.py --view cross-encoder
    uv run python scripts/analysis/best_layer.py --view summary --filter VGMIDITVar
    uv run python scripts/analysis/best_layer.py --view meanall-gap
    uv run python scripts/analysis/best_layer.py --view consistency --encoder OMARRQ-multifeature-25hz-nonfsq
    uv run python scripts/analysis/best_layer.py --group "MERT-v1-95M / SHS100K"
    uv run python scripts/analysis/best_layer.py --view cross-encoder --csv /tmp/cross.csv
"""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict

# Headline metric priority — first match wins per task.
METRIC_PRIORITY: list[str] = [
    "test/weighted_score",  # GS, HookTheoryKey
    "test/MAP",  # Covers80, SHS100K, VGMIDITVar
    "test/map",  # alias (lower-case map shown in some runs)
    "test/MRR",  # retrieval fallback
    "test/beat_f1",  # GTZANBeatTracking
    "test/acc_rpa",  # HookTheoryMelody
    "test/macro_f1",  # HookTheoryStructure
    "test/f1",
    "test/auc",
    "test/acc",  # NSynth, GTZANGenre
]


# ──────────────────────────────────────────────────────────────────────────────
# Run-introspection helpers
# ──────────────────────────────────────────────────────────────────────────────


def _layer_from_run(run) -> int | None:
    """Extract layer index from a run name (`layer-N-fit` / `layer-N-test`)."""
    if run.name:
        m = re.search(r"layer-(\d+)", run.name)
        if m:
            return int(m.group(1))
    try:
        cfg = run.config or {}
        et = cfg["model"]["init_args"]["emb_transforms"]
        if isinstance(et, list) and et and "init_args" in et[0]:
            layers = et[0]["init_args"].get("layers")
            if isinstance(layers, list) and len(layers) == 1:
                return int(layers[0])
    except (KeyError, TypeError, IndexError):
        pass
    return None


def _is_meanall(run) -> bool:
    """Tag-driven meanall detection (Fix-#5 conventions).

    Meanall runs now live in the same group as the per-layer sweep,
    distinguished only by tags / run name. Detection signals:
      - tag `mean-all` / `mean-agg` (legacy) / `layer-meanall`
      - run name contains `meanall` (e.g. `layer-meanall-test`)
    """
    tags = set(run.tags or [])
    if {"mean-all", "mean-agg", "layer-meanall"} & tags:
        return True
    return bool(run.name and "meanall" in run.name)


def _pick_metric(summary_keys: list[str], override: str | None) -> str | None:
    if override:
        return override if override in summary_keys else None
    for m in METRIC_PRIORITY:
        if m in summary_keys:
            return m
    for k in summary_keys:
        if k.startswith("test/") and "loss" not in k.lower():
            return k
    return None


def _split_group(group: str) -> tuple[str, str] | None:
    """`<encoder> / <task>` → (encoder, task). None if no ' / '."""
    if " / " not in group:
        return None
    enc, task = group.split(" / ", 1)
    return enc, task


def _encoder_family(enc: str) -> str:
    """Strip variant + aggregation suffixes for cross-encoder grouping.

    After the 2026-05-14 taxonomy refactor meanall runs live in the
    per-layer group, so there's no `-meanall` encoder suffix to strip in
    fresh data. Kept as a safety net for any historical group that
    slipped past `fix_wandb_runs.py`.
    """
    base = enc
    for suffix in ("-meanall",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────


def collect_runs(
    api, project_path: str, per_page: int, metric_override: str | None, filter_substr: str | None
):
    """Walk all runs and bucket them by (group, layer).

    Returns
    -------
    by_group : {group: {layer_int: (metric, value, name, run_id)}}
        Per-layer best score for each sweep group.
    meanall_scores : {group: (metric, value, name)}
        Mean-of-all-layers score per group (now keyed by the SAME group
        as the per-layer entries, since meanall lives in the parent
        sweep group as of the 2026-05-14 taxonomy refactor). Legacy
        `<encoder>-meanall / <task>` groups are mapped back to their
        canonical parent here so downstream views see one consistent
        key per (encoder, task).
    """
    runs = list(api.runs(project_path, per_page=per_page))

    by_group: dict[str, dict[int, tuple[str, float, str, str]]] = defaultdict(dict)
    meanall_scores: dict[str, tuple[str, float, str]] = {}

    for r in runs:
        group = r.group
        if group is None:
            continue
        if filter_substr and filter_substr.lower() not in group.lower():
            continue

        keys = list(r.summary.keys())
        metric = _pick_metric(keys, metric_override)
        if metric is None:
            continue
        try:
            value = float(r.summary[metric])
        except (TypeError, ValueError):
            continue

        if _is_meanall(r):
            # Canonicalize: if any historical run still sits in a legacy
            # `<encoder>-meanall / <task>` group, fold it back into the
            # parent `<encoder> / <task>` key so downstream views show
            # the meanall alongside the per-layer sweep entries.
            sp = _split_group(group)
            if sp is not None:
                enc, task = sp
                canonical_group = f"{enc.removesuffix('-meanall')} / {task}"
            else:
                canonical_group = group
            prev = meanall_scores.get(canonical_group)
            if prev is None or value > prev[1]:
                meanall_scores[canonical_group] = (metric, value, r.name)
        else:
            layer = _layer_from_run(r)
            if layer is None:
                continue
            prev = by_group[group].get(layer)
            if prev is None or value > prev[1]:
                by_group[group][layer] = (metric, value, r.name, r.id)

    return by_group, meanall_scores


# ──────────────────────────────────────────────────────────────────────────────
# Views
# ──────────────────────────────────────────────────────────────────────────────


def view_best(by_group, meanall_scores=None, csv_path=None):
    """Best layer per group, with the mean-of-all-layers baseline shown
    alongside (a `★` marks whether meanall beats the best single layer)."""
    meanall_scores = meanall_scores or {}
    rows = []

    # Union of groups that have either per-layer or meanall data
    all_groups = sorted(set(by_group) | set(meanall_scores))
    for group in all_groups:
        layer_map = by_group.get(group, {})
        ma = meanall_scores.get(group)
        if not layer_map and ma is None:
            continue
        if layer_map:
            best_layer = max(layer_map, key=lambda l: layer_map[l][1])
            m, v, _, _ = layer_map[best_layer]
        else:
            best_layer, m, v = None, (ma[0] if ma else "?"), None
        ma_v = ma[1] if ma else None
        # Mark which choice is best end-to-end
        if ma_v is not None and v is not None:
            winner = "meanall" if ma_v > v else f"L{best_layer}"
        elif ma_v is not None:
            winner = "meanall"
        else:
            winner = f"L{best_layer}"
        rows.append(
            {
                "group": group,
                "best_layer": best_layer,
                "metric": m.removeprefix("test/") if m else "",
                "best_value": None if v is None else round(v, 4),
                "meanall_value": None if ma_v is None else round(ma_v, 4),
                "winner": winner,
            }
        )

    print(
        f"\n{'Group':<48} {'Layer':>5}  {'best':>8}  {'meanall':>8}  {'winner':<9}  {'metric':<18}"
    )
    print("-" * 110)
    for r in rows:
        bl = "—" if r["best_layer"] is None else str(r["best_layer"])
        bv = "—" if r["best_value"] is None else f"{r['best_value']:>8.4f}"
        mv = "—" if r["meanall_value"] is None else f"{r['meanall_value']:>8.4f}"
        marker = " ★" if r["winner"] == "meanall" else "  "
        print(
            f"{r['group']:<48} {bl:>5}  {bv:>8}  {mv:>8} {marker}{r['winner']:<7}"
            f"  {r['metric']:<18}"
        )
    print(f"\n  {len(rows)} group(s).  ★ = meanall beats the best single layer.")
    if csv_path:
        _write_csv(csv_path, rows)


def view_drill(group_name, by_group):
    if group_name not in by_group:
        print(f"Group not found: {group_name!r}")
        print(f"Available ({len(by_group)}):")
        for g in sorted(by_group):
            print(f"  {g}")
        sys.exit(1)
    layer_map = by_group[group_name]
    best_layer = max(layer_map, key=lambda l: layer_map[l][1])
    metric = layer_map[best_layer][0]
    print(f"\n{group_name}  ({len(layer_map)} layers, metric: {metric})")
    print("-" * 64)
    print(f"  {'layer':>5}  {'value':>10}  run")
    for layer in sorted(layer_map):
        m, v, name, rid = layer_map[layer]
        marker = " ← best" if layer == best_layer else ""
        print(f"  {layer:>5}  {v:>10.4f}  {name}{marker}")


def view_cross_encoder(by_group, csv_path=None):
    """Rows = tasks, columns = encoders, cells = '<best_L>  <value>'."""
    # encoder set + task set
    enc_set: set[str] = set()
    task_set: set[str] = set()
    cell: dict[tuple[str, str], tuple[int, float]] = {}  # (encoder, task) → (best_L, value)

    for group, layer_map in by_group.items():
        sp = _split_group(group)
        if sp is None:
            continue
        enc, task = sp
        enc_set.add(enc)
        task_set.add(task)
        best_layer = max(layer_map, key=lambda l: layer_map[l][1])
        v = layer_map[best_layer][1]
        cell[(enc, task)] = (best_layer, v)

    encs = sorted(enc_set)
    tasks = sorted(task_set)

    # Print table
    col_w = 14
    header = "Task".ljust(22) + "".join(e[: col_w - 1].ljust(col_w) for e in encs)
    print("\n" + header)
    print("-" * len(header))
    for task in tasks:
        row_vals = []
        # Find the best encoder for this row (to mark with *)
        row_cells = {e: cell.get((e, task)) for e in encs if (e, task) in cell}
        best_enc = max(row_cells, key=lambda e: row_cells[e][1]) if row_cells else None
        for enc in encs:
            c = cell.get((enc, task))
            if c is None:
                row_vals.append("—".ljust(col_w))
            else:
                bl, v = c
                marker = "*" if enc == best_enc else " "
                row_vals.append(f"{marker}L{bl}  {v:.3f}".ljust(col_w))
        print(f"{task:<22}" + "".join(row_vals))
    print(f"\n  {len(tasks)} task(s) × {len(encs)} encoder(s).  * = best encoder for the task.")

    if csv_path:
        rows = [
            {
                "task": task,
                **{
                    enc: f"L{cell[(enc, task)][0]}={cell[(enc, task)][1]:.4f}"
                    if (enc, task) in cell
                    else ""
                    for enc in encs
                },
            }
            for task in tasks
        ]
        _write_csv(csv_path, rows)


def view_summary(by_group, csv_path=None):
    """Per-group summary statistics — useful to see how flat or peaked
    each layer profile is."""
    print(
        f"\n{'Group':<55} {'min':>7} {'med':>7} {'max':>7} {'std':>7} {'gain':>7} {'L*':>4} top-3"
    )
    print("-" * 110)
    rows = []
    for group in sorted(by_group):
        layer_map = by_group[group]
        if len(layer_map) < 2:
            continue
        values = [v for (_, v, _, _) in layer_map.values()]
        vmin = min(values)
        vmax = max(values)
        vmed = statistics.median(values)
        vstd = statistics.pstdev(values) if len(values) > 1 else 0.0
        gain = vmax - vmin
        best_layer = max(layer_map, key=lambda l: layer_map[l][1])
        top3 = sorted(layer_map, key=lambda l: -layer_map[l][1])[:3]
        rows.append(
            {
                "group": group,
                "min": vmin,
                "median": vmed,
                "max": vmax,
                "std": vstd,
                "gain": gain,
                "best_layer": best_layer,
                "top3_layers": top3,
            }
        )
        print(
            f"{group:<55} {vmin:>7.4f} {vmed:>7.4f} {vmax:>7.4f} "
            f"{vstd:>7.4f} {gain:>7.4f} {best_layer:>4} {top3}"
        )
    if csv_path:
        _write_csv(csv_path, rows)


def view_meanall_gap(by_group, meanall_scores, csv_path=None):
    """For each (encoder, task), compare best-layer-sweep score vs the
    meanall score. After the 2026-05-14 taxonomy refactor the meanall
    score is keyed by the SAME group as the per-layer sweep, so this is
    a direct lookup."""
    print(
        f"\n{'Encoder':<42} {'Task':<22} {'L*':>4} {'sweep':>8} {'meanall':>9} {'gain':>7} verdict"
    )
    print("-" * 110)
    rows = []
    for group, layer_map in sorted(by_group.items()):
        sp = _split_group(group)
        if sp is None:
            continue
        enc, task = sp
        best_layer = max(layer_map, key=lambda l: layer_map[l][1])
        _, sweep_v, _, _ = layer_map[best_layer]
        ma = meanall_scores.get(group)
        if ma is None:
            mv = None
            rel = None
            verdict = "no meanall run"
        else:
            mv = ma[1]
            rel = (sweep_v - mv) / max(mv, 1e-12)
            if abs(rel) < 0.02:
                verdict = "meanall ≈ best"
            elif rel < 0.10:
                verdict = "modest sweep gain"
            else:
                verdict = "sweep matters"
        rows.append(
            {
                "encoder": enc,
                "task": task,
                "best_layer": best_layer,
                "sweep": sweep_v,
                "meanall": mv,
                "gain_relative": rel,
                "verdict": verdict,
            }
        )
        ma_str = f"{mv:>9.4f}" if mv is not None else f"{'—':>9}"
        rel_str = f"{rel:+.1%}" if rel is not None else "—"
        print(
            f"{enc:<42} {task:<22} {best_layer:>4} {sweep_v:>8.4f} {ma_str} {rel_str:>7} {verdict}"
        )
    if csv_path:
        _write_csv(csv_path, rows)


def view_consistency(by_group, encoder_filter: str | None, csv_path=None):
    """Per-encoder: which layer wins across tasks?"""
    # Group rows by encoder
    by_enc: dict[str, list[tuple[str, int, list[int]]]] = defaultdict(list)
    for group, layer_map in by_group.items():
        sp = _split_group(group)
        if sp is None:
            continue
        enc, task = sp
        if encoder_filter and encoder_filter not in enc:
            continue
        best_layer = max(layer_map, key=lambda l: layer_map[l][1])
        top3 = sorted(layer_map, key=lambda l: -layer_map[l][1])[:3]
        by_enc[enc].append((task, best_layer, top3))

    rows = []
    for enc in sorted(by_enc):
        records = by_enc[enc]
        print(f"\n{enc}  ({len(records)} task(s))")
        print(f"  {'Task':<25} {'best_L':>6}   top-3 layers")
        for task, best_L, top3 in sorted(records):
            print(f"  {task:<25} {best_L:>6}   {top3}")
            rows.append({"encoder": enc, "task": task, "best_layer": best_L, "top3": top3})
        # Quick summary: median + spread of best layers
        bests = [bl for _, bl, _ in records]
        if len(bests) > 1:
            print(
                f"  → best-layer median: {statistics.median(bests):.0f}  "
                f"spread {min(bests)}..{max(bests)}"
            )
    if csv_path:
        _write_csv(csv_path, rows)


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        print(f"  (no rows to write to {path})")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--project", default="marble")
    ap.add_argument("--entity", default=None)
    ap.add_argument(
        "--view",
        choices=[
            "best",
            "cross-encoder",
            "summary",
            "meanall-gap",
            "consistency",
        ],
        default="best",
        help="Analysis view (default: best)",
    )
    ap.add_argument("--group", default=None, help="(view=best) Drill into a single sweep group.")
    ap.add_argument(
        "--filter", default=None, help="Substring filter on group names (case-insensitive)."
    )
    ap.add_argument(
        "--encoder", default=None, help="(view=consistency) Filter to a specific encoder."
    )
    ap.add_argument("--metric", default=None, help="Override the headline metric (e.g. test/MAP).")
    ap.add_argument("--per-page", type=int, default=500)
    ap.add_argument("--csv", default=None, help="Also write the view as CSV to this path.")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb not installed.", file=sys.stderr)
        sys.exit(2)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    by_group, meanall_scores = collect_runs(
        api,
        project_path,
        args.per_page,
        args.metric,
        args.filter,
    )

    if not by_group and not meanall_scores:
        print("No completed sweep / meanall runs found (with test/* metrics).")
        sys.exit(0)

    if args.group:
        view_drill(args.group, by_group)
        return

    if args.view == "best":
        view_best(by_group, meanall_scores, csv_path=args.csv)
    elif args.view == "cross-encoder":
        view_cross_encoder(by_group, csv_path=args.csv)
    elif args.view == "summary":
        view_summary(by_group, csv_path=args.csv)
    elif args.view == "meanall-gap":
        view_meanall_gap(by_group, meanall_scores, csv_path=args.csv)
    elif args.view == "consistency":
        view_consistency(by_group, args.encoder, csv_path=args.csv)


if __name__ == "__main__":
    main()
