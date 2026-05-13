#!/usr/bin/env python3
"""scripts/analysis/best_layer.py
─────────────────────────
Query the WandB project for completed layer-sweep runs and report the
best layer per sweep (group).

A "completed" run is one whose summary contains at least one `test/*`
key (the same signal `_layer_done` uses to mark sweeps as done).

Examples
--------
    # Best layer for every sweep group in the project
    uv run python scripts/analysis/best_layer.py

    # Drill into one sweep — print every layer's metric
    uv run python scripts/analysis/best_layer.py --group "MERT-v1-95M / SHS100K"

    # Filter groups by substring (case-insensitive)
    uv run python scripts/analysis/best_layer.py --filter NSynth

    # Override the metric pick (default: auto-detect)
    uv run python scripts/analysis/best_layer.py --metric test/MAP
"""

import argparse
import re
import sys
from collections import defaultdict


# Headline metric priority — first match wins per task. Add new tasks here
# as they get wired up.
METRIC_PRIORITY: list[str] = [
    "test/weighted_score",   # GS, HookTheoryKey
    "test/MAP",              # Covers80, SHS100K, VGMIDITVar
    "test/MRR",              # retrieval fallback
    "test/beat_f1",          # GTZANBeatTracking
    "test/acc_rpa",          # HookTheoryMelody
    "test/macro_f1",         # HookTheoryStructure
    "test/f1",
    "test/auc",
    "test/acc",              # NSynth, GTZANGenre
]


def _layer_from_run(run) -> int | None:
    """Extract layer index from a run. Tries name first ("layer-N-*"),
    then the config (`model.init_args.emb_transforms[0].init_args.layers`)."""
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


def _pick_metric(summary_keys: list[str], override: str | None) -> str | None:
    """Pick the headline test metric for a run."""
    if override:
        return override if override in summary_keys else None
    for m in METRIC_PRIORITY:
        if m in summary_keys:
            return m
    # Last resort: any test/* key, but skip *_loss
    for k in summary_keys:
        if k.startswith("test/") and "loss" not in k.lower():
            return k
    return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--project", default="marble",
                    help="WandB project name (default: marble)")
    ap.add_argument("--entity", default=None,
                    help="WandB entity / username (default: auth's default)")
    ap.add_argument("--group", default=None,
                    help="Drill into a single sweep group — print every "
                         "layer's metric, sorted by layer index.")
    ap.add_argument("--filter", default=None,
                    help="Substring filter on group names (case-insensitive).")
    ap.add_argument("--metric", default=None,
                    help="Override the headline metric (e.g. test/MAP). "
                         "Default: auto-pick by METRIC_PRIORITY.")
    ap.add_argument("--per-page", type=int, default=500,
                    help="WandB API page size (default: 500).")
    args = ap.parse_args()

    try:
        import wandb
    except ImportError:
        print("wandb not installed in this environment — `uv add wandb` "
              "or run inside a venv that has it.", file=sys.stderr)
        sys.exit(2)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(project_path, per_page=args.per_page))

    # ── Collect: group → layer → (metric_name, value, run_name, run_id) ──────
    by_group: dict[str, dict[int, tuple[str, float, str, str]]] = defaultdict(dict)
    for r in runs:
        group = r.group
        if group is None:
            continue
        if args.filter and args.filter.lower() not in group.lower():
            continue
        layer = _layer_from_run(r)
        if layer is None:
            continue
        keys = list(r.summary.keys())
        metric = _pick_metric(keys, args.metric)
        if metric is None:
            continue
        try:
            value = float(r.summary[metric])
        except (TypeError, ValueError):
            continue
        # Keep the latest run per (group, layer) — there may be reruns.
        prev = by_group[group].get(layer)
        if prev is None or value > prev[1]:
            by_group[group][layer] = (metric, value, r.name, r.id)

    if not by_group:
        print("No completed sweep runs found (with test/* metrics).")
        sys.exit(0)

    # ── Single-group drill-down ──────────────────────────────────────────────
    if args.group:
        if args.group not in by_group:
            print(f"Group not found: {args.group!r}")
            available = sorted(by_group.keys())
            print(f"Available ({len(available)}):")
            for g in available:
                print(f"  {g}")
            sys.exit(1)
        layer_map = by_group[args.group]
        best_layer = max(layer_map, key=lambda l: layer_map[l][1])
        metric = layer_map[best_layer][0]
        print(f"\n{args.group}  ({len(layer_map)} layers, metric: {metric})")
        print("-" * 64)
        print(f"  {'layer':>5}  {'value':>10}  run")
        for layer in sorted(layer_map):
            m, v, name, rid = layer_map[layer]
            marker = " ← best" if layer == best_layer else ""
            print(f"  {layer:>5}  {v:>10.4f}  {name}{marker}")
        return

    # ── Summary across all groups ────────────────────────────────────────────
    print(f"\n{'Group':<48} {'Layer':>5}  {'Metric':<22} {'Value':>9}")
    print("-" * 96)
    for group in sorted(by_group):
        layer_map = by_group[group]
        best_layer = max(layer_map, key=lambda l: layer_map[l][1])
        m, v, _name, _rid = layer_map[best_layer]
        # Truncate metric name for display
        m_disp = m.removeprefix("test/")
        print(f"{group:<48} {best_layer:>5}  {m_disp:<22} {v:>9.4f}")
    print()
    print(f"  {len(by_group)} sweep group(s) found.")
    print(f"  Drill into one with: scripts/analysis/best_layer.py --group \"<name>\"")


if __name__ == "__main__":
    main()
