"""Aggregate the JKUPDD within-piece motif RETRIEVAL layer sweep.

JKUPDD (JKU Patterns Development Database, Collins 2013) ground truth as a
zero-shot retrieval task: **78 byte-dedup'd** annotated pattern-occurrence
windows (165 raw, before within-group byte-dedup) across 20 groups from 5
cross-composer pieces (Bach / Beethoven / Chopin / Gibbons / Mozart). Each
window is a query; relevance = same ``(piece, annotator, pattern)`` group.

Unlike the BPS-Motif retrieval sweep, JKUPDD has **no CV folds** — the whole
benchmark is one within-piece test set, so there is exactly **one run per
layer**. Dirs are therefore::

    output/probe.JKUPDDRetrieval.CLaMP3-symbolic-layers.layer{N}/...   (per-layer)
    output/probe.JKUPDDRetrieval.CLaMP3-symbolic-meanall/...           (meanall)

with NO ``.fold{F}`` suffix. This script globs every ``wandb-summary.json``
underneath, picks the latest completed test run per layer, and reports:
  1. per-layer ``test/map`` (raw / centered / whitened), best layer flagged
  2. the meanall baseline vs best layer
  3. recall@K + secondary metrics at the best layer

It writes a leaderboard CSV (ranked best-first by raw MAP) and a JSON payload
for downstream plotting. Pure stdlib so it runs anywhere python3 does.

Usage:
  python3 scripts/sweeps/jkupdd_retrieval_summary.py [--base DIR]
      [--out-json PATH] [--out-md PATH] [--out-csv PATH]
"""

import argparse
import csv
import glob
import json
import os
import re

# Per-layer dirs: ``...CLaMP3-symbolic-layers.layer{N}`` with NO fold suffix.
# Anchor the layer token to the end (or a non-``.fold`` boundary) so this never
# matches a stray fold-style dir from a different (BPS) sweep that happens to
# share the output root.
_LAYER_RX = re.compile(r"-layers\.layer(\d+)(?!\.fold)(?:$|[^0-9])")

# Metrics reported in the recall@K / secondary tables at the best layer.
RECALL_KS = (1, 5, 10, 50, 100)
SECONDARY = ("r_precision", "median_rank", "mrr", "map@1", "hit_rate@10")
VARIANTS = ("", "_centered", "_whitened")


def parse_layer(name: str) -> int | None:
    """Return the layer index from a JKUPDD per-layer sweep dir name, or None.

    JKUPDD has no folds, so the dir ends with ``.layer{N}`` (no ``.fold{F}``).
    A trailing ``.fold{F}`` (a BPS-style dir) or a ``-meanall`` dir returns
    None so they are never mistaken for a JKUPDD per-layer cell.

    >>> parse_layer("probe.JKUPDDRetrieval.CLaMP3-symbolic-layers.layer6")
    6
    >>> parse_layer("x.CLaMP3-symbolic-layers.layer12")
    12
    >>> parse_layer("probe.BPSMotifRetrieval.CLaMP3-symbolic-layers.layer6.fold3") is None
    True
    >>> parse_layer("probe.JKUPDDRetrieval.CLaMP3-symbolic-meanall") is None
    True
    """
    m = _LAYER_RX.search(name)
    return int(m.group(1)) if m else None


def _latest_test_summary(run_glob: str) -> dict | None:
    """Return the test-metric summary from the latest run dir matching
    ``run_glob`` (ascending sort -> last wins), or None. Mirrors the BPS
    aggregator so a re-test deterministically supersedes an earlier run."""
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


def load_layers(base: str) -> dict[int, dict]:
    """Return ``{layer: summary}`` — one completed test run per layer."""
    cells: dict[int, dict] = {}
    root = os.path.join(base, "output", "probe.JKUPDDRetrieval.CLaMP3-symbolic-layers.*")
    for d in sorted(glob.glob(root)):
        layer = parse_layer(d)
        if layer is None:
            continue
        g = os.path.join(d, "wandb", "run-*", "files", "wandb-summary.json")
        summ = _latest_test_summary(g)
        if summ is not None:
            cells[layer] = summ
    return cells


def load_meanall(base: str) -> dict | None:
    """Return the meanall test summary (no folds → a single run), or None."""
    root = os.path.join(base, "output", "probe.JKUPDDRetrieval.CLaMP3-symbolic-meanall")
    g = os.path.join(root, "wandb", "run-*", "files", "wandb-summary.json")
    return _latest_test_summary(g)


def _f(x, p=4):
    return f"{x:.{p}f}" if x is not None else "  --  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.expanduser("~/developer/python/marble"))
    ap.add_argument("--out-json", default="/tmp/jkupdd_retr_summary.json")
    ap.add_argument("--out-md", default="/tmp/jkupdd_retr_summary.md")
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Also write a leaderboard CSV ranked best-first by raw test/map.",
    )
    args = ap.parse_args()

    layers = load_layers(args.base)
    meanall = load_meanall(args.base)
    if not layers:
        print("no layer runs found under", args.base)
        return

    layer_ids = sorted(layers)

    def cell(layer: int, key: str):
        d = layers.get(layer)
        return d.get(key) if d else None

    def ma(key: str):
        return meanall.get(key) if meanall else None

    ranked = sorted(layer_ids, key=lambda l: cell(l, "test/map") or -1, reverse=True)
    best = ranked[0]

    out = []
    out.append("# JKUPDD Retrieval — CLaMP3-symbolic layer sweep (no folds)")
    out.append("")
    out.append(
        "Within-piece motif retrieval, 5 cross-composer pieces, 78 byte-dedup'd "
        "occurrence windows / 20 groups, **one test set (no CV folds)**. Primary "
        "metric: **test/map**."
    )
    out.append("")

    # 1. per-layer raw / centered / whitened MAP
    out.append("## Per-layer test/map: raw | centered | whitened")
    out.append("")
    out.append("| layer | raw | centered | whitened |")
    out.append("|------:|----:|---------:|---------:|")
    for l in layer_ids:
        flag = " ⭐" if l == best else ""
        out.append(
            f"| {l}{flag} | {_f(cell(l, 'test/map'))} | "
            f"{_f(cell(l, 'test/map_centered'))} | {_f(cell(l, 'test/map_whitened'))} |"
        )
    if meanall is not None:
        out.append(
            f"| **meanall** | {_f(ma('test/map'))} | "
            f"{_f(ma('test/map_centered'))} | {_f(ma('test/map_whitened'))} |"
        )
    out.append("")
    out.append(
        f"**Best layer: {best} "
        f"(MAP {_f(cell(best, 'test/map'))} raw, "
        f"{_f(cell(best, 'test/map_centered'))} centered).**"
    )
    out.append("")

    # 2. best vs meanall
    if meanall is not None:
        out.append("## Best layer vs meanall")
        out.append("")
        out.append(f"| metric | best layer ({best}) | meanall | Δ (best−meanall) |")
        out.append("|---|---|---|---|")
        for key in ("map", "map_centered", "map_whitened"):
            bl = cell(best, f"test/{key}")
            mv = ma(f"test/{key}")
            delta = (bl - mv) if (bl is not None and mv is not None) else None
            out.append(f"| {key} | {_f(bl)} | {_f(mv)} | {_f(delta)} |")
        out.append("")

    # 3. recall@K at best layer
    out.append(f"## recall@K (best layer {best})")
    out.append("")
    out.append("| K | raw | centered | whitened |")
    out.append("|---:|----:|---------:|---------:|")
    for k in RECALL_KS:
        out.append(
            f"| {k} | {_f(cell(best, f'test/recall@{k}'))} | "
            f"{_f(cell(best, f'test/recall@{k}_centered'))} | "
            f"{_f(cell(best, f'test/recall@{k}_whitened'))} |"
        )
    out.append("")

    # 4. secondary metrics at best layer
    out.append(f"## Secondary metrics (best layer {best}) — raw / centered / whitened")
    out.append("")
    for key in SECONDARY:
        out.append(
            f"- `{key}`: {_f(cell(best, f'test/{key}'))} / "
            f"{_f(cell(best, f'test/{key}_centered'))} / "
            f"{_f(cell(best, f'test/{key}_whitened'))}"
        )
    out.append("")

    # 5. coverage
    out.append("## Coverage")
    out.append("")
    out.append(f"- layer cells: {len(layer_ids)}/13 layers found")
    out.append(f"- meanall: {'present' if meanall is not None else 'MISSING'}")
    out.append("")

    md = "\n".join(out)
    print(md)
    with open(args.out_md, "w") as fh:
        fh.write(md)

    payload = {
        "best_layer": best,
        "ranking": ranked,
        "layers": layer_ids,
        "per_layer": {
            str(l): {k: layers[l].get(k) for k in layers[l] if k.startswith("test/")}
            for l in layer_ids
        },
        "meanall": (
            {k: meanall.get(k) for k in meanall if k.startswith("test/")} if meanall else None
        ),
    }
    with open(args.out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[wrote {args.out_json} and {args.out_md}]")

    if args.out_csv:

        def rnd(x):
            return round(x, 4) if x is not None else ""

        cols = [f"map{v}" for v in VARIANTS] + [f"recall@{k}" for k in RECALL_KS]
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "layer"] + cols)
            for i, l in enumerate(ranked, 1):
                row = [i, l]
                row += [rnd(cell(l, f"test/map{v}")) for v in VARIANTS]
                row += [rnd(cell(l, f"test/recall@{k}")) for k in RECALL_KS]
                w.writerow(row)
            if meanall is not None:
                row = ["", "meanall"]
                row += [rnd(ma(f"test/map{v}")) for v in VARIANTS]
                row += [rnd(ma(f"test/recall@{k}")) for k in RECALL_KS]
                w.writerow(row)
        print(f"[wrote {args.out_csv}]")


if __name__ == "__main__":
    main()
