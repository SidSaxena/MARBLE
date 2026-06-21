"""Aggregate the BPS-Motif Retrieval layer sweep into readable tables.

Reads every wandb-summary.json under
  output/probe.BPSMotifRetrieval.CLaMP3-symbolic-layers.fold{F}.layer{L}/wandb/run-*/files/
and prints:
  1. test/map by layer x fold (+ mean), with the best layer flagged
  2. per-layer mean MAP: raw vs centered vs whitened
  3. recall@K at the best layer (raw / centered / whitened)
  4. secondary metrics at the best layer
Also writes the same text to /tmp/bps_retr_summary.txt.
"""

import glob
import json
import os
import re

BASE = os.path.expanduser("~/developer/python/marble")
PAT = os.path.join(
    BASE,
    "output",
    "probe.BPSMotifRetrieval.CLaMP3-symbolic-layers.*",
    "wandb",
    "run-*",
    "files",
    "wandb-summary.json",
)
# Accept both dir orderings: layer-primary "...layer6.fold3" (current) and
# fold-primary "...fold3.layer6" (legacy).
_LAYER_PRIMARY_RX = re.compile(r"\.layer(\d+)\.fold(\d+)")
_FOLD_PRIMARY_RX = re.compile(r"\.fold(\d+)\.layer(\d+)")


def _parse_fold_layer(path: str):
    m = _LAYER_PRIMARY_RX.search(path)
    if m:
        return int(m.group(2)), int(m.group(1))
    m = _FOLD_PRIMARY_RX.search(path)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def load_runs():
    best: dict[tuple[int, int], dict] = {}
    # sorted() so that if a (fold, layer) dir holds multiple completed runs
    # (e.g. after a re-test), the pick is deterministic rather than glob-order.
    for p in sorted(glob.glob(PAT)):
        fl = _parse_fold_layer(p)
        if fl is None:
            continue
        f, l = fl
        try:
            with open(p) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if "test/map" in d and ((f, l) not in best or "test/map" not in best[(f, l)]):
            best[(f, l)] = d
    return best


def main():
    best = load_runs()
    if not best:
        print("no runs found under", PAT)
        return
    folds = sorted({f for f, _ in best})
    layers = sorted({l for _, l in best})

    def mean(layer, key):
        vs = [
            best[(f, layer)][key] for f in folds if (f, layer) in best and key in best[(f, layer)]
        ]
        return sum(vs) / len(vs) if vs else None

    def fmt(x):
        return f"{x:.4f}" if x is not None else "  --  "

    out = []
    out.append("=== test/map by layer x fold ===")
    out.append("layer | " + " | ".join(f"fold{f}" for f in folds) + " |  MEAN")
    for l in layers:
        row = [f"  {l:>3} "] + [fmt(best.get((f, l), {}).get("test/map")) for f in folds]
        row.append(fmt(mean(l, "test/map")))
        out.append(" | ".join(row))
    bl = max(layers, key=lambda l: (mean(l, "test/map") or -1))
    out.append(
        f"\nBEST LAYER by mean MAP: layer {bl} ({fmt(mean(bl, 'test/map'))} raw, "
        f"{fmt(mean(bl, 'test/map_centered'))} centered)\n"
    )

    out.append("=== per-layer mean MAP: raw | centered | whitened ===")
    out.append("layer |   raw   | centered | whitened")
    for l in layers:
        out.append(
            f"  {l:>3} | {fmt(mean(l, 'test/map'))}  |  "
            f"{fmt(mean(l, 'test/map_centered'))}  |  {fmt(mean(l, 'test/map_whitened'))}"
        )

    out.append(f"\n=== best layer {bl}: recall@K (mean) raw | centered | whitened ===")
    out.append("  K  |   raw   | centered | whitened")
    for k in (1, 5, 10, 50, 100):
        out.append(
            f"  {k:>3}| {fmt(mean(bl, f'test/recall@{k}'))}  |  "
            f"{fmt(mean(bl, f'test/recall@{k}_centered'))}  |  "
            f"{fmt(mean(bl, f'test/recall@{k}_whitened'))}"
        )

    out.append(f"\n=== best layer {bl}: secondary metrics (mean) raw / centered / whitened ===")
    for key in ("r_precision", "median_rank", "mrr", "map@1", "hit_rate@10"):
        out.append(
            f"  {key:<14} {fmt(mean(bl, f'test/{key}'))} / "
            f"{fmt(mean(bl, f'test/{key}_centered'))} / "
            f"{fmt(mean(bl, f'test/{key}_whitened'))}"
        )

    got = sum(1 for k in best if "test/map" in best[k])
    out.append(
        f"\ncoverage: {got} runs with test/map "
        f"(expected {len(folds)} folds x {len(layers)} layers = {len(folds) * len(layers)})"
    )

    text = "\n".join(out)
    with open("/tmp/bps_retr_summary.txt", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
