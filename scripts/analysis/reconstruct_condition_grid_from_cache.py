#!/usr/bin/env python3
"""
scripts/analysis/reconstruct_condition_grid_from_cache.py
─────────────────────────────────────────────────────────
Rebuild the per-condition MAP grid (and CSV + PNG heatmap) for a
completed VGMIDITVar-timbre meanall run, using only the cached per-clip
embeddings.

This is the backfill for runs that completed BEFORE probe.py started
logging ``test/map_grid/<q>_to_<t>`` cells unconditionally. The full
(8, 8) grid is computed offline in ~minutes instead of re-running the
encoder pass.

What it does
------------
1. Parses the task JSONL for ``audio_path``, ``work_id``, ``gm_program``.
2. Walks ``output/.emb_cache/<encoder>/<task>__<hash>/`` for cached
   per-clip ``(L, H)`` tensors and averages them per file.
3. Replicates the probe's mean-pool over layers (meanall configs use
   ``LayerSelector(mode='mean')``), then mean-pool over clips of the
   same file, then L2-normalise → ``(N, H)`` per-file embedding matrix.
4. Centers, computes cosine similarity, runs
   ``compute_perpair_map_all`` with the same conventions as the live
   probe (centered sim, -inf self-exclusion).
5. Writes ``condition_grid.csv`` and ``condition_grid.png`` (heatmap)
   next to ``--out-dir``.

Usage
-----
  python scripts/analysis/reconstruct_condition_grid_from_cache.py \\
      --encoder MERT-v1-95M \\
      --jsonl data/VGMIDITVar-timbre/VGMIDITVar.jsonl \\
      --cache-dir output/.emb_cache/MERT-v1-95M/VGMIDITVar-timbre__75764d50 \\
      --out-dir output/recon/MERT-v1-95M-VGMIDITVar-timbre
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from marble.utils.emb_cache import make_clip_id  # noqa: E402
from marble.utils.path_compat import load_jsonl  # noqa: E402
from marble.utils.retrieval_metrics import compute_perpair_map_all  # noqa: E402


def _collect_per_file_embeddings(
    records: list[dict], cache_dir: Path
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Return ``(N, H)`` file embeddings + ``work_ids`` + ``conditions``.

    For each JSONL record, glob the cache dir for ``<clip_id>.pt`` files
    sharing the record's audio_path-hash. Take mean over layers (L)
    then mean over slices (clips), L2-normalise. Records without any
    cached tensor are skipped (warned).
    """
    file_embs: list[torch.Tensor] = []
    file_wids: list[int] = []
    file_conds: list[int] = []
    n_missing = 0
    n_total = 0

    for rec in records:
        n_total += 1
        # clip_id format: <stem>__<sha1(audio_path)[:8]>__c<slice_idx>
        base_id = make_clip_id(rec["audio_path"], 0).rsplit("__c", 1)[0]
        # Glob all slices for this file.
        slice_paths = sorted(cache_dir.glob(f"{base_id}__c*.pt"))
        # Defensive: cache stores filesystem-safe names. The path_for
        # method in emb_cache.py replaces / and \ with _, which can
        # affect the stem if the audio_path stem itself contained those.
        # For VGMIDITVar-timbre stems they're letters/digits/dashes only.
        if not slice_paths:
            n_missing += 1
            continue
        per_slice: list[torch.Tensor] = []
        for p in slice_paths:
            t = torch.load(p, map_location="cpu", weights_only=True)
            # Cached shape is (L, H) for pool_time=True. Mean over L to
            # replicate LayerSelector(mode='mean') + TimeAvgPool.
            if t.dim() == 2:
                per_slice.append(t.mean(dim=0))  # (H,)
            elif t.dim() == 3:
                # frame-level cache (L, T, H) — average over both L and T.
                per_slice.append(t.mean(dim=(0, 1)))
            else:
                per_slice.append(t.flatten())
        mean_emb = torch.stack(per_slice).mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=-1)
        file_embs.append(mean_emb)
        file_wids.append(int(rec["work_id"]))
        cond_raw = rec.get("gm_program")
        if cond_raw is None:
            cond_raw = rec.get("soundfont_id")
        file_conds.append(int(cond_raw) if cond_raw is not None else -1)

    if n_missing:
        print(
            f"WARN: {n_missing}/{n_total} JSONL records had no cached embedding in {cache_dir}",
            file=sys.stderr,
        )
    if not file_embs:
        raise RuntimeError(f"No cached embeddings found under {cache_dir}")
    return torch.stack(file_embs), file_wids, file_conds


def _write_csv(
    csv_path: Path,
    unique_conds: list[int],
    cell_results: dict[tuple[int, int], tuple[float, int]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_program", "target_program", "map", "n_queries"])
        for q in unique_conds:
            for t in unique_conds:
                ap, n = cell_results.get((q, t), (0.0, 0))
                w.writerow([q, t, f"{ap:.6f}", n])


def _write_png(
    png_path: Path,
    unique_conds: list[int],
    cell_results: dict[tuple[int, int], tuple[float, int]],
    title: str,
) -> None:
    try:
        import matplotlib
    except ImportError:
        print("WARN: matplotlib not installed; skipping PNG", file=sys.stderr)
        return
    matplotlib.use("Agg")
    import numpy as np
    from matplotlib import pyplot as plt

    n_c = len(unique_conds)
    grid = np.full((n_c, n_c), np.nan, dtype=float)
    for i, q in enumerate(unique_conds):
        for j, t in enumerate(unique_conds):
            ap, n = cell_results.get((q, t), (0.0, 0))
            if n > 0:
                grid[i, j] = ap

    fig, ax = plt.subplots(figsize=(max(4, n_c * 0.7), max(4, n_c * 0.7)))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.0)
    ax.set_xticks(range(n_c))
    ax.set_yticks(range(n_c))
    ax.set_xticklabels([str(c) for c in unique_conds], rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in unique_conds])
    ax.set_xlabel("target condition (gm_program)")
    ax.set_ylabel("query condition (gm_program)")
    ax.set_title(title)
    for i in range(n_c):
        for j in range(n_c):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.5 else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=ax, label="MAP")
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--encoder", required=True, help="Encoder slug (for the heatmap title)")
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: jsonl not found: {args.jsonl}", file=sys.stderr)
        return 1
    if not args.cache_dir.is_dir():
        print(f"ERROR: cache_dir not found: {args.cache_dir}", file=sys.stderr)
        return 1

    print(f"Loading JSONL: {args.jsonl}")
    records = load_jsonl(args.jsonl)
    print(f"  {len(records)} records")

    print(f"Loading per-clip embeddings from cache: {args.cache_dir}")
    embs, wids, conds = _collect_per_file_embeddings(records, args.cache_dir)
    print(f"  {embs.shape[0]} files, {embs.shape[1]}-dim embeddings")

    if not any(c != -1 for c in conds):
        print("ERROR: no records carry a condition field; nothing to do", file=sys.stderr)
        return 1
    unique_conds = sorted({c for c in conds if c != -1})
    print(f"  {len(unique_conds)} unique conditions: {unique_conds}")

    # Center + cosine sim (same as probe.py).
    print("Computing centered cosine similarity ...")
    embs_c = embs - embs.mean(dim=0, keepdim=True)
    embs_c = F.normalize(embs_c, dim=-1)
    sim_c = embs_c @ embs_c.T

    print("Running compute_perpair_map_all ...")
    cell_results = compute_perpair_map_all(
        sim_c, wids, conds, query_conds=unique_conds, target_conds=unique_conds
    )

    # Summarise.
    same_aps = [ap for (q, t), (ap, n) in cell_results.items() if q == t and n > 0]
    cross_aps = [ap for (q, t), (ap, n) in cell_results.items() if q != t and n > 0]
    if same_aps:
        print(f"  MAP same-condition  = {sum(same_aps) / len(same_aps):.4f}")
    if cross_aps:
        print(f"  MAP cross-condition = {sum(cross_aps) / len(cross_aps):.4f}")
    if same_aps and cross_aps:
        gap = sum(same_aps) / len(same_aps) - sum(cross_aps) / len(cross_aps)
        print(f"  condition_gap       = {gap:+.4f}")

    csv_path = args.out_dir / "condition_grid.csv"
    png_path = args.out_dir / "condition_grid.png"
    _write_csv(csv_path, unique_conds, cell_results)
    print(f"Wrote {csv_path}")
    _write_png(
        png_path,
        unique_conds,
        cell_results,
        title=f"{args.encoder} × VGMIDITVar-timbre — per-condition MAP",
    )
    print(f"Wrote {png_path}")

    # Also emit the raw cell_results as JSON for downstream tools.
    json_path = args.out_dir / "condition_grid.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "encoder": args.encoder,
                "jsonl": str(args.jsonl),
                "cache_dir": str(args.cache_dir),
                "n_files": embs.shape[0],
                "unique_conditions": unique_conds,
                "cells": [
                    {"query": q, "target": t, "map": ap, "n_queries": n}
                    for (q, t), (ap, n) in sorted(cell_results.items())
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
