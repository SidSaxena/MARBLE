#!/usr/bin/env python3
"""
scripts/analysis/vgmiditvar_leitmotif_sweep.py
──────────────────────────────────────────────
Reproducible end-to-end analysis for the VGMIDITVar-leitmotif benchmark
across all encoders and ALL layers (no selection bias).

For each (encoder, layer) cell we compute:
  • Aggregate retrieval MAP (full candidate pool — same as the WandB number)
  • Per-pair MAP grid sliced by (query gm_program, target gm_program)
  • Mean of diagonal cells (same-instrument MAP — small N, see notes)
  • Mean of off-diagonal cells (cross-instrument MAP — the headline metric
    for genuine leitmotif invariance)

Why all-layers, not top-K? Selection by aggregate MAP is exactly the bias
we discovered with MuQ: aggregate-best L12 is NOT cross-instrument-best
(L8 wins per-pair). Picking layers by aggregate before computing per-pair
hides that. The cached embeddings make full-layer enumeration cheap —
each layer is a tensor slice + matrix product over ~1700 files.

Outputs (all under <out-dir>, default output/analysis/leitmotif/):
  • per_pair_map.csv          long format: (encoder, layer, q_program, t_program, map, n)
  • per_layer_summary.csv     wide format: (encoder, layer, agg, same, cross)
  • cross_encoder_summary.csv per-encoder best-layer per metric, plus the WandB
                              aggregate from VGMIDITVar / VGMIDITVar-multisf
                              for direct three-task comparison
  • dashboard_<encoder>.png   one PNG per encoder: layer profile + best-layer
                              pair heatmap + tri-task overlay
  • cross_encoder_summary.png cross-encoder bar chart at each encoder's
                              cross-instrument-best layer

Reproducibility:
  • All numerical inputs come from on-disk cache files + WandB (deterministic).
  • Computation is pure linear algebra; no RNG.
  • All intermediate data is dumped as CSV — plots can be regenerated from
    CSVs without re-loading the cache.

CLaMP3-symbolic note: the symbolic dataset doesn't emit clip_ids, so the
cache mechanism never fires for it (cache dir is empty). This script
includes symbolic in the WandB-aggregate side of cross_encoder_summary
but cannot do per-pair analysis for it. A separate one-off extraction
script is needed if per-pair symbolic numbers are wanted.

Usage:
    uv run python scripts/analysis/vgmiditvar_leitmotif_sweep.py
    # (auto-discovers cache dirs under output/.emb_cache/<encoder>/...)

    uv run python scripts/analysis/vgmiditvar_leitmotif_sweep.py \\
        --cache-root output/.emb_cache \\
        --jsonl data/VGMIDITVar-leitmotif/VGMIDITVar.jsonl \\
        --out-dir output/analysis/leitmotif \\
        --no-wandb           # skip WandB pulls if running offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

# Reuse helpers from the single-layer breakdown script (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vgmiditvar_leitmotif_breakdown import (  # noqa: E402
    _clip_id,
    _load_jsonl,
    _map_for_subset,
)

log = logging.getLogger(__name__)

# Audio encoders we sweep (CLaMP3-symbolic excluded — cache is empty by
# design; covered separately in the WandB-aggregate side).
AUDIO_ENCODERS: list[str] = [
    "CLaMP3",
    "MERT-v1-95M",
    "MuQ",
    "OMARRQ-multifeature-25hz",
]

# Encoders we ALSO pull WandB aggregate metrics for (for the cross-task
# comparison). Symbolic is here because we want it on the cross-encoder
# summary even though we can't do per-pair for it.
WANDB_ENCODERS: list[str] = AUDIO_ENCODERS + ["CLaMP3-symbolic"]

GM_LABELS: dict[int, str] = {0: "Piano", 48: "Strings", 56: "Trumpet", 60: "Horn", 73: "Flute"}

# Sentinel for the aggregate (full-pool) row in per_pair_map.csv — keeps
# everything in one long-format DataFrame.
AGG_SENTINEL = -1


# ─── Cache discovery + load ──────────────────────────────────────────────────


def discover_cache_dir(cache_root: Path, encoder: str) -> Path | None:
    """Find the most recent VGMIDITVar-leitmotif cache dir for an encoder.

    Returns None if no dir exists OR the dir is empty (e.g. CLaMP3-symbolic).
    """
    enc_dir = cache_root / encoder
    if not enc_dir.exists():
        log.warning("No cache dir for %s under %s", encoder, cache_root)
        return None
    cands = sorted(
        enc_dir.glob("VGMIDITVar-leitmotif__*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        log.warning("No VGMIDITVar-leitmotif cache dir under %s", enc_dir)
        return None
    chosen = cands[0]
    n_pt = sum(1 for _ in chosen.glob("*.pt"))
    if n_pt == 0:
        log.warning("Cache dir %s is empty (encoder bypasses cache?)", chosen)
        return None
    log.info("  %s: cache=%s (%d .pt files)", encoder, chosen.name, n_pt)
    return chosen


def discover_layer_count(cache_dir: Path) -> int:
    """Inspect the first .pt file to get L from the (L, H) tensor."""
    for p in cache_dir.glob("*.pt"):
        obj = torch.load(p, map_location="cpu", weights_only=True)
        tensor = obj["embedding"] if isinstance(obj, dict) else obj
        return int(tensor.size(0))
    return 0


def load_layered_embeddings(
    records: list[dict],
    cache_dir: Path,
    n_layers: int,
) -> tuple[torch.Tensor, list[int], list[int]] | None:
    """Load (N_paths, L, H) tensor + per-path work_ids and gm_programs.

    Mean-pools clip embeddings per audio_path (multiple clips per path can
    arise from clip_seconds slicing, though VGMIDITVar typically yields 1
    clip per file). The per-layer L2 normalisation happens later, per layer.
    """
    path2info: dict[str, dict] = defaultdict(lambda: {"embs": [], "work_id": None, "program": None})
    missing = 0
    bad_shape = 0
    for rec in records:
        audio_path = rec["audio_path"]
        cid = _clip_id(audio_path, 0)
        p = cache_dir / f"{cid}.pt"
        if not p.exists():
            missing += 1
            continue
        obj = torch.load(p, map_location="cpu", weights_only=True)
        tensor = obj["embedding"] if isinstance(obj, dict) else obj
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 or tensor.size(0) != n_layers:
            bad_shape += 1
            continue
        path2info[audio_path]["embs"].append(tensor)
        path2info[audio_path]["work_id"] = int(rec["work_id"])
        path2info[audio_path]["program"] = int(rec.get("gm_program", 0))

    if missing:
        log.warning("    %d/%d cache misses", missing, len(records))
    if bad_shape:
        log.warning("    %d/%d cache files had bad shape", bad_shape, len(records))

    paths = sorted(path2info)
    if not paths:
        return None

    layered = []
    work_ids: list[int] = []
    programs: list[int] = []
    for path in paths:
        info = path2info[path]
        stacked = torch.stack(info["embs"]).mean(0)  # (L, H)
        layered.append(stacked)
        work_ids.append(info["work_id"])
        programs.append(info["program"])

    return torch.stack(layered), work_ids, programs  # (N, L, H), [N], [N]


# ─── Per-layer per-pair MAP ──────────────────────────────────────────────────


def compute_per_pair_for_encoder(
    layered: torch.Tensor,
    work_ids: list[int],
    programs: list[int],
    encoder: str,
) -> pd.DataFrame:
    """For each layer, compute aggregate MAP + per-pair grid. Long-format DF."""
    n, n_layers, _ = layered.shape
    progs_sorted = sorted({p for p in programs if p is not None})
    rows: list[dict] = []

    for L in range(n_layers):
        layer_embs = F.normalize(layered[:, L, :], dim=-1)
        sim = layer_embs @ layer_embs.T

        # Aggregate (full pool, no program filter)
        agg_map, agg_n = _map_for_subset(sim, work_ids, programs, None, None)
        rows.append(
            {
                "encoder": encoder,
                "layer": L,
                "query_program": AGG_SENTINEL,
                "target_program": AGG_SENTINEL,
                "map": agg_map,
                "n_queries": agg_n,
            }
        )

        # Per-pair grid
        for q in progs_sorted:
            for t in progs_sorted:
                m, nq = _map_for_subset(sim, work_ids, programs, q, t)
                rows.append(
                    {
                        "encoder": encoder,
                        "layer": L,
                        "query_program": q,
                        "target_program": t,
                        "map": m,
                        "n_queries": nq,
                    }
                )
        log.info("    L%d: agg=%.4f (N=%d)", L, agg_map, agg_n)

    return pd.DataFrame(rows)


def summarise_per_layer(per_pair: pd.DataFrame) -> pd.DataFrame:
    """Collapse per_pair → per (encoder, layer) summary with agg / same / cross."""
    rows: list[dict] = []
    for (enc, L), sub in per_pair.groupby(["encoder", "layer"]):
        agg = sub[sub["query_program"] == AGG_SENTINEL].iloc[0]
        non_agg = sub[sub["query_program"] != AGG_SENTINEL]
        same = non_agg[non_agg["query_program"] == non_agg["target_program"]]
        cross = non_agg[non_agg["query_program"] != non_agg["target_program"]]

        # Cells with N=0 don't contribute to the mean — they reflect
        # work-id structure (some pairs have no positives). Mask them out
        # to avoid biasing the mean toward zero.
        same_nz = same[same["n_queries"] > 0]
        cross_nz = cross[cross["n_queries"] > 0]

        rows.append(
            {
                "encoder": enc,
                "layer": int(L),
                "aggregate_map": float(agg["map"]),
                "aggregate_n": int(agg["n_queries"]),
                "same_instr_map_mean": float(same_nz["map"].mean())
                if len(same_nz)
                else float("nan"),
                "same_instr_n_cells": int(len(same_nz)),
                "cross_instr_map_mean": float(cross_nz["map"].mean())
                if len(cross_nz)
                else float("nan"),
                "cross_instr_n_cells": int(len(cross_nz)),
            }
        )
    return pd.DataFrame(rows).sort_values(["encoder", "layer"]).reset_index(drop=True)


# ─── WandB aggregate pulls (for the tri-task comparison) ─────────────────────


def pull_wandb_aggregates(
    encoders: list[str],
    variants: tuple[str, ...] = ("VGMIDITVar", "VGMIDITVar-multisf", "VGMIDITVar-leitmotif"),
) -> pd.DataFrame:
    """Pull test/map per (encoder, variant, layer) from WandB. Returns long DF."""
    import re

    import wandb

    api = wandb.Api()
    rows: list[dict] = []
    for enc in encoders:
        for v in variants:
            try:
                runs = list(api.runs("marble", filters={"group": f"{enc} / {v}"}))
            except Exception as e:
                log.warning("WandB pull failed for %s / %s: %s", enc, v, e)
                continue
            for r in runs:
                m = re.match(r"layer-(\d+|meanall)-test", r.name or "")
                if not m:
                    continue
                summ = r.summary._json_dict if hasattr(r.summary, "_json_dict") else dict(r.summary)
                mp = summ.get("test/map")
                if not isinstance(mp, (int, float)):
                    continue
                layer_token = m.group(1)
                rows.append(
                    {
                        "encoder": enc,
                        "variant": v,
                        "layer_token": layer_token,
                        "layer": -1 if layer_token == "meanall" else int(layer_token),
                        "test_map": float(mp),
                        "test_mrr": float(summ.get("test/mrr", float("nan")) or float("nan")),
                        "test_map_at_1": float(
                            summ.get("test/map@1", float("nan")) or float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


# ─── Cross-encoder summary ───────────────────────────────────────────────────


def build_cross_encoder_summary(
    per_layer: pd.DataFrame,
    wandb_agg: pd.DataFrame,
) -> pd.DataFrame:
    """For each encoder, surface best-layer-per-metric across the three variants
    and the per-pair metrics. One row per encoder."""
    rows: list[dict] = []

    def _best_from_wnd(wnd_subset: pd.DataFrame, variant: str) -> tuple[int | None, float | None]:
        """Return (best_layer, best_test_map) for one (encoder, variant) slice.
        Lifted above the loop and given the slice as a parameter so we don't
        close over loop-local DataFrames (B023)."""
        sub = wnd_subset[(wnd_subset["variant"] == variant) & (wnd_subset["layer"] >= 0)]
        if sub.empty:
            return None, None
        row = sub.loc[sub["test_map"].idxmax()]
        return int(row["layer"]), float(row["test_map"])

    for enc in WANDB_ENCODERS:
        wnd = wandb_agg[wandb_agg["encoder"] == enc]
        per = per_layer[per_layer["encoder"] == enc]

        b_orig = _best_from_wnd(wnd, "VGMIDITVar")
        b_ms = _best_from_wnd(wnd, "VGMIDITVar-multisf")
        b_lm = _best_from_wnd(wnd, "VGMIDITVar-leitmotif")

        if not per.empty:
            best_cross_row = per.loc[per["cross_instr_map_mean"].idxmax()]
            best_agg_row = per.loc[per["aggregate_map"].idxmax()]
            best_cross = (
                int(best_cross_row["layer"]),
                float(best_cross_row["cross_instr_map_mean"]),
            )
            best_agg_per = (int(best_agg_row["layer"]), float(best_agg_row["aggregate_map"]))
        else:
            best_cross = (None, None)
            best_agg_per = (None, None)

        rows.append(
            {
                "encoder": enc,
                "wandb_best_layer_orig": b_orig[0],
                "wandb_best_map_orig": b_orig[1],
                "wandb_best_layer_multisf": b_ms[0],
                "wandb_best_map_multisf": b_ms[1],
                "wandb_best_layer_leitmotif": b_lm[0],
                "wandb_best_map_leitmotif": b_lm[1],
                "perpair_best_layer_aggregate": best_agg_per[0],
                "perpair_best_map_aggregate": best_agg_per[1],
                "perpair_best_layer_cross_instrument": best_cross[0],
                "perpair_best_map_cross_instrument": best_cross[1],
            }
        )
    return pd.DataFrame(rows)


# ─── Plotting ────────────────────────────────────────────────────────────────


def plot_dashboard_for_encoder(
    encoder: str,
    per_pair: pd.DataFrame,
    per_layer: pd.DataFrame,
    wandb_agg: pd.DataFrame,
    out_dir: Path,
) -> None:
    """One dashboard PNG per encoder: 2x2 panels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"VGMIDITVar-leitmotif breakdown — {encoder}",
        fontsize=14,
        fontweight="bold",
    )

    enc_per_layer = per_layer[per_layer["encoder"] == encoder].sort_values("layer")
    enc_per_pair = per_pair[per_pair["encoder"] == encoder]

    # Panel A — layer profile (3 metrics on one axis)
    ax = axes[0, 0]
    ax.plot(
        enc_per_layer["layer"],
        enc_per_layer["aggregate_map"],
        marker="o",
        label="Aggregate (full pool)",
        color="C0",
    )
    ax.plot(
        enc_per_layer["layer"],
        enc_per_layer["cross_instr_map_mean"],
        marker="s",
        label="Cross-instrument (off-diag mean)",
        color="C1",
    )
    ax.plot(
        enc_per_layer["layer"],
        enc_per_layer["same_instr_map_mean"],
        marker="^",
        label="Same-instrument (diag mean, low N)",
        color="C2",
        alpha=0.5,
        linestyle="--",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("MAP")
    ax.set_title("Per-layer MAP profile (leitmotif task)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    # Mark best layer per metric
    if not enc_per_layer["aggregate_map"].isna().all():
        idx = enc_per_layer["aggregate_map"].idxmax()
        ax.axvline(enc_per_layer.loc[idx, "layer"], color="C0", alpha=0.3, linestyle=":")
    if not enc_per_layer["cross_instr_map_mean"].isna().all():
        idx = enc_per_layer["cross_instr_map_mean"].idxmax()
        ax.axvline(enc_per_layer.loc[idx, "layer"], color="C1", alpha=0.3, linestyle=":")

    # Panel B — pair heatmap at cross-instrument-best layer
    ax = axes[0, 1]
    if not enc_per_layer["cross_instr_map_mean"].isna().all():
        best_L = int(enc_per_layer.loc[enc_per_layer["cross_instr_map_mean"].idxmax(), "layer"])
        sub = enc_per_pair[
            (enc_per_pair["layer"] == best_L) & (enc_per_pair["query_program"] != AGG_SENTINEL)
        ]
        progs = sorted(set(sub["query_program"].tolist()))
        grid = np.full((len(progs), len(progs)), np.nan)
        n_grid = np.zeros_like(grid, dtype=int)
        for _, row in sub.iterrows():
            qi = progs.index(row["query_program"])
            ti = progs.index(row["target_program"])
            if row["n_queries"] > 0:
                grid[qi, ti] = row["map"]
            n_grid[qi, ti] = row["n_queries"]
        labels = [GM_LABELS.get(p, str(p)) for p in progs]
        annot = np.array(
            [
                [
                    f"{grid[i, j]:.3f}\nN={n_grid[i, j]}" if not np.isnan(grid[i, j]) else "—"
                    for j in range(len(progs))
                ]
                for i in range(len(progs))
            ]
        )
        sns.heatmap(
            grid,
            annot=annot,
            fmt="",
            cmap="viridis",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "MAP"},
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Per-pair MAP at L{best_L} (cross-instr best)")
        ax.set_xlabel("Target instrument →")
        ax.set_ylabel("← Query instrument")
    else:
        ax.text(0.5, 0.5, "no per-pair data", ha="center", va="center")
        ax.set_axis_off()

    # Panel C — tri-task overlay (orig / multisf / leitmotif aggregate)
    ax = axes[1, 0]
    wnd = wandb_agg[(wandb_agg["encoder"] == encoder) & (wandb_agg["layer"] >= 0)]
    for variant, color, marker in [
        ("VGMIDITVar", "C0", "o"),
        ("VGMIDITVar-multisf", "C1", "s"),
        ("VGMIDITVar-leitmotif", "C3", "^"),
    ]:
        sub = wnd[wnd["variant"] == variant].sort_values("layer")
        if sub.empty:
            continue
        ax.plot(
            sub["layer"], sub["test_map"], marker=marker, label=variant, color=color, alpha=0.85
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("test/MAP (aggregate)")
    ax.set_title("Tri-task aggregate MAP across VGMIDITVar variants")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    # Panel D — pair heatmap at AGGREGATE-best layer (might differ from C)
    ax = axes[1, 1]
    if not enc_per_layer["aggregate_map"].isna().all():
        best_L = int(enc_per_layer.loc[enc_per_layer["aggregate_map"].idxmax(), "layer"])
        sub = enc_per_pair[
            (enc_per_pair["layer"] == best_L) & (enc_per_pair["query_program"] != AGG_SENTINEL)
        ]
        progs = sorted(set(sub["query_program"].tolist()))
        grid = np.full((len(progs), len(progs)), np.nan)
        n_grid = np.zeros_like(grid, dtype=int)
        for _, row in sub.iterrows():
            qi = progs.index(row["query_program"])
            ti = progs.index(row["target_program"])
            if row["n_queries"] > 0:
                grid[qi, ti] = row["map"]
            n_grid[qi, ti] = row["n_queries"]
        labels = [GM_LABELS.get(p, str(p)) for p in progs]
        annot = np.array(
            [
                [
                    f"{grid[i, j]:.3f}\nN={n_grid[i, j]}" if not np.isnan(grid[i, j]) else "—"
                    for j in range(len(progs))
                ]
                for i in range(len(progs))
            ]
        )
        sns.heatmap(
            grid,
            annot=annot,
            fmt="",
            cmap="viridis",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "MAP"},
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Per-pair MAP at L{best_L} (aggregate best)")
        ax.set_xlabel("Target instrument →")
        ax.set_ylabel("← Query instrument")
    else:
        ax.text(0.5, 0.5, "no per-pair data", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / f"dashboard_{encoder}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", out_path)


def plot_cross_encoder_comparison(
    summary: pd.DataFrame,
    per_layer: pd.DataFrame,
    wandb_agg: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Two-panel comparison across all encoders."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Cross-encoder VGMIDITVar comparison",
        fontsize=14,
        fontweight="bold",
    )

    # Panel A — best aggregate MAP per (encoder, variant)
    ax = axes[0]
    encoders = list(summary["encoder"])
    x = np.arange(len(encoders))
    width = 0.27
    bars_data = [
        ("VGMIDITVar (FluidR3)", summary["wandb_best_map_orig"], "C0"),
        ("VGMIDITVar-multisf", summary["wandb_best_map_multisf"], "C1"),
        ("VGMIDITVar-leitmotif", summary["wandb_best_map_leitmotif"], "C3"),
    ]
    for i, (label, vals, color) in enumerate(bars_data):
        positions = x + (i - 1) * width
        ax.bar(positions, vals.fillna(0), width, label=label, color=color, alpha=0.8)
        # Annotate each bar with the layer index
        layer_col = {
            "VGMIDITVar (FluidR3)": "wandb_best_layer_orig",
            "VGMIDITVar-multisf": "wandb_best_layer_multisf",
            "VGMIDITVar-leitmotif": "wandb_best_layer_leitmotif",
        }[label]
        for j, (v, L) in enumerate(zip(vals.tolist(), summary[layer_col].tolist(), strict=True)):
            if v is not None and not pd.isna(v) and L is not None and not pd.isna(L):
                ax.text(positions[j], v + 0.005, f"L{int(L)}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=20, ha="right")
    ax.set_ylabel("Best test/MAP")
    ax.set_title("Aggregate MAP at each encoder's best layer per variant")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel B — best CROSS-INSTRUMENT MAP per encoder (per-pair analysis)
    ax = axes[1]
    audio_only = summary[summary["encoder"].isin(AUDIO_ENCODERS)].copy()
    audio_only = audio_only.sort_values("perpair_best_map_cross_instrument", ascending=False)
    encs = audio_only["encoder"].tolist()
    vals = audio_only["perpair_best_map_cross_instrument"].tolist()
    layers = audio_only["perpair_best_layer_cross_instrument"].tolist()
    colors = ["C2"] * len(encs)
    bars = ax.bar(encs, [v if v is not None else 0 for v in vals], color=colors, alpha=0.85)
    for bar, v, L in zip(bars, vals, layers, strict=True):
        if v is not None and not pd.isna(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"L{int(L)}\n{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticklabels(encs, rotation=20, ha="right")
    ax.set_ylabel("Cross-instrument MAP (off-diagonal mean)")
    ax.set_title("Cross-instrument retrieval ability (audio encoders, best layer per encoder)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / "cross_encoder_summary.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", out_path)


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    sns.set_theme(style="whitegrid", context="paper")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--cache-root",
        default="output/.emb_cache",
        help="MARBLE cache root (default: %(default)s)",
    )
    ap.add_argument(
        "--jsonl",
        default="data/VGMIDITVar-leitmotif/VGMIDITVar.jsonl",
        help="Leitmotif JSONL (must have gm_program field)",
    )
    ap.add_argument(
        "--out-dir",
        default="output/analysis/leitmotif",
        help="Output directory for CSVs and PNGs",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "all"),
    )
    ap.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip the WandB pull. Disables tri-task overlay panel.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_root)
    if not cache_root.exists():
        log.error("Cache root not found: %s", cache_root)
        sys.exit(1)

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        log.error("JSONL not found: %s", jsonl_path)
        sys.exit(1)

    log.info("Loading JSONL: %s", jsonl_path)
    records = _load_jsonl(jsonl_path)
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]
    log.info("  %d records (split=%s)", len(records), args.split)
    if any("gm_program" not in r for r in records):
        log.warning(
            "Some records lack `gm_program`; they will default to 0 (Piano). "
            "Re-render the audio with `--instrument-map` to fix."
        )

    # ── Per-pair sweep across all (encoder, layer) cells ────────────────────
    log.info("\n── Per-pair sweep (audio encoders) ──")
    all_per_pair: list[pd.DataFrame] = []
    for enc in AUDIO_ENCODERS:
        log.info("encoder=%s", enc)
        cache_dir = discover_cache_dir(cache_root, enc)
        if cache_dir is None:
            continue
        n_layers = discover_layer_count(cache_dir)
        if n_layers == 0:
            continue
        log.info("  L=%d", n_layers)
        loaded = load_layered_embeddings(records, cache_dir, n_layers)
        if loaded is None:
            log.warning("  no usable embeddings for %s; skipping", enc)
            continue
        layered, work_ids, programs = loaded
        log.info("  loaded (%d files, L=%d, H=%d)", *layered.shape)
        df = compute_per_pair_for_encoder(layered, work_ids, programs, enc)
        all_per_pair.append(df)

    if not all_per_pair:
        log.error("No per-pair data computed. Aborting.")
        sys.exit(2)
    per_pair = pd.concat(all_per_pair, ignore_index=True)

    # ── Per-layer summary ───────────────────────────────────────────────────
    per_layer = summarise_per_layer(per_pair)

    # ── WandB pull for tri-task overlay + cross-encoder summary ─────────────
    if args.no_wandb:
        wandb_agg = pd.DataFrame(columns=["encoder", "variant", "layer", "test_map"])
    else:
        log.info("\n── WandB aggregate pull (3 variants × %d encoders) ──", len(WANDB_ENCODERS))
        wandb_agg = pull_wandb_aggregates(WANDB_ENCODERS)
        log.info("  pulled %d run summaries", len(wandb_agg))

    summary = build_cross_encoder_summary(per_layer, wandb_agg)

    # ── Save CSVs ───────────────────────────────────────────────────────────
    per_pair.to_csv(out_dir / "per_pair_map.csv", index=False)
    per_layer.to_csv(out_dir / "per_layer_summary.csv", index=False)
    summary.to_csv(out_dir / "cross_encoder_summary.csv", index=False)
    if not wandb_agg.empty:
        wandb_agg.to_csv(out_dir / "wandb_aggregate.csv", index=False)
    log.info("\nWrote CSVs:")
    for fname in (
        "per_pair_map.csv",
        "per_layer_summary.csv",
        "cross_encoder_summary.csv",
        "wandb_aggregate.csv",
    ):
        if (out_dir / fname).exists():
            log.info(
                "  %s (%d rows)", out_dir / fname, sum(1 for _ in (out_dir / fname).open()) - 1
            )

    # ── Plots ───────────────────────────────────────────────────────────────
    log.info("\nWriting plots:")
    for enc in summary["encoder"]:
        if enc in AUDIO_ENCODERS:
            plot_dashboard_for_encoder(enc, per_pair, per_layer, wandb_agg, out_dir)
    plot_cross_encoder_comparison(summary, per_layer, wandb_agg, out_dir)

    # ── Stdout summary ──────────────────────────────────────────────────────
    log.info("\n══════════════════════════════════════════════════════════════")
    log.info("Cross-encoder summary")
    log.info("══════════════════════════════════════════════════════════════")
    fmt = pd.option_context("display.max_columns", None, "display.width", 160)
    with fmt:
        print()
        print(summary.to_string(index=False))
        print()
        print("Per-layer headline metrics (audio encoders):")
        print(
            per_layer[
                ["encoder", "layer", "aggregate_map", "cross_instr_map_mean", "same_instr_map_mean"]
            ]
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
