"""Cross-encoder VGMIDITVar-timbre layer-sweep analysis.

Pulls per-layer condition_grid.csv and wandb-summary.json from
``output/probe.VGMIDITVar-timbre.*.layer*/wandb/run-*/files/`` for any
encoder that has finished a sweep, then renders a unified set of
comparison figures and a leaderboard CSV.

Designed to be run repeatedly as new encoders complete — just pass
``--encoder-tag`` for each you want included. When multiple wandb run
dirs exist per layer (from crashed retries), automatically picks the
one whose summary has actual ``test/*`` keys.

Outputs (under ``--out-dir``):
  per_encoder_curves.png   — MAP same/cross/gap, recall, anisotropy curves
  best_layer_grids.png     — 8×8 heatmaps of each encoder's best layer
  per_instrument.png       — per-GM-program diagonal MAP, best layer per encoder
  best_layer_asymmetry.png — cell[q,t] - cell[t,q] heatmaps
  per_layer_tables.txt     — full per-layer table per encoder
  encoder_leaderboard.csv  — one-row-per-encoder leaderboard

Usage (from repo root):

    uv run python scripts/analysis/compare_encoders_vgmiditvar_timbre.py \
        --encoder-tag CLaMP3-layers \
        --encoder-tag MERT-v1-95M-layers \
        --encoder-tag MuQ \
        --encoder-tag OMARRQ-multifeature-25hz \
        --out-dir docs/figures/vgmiditvar_timbre_4enc

Encoder tags are the directory-name portion between
``probe.VGMIDITVar-timbre.`` and ``.layer<N>`` (or just ``.layer<N>`` for
MuQ-style naming with no ``-layers`` suffix). The script auto-handles
both naming conventions.

Display names default to a sensible mapping; pass
``--display-name CLaMP3-layers:CLaMP3`` to override.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GM program codes used in the VGMIDITVar-timbre rendering pipeline.
GM_NAMES = {
    0: "Piano",
    24: "Guitar",
    48: "Strings",
    52: "Choir",
    60: "Horn",
    73: "Flute",
    80: "Lead",
    89: "Pad",
}
GM_ORDER = [0, 24, 48, 52, 60, 73, 80, 89]

# Default tag → display-name mapping. Override via --display-name on the CLI.
DEFAULT_DISPLAY_NAMES = {
    "CLaMP3-layers": "CLaMP3",
    "MERT-v1-95M-layers": "MERT-v1-95M",
    "MuQ": "MuQ",
    "OMARRQ-multifeature-25hz": "OMARRQ-25hz",
}


def find_layer_dirs(root: Path, enc_tag: str) -> dict[int, Path]:
    """Map layer_idx → wandb run-files dir (latest valid one if duplicates)."""
    out: dict[int, list[Path]] = defaultdict(list)
    # Two naming conventions seen in practice:
    #   - probe.VGMIDITVar-timbre.CLaMP3-layers.layer<N>   (CLaMP3, MERT)
    #   - probe.VGMIDITVar-timbre.MuQ.layer<N>             (MuQ)
    # The tag matches the ``.layer<N>`` boundary in both cases.
    pat = re.compile(rf"VGMIDITVar-timbre\.{re.escape(enc_tag)}\.layer(\d+)")
    for d in root.glob(f"probe.VGMIDITVar-timbre.{enc_tag}.layer*/wandb/run-*/files"):
        m = pat.search(str(d))
        if m:
            out[int(m.group(1))].append(d)
    # For each layer, pick the most recent run dir whose summary actually
    # has test/* keys (skips crashed-pre-test attempts).
    result: dict[int, Path] = {}
    for layer, dirs in out.items():
        best = None
        for d in sorted(dirs, reverse=True):
            sj = d / "wandb-summary.json"
            if not sj.exists():
                continue
            try:
                data = json.loads(sj.read_text())
            except Exception:
                continue
            if any(k.startswith("test/") for k in data):
                best = d
                break
        if best is None and dirs:
            best = sorted(dirs)[-1]
        if best is not None:
            result[layer] = best
    return dict(sorted(result.items()))


def load_layer(d: Path) -> dict:
    df = pd.read_csv(d / "condition_grid.csv")
    grid = np.zeros((8, 8))
    n_grid = np.zeros((8, 8), dtype=int)
    for _, r in df.iterrows():
        qi = GM_ORDER.index(int(r["query_program"]))
        ti = GM_ORDER.index(int(r["target_program"]))
        grid[qi, ti] = r["map"]
        n_grid[qi, ti] = int(r["n_queries"])
    summary = json.loads((d / "wandb-summary.json").read_text())
    return {"grid": grid, "n_grid": n_grid, "summary": summary}


def aggregate(grid: np.ndarray) -> tuple[float, float, float]:
    """Return (diag_mean, off_diag_mean, gap=diag−off)."""
    d = float(np.diag(grid).mean())
    o = float((grid.sum() - np.diag(grid).sum()) / 56)
    return d, o, d - o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--encoder-tag",
        action="append",
        required=True,
        help="Encoder directory tag (between 'VGMIDITVar-timbre.' and "
        "'.layer<N>'). Pass multiple times to include multiple encoders.",
    )
    ap.add_argument(
        "--display-name",
        action="append",
        default=[],
        help="Override display name for an encoder, format 'tag:Display Name'. "
        "Pass multiple times. Falls back to a default mapping for known tags.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("output"),
        help="Root dir to search for probe.VGMIDITVar-timbre.* directories (default: ./output).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for figures and CSVs.",
    )
    args = ap.parse_args()

    display_map = dict(DEFAULT_DISPLAY_NAMES)
    for spec in args.display_name:
        tag, name = spec.split(":", 1)
        display_map[tag] = name

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all encoders ─────────────────────────────────────────────
    all_data: dict[str, dict] = {}
    for tag in args.encoder_tag:
        name = display_map.get(tag, tag)
        layers = find_layer_dirs(args.root, tag)
        if not layers:
            print(f"[{name}] no layers found in {args.root}; skipping")
            continue
        rows = []
        grids: dict[int, np.ndarray] = {}
        for layer_idx, d in layers.items():
            L = load_layer(d)
            diag, off, gap = aggregate(L["grid"])
            s = L["summary"]
            n = len(layers)
            rows.append(
                {
                    "encoder": name,
                    "layer": layer_idx,
                    "depth_frac": layer_idx / (n - 1) if n > 1 else 0,
                    "diag": diag,
                    "off": off,
                    "gap": gap,
                    "map_raw": s.get("test/map", np.nan),
                    "map_centered": s.get("test/map_centered", np.nan),
                    "recall@10": s.get("test/recall@10", np.nan),
                    "median_rank": s.get("test/median_rank", np.nan),
                    "r_precision": s.get("test/r_precision", np.nan),
                    "mean_vec_norm": s.get("test/anisotropy/mean_vec_norm", np.nan),
                    "effective_rank": s.get("test/anisotropy/effective_rank", np.nan),
                    "top1_sv_share": s.get("test/anisotropy/top1_sv_share", np.nan),
                    "avg_pair_cos": s.get("test/anisotropy/avg_pair_cos", np.nan),
                }
            )
            grids[layer_idx] = L["grid"]
        all_data[name] = {
            "df": pd.DataFrame(rows).sort_values("layer").reset_index(drop=True),
            "grids": grids,
        }
        print(f"[{name}] loaded {len(rows)} layers")

    if not all_data:
        print("no encoders loaded; nothing to do")
        return

    # ── Per-encoder leaderboard ───────────────────────────────────────
    leaderboard = []
    for name, info in all_data.items():
        df = info["df"]
        leaderboard.append(
            {
                "encoder": name,
                "best_off_layer": int(df.loc[df["off"].idxmax(), "layer"]),
                "max_off": float(df["off"].max()),
                "max_off_diag": float(df.loc[df["off"].idxmax(), "diag"]),
                "min_gap_layer": int(df.loc[df["gap"].idxmin(), "layer"]),
                "min_gap": float(df["gap"].min()),
                "best_map_layer": int(df.loc[df["map_centered"].idxmax(), "layer"]),
                "best_map_centered": float(df["map_centered"].max()),
                "best_map_raw": float(df["map_raw"].max()),
                "best_recall@10": float(df["recall@10"].max()),
                "best_r_precision": float(df["r_precision"].max()),
                "min_median_rank": float(df["median_rank"].min()),
                "max_eff_rank": float(df["effective_rank"].max()),
                "min_mean_vec_norm": float(df["mean_vec_norm"].min()),
            }
        )
    lb_df = pd.DataFrame(leaderboard)
    lb_df.to_csv(args.out_dir / "encoder_leaderboard.csv", index=False)
    print("\nLEADERBOARD")
    print(lb_df.to_string(index=False, float_format="%.4f"))

    # Color cycling for an arbitrary number of encoders.
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    enc_color = {name: colors[i % len(colors)] for i, name in enumerate(all_data)}
    enc_marker = {name: markers[i % len(markers)] for i, name in enumerate(all_data)}

    # ── Figure 1: Per-encoder trajectory curves ──────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(
        f"{' vs '.join(all_data.keys())} — VGMIDITVar-timbre layer sweep",
        fontsize=14,
        fontweight="bold",
    )

    panels = [
        ("off", "Cross-instrument MAP (off-diagonal mean)  ← key metric", "cross-condition MAP"),
        ("diag", "Within-instrument MAP (diagonal mean)", "same-condition MAP"),
        ("gap", "Timbre-dependence gap  ← lower = better cross-timbre", "MAP same − MAP cross"),
        ("recall@10", "Recall@10 (higher = surfacing relevants in top-10)", "recall@10"),
        ("mean_vec_norm", "Cone collapse (closer to 1 = worse)", "mean_vec_norm"),
        ("effective_rank", "Effective rank (higher = more diverse residual)", "effective_rank"),
    ]
    for ax, (col, title, ylabel) in zip(axes.flat, panels, strict=False):
        for name, info in all_data.items():
            df = info["df"]
            ax.plot(
                df["depth_frac"],
                df[col],
                marker=enc_marker[name],
                color=enc_color[name],
                linewidth=2,
                label=name,
                markersize=7,
            )
        ax.set_xlabel("layer depth (fraction)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
        if col == "gap":
            ax.axhline(0, color="black", linewidth=0.5)
        if col == "mean_vec_norm":
            ax.set_ylim(0, 1)
    plt.savefig(args.out_dir / "per_encoder_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Best-layer 8x8 heatmaps side by side ───────────────
    best_grids = {}
    for name, info in all_data.items():
        df = info["df"]
        best_layer = int(df.loc[df["off"].idxmax(), "layer"])
        best_grids[name] = (best_layer, info["grids"][best_layer])
    n_enc = len(best_grids)
    fig, axes = plt.subplots(1, n_enc, figsize=(6 * n_enc, 6), constrained_layout=True)
    if n_enc == 1:
        axes = [axes]
    fig.suptitle(
        "Best layer per encoder — cross-instrument MAP grid (rows=query, cols=target)",
        fontsize=13,
        fontweight="bold",
    )
    vmin, vmax = 0.0, max(g.max() for _, g in best_grids.values())
    for ax, (name, (best_layer, g)) in zip(axes, best_grids.items(), strict=False):
        im = ax.imshow(g, cmap="viridis", vmin=vmin, vmax=vmax)
        d, o, gap = aggregate(g)
        ax.set_title(
            f"{name}  L{best_layer}\ndiag={d:.3f}  off={o:.3f}  gap={gap:.3f}",
            fontsize=11,
        )
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([GM_NAMES[p] for p in GM_ORDER], rotation=45, ha="right")
        ax.set_yticklabels([GM_NAMES[p] for p in GM_ORDER])
        ax.set_xlabel("target")
        ax.set_ylabel("query")
        for i in range(8):
            for j in range(8):
                v = g[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.5 * vmax else "black",
                    fontsize=8,
                )
    fig.colorbar(im, ax=axes, shrink=0.7, label="MAP")
    plt.savefig(args.out_dir / "best_layer_grids.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Per-instrument diagonal MAP, best layer per encoder ─
    fig, ax = plt.subplots(1, 1, figsize=(max(11, 1.5 * n_enc * 4), 6))
    width = 0.85 / n_enc
    x = np.arange(len(GM_ORDER))
    for idx, (name, (best_layer, g)) in enumerate(best_grids.items()):
        diag_vals = np.diag(g)
        ax.bar(
            x + (idx - (n_enc - 1) / 2) * width,
            diag_vals,
            width=width,
            label=f"{name} (L{best_layer})",
            color=enc_color[name],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([GM_NAMES[p] for p in GM_ORDER], rotation=45, ha="right")
    ax.set_ylabel("within-timbre MAP (diagonal cell)")
    ax.set_title("Per-instrument within-timbre MAP at each encoder's best layer")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(args.out_dir / "per_instrument.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: Asymmetry of best-layer grids ──────────────────────
    fig, axes = plt.subplots(1, n_enc, figsize=(6 * n_enc, 6), constrained_layout=True)
    if n_enc == 1:
        axes = [axes]
    fig.suptitle(
        "Asymmetry of best-layer grids: cell[q,t] − cell[t,q]\n"
        "(red = q→t works better than t→q; ideally all near 0)",
        fontsize=13,
        fontweight="bold",
    )
    asyms = [(g - g.T) for _, g in best_grids.values()]
    amax = max(float(np.abs(a).max()) for a in asyms)
    for ax, ((name, (best_layer, _g)), asym) in zip(
        axes, zip(best_grids.items(), asyms, strict=False), strict=False
    ):
        im = ax.imshow(asym, cmap="RdBu_r", vmin=-amax, vmax=amax)
        ax.set_title(
            f"{name}  L{best_layer}\nmean |asymmetry| = {float(np.abs(asym).mean()):.3f}",
            fontsize=11,
        )
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.set_xticklabels([GM_NAMES[p] for p in GM_ORDER], rotation=45, ha="right")
        ax.set_yticklabels([GM_NAMES[p] for p in GM_ORDER])
        ax.set_xlabel("target")
        ax.set_ylabel("query")
    fig.colorbar(im, ax=axes, shrink=0.7, label="MAP[q,t] − MAP[t,q]")
    plt.savefig(args.out_dir / "best_layer_asymmetry.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Per-layer text tables ────────────────────────────────────────
    text_lines = []
    for name, info in all_data.items():
        df = info["df"]
        text_lines.append(f"\n=== {name} ===")
        cols = [
            "layer",
            "diag",
            "off",
            "gap",
            "map_raw",
            "map_centered",
            "recall@10",
            "median_rank",
            "r_precision",
            "mean_vec_norm",
            "effective_rank",
        ]
        text_lines.append(df[cols].to_string(index=False, float_format="%.4f"))
    (args.out_dir / "per_layer_tables.txt").write_text("\n".join(text_lines))

    print(f"\nfigures + tables written to {args.out_dir}")


if __name__ == "__main__":
    main()
