"""Aggregate the BPS-Motif within-piece WINDOW-SIZE breaking-point sweep.

Reads every ``wandb-summary.json`` under the per-N within-piece sweep output
dirs and produces, for each window size N (in bars):

  * the per-layer ``test/map`` (raw) curve,
  * the BEST-layer MAP (raw) + which layer + that layer's centered MAP,
  * the meanall MAP (raw + centered).

Then it builds the breaking-point analysis:

  1. a layer x N table of raw MAP,
  2. the **breaking-point curve** (best-layer raw MAP vs N) + the best layer per
     N (watch for the layer drifting deeper as N grows),
  3. a **layer x N heatmap** PNG,
  4. the breaking-point-curve PNG (best-layer MAP vs N, with the best layer
     annotated at each point, + the meanall line),

saves both PNGs under ``docs/`` (default) and, unless ``--no-wandb``, logs a
wandb analysis run (project ``marble``, group
``CLaMP3-symbolic-abc / BPSMotifWithinPiece-Nsweep``) carrying both plots + the
per-N best-layer table.

Output dir convention (set by the per-N configs' save_dir + gen_sweep_configs'
``.layer{L}`` patch):

  output/probe.BPSMotifWithinPieceN{N}.CLaMP3-symbolic-abc-layers.layer{L}/...
  output/probe.BPSMotifWithinPieceN{N}.CLaMP3-symbolic-abc-meanall/...

Usage::

    python scripts/sweeps/bps_within_piece_n_summary.py \
        --windows 1 2 4 6 8 12 16 24 32 \
        [--output-root output] [--docs-dir docs] [--no-wandb]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_LAYER_RX = re.compile(r"\.CLaMP3-symbolic-abc-layers\.layer(\d+)(?:\b|/)")


def _has_test_map(d: dict) -> bool:
    return "test/map" in d


def _load_summary(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def collect_layer_maps(output_root: Path, n: int) -> dict[int, dict]:
    """layer -> wandb summary dict (with test/map etc.) for window size N."""
    pat = str(
        output_root
        / f"probe.BPSMotifWithinPieceN{n}.CLaMP3-symbolic-abc-layers.layer*"
        / "wandb"
        / "run-*"
        / "files"
        / "wandb-summary.json"
    )
    out: dict[int, dict] = {}
    for p in sorted(glob.glob(pat)):
        m = _LAYER_RX.search(p)
        if not m:
            continue
        layer = int(m.group(1))
        d = _load_summary(p)
        if d is None or not _has_test_map(d):
            continue
        # Keep the latest complete run for a layer (sorted glob → last wins).
        out[layer] = d
    return out


def collect_meanall_map(output_root: Path, n: int) -> dict | None:
    pat = str(
        output_root
        / f"probe.BPSMotifWithinPieceN{n}.CLaMP3-symbolic-abc-meanall"
        / "wandb"
        / "run-*"
        / "files"
        / "wandb-summary.json"
    )
    best = None
    for p in sorted(glob.glob(pat)):
        d = _load_summary(p)
        if d is not None and _has_test_map(d):
            best = d
    return best


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "  --  "


def build_tables(windows: list[int], output_root: Path):
    """Return (per_n, all_layers) where per_n[N] = {layer: summary, 'meanall': summary|None}."""
    per_n: dict[int, dict] = {}
    all_layers: set[int] = set()
    for n in windows:
        layer_maps = collect_layer_maps(output_root, n)
        meanall = collect_meanall_map(output_root, n)
        per_n[n] = {"layers": layer_maps, "meanall": meanall}
        all_layers |= set(layer_maps)
    return per_n, sorted(all_layers)


def print_summary(windows, per_n, all_layers) -> list[str]:
    out: list[str] = []
    out.append("=== BPS-Motif within-piece WINDOW-SIZE sweep: test/map (raw) layer x N ===")
    header = "layer | " + " | ".join(f"N={n:<3}" for n in windows)
    out.append(header)
    out.append("-" * len(header))
    for l in all_layers:
        row = [f"  {l:>3} "]
        for n in windows:
            d = per_n[n]["layers"].get(l)
            row.append(_fmt(d.get("test/map") if d else None).rjust(6))
        out.append(" | ".join(row))
    # meanall row
    mrow = ["mean "]
    for n in windows:
        d = per_n[n]["meanall"]
        mrow.append(_fmt(d.get("test/map") if d else None).rjust(6))
    out.append("-" * len(header))
    out.append(" | ".join(mrow))

    out.append("\n=== Breaking-point: best layer per N ===")
    out.append("  N | n_windows | best_layer | best_MAP(raw) | best_MAP(centered) | meanall(raw)")
    best_curve: list[tuple[int, int, float]] = []
    for n in windows:
        layer_maps = per_n[n]["layers"]
        if not layer_maps:
            out.append(
                f"  {n:>2} |    --     |     --     |       --       |        --         |    --"
            )
            continue
        best_layer = max(layer_maps, key=lambda l: layer_maps[l].get("test/map", -1))
        bd = layer_maps[best_layer]
        # n_windows is logged by the probe print, not the summary; try the run
        # config/summary if present, else leave blank.
        nw = bd.get("test/n_windows") or bd.get("n_windows") or "--"
        meanall = per_n[n]["meanall"]
        out.append(
            f"  {n:>2} | {str(nw):>9} | {best_layer:>10} | "
            f"{_fmt(bd.get('test/map')):>14} | {_fmt(bd.get('test/map_centered')):>18} | "
            f"{_fmt(meanall.get('test/map') if meanall else None)}"
        )
        best_curve.append((n, best_layer, bd.get("test/map", float("nan"))))

    # breaking-point verdict
    if len(best_curve) >= 2:
        peak = max(best_curve, key=lambda t: t[2])
        out.append(
            f"\nPEAK: N={peak[0]} at layer {peak[1]} (MAP={peak[2]:.4f}). "
            "See the curve to judge plateau vs turnover."
        )
        layers_seq = [bl for _, bl, _ in best_curve]
        if layers_seq == sorted(layers_seq):
            out.append(
                "Best layer is (weakly) monotonic non-decreasing in N → layer DRIFTS DEEPER as windows grow."
            )
    return out


def plot_heatmap(windows, per_n, all_layers, docs_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    M = np.full((len(all_layers), len(windows)), np.nan)
    for j, n in enumerate(windows):
        for i, l in enumerate(all_layers):
            d = per_n[n]["layers"].get(l)
            if d and isinstance(d.get("test/map"), (int, float)):
                M[i, j] = d["test/map"]

    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(windows), 1.0 + 0.4 * len(all_layers)))
    im = ax.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"N={n}" for n in windows])
    ax.set_yticks(range(len(all_layers)))
    ax.set_yticklabels([str(l) for l in all_layers])
    ax.set_xlabel("window size N (bars)")
    ax.set_ylabel("CLaMP3 layer")
    ax.set_title("BPS-Motif within-piece (clip-isolated) raw MAP: layer x N")
    for i in range(len(all_layers)):
        for j in range(len(windows)):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if M[i, j] < np.nanmean(M) else "black",
                )
    fig.colorbar(im, ax=ax, label="test/map (raw)")
    fig.tight_layout()
    out = docs_dir / "bps_within_piece_n_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_breaking_point(windows, per_n, docs_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, best_map, best_layer, meanall_map = [], [], [], []
    for n in windows:
        layer_maps = per_n[n]["layers"]
        if not layer_maps:
            continue
        bl = max(layer_maps, key=lambda l: layer_maps[l].get("test/map", -1))
        xs.append(n)
        best_map.append(layer_maps[bl].get("test/map"))
        best_layer.append(bl)
        ma = per_n[n]["meanall"]
        meanall_map.append(ma.get("test/map") if ma else None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, best_map, "o-", label="best-layer MAP (raw)", color="C0")
    for x, y, l in zip(xs, best_map, best_layer, strict=False):
        if y is not None:
            ax.annotate(
                f"L{l}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                color="C0",
            )
    if any(m is not None for m in meanall_map):
        mm_x = [x for x, m in zip(xs, meanall_map, strict=False) if m is not None]
        mm_y = [m for m in meanall_map if m is not None]
        ax.plot(mm_x, mm_y, "s--", label="meanall MAP (raw)", color="C1", alpha=0.7)
    ax.set_xlabel("window size N (bars)")
    ax.set_ylabel("within-movement same-motif MAP (raw)")
    ax.set_title("BPS-Motif within-piece breaking-point curve (best-layer MAP vs N)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(n) for n in xs])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = docs_dir / "bps_within_piece_n_breaking_point.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12, 16, 24, 32])
    ap.add_argument("--output-root", default=str(REPO / "output"))
    ap.add_argument("--docs-dir", default=str(REPO / "docs"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    windows = sorted(set(args.windows))
    output_root = Path(args.output_root)
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    per_n, all_layers = build_tables(windows, output_root)
    lines = print_summary(windows, per_n, all_layers)
    text = "\n".join(lines)
    print(text)
    (docs_dir / "bps_within_piece_n_summary.txt").write_text(text + "\n")

    heatmap = plot_heatmap(windows, per_n, all_layers, docs_dir)
    curve = plot_breaking_point(windows, per_n, docs_dir)
    print(f"\nplots:\n  {heatmap}\n  {curve}")

    if not args.no_wandb:
        try:
            import wandb

            run = wandb.init(
                project="marble",
                group="CLaMP3-symbolic-abc / BPSMotifWithinPiece-Nsweep",
                job_type="analysis",
                name="within-piece-Nsweep-summary",
                tags=["CLaMP3-symbolic-abc", "within-piece", "window-size-sweep", "analysis"],
            )
            run.log(
                {
                    "breaking_point_curve": wandb.Image(str(curve)),
                    "layer_x_N_heatmap": wandb.Image(str(heatmap)),
                }
            )
            # per-N best-layer table
            tbl = wandb.Table(
                columns=["N", "best_layer", "best_map_raw", "best_map_centered", "meanall_raw"]
            )
            for n in windows:
                lm = per_n[n]["layers"]
                if not lm:
                    continue
                bl = max(lm, key=lambda l: lm[l].get("test/map", -1))
                ma = per_n[n]["meanall"]
                tbl.add_data(
                    n,
                    bl,
                    lm[bl].get("test/map"),
                    lm[bl].get("test/map_centered"),
                    ma.get("test/map") if ma else None,
                )
            run.log({"per_N_best_layer": tbl})
            print(f"wandb analysis run: {run.url}")
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"(wandb logging skipped: {e})")


if __name__ == "__main__":
    main()
