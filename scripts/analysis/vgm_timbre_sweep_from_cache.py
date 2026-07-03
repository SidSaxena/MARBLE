#!/usr/bin/env python
"""Offline VGMIDITVar-timbre layer sweep straight from the embedding cache.

Why this exists: the live ``cli.py test`` path pays, PER LAYER, a ~2.5 min
Lightning cache-read plus a ~50 min CPU-bound metric stage (up to four
separate full N×N streaming passes whose aggregation ran on CPU). But the
cache already stores the FULL layer stack — each clip ``.pt`` is
``(L, H) = (13, 1024)`` time-pooled — so one cache read serves every layer,
and the fused GPU-resident pass (``marble/utils/retrieval_fused.py``)
computes base + grid + varctl + score metrics in ONE chunked pass per
geometry with aggregation on the GPU. Net: all 13 layers ≈ well under an
hour instead of ~13 h serial.

Pipeline parity: replicates ``CoverRetrievalTask.on_test_epoch_end`` exactly —
clip→file mean + L2-norm, raw/centered/whitened geometries, identical metric
keys, identical artifacts (condition_grid.csv, retrieval_score_*), one wandb
run per layer following the sweep naming convention. ``variation`` /
``gm_program`` come straight from the jsonl (no filename regex needed).

Validation: ``--oracle`` runs layer 11 FIRST and asserts every core metric
against the audited live-run numbers (wandb run di81drwb, 2026-07-03) before
any other layer is computed. Unit-level equivalence is pinned by
``tests/test_retrieval_fused.py``.

Example (PC, repo root):
    .venv/bin/python scripts/analysis/vgm_timbre_sweep_from_cache.py \
        --cache-dir output/.emb_cache/MuQ/VGMIDITVar-timbre__005f6eeb \
        --jsonl data/VGMIDITVar-timbre/VGMIDITVar.jsonl \
        --layers 0-12 --oracle --wandb-mode online
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marble.tasks.Covers80.probe import auto_whiten_params  # noqa: E402
from marble.utils.emb_cache import make_clip_id  # noqa: E402
from marble.utils.path_compat import load_jsonl  # noqa: E402
from marble.utils.retrieval_fused import fused_retrieval_pass  # noqa: E402
from marble.utils.retrieval_metrics import anisotropy_metrics, zca_whiten  # noqa: E402

# ── Oracle: audited MuQ-L11 live-run numbers (wandb di81drwb, 2026-07-03) ──
ORACLE_LAYER = 11
ORACLE_EXPECTED: dict[str, float] = {
    "test/map": 0.16983793675899506,
    "test/map_centered": 0.1777883619070053,
    "test/map_whitened": 0.3815738558769226,
    "test/map_whitened_alpha": 1.0,
    "test/map_whitened_eps_rel": 0.0,
    "test/recall@10": 0.16281619668006897,
    "test/r_precision": 0.2370862513780594,
    "test/median_rank": 3.0,
    "test/map_same_condition": 0.2899128496646881,
    "test/map_cross_condition": 0.47683149576187134,
    "test/condition_gap": -0.18691864609718323,
    "test/map_same_condition_varctl": 0.2899128496646881,
    "test/map_cross_condition_varctl": 0.2174838185310364,
    "test/condition_gap_varctl": 0.07242902368307114,
    "test/map_same_condition_varctl_n": 102960.0,
    "test/map_cross_condition_varctl_n": 720720.0,
    "test/score_sep_overall": 0.4963960647583008,
    "test/score_sep_within": 0.5298025608062744,
    "test/score_sep_cross": 0.5011856555938721,
    "test/score_sep_overall_varctl": 0.4373486340045929,
    "test/score_sep_within_varctl": 0.5298025608062744,
    "test/score_sep_cross_varctl": 0.4241409301757813,
    "test/anisotropy/mean_vec_norm": 0.8761718273162842,
    "test/anisotropy/avg_pair_cos": 0.76800537109375,
    "test/anisotropy/top1_sv_share": 0.11518090963363647,
    "test/anisotropy/effective_rank": 78.53166961669922,
}
# Anisotropy uses seeded random pairs + SVD on slightly different fp paths;
# effective_rank is scale ~80 so it gets a wider absolute band.
ORACLE_TOL: dict[str, float] = {
    "test/median_rank": 1.0,
    "test/map_same_condition_varctl_n": 0.0,
    "test/map_cross_condition_varctl_n": 0.0,
    "test/anisotropy/effective_rank": 0.5,
    "test/anisotropy/avg_pair_cos": 5e-3,
}
ORACLE_DEFAULT_TOL = 2.5e-3


def _parse_layers(spec: str) -> list[int | str]:
    out: list[int | str] = []
    for part in spec.split(","):
        part = part.strip()
        if part == "meanall":
            out.append("meanall")
        elif "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


# ── Stage 1: consolidated per-file embeddings from the clip cache ─────────


def build_or_load_file_embs(
    jsonl_path: str, cache_dir: Path, consolidated: Path, n_load_threads: int = 16
):
    """Return (rows, embs_all (N, L, H) fp32). Loads the consolidated tensor
    if present, else assembles it from the per-clip cache and saves it."""
    rows = load_jsonl(jsonl_path)
    n = len(rows)
    if consolidated.exists():
        print(f"[from-cache] loading consolidated embeddings {consolidated}")
        blob = torch.load(consolidated, map_location="cpu")
        assert blob["n_rows"] == n, "consolidated tensor does not match jsonl length"
        return rows, blob["embs"]

    print(f"[from-cache] scanning cache dir {cache_dir}")
    t0 = time.time()
    groups: dict[str, list[str]] = {}
    with os.scandir(cache_dir) as it:
        for e in it:
            if not e.name.endswith(".pt"):
                continue
            prefix = e.name[: -len(".pt")].rsplit("__c", 1)[0]
            groups.setdefault(prefix, []).append(e.path)
    print(
        f"[from-cache] {sum(len(v) for v in groups.values())} clip files, "
        f"{len(groups)} file groups ({time.time() - t0:.1f}s)"
    )

    # Map each jsonl row to its clip files via the exact live-path key fn.
    file_lists: list[list[str]] = []
    missing = 0
    for r in rows:
        prefix = make_clip_id(r["audio_path"], 0).rsplit("__c", 1)[0]
        fl = groups.get(prefix)
        if not fl:
            missing += 1
            file_lists.append([])
        else:
            file_lists.append(sorted(fl))
    if missing:
        raise SystemExit(
            f"[from-cache] FATAL: {missing}/{n} jsonl rows have no cached clips — "
            "cache/key mismatch, refusing to continue."
        )

    # Peek one tensor for (L, H).
    probe = torch.load(file_lists[0][0], map_location="cpu")
    probe_t = probe["embedding"] if isinstance(probe, dict) else probe
    L, H = probe_t.shape
    print(f"[from-cache] clip tensors are (L={L}, H={H}); assembling {n} file embeddings")

    embs_all = torch.empty(n, L, H, dtype=torch.float32)

    def load_one(i: int) -> None:
        ts = []
        for fp in file_lists[i]:
            t = torch.load(fp, map_location="cpu")
            ts.append(t["embedding"] if isinstance(t, dict) else t)
        embs_all[i] = torch.stack(ts).mean(0)  # mean over clips (probe: mean THEN normalize)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_load_threads) as ex:
        for done, _ in enumerate(ex.map(load_one, range(n), chunksize=256), start=1):
            if done % 20000 == 0:
                print(f"[from-cache]   {done}/{n} files ({time.time() - t0:.0f}s)")
    print(f"[from-cache] assembled in {time.time() - t0:.0f}s; saving {consolidated}")
    torch.save({"n_rows": n, "embs": embs_all}, consolidated)
    return rows, embs_all


# ── Stage 2: per-layer metric computation (probe-parity) ──────────────────


def evaluate_layer(
    embs_file: torch.Tensor,  # (N, H) — file-level, UNnormalised clip-mean
    work_ids: torch.Tensor,
    conditions: torch.Tensor,
    variations: torch.Tensor,
    *,
    extended: bool,
    device: str,
    batch: int,
    score_n_bins: int = 50,
) -> tuple[dict[str, float], dict, dict, dict]:
    """Compute the full metric suite for one layer. Returns
    (metrics_dict, grid_cells, varctl_cells, scores_result)."""
    N, H = embs_file.shape
    m: dict[str, float] = {}

    embs = F.normalize(embs_file, dim=-1)  # probe: normalize(mean(clips))

    # Anisotropy diagnostics (probe logs these on the raw normalized embs).
    ani = anisotropy_metrics(embs)
    for k, v in ani.items():
        m[f"test/anisotropy/{k}"] = float(v)

    # Base-metric kwargs — replicate the probe's raw/centered/whitened sets.
    recall_ks_raw = [10] if N > 10 else []
    if extended:
        recall_ks_raw = sorted(set(recall_ks_raw + [k for k in (1, 5, 50, 100) if k < N]))
    raw_kwargs = dict(
        recall_ks=recall_ks_raw,
        hit_ks=[k for k in (1, 5, 10) if k < N] if extended else [],
        include_r_precision=True,
        include_median_rank=True,
        include_map=True,
        map_at_ks=[1] if extended and N > 1 else [],
        include_mrr=extended,
    )
    ext_kwargs = dict(
        recall_ks=[k for k in (1, 5, 10, 50, 100) if k < N] if extended else [],
        hit_ks=[k for k in (1, 5, 10) if k < N] if extended else [],
        include_r_precision=extended,
        include_median_rank=extended,
        include_map=True,
        map_at_ks=[1] if extended and N > 1 else [],
        include_mrr=extended,
    )

    def _log_base(prefix_suffix: str, res: dict) -> None:
        """Replicate the probe's per-geometry logging key layout."""
        sfx = prefix_suffix
        m[f"test/map{sfx}"] = res["map"]
        for k in (1, 5, 10, 50, 100):
            if f"recall@{k}" in res:
                m[f"test/recall@{k}{sfx}"] = res[f"recall@{k}"]
        for k in (1, 5, 10):
            if f"hit_rate@{k}" in res:
                m[f"test/hit_rate@{k}{sfx}"] = res[f"hit_rate@{k}"]
        if "r_precision" in res:
            m[f"test/r_precision{sfx}"] = res["r_precision"]
        if "median_rank" in res:
            m[f"test/median_rank{sfx}"] = res["median_rank"]
        if "map@1" in res:
            m[f"test/map@1{sfx}"] = res["map@1"]
        if "mrr" in res:
            m[f"test/mrr{sfx}"] = res["mrr"]

    # ── raw geometry: base metrics only ──
    t0 = time.time()
    raw = fused_retrieval_pass(
        embs,
        work_ids,
        base_kwargs=raw_kwargs,
        with_grid=False,
        with_varctl=False,
        with_scores=False,
        device=device,
        batch=batch,
    )
    _log_base("", raw["base"])
    t_raw = time.time() - t0

    # ── centered geometry: base + grid + varctl + scores in ONE pass ──
    t0 = time.time()
    embs_c = F.normalize(embs - embs.mean(dim=0, keepdim=True), dim=-1)
    cen = fused_retrieval_pass(
        embs_c,
        work_ids,
        conditions=conditions,
        variations=variations,
        base_kwargs=ext_kwargs,
        with_grid=True,
        with_varctl=True,
        with_scores=True,
        score_n_bins=score_n_bins,
        device=device,
        batch=batch,
    )
    _log_base("_centered", cen["base"])
    t_cen = time.time() - t0

    # ── whitened geometry: base metrics only ──
    t0 = time.time()
    n_works = int(torch.unique(work_ids).numel())
    w_alpha, w_eps_rel = auto_whiten_params(n_works, N, H)
    m["test/map_whitened_alpha"] = float(w_alpha)
    m["test/map_whitened_eps_rel"] = float(w_eps_rel)
    embs_w = F.normalize(zca_whiten(embs, alpha=w_alpha, eps_rel=w_eps_rel), dim=-1)
    wht = fused_retrieval_pass(
        embs_w,
        work_ids,
        base_kwargs=ext_kwargs,
        with_grid=False,
        with_varctl=False,
        with_scores=False,
        device=device,
        batch=batch,
    )
    _log_base("_whitened", wht["base"])
    t_wht = time.time() - t0

    # ── condition-grid summaries (probe parity: mean over cells with n>0) ──
    grid = cen["grid"] or {}
    varctl = cen["grid_varctl"] or {}
    unique_conds = sorted({q for q, _ in grid})

    def _summarize(cells: dict, key_sfx: str, with_n: bool) -> None:
        same, cross = [], []
        same_n = cross_n = 0
        for (q, t), (ap, nq) in cells.items():
            if nq == 0:
                continue
            if q == t:
                same.append(ap)
                same_n += nq
            else:
                cross.append(ap)
                cross_n += nq
        if with_n:
            m[f"test/map_same_condition{key_sfx}_n"] = float(same_n)
            m[f"test/map_cross_condition{key_sfx}_n"] = float(cross_n)
        if same:
            m[f"test/map_same_condition{key_sfx}"] = float(sum(same) / len(same))
        if cross:
            m[f"test/map_cross_condition{key_sfx}"] = float(sum(cross) / len(cross))
        if same and cross:
            m[f"test/condition_gap{key_sfx}"] = float(
                sum(same) / len(same) - sum(cross) / len(cross)
            )

    if grid:
        _summarize(grid, "", with_n=False)
        for q in unique_conds:
            for t in unique_conds:
                ap, nq = grid.get((q, t), (0.0, 0))
                if nq == 0:
                    continue
                m[f"test/map_grid/{q}_to_{t}"] = ap
                m[f"test/map_grid/{q}_to_{t}_n"] = float(nq)
    if varctl:
        _summarize(varctl, "_varctl", with_n=True)

    # ── score-distribution summaries (probe parity) ──
    scores = cen["scores"]
    if scores is not None:
        ov = scores["overall"]
        if ov["separation"] is not None:
            m["test/score_sep_overall"] = float(ov["separation"])
        if ov["separation_varctl"] is not None:
            m["test/score_sep_overall_varctl"] = float(ov["separation_varctl"])
        for name, field in (
            ("within", "separation"),
            ("cross", "separation"),
            ("within_varctl", "separation_varctl"),
            ("cross_varctl", "separation_varctl"),
        ):
            diag = name.startswith("within")
            vals = [
                v[field]
                for k, v in scores["cells"].items()
                if (k[0] == k[1]) == diag and v[field] is not None
            ]
            if vals:
                m[f"test/score_sep_{name}"] = float(sum(vals) / len(vals))

    print(
        f"    pass timings: raw {t_raw:.0f}s | centered(fused) {t_cen:.0f}s | whitened {t_wht:.0f}s"
    )
    return m, grid, varctl, scores


# ── Artifacts (formats copied from the probe's writers) ───────────────────


def write_artifacts(out_dir: Path, grid: dict, varctl: dict, scores: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_conds = sorted({q for q, _ in grid}) if grid else []
    if grid:
        with open(out_dir / "condition_grid.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_program", "target_program", "map", "n_queries"])
            for q in unique_conds:
                for t in unique_conds:
                    ap, nq = grid.get((q, t), (0.0, 0))
                    w.writerow([q, t, f"{ap:.6f}", nq])
    if varctl:
        with open(out_dir / "condition_grid_varctl.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_program", "target_program", "map", "n_queries"])
            for q in unique_conds:
                for t in unique_conds:
                    ap, nq = varctl.get((q, t), (0.0, 0))
                    w.writerow([q, t, f"{ap:.6f}", nq])
    if scores is not None:
        json_res = {
            "overall": scores["overall"],
            "cells": {f"{q}_to_{t}": v for (q, t), v in scores["cells"].items()},
        }
        with open(out_dir / "retrieval_score_distributions.json", "w", encoding="utf-8") as f:
            json.dump(json_res, f, indent=2)
        with open(out_dir / "retrieval_score_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "cell",
                    "n_relevant",
                    "n_relevant_diffvar",
                    "n_distractor",
                    "mean_relevant",
                    "mean_relevant_diffvar",
                    "mean_distractor",
                    "separation",
                    "separation_varctl",
                ]
            )

            def _f(x):
                return "" if x is None else f"{x:.6f}"

            rows_out = [("overall", scores["overall"])] + sorted(
                ((f"{q}_to_{t}", v) for (q, t), v in scores["cells"].items()),
                key=lambda x: x[0],
            )
            for name, v in rows_out:
                w.writerow(
                    [
                        name,
                        v["relevant"]["n"],
                        v["relevant_diffvar"]["n"],
                        v["distractor"]["n"],
                        _f(v["relevant"]["mean"]),
                        _f(v["relevant_diffvar"]["mean"]),
                        _f(v["distractor"]["mean"]),
                        _f(v["separation"]),
                        _f(v["separation_varctl"]),
                    ]
                )


def check_oracle(m: dict[str, float]) -> bool:
    print("\n[oracle] comparing layer 11 against the audited live-run numbers:")
    ok = True
    for k, exp in ORACLE_EXPECTED.items():
        tol = ORACLE_TOL.get(k, ORACLE_DEFAULT_TOL)
        got = m.get(k)
        if got is None:
            print(f"  MISSING {k} (expected {exp})")
            ok = False
            continue
        d = abs(got - exp)
        status = "OK " if d <= tol else "FAIL"
        if d > tol:
            ok = False
        print(f"  {status} {k}: got {got:.6f} exp {exp:.6f} |d|={d:.2e} tol={tol:g}")
    print(f"[oracle] {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", default="data/VGMIDITVar-timbre/VGMIDITVar.jsonl")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--layers", default="0-12")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--extended", action="store_true", default=True)
    ap.add_argument("--no-extended", dest="extended", action="store_false")
    ap.add_argument("--oracle", action="store_true", help="validate layer 11 first; abort on FAIL")
    ap.add_argument("--consolidated", default=None)
    ap.add_argument("--out-dir", default="output/vgm_timbre_from_cache")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="marble")
    ap.add_argument("--wandb-group", default="MuQ / VGMIDITVar-timbre")
    ap.add_argument("--load-threads", type=int, default=16)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    consolidated = (
        Path(args.consolidated) if args.consolidated else cache_dir / "consolidated_file_embs.pt"
    )
    layers = _parse_layers(args.layers)
    if args.oracle and ORACLE_LAYER in layers:
        layers = [ORACLE_LAYER] + [x for x in layers if x != ORACLE_LAYER]
    elif args.oracle:
        layers = [ORACLE_LAYER] + layers

    rows, embs_all = build_or_load_file_embs(
        args.jsonl, cache_dir, consolidated, n_load_threads=args.load_threads
    )
    work_ids = torch.tensor([int(r["work_id"]) for r in rows])
    conditions = torch.tensor([int(r.get("gm_program", -1)) for r in rows])
    variations = torch.tensor([int(r.get("variation", -1)) for r in rows])
    print(
        f"[from-cache] N={len(rows)} works={int(torch.unique(work_ids).numel())} "
        f"conds={sorted(set(conditions.tolist()))} layers={layers} device={args.device}"
    )

    for li, layer in enumerate(layers):
        t0 = time.time()
        print(f"\n===== layer {layer} ({li + 1}/{len(layers)}) =====")
        # "meanall" = LayerSelector mode=mean over the full stack. Mean over
        # layers commutes with the clip-mean, so layer-mean of the cached
        # per-clip stack == the live meanall pipeline.
        emb_slice = embs_all.mean(dim=1) if layer == "meanall" else embs_all[:, int(layer), :]
        metrics, grid, varctl, scores = evaluate_layer(
            emb_slice,
            work_ids,
            conditions,
            variations,
            extended=args.extended,
            device=args.device,
            batch=args.batch,
        )
        headline = {
            k: metrics.get(k)
            for k in (
                "test/map",
                "test/map_centered",
                "test/map_whitened",
                "test/map_same_condition",
                "test/map_cross_condition",
                "test/condition_gap",
                "test/map_cross_condition_varctl",
                "test/condition_gap_varctl",
            )
        }
        print(
            "    "
            + " | ".join(
                f"{k.split('/')[-1]}={v:.4f}" for k, v in headline.items() if v is not None
            )
        )

        if args.oracle and layer == ORACLE_LAYER and not check_oracle(metrics):
            raise SystemExit("[oracle] FAILED — not proceeding to other layers.")

        layer_dir = Path(args.out_dir) / f"layer{layer}"
        write_artifacts(layer_dir, grid, varctl, scores)
        with open(layer_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=1, sort_keys=True)

        if args.wandb_mode != "disabled":
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                name=f"layer-{layer}-test",
                job_type="test",
                tags=["MuQ", "VGMIDITVar-timbre", "layer-sweep", "probe", "varctl", "from-cache"],
                config={
                    "layer": layer,
                    "from_cache": True,
                    "extended": args.extended,
                    "batch": args.batch,
                    "device": args.device,
                    "cache_dir": str(cache_dir),
                },
                mode=args.wandb_mode,
                reinit=True,
            )
            wandb.log(metrics)
            for art in layer_dir.iterdir():
                wandb.save(str(art), base_path=str(layer_dir))
            run.finish()
        print(f"    layer {layer} done in {time.time() - t0:.0f}s")

    print("\n=== VGM_TIMBRE_FROM_CACHE_SWEEP_DONE ===")


if __name__ == "__main__":
    main()
