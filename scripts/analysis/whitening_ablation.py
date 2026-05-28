#!/usr/bin/env python3
"""
scripts/analysis/whitening_ablation.py
──────────────────────────────────────
Whitening / dimensionality-postprocessing ablation for the VGMIDITVar-
timbre retrieval task.

Given one (encoder, layer) pair with a populated embedding cache, this
script reproduces the probe's per-file L2-normalised embedding matrix,
fits PCA on it, and re-runs the streaming MAP + per-condition grid
under several **embedding transformations**:

    raw            — just L2-normalise (sanity baseline).
    centered       — subtract the corpus mean, L2-normalise.
    abtt-K         — subtract the projection onto the top-K principal
                     components of the centered data (Mu & Viswanath
                     ICLR 2018, "All But The Top"), then L2-normalise.
                     K ∈ {1, 3, 10} are the standard ablation choices.
    whiten-aα      — fractional ZCA whitening with exponent α ∈ [0, 1]:
                     ``e_w = U Λ^(−α/2) U^T (e − μ)``. α=0 collapses to
                     ``centered``; α=1 is full whitening (residual cov
                     = I before the L2-norm step). Default sweep:
                     α ∈ {0.5, 0.7, 1.0}.
    whiten-a1.0-erel-E
                   — full whitening with relative Tikhonov ridge:
                     ``e_w = U (Λ + E · λ_max · I)^(−1/2) U^T (e − μ)``.
                     Tames the noise-amplification risk on small
                     eigenvalues. E ∈ {1e-3, 1e-2} are reasonable.

After the transform, every variant is **L2-normalised** before the
metric pass. This means "whitening" here is a cosine-retrieval
preprocessing step (the post-norm vectors do NOT have identity
covariance — they live on the sphere); the meaningful effect is the
relative reweighting of principal directions before normalisation.

Crucial correctness gate: ``--verify`` cross-checks the ``raw`` and
``centered`` MAP against the matching wandb-summary.json from the
original sweep. The script aborts if the delta exceeds ``5e-4`` —
proving the load/aggregate/metric pipeline matches the live probe.

Outputs one CSV row per treatment with metric values + provenance
(encoder_model_id, config_hash, λ_max, λ_min, effective_rank).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from marble.utils.emb_cache import make_clip_id  # noqa: E402
from marble.utils.path_compat import load_jsonl  # noqa: E402
from marble.utils.retrieval_metrics import (  # noqa: E402
    anisotropy_metrics,
    compute_perpair_map_all_streaming,
    compute_retrieval_metrics_streaming,
)

# ──────────────────────────────────────────────────────────────────────
# Aggregation — mirrors the probe's per-file embedding construction
# ──────────────────────────────────────────────────────────────────────


def _load_per_file_embeddings(
    records: list[dict],
    cache_dir: Path,
    layer_idx: int,
) -> tuple[torch.Tensor, list[int], list[int], int, int]:
    """Return ``(embs, work_ids, conditions, n_files, n_missing)``.

    Matches the probe's ``on_test_epoch_end`` aggregation:

    1. For each JSONL record, glob all cached clip files for that audio
       path: ``cache_dir / "<stem>__<sha1>__c*.pt"``.
    2. Each ``.pt`` stores ``{"embedding": tensor}`` with shape
       ``(L, H)`` — full hidden-state tuple from the encoder, post-
       TimeAvgPool. Slice ``[layer_idx]`` → ``(H,)``.
    3. **L2-normalise EACH clip embedding first** (matches probe
       forward, marble/tasks/Covers80/probe.py line 138).
    4. Stack per-file clips → mean over clips → re-L2-normalise per
       file (matches probe on_test_epoch_end line 221).

    Records whose cache glob is empty are skipped and counted in
    ``n_missing``. Conditions follow the same fallback chain as the
    datamodule: ``gm_program → soundfont_id → -1``.
    """
    file_embs: list[torch.Tensor] = []
    file_wids: list[int] = []
    file_conds: list[int] = []
    n_missing = 0
    n_total = 0

    for rec in records:
        n_total += 1
        # Clip id format: <stem>__<sha1(audio_path)[:8]>__c<slice_idx>
        base_id = make_clip_id(rec["audio_path"], 0).rsplit("__c", 1)[0]
        slice_paths = sorted(cache_dir.glob(f"{base_id}__c*.pt"))
        if not slice_paths:
            n_missing += 1
            continue

        per_slice_unit_norm: list[torch.Tensor] = []
        for p in slice_paths:
            blob = torch.load(p, map_location="cpu", weights_only=True)
            if isinstance(blob, dict):
                t = blob.get("embedding")
                if t is None:
                    for v in blob.values():
                        if isinstance(v, torch.Tensor):
                            t = v
                            break
                if t is None:
                    continue
            else:
                t = blob
            # Cache shape: (L, H) for pool_time=True. Slice the target layer.
            # If the cache is frame-level (L, T, H), mean over T first.
            if t.dim() == 3:
                t = t.mean(dim=1)  # (L, H)
            if t.dim() != 2:
                raise ValueError(
                    f"Unexpected cached embedding shape {tuple(t.shape)} in {p}; "
                    f"expected (L, H) post-time-pool."
                )
            if layer_idx < 0 or layer_idx >= t.shape[0]:
                raise IndexError(
                    f"--layer {layer_idx} out of range for cache with L={t.shape[0]} "
                    f"(file {p.name}). Use 0..{t.shape[0] - 1}."
                )
            clip_emb = t[layer_idx].float()  # (H,)
            # Per-clip L2 — matches probe forward line 138.
            per_slice_unit_norm.append(F.normalize(clip_emb, dim=-1))

        if not per_slice_unit_norm:
            n_missing += 1
            continue

        # Mean of unit-norm clip vectors → re-normalise → per-file unit vector.
        mean_emb = torch.stack(per_slice_unit_norm).mean(dim=0)
        mean_emb = F.normalize(mean_emb, dim=-1)

        file_embs.append(mean_emb)
        file_wids.append(int(rec["work_id"]))
        cond_raw = rec.get("gm_program")
        if cond_raw is None:
            cond_raw = rec.get("soundfont_id")
        file_conds.append(int(cond_raw) if cond_raw is not None else -1)

    if not file_embs:
        raise RuntimeError(f"No cached embeddings found under {cache_dir}")
    return (torch.stack(file_embs), file_wids, file_conds, n_total, n_missing)


# ──────────────────────────────────────────────────────────────────────
# PCA + treatments
# ──────────────────────────────────────────────────────────────────────


def _compute_corpus_pca(
    embs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(mu, eigvals_desc, eigvecs_desc)`` as fp32 tensors.

    The eigendecomposition is performed in **fp64** on CPU to avoid
    fp32 blow-up on small eigenvalues (the cone-collapsed encoders here
    typically have λ_min/λ_max < 1e-6, which is dangerous for
    ``Λ^(−1/2)``). Outputs are cast back to fp32 for downstream use.

    The returned ``eigvals_desc`` is sorted descending (largest first).
    ``eigvecs_desc[:, k]`` is the k-th principal direction (top-1 first).
    """
    if embs.dim() != 2:
        raise ValueError(f"Expected 2D (N, H) embeddings; got {tuple(embs.shape)}")
    n, h = embs.shape
    if n < h + 1:
        # Σ has at most N-1 non-zero eigenvalues. We still eigh the full
        # (H, H) matrix; the zero tail will be cleanly handled below.
        pass

    embs64 = embs.detach().to(torch.float64)
    mu64 = embs64.mean(dim=0, keepdim=True)
    centered = embs64 - mu64
    sigma = (centered.T @ centered) / float(n)
    # eigh returns ascending eigenvalues + matching eigenvector columns.
    # Symmetric PSD by construction → all eigenvalues ≥ 0 in exact math;
    # tiny negative values from fp roundoff are clamped to 0 below.
    eigvals_asc, eigvecs_asc = torch.linalg.eigh(sigma)
    # Flip to descending.
    eigvals_desc = eigvals_asc.flip(0).clamp(min=0.0)
    eigvecs_desc = eigvecs_asc.flip(1)

    return (
        mu64.squeeze(0).to(torch.float32),
        eigvals_desc.to(torch.float32),
        eigvecs_desc.to(torch.float32),
    )


# Strict treatment-name regexes — unrecognised tokens fail loudly via _parse_treatments.
_RE_RAW = re.compile(r"^raw$")
_RE_CENTERED = re.compile(r"^centered$")
_RE_ABTT = re.compile(r"^abtt-(\d+)$")
_RE_WHITEN_PLAIN = re.compile(r"^whiten-a(\d+(?:\.\d+)?)$")
_RE_WHITEN_REG = re.compile(r"^whiten-a(\d+(?:\.\d+)?)-erel-(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")


def _apply_treatment(
    treatment: str,
    embs: torch.Tensor,  # (N, H), per-file L2-normed
    mu: torch.Tensor,  # (H,)
    eigvals: torch.Tensor,  # (H,) descending
    eigvecs: torch.Tensor,  # (H, H) columns are PCs in descending order
) -> torch.Tensor:
    """Build the transformed ``(N, H)`` matrix for ``treatment``.

    Output is **NOT yet L2-normed** — the caller renormalises before
    handing off to the streaming MAP. Treatments use fp32 throughout
    (eigendecomp was fp64, eigvecs already downcasted).
    """
    if _RE_RAW.match(treatment):
        return embs.clone()

    if _RE_CENTERED.match(treatment):
        return embs - mu

    m = _RE_ABTT.match(treatment)
    if m:
        k = int(m.group(1))
        if k < 0:
            raise ValueError(f"abtt-K requires K ≥ 0; got {k}")
        if k > eigvecs.shape[1]:
            raise ValueError(f"abtt-{k} requires K ≤ H={eigvecs.shape[1]}; got K={k}")
        centered = embs - mu
        if k == 0:
            return centered  # degenerate: equivalent to ``centered``
        # Subtract the projection onto the top-K principal components.
        # Each PC is a column of eigvecs[:, :k]. Projection of x onto
        # U_k is U_k @ (U_k.T @ x). For a batch this is centered @ U_k @ U_k.T.
        top_k = eigvecs[:, :k]  # (H, K)
        proj = centered @ top_k @ top_k.T  # (N, H)
        return centered - proj

    m = _RE_WHITEN_PLAIN.match(treatment)
    if m:
        alpha = float(m.group(1))
        if alpha < 0.0 or alpha > 2.0:
            raise ValueError(f"whiten-a requires α in [0, 2]; got {alpha}")
        centered = embs - mu
        if alpha == 0.0:
            return centered  # degenerate: equivalent to ``centered``
        # Λ^(−α/2). Floor at a tiny absolute value to avoid div-by-zero
        # in the unregularised case; the user should use whiten-a*-erel
        # for serious work.
        eps_floor = 1e-12
        scale = eigvals.clamp(min=eps_floor).pow(-alpha / 2.0)  # (H,)
        # ZCA: U Λ^(−α/2) U^T. The dot product is (centered) @ V where
        # V = U Λ^(−α/2) U^T. Equivalently: project to PC basis, scale,
        # rotate back.
        in_pc = centered @ eigvecs  # (N, H)
        in_pc = in_pc * scale.unsqueeze(0)  # broadcast (1, H)
        return in_pc @ eigvecs.T  # back to original basis

    m = _RE_WHITEN_REG.match(treatment)
    if m:
        alpha = float(m.group(1))
        eps_rel = float(m.group(2))
        if alpha < 0.0 or alpha > 2.0:
            raise ValueError(f"whiten-a requires α in [0, 2]; got {alpha}")
        if eps_rel < 0.0:
            raise ValueError(f"-erel-E requires E ≥ 0; got {eps_rel}")
        centered = embs - mu
        lambda_max = float(eigvals.max())
        # Relative Tikhonov: Λ + ε * λ_max * I. ε is dimensionless and
        # comparable across encoders.
        eigvals_reg = eigvals + eps_rel * lambda_max
        scale = eigvals_reg.pow(-alpha / 2.0)
        in_pc = centered @ eigvecs
        in_pc = in_pc * scale.unsqueeze(0)
        return in_pc @ eigvecs.T

    raise ValueError(
        f"unrecognised treatment name {treatment!r}. Valid: raw, centered, "
        f"abtt-K (K integer), whiten-aα (α float in [0, 2]), "
        f"whiten-aα-erel-E (E float ≥ 0). Example: whiten-a1.0-erel-1e-3."
    )


def _parse_treatments(specs: list[str]) -> list[str]:
    """Validate every treatment spec; raise loudly on the first bad token."""
    validators: list[re.Pattern[str]] = [
        _RE_RAW,
        _RE_CENTERED,
        _RE_ABTT,
        _RE_WHITEN_PLAIN,
        _RE_WHITEN_REG,
    ]
    for spec in specs:
        if not any(p.match(spec) for p in validators):
            raise SystemExit(
                f"ERROR: unrecognised treatment {spec!r}. Valid forms:\n"
                f"  raw\n  centered\n  abtt-K          (K integer, e.g. abtt-1, abtt-10)\n"
                f"  whiten-aα       (α in [0, 2], e.g. whiten-a0.5, whiten-a1.0)\n"
                f"  whiten-aα-erel-E (relative Tikhonov; e.g. whiten-a1.0-erel-1e-3)"
            )
    return specs


# ──────────────────────────────────────────────────────────────────────
# Cache metadata
# ──────────────────────────────────────────────────────────────────────


def _load_cache_meta(cache_dir: Path) -> dict:
    """Read ``_meta.json`` from the cache dir. Refuse to proceed if missing."""
    meta_path = cache_dir / "_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"ERROR: cache dir {cache_dir} has no _meta.json. Either the "
            f"cache was built before metadata logging landed, or you're "
            f"pointing at the wrong directory. Refusing to proceed."
        )
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"ERROR: failed to parse {meta_path}: {e}") from e


def _find_wandb_summary(
    repo_root: Path, task_tag: str, encoder_tag: str, layer: int
) -> Path | None:
    """Locate the wandb-summary.json from the matching sweep run.

    Pattern: ``output/probe.<task>.<encoder-tag>.layer<N>/wandb/run-*/files/wandb-summary.json``
    Picks the most-recently-modified summary that contains ``test/*`` keys.
    """
    layer_dir = repo_root / "output" / f"probe.{task_tag}.{encoder_tag}.layer{layer}"
    if not layer_dir.exists():
        return None
    candidates = sorted(
        layer_dir.glob("wandb/run-*/files/wandb-summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
            if any(k.startswith("test/") for k in data):
                return c
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────────────────────────────
# CSV
# ──────────────────────────────────────────────────────────────────────


CSV_COLUMNS = [
    # Provenance
    "encoder",
    "encoder_model_id",
    "config_hash",
    "task_tag",
    "layer",
    "n_files",
    "n_unique_conditions",
    # PCA diagnostics
    "lambda_max",
    "lambda_min",
    "lambda_ratio",
    "effective_rank_centered",
    # Treatment
    "treatment",
    # Metrics
    "map",
    "recall@10",
    "r_precision",
    "median_rank",
    "diag_mean",
    "off_mean",
    "gap",
    "mean_vec_norm",
    "effective_rank",
    "top1_sv_share",
    "avg_pair_cos",
    "elapsed_sec",
]


def _write_csv_row(csv_path: Path, row: dict, append: bool) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    mode = "a" if (append or file_exists) else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if mode == "w" or not file_exists:
            w.writeheader()
        w.writerow(row)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def _stratified_max_works(
    work_ids: list[int],
    file_indices_to_keep: int,
    seed: int,
) -> list[int]:
    """Return file indices that keep all clips of `file_indices_to_keep`
    randomly-chosen work_ids. Avoids deflating recall vs naive subsample."""
    rng = torch.Generator().manual_seed(seed)
    unique = sorted(set(work_ids))
    perm = torch.randperm(len(unique), generator=rng).tolist()
    chosen = set(unique[i] for i in perm[:file_indices_to_keep])
    return [i for i, w in enumerate(work_ids) if w in chosen]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--encoder", required=True, help="Display name (e.g. CLaMP3).")
    ap.add_argument(
        "--encoder-tag",
        required=True,
        help="Sweep output-dir tag (e.g. 'CLaMP3-layers', 'MERT-v1-95M-layers', 'MuQ').",
    )
    ap.add_argument(
        "--task-tag",
        default="VGMIDITVar-timbre",
        help="Task tag (default: VGMIDITVar-timbre).",
    )
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument(
        "--treatments",
        nargs="+",
        default=[
            "raw",
            "centered",
            "abtt-1",
            "abtt-3",
            "abtt-10",
            "whiten-a0.5",
            "whiten-a0.7",
            "whiten-a1.0",
            "whiten-a1.0-erel-1e-3",
            "whiten-a1.0-erel-1e-2",
        ],
        help="Treatment names. Full default suite if omitted.",
    )
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument(
        "--max-works",
        type=int,
        default=None,
        help="Subsample to first-K random work_ids (keeps all instances, "
        "stratified). Useful for smoke testing.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load + shape data, fit PCA, skip the metric pass. Reports diagnostics only.",
    )
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument(
        "--force-append",
        action="store_true",
        help="Append rows to --out-csv instead of overwriting on first row.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Cross-check raw + centered MAP against the matching "
        "wandb-summary.json. Fail if delta > 5e-4.",
    )
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"ERROR: jsonl not found: {args.jsonl}", file=sys.stderr)
        return 1
    if not args.cache_dir.is_dir():
        print(f"ERROR: cache_dir not found: {args.cache_dir}", file=sys.stderr)
        return 1

    treatments = _parse_treatments(args.treatments)

    if args.out_csv is None:
        args.out_csv = (
            REPO / "docs" / "figures" / "whitening_ablation" / f"{args.encoder}_L{args.layer}.csv"
        )

    # ── Cache metadata ────────────────────────────────────────────────
    meta = _load_cache_meta(args.cache_dir)
    encoder_model_id = meta.get("encoder_model_id", "<unknown>")
    config_hash = args.cache_dir.name  # `<task>__<hash>` directory name

    # ── Load embeddings ──────────────────────────────────────────────
    print(f"Loading JSONL: {args.jsonl}")
    records = load_jsonl(str(args.jsonl))
    print(f"  {len(records)} records")

    print(f"Loading per-clip embeddings (layer {args.layer}) from {args.cache_dir}")
    t0 = time.perf_counter()
    embs, wids, conds, n_total, n_missing = _load_per_file_embeddings(
        records, args.cache_dir, args.layer
    )
    t_load = time.perf_counter() - t0
    print(
        f"  {embs.shape[0]} per-file embeddings, H={embs.shape[1]}, "
        f"{n_missing}/{n_total} records missing  ({t_load:.1f}s)"
    )
    if n_missing > n_total * 0.05:
        raise SystemExit(
            f"ERROR: {n_missing}/{n_total} = {100 * n_missing / n_total:.1f}% "
            f"of records missing from cache. Threshold is 5%. Likely a "
            f"path-normalisation mismatch between JSONL and cache; OR a "
            f"stale cache dir. Investigate before proceeding."
        )

    # ── Optional stratified subsample by work_id ─────────────────────
    if args.max_works is not None and args.max_works < len(set(wids)):
        keep_idx = _stratified_max_works(wids, args.max_works, args.seed)
        embs = embs[keep_idx]
        wids = [wids[i] for i in keep_idx]
        conds = [conds[i] for i in keep_idx]
        print(f"  subsampled to {args.max_works} work_ids → {embs.shape[0]} files")

    if not any(c != -1 for c in conds):
        print("WARN: no records carry a condition; per-condition grid disabled.")
    unique_conds = sorted({c for c in conds if c != -1})
    print(f"  {len(unique_conds)} unique conditions: {unique_conds}")

    # ── PCA fit ──────────────────────────────────────────────────────
    print("Fitting PCA in fp64 ...")
    t0 = time.perf_counter()
    mu, eigvals_desc, eigvecs_desc = _compute_corpus_pca(embs)
    t_pca = time.perf_counter() - t0
    lambda_max = float(eigvals_desc[0])
    nonzero_eigs = eigvals_desc[eigvals_desc > 1e-10 * lambda_max]
    lambda_min_nonzero = float(nonzero_eigs[-1]) if len(nonzero_eigs) else 0.0
    print(
        f"  λ_max={lambda_max:.4e}  λ_min(non-zero)={lambda_min_nonzero:.4e}  "
        f"ratio={lambda_max / max(lambda_min_nonzero, 1e-20):.2e}  ({t_pca:.2f}s)"
    )
    eff_rank_centered = float(
        nonzero_eigs.sum() ** 2 / (nonzero_eigs**2).sum() if len(nonzero_eigs) else float("nan")
    )

    if args.dry_run:
        print("--dry-run: shapes + PCA fit OK; skipping metric pass.")
        return 0

    # ── Treatment loop ───────────────────────────────────────────────
    work_ids_t = torch.tensor(wids, dtype=torch.long)
    conds_t = torch.tensor(conds, dtype=torch.long)
    written_so_far = False
    verify_failures: list[str] = []

    for treatment in treatments:
        print(f"\n=== treatment: {treatment} ===")
        t0 = time.perf_counter()

        transformed = _apply_treatment(treatment, embs, mu, eigvals_desc, eigvecs_desc)
        # L2-normalise — every treatment ends here so cosine retrieval
        # interprets the rows uniformly.
        e_t = F.normalize(transformed, dim=-1)

        # Overall MAP + secondary metrics.
        m = compute_retrieval_metrics_streaming(
            e_t,
            work_ids_t,
            recall_ks=(10,),
            include_r_precision=True,
            include_median_rank=True,
            include_map=True,
            device=args.device,
            batch=args.batch,
        )

        # Per-condition grid (skip if no conditions).
        if unique_conds:
            cell_results = compute_perpair_map_all_streaming(
                e_t,
                work_ids_t,
                conds_t,
                query_conds=unique_conds,
                target_conds=unique_conds,
                device=args.device,
                batch=args.batch,
            )
            diag_aps = [ap for (q, t), (ap, n) in cell_results.items() if q == t and n > 0]
            off_aps = [ap for (q, t), (ap, n) in cell_results.items() if q != t and n > 0]
            diag_mean = sum(diag_aps) / len(diag_aps) if diag_aps else float("nan")
            off_mean = sum(off_aps) / len(off_aps) if off_aps else float("nan")
            gap = diag_mean - off_mean
        else:
            diag_mean = float("nan")
            off_mean = float("nan")
            gap = float("nan")

        # Anisotropy diagnostics (always computed on the L2-normed transformed embs).
        ani = anisotropy_metrics(e_t, seed=args.seed)

        elapsed = time.perf_counter() - t0
        print(
            f"  map={m['map']:.4f}  recall@10={m['recall@10']:.4f}  "
            f"r_prec={m['r_precision']:.4f}  median_rank={m['median_rank']:.1f}"
        )
        print(
            f"  diag={diag_mean:.4f}  off={off_mean:.4f}  gap={gap:+.4f}  "
            f"mvn={ani['mean_vec_norm']:.3f}  eff_rank={ani['effective_rank']:.1f}  "
            f"({elapsed:.1f}s)"
        )

        row = {
            "encoder": args.encoder,
            "encoder_model_id": encoder_model_id,
            "config_hash": config_hash,
            "task_tag": args.task_tag,
            "layer": args.layer,
            "n_files": embs.shape[0],
            "n_unique_conditions": len(unique_conds),
            "lambda_max": lambda_max,
            "lambda_min": lambda_min_nonzero,
            "lambda_ratio": lambda_max / max(lambda_min_nonzero, 1e-20),
            "effective_rank_centered": eff_rank_centered,
            "treatment": treatment,
            "map": m["map"],
            "recall@10": m["recall@10"],
            "r_precision": m["r_precision"],
            "median_rank": m["median_rank"],
            "diag_mean": diag_mean,
            "off_mean": off_mean,
            "gap": gap,
            "mean_vec_norm": ani["mean_vec_norm"],
            "effective_rank": ani["effective_rank"],
            "top1_sv_share": ani["top1_sv_share"],
            "avg_pair_cos": ani["avg_pair_cos"],
            "elapsed_sec": elapsed,
        }
        _write_csv_row(
            args.out_csv,
            row,
            append=args.force_append or written_so_far,
        )
        written_so_far = True

        # ── --verify cross-check ──────────────────────────────────────
        if args.verify and treatment in ("raw", "centered"):
            summary_path = _find_wandb_summary(REPO, args.task_tag, args.encoder_tag, args.layer)
            if summary_path is None:
                print(
                    f"  WARN: --verify requested but no wandb-summary.json "
                    f"found for {args.task_tag}.{args.encoder_tag}.layer{args.layer}; "
                    f"skipping cross-check."
                )
            else:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                key = "test/map" if treatment == "raw" else "test/map_centered"
                logged = summary.get(key)
                if logged is None:
                    print(f"  WARN: --verify: {key} missing from summary; skipping.")
                else:
                    delta = abs(m["map"] - logged)
                    tag = "OK" if delta <= 5e-4 else "FAIL"
                    print(
                        f"  --verify [{tag}] {treatment}: ours={m['map']:.6f} "
                        f"logged={logged:.6f}  Δ={delta:.2e}"
                    )
                    if delta > 5e-4:
                        verify_failures.append(
                            f"{treatment}: ours={m['map']:.6f} logged={logged:.6f} Δ={delta:.2e}"
                        )

    print(f"\nWrote: {args.out_csv}")
    if verify_failures:
        print("\nVERIFY FAILURES:", file=sys.stderr)
        for fail in verify_failures:
            print(f"  - {fail}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
