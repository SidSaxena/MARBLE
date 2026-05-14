#!/usr/bin/env python3
"""scripts/diagnostics/anisotropy_diag.py
─────────────────────────────────────────
Unified isotropy / rank-collapse diagnostic for MARBLE encoders.

Builds on the methodology of
  /Users/sid/leitmotifs/scripts/diagnose_anisotropy.py
and extends it with:
  - rank-collapse markers (top-1 singular value share, effective rank)
  - side-by-side comparison across OMAR-RQ / MERT / CLaMP3 layers
  - a uniform-sphere reference baseline so thresholds are principled
    rather than arbitrary

Output for each (encoder, layer) row:
  avg_random_pair_cosine            — Ethayarajh-style cone-effect
  mean_vector_norm                  — norm of normalized-frame centroid
  top1_singular_share               — σ₁² / Σσᵢ²
  effective_rank                    — exp(H(σ²)) on the centered covariance
  All four also computed AFTER centering (subtracting corpus mean).

Usage
-----
  # Run all three encoders against the leitmotif clips
  uv run python scripts/diagnostics/anisotropy_diag.py

  # Just one encoder, custom clips
  uv run python scripts/diagnostics/anisotropy_diag.py \\
      --encoders omarrq --clips-dir /path/to/wavs --n-clips 32

  # Use only a subset of layers (faster)
  uv run python scripts/diagnostics/anisotropy_diag.py --layers 0 6 12 23
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# ── Encoder registry ───────────────────────────────────────────────────────
# (label, module_path:class_name, sample_rate, num_layers)
ENCODERS = {
    "omarrq": ("marble.encoders.OMAR_RQ.model:OMARRQ_Multifeature25hz_Encoder", 24000, 24),
    "mert": ("marble.encoders.MERT.model:MERT_v1_95M_Encoder", 24000, 13),
    "clamp3": ("marble.encoders.CLaMP3.model:CLaMP3_Encoder", 24000, 13),
}


def _load_clips(clips_dir: Path, n_clips: int, sample_rate: int, secs: float = 5.0) -> torch.Tensor:
    """Load N audio clips at the target sample rate as a (B, 1, T) tensor."""
    paths = sorted(p for p in clips_dir.rglob("*.wav") if not p.name.startswith("._"))[:n_clips]
    if not paths:
        raise FileNotFoundError(f"No .wav files under {clips_dir}")
    target = int(secs * sample_rate)
    wavs = []
    for p in paths:
        w, sr = torchaudio.load(str(p))
        if w.size(0) > 1:
            w = w.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            w = torchaudio.functional.resample(w, sr, sample_rate)
        if w.size(1) >= target:
            w = w[:, :target]
        else:
            w = torch.nn.functional.pad(w, (0, target - w.size(1)))
        wavs.append(w)
    return torch.stack(wavs, dim=0)


def _import(spec: str):
    """Import 'pkg.mod:ClassName' → ClassName."""
    mod, cls = spec.split(":")
    m = __import__(mod, fromlist=[cls])
    return getattr(m, cls)


def _extract_layer_embeddings(encoder_spec: str, audio: torch.Tensor) -> torch.Tensor:
    """Run an encoder, return (L, N_frames, C) — concatenated frames across batch."""
    Encoder = _import(encoder_spec)
    enc = Encoder().eval()
    with torch.no_grad():
        # Different encoders expect different shapes. Try (B, T) first (squeezed),
        # then fall back to (B, 1, T).
        try:
            x = audio.squeeze(1)  # (B, T)
            outs = enc(x)
        except Exception:
            outs = enc(audio)
    # Normalize to a tuple/list of (B, T, C) tensors
    if isinstance(outs, torch.Tensor):
        outs = [outs]
    elif isinstance(outs, tuple):
        outs = list(outs)
    # MERT returns the all-layers as a stacked tensor; handle that case.
    if len(outs) == 1 and outs[0].dim() == 4:
        # (L, B, T, C) → list of (B, T, C)
        outs = [outs[0][i] for i in range(outs[0].shape[0])]
    # Reshape each layer's (B, T, C) → (N_frames, C) by flattening B and T
    flat = [t.reshape(-1, t.shape[-1]) for t in outs]
    # Stack into (L, N_frames, C). All layers should share N_frames; if not,
    # take the min.
    n = min(f.shape[0] for f in flat)
    return torch.stack([f[:n] for f in flat], dim=0)


# ── Diagnostic metrics ──────────────────────────────────────────────────────


def _isotropy_metrics(frames: np.ndarray, n_pairs: int = 5000, seed: int = 0) -> dict:
    """
    Args:
        frames: (N, C) embeddings (NOT yet centered, NOT yet L2-normalized)
    Returns:
        dict with avg pair cosine, mean-vector-norm, top1 singular share,
        effective rank — all for the L2-normalized embedding.
    """
    rng = np.random.default_rng(seed)
    n, c = frames.shape

    # L2 normalize
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    normed = frames / np.clip(norms, 1e-8, None)

    # avg pair cosine via random sampling without replacement
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    a = rng.choice(n, size=n_pairs, replace=True)
    b = rng.choice(n, size=n_pairs, replace=True)
    same = a == b
    b[same] = (b[same] + 1) % n
    sims = (normed[a] * normed[b]).sum(axis=1)

    # mean-vector norm (cone effect)
    mean_vec = normed.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))

    # Rank collapse via SVD of the centered embedding matrix.
    # Sample-cap because SVD on 7M frames is expensive.
    n_svd = min(n, 4096)
    sub_idx = rng.choice(n, size=n_svd, replace=False) if n_svd < n else np.arange(n)
    sub = frames[sub_idx] - frames[sub_idx].mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(sub, compute_uv=False)
    except np.linalg.LinAlgError:
        sv = np.zeros(min(n_svd, c))
    sv2 = sv * sv
    if sv2.sum() > 0:
        share = sv2 / sv2.sum()
        top1 = float(share[0])
        # entropy in nats → effective rank = exp(H)
        h = -np.sum(share * np.log(share + 1e-12))
        eff_rank = float(np.exp(h))
    else:
        top1 = float("nan")
        eff_rank = float("nan")

    # Expected isotropic baselines:
    # Uniform-on-sphere in C dims: avg_cos = 0, std = 1/sqrt(C)
    # Mean of N iid unit vectors: ||mean|| ≈ 1/sqrt(N)
    expected_pair_std = 1.0 / math.sqrt(c)
    expected_mean_norm = 1.0 / math.sqrt(n) if n > 0 else float("nan")

    return {
        "n_frames": n,
        "n_dim": c,
        "avg_pair_cos": float(sims.mean()),
        "std_pair_cos": float(sims.std()),
        "mean_vec_norm": mean_norm,
        "top1_sv_share": top1,
        "effective_rank": eff_rank,
        "iso_pair_std_baseline": expected_pair_std,
        "iso_mean_norm_baseline": expected_mean_norm,
    }


def _diagnose_layer(frames: torch.Tensor, n_pairs: int) -> dict:
    """Compute metrics both with and without centering."""
    f = frames.detach().cpu().float().numpy()
    raw = _isotropy_metrics(f, n_pairs=n_pairs)
    centered = _isotropy_metrics(f - f.mean(axis=0, keepdims=True), n_pairs=n_pairs)
    return {"raw": raw, "centered": centered}


def _judgement(raw: dict, centered: dict) -> str:
    """Heuristic verdict based on the metrics."""
    # Strong cone effect / high anisotropy in raw
    if raw["avg_pair_cos"] > 0.5 or raw["mean_vec_norm"] > 0.5:
        if centered["avg_pair_cos"] > 0.3:
            return "ANISOTROPIC (severe)"
        return "ANISOTROPIC (cone — fixable by centering)"
    if raw["top1_sv_share"] > 0.5:
        return "RANK-COLLAPSED"
    if raw["effective_rank"] < raw["n_dim"] * 0.1:
        return "LOW EFFECTIVE RANK"
    # Mild anisotropy: pair cosine clearly above baseline
    if raw["avg_pair_cos"] > 5 * raw["iso_pair_std_baseline"]:
        return "MILD anisotropy"
    return "isotropic"


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--encoders",
        nargs="+",
        default=list(ENCODERS),
        choices=list(ENCODERS),
        help="Encoders to diagnose",
    )
    ap.add_argument(
        "--clips-dir",
        type=Path,
        default=Path("/Users/sid/leitmotifs/results/smoke_test/browsable/mert95m_L7_5s/clips"),
    )
    ap.add_argument("--n-clips", type=int, default=16)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="*",
        default=None,
        help="Subset of layers to evaluate (default: all)",
    )
    ap.add_argument("--n-pairs", type=int, default=5000)
    ap.add_argument("--out-json", type=Path, default=Path("/tmp/anisotropy_diag.json"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    all_results: dict[str, dict] = {}
    for enc_name in args.encoders:
        spec, sr, n_layers = ENCODERS[enc_name]
        print(f"\n{'═' * 72}")
        print(f"  {enc_name}  ({n_layers} layers @ {sr} Hz)")
        print(f"{'═' * 72}")

        audio = _load_clips(args.clips_dir, args.n_clips, sr, args.clip_seconds)
        print(f"  audio: {tuple(audio.shape)}  ({args.n_clips} clips × {args.clip_seconds}s)")

        try:
            stacked = _extract_layer_embeddings(spec, audio)  # (L, N_frames, C)
        except Exception as e:
            print(f"  ✗ encoder failed: {type(e).__name__}: {e}")
            continue

        L, N, C = stacked.shape
        print(f"  embeddings: ({L}, {N}, {C})")

        layer_indices = args.layers if args.layers is not None else list(range(L))
        results = {}
        print(
            f"\n  {'L':>3}  {'avg_pair':>8}  {'std_pair':>8}  {'mean_norm':>9}  "
            f"{'top1_sv':>7}  {'eff_rank':>8}  {'centered_pair':>13}  verdict"
        )
        for li in layer_indices:
            if li >= L:
                continue
            d = _diagnose_layer(stacked[li], args.n_pairs)
            results[li] = d
            r, c = d["raw"], d["centered"]
            verdict = _judgement(r, c)
            print(
                f"  {li:>3}  {r['avg_pair_cos']:>8.4f}  {r['std_pair_cos']:>8.4f}  "
                f"{r['mean_vec_norm']:>9.4f}  {r['top1_sv_share']:>7.4f}  "
                f"{r['effective_rank']:>8.1f}  {c['avg_pair_cos']:>13.4f}  {verdict}"
            )

        all_results[enc_name] = results

        # Free GPU/CPU mem before next encoder
        del stacked
        import gc

        gc.collect()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n→ Full results saved to {args.out_json}")


if __name__ == "__main__":
    main()
