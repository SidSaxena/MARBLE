#!/usr/bin/env python3
"""scripts/diagnostics/probe_omarrq_shapes.py
──────────────────────────────────────────────
Step-1 audit for OMAR-RQ: load the encoder exactly the way the MARBLE
probe does, push a real audio batch through it, and print every
intermediate tensor shape and per-layer statistic.

Resolves several open questions from the audit:
  - Is the output shape (24, B, T, 1024) or (24, B, T, 512)?
  - Are layer indices ordered correctly (0 = first block, 23 = last)?
  - Does our FeatureExtractor + encoder.forward double-squeeze?
  - Are per-layer activations distinguishable (mean / std spread)?
  - Are token rate and embed_dim what the wrapper claims?

Usage
-----
    uv run python scripts/diagnostics/probe_omarrq_shapes.py
    uv run python scripts/diagnostics/probe_omarrq_shapes.py --clips-dir /Users/sid/leitmotifs/results/smoke_test/browsable/mert95m_L7_5s/clips --n-clips 8

Output: a printed report; also saves per-layer mean-pooled embeddings
to /tmp/omarrq_layer_embeddings.pt for later use by anisotropy_diag.py.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio


def _gather_clips(clips_dir: Path, n_clips: int, sample_rate: int) -> tuple[torch.Tensor, list[Path]]:
    """Load N audio files (resample to target sr, mono, 5s clip)."""
    paths = sorted(p for p in clips_dir.rglob("*.wav") if not p.name.startswith("._"))[:n_clips]
    if not paths:
        raise FileNotFoundError(f"No .wav files under {clips_dir}")

    wavs = []
    for p in paths:
        w, sr = torchaudio.load(str(p))
        if w.size(0) > 1:
            w = w.mean(dim=0, keepdim=True)   # mono
        if sr != sample_rate:
            w = torchaudio.functional.resample(w, sr, sample_rate)
        # Truncate or pad to exactly 5s
        target = 5 * sample_rate
        if w.size(1) >= target:
            w = w[:, :target]
        else:
            w = torch.nn.functional.pad(w, (0, target - w.size(1)))
        wavs.append(w)

    batch = torch.stack(wavs, dim=0)   # (B, 1, T)
    return batch, paths


def _stat_row(name: str, t: torch.Tensor) -> str:
    return (
        f"  {name:<32}  shape={tuple(t.shape)}  "
        f"dtype={str(t.dtype).removeprefix('torch.'):<10}  "
        f"mean={t.float().mean().item():+.4f}  "
        f"std={t.float().std().item():.4f}  "
        f"min={t.float().min().item():+.4f}  "
        f"max={t.float().max().item():+.4f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", type=Path,
                    default=Path("/Users/sid/leitmotifs/results/smoke_test/browsable/mert95m_L7_5s/clips"),
                    help="Directory containing .wav clips to feed the encoder")
    ap.add_argument("--n-clips", type=int, default=8)
    ap.add_argument("--save-embeddings", type=Path,
                    default=Path("/tmp/omarrq_layer_embeddings.pt"))
    args = ap.parse_args()

    # ── Import the MARBLE encoder ───────────────────────────────────────────
    print("→ Importing OMARRQ_Multifeature25hz_Encoder from MARBLE ...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from marble.encoders.OMAR_RQ.model import (
        OMARRQ_FeatureExtractor,
        OMARRQ_Multifeature25hz_Encoder,
    )

    sample_rate = OMARRQ_Multifeature25hz_Encoder.SAMPLING_RATE
    print(f"  declared SAMPLING_RATE: {sample_rate}")
    print(f"  declared TOKEN_RATE:    {OMARRQ_Multifeature25hz_Encoder.TOKEN_RATE}")
    print(f"  declared NUM_FEATURES:  {OMARRQ_Multifeature25hz_Encoder.NUM_FEATURES}")
    print(f"  declared N_LAYERS:      {OMARRQ_Multifeature25hz_Encoder.N_TRANSFORMER_LAYERS}")

    # ── Load audio ──────────────────────────────────────────────────────────
    print(f"\n→ Loading {args.n_clips} clips at {sample_rate} Hz, mono, 5s ...")
    batch, paths = _gather_clips(args.clips_dir, args.n_clips, sample_rate)
    print(_stat_row("raw waveform batch", batch))
    for p in paths:
        print(f"    {p.name}")

    # ── Build the encoder ───────────────────────────────────────────────────
    print("\n→ Constructing OMARRQ encoder (downloads weights if needed) ...")
    enc = OMARRQ_Multifeature25hz_Encoder()
    enc.eval()

    # ── Mimic the dataloader's audio_transform application ──────────────────
    feat = OMARRQ_FeatureExtractor()
    # MARBLE's collator applies the transform PER ITEM via a "input_features" dict.
    # Re-create that contract here.
    transformed = []
    for i in range(batch.size(0)):
        item_in = batch[i]  # (1, T)
        sample = {"input_features": item_in}
        sample = feat(sample)
        transformed.append(sample["input_features"])

    # Now: what does the dataloader collate? If each item is (T,) it stacks to (B, T).
    if all(t.dim() == 1 for t in transformed):
        post_feat = torch.stack(transformed, dim=0)   # (B, T)
    else:
        post_feat = torch.stack(transformed, dim=0)
    print(_stat_row("after FeatureExtractor", post_feat))

    # ── Push through the encoder ────────────────────────────────────────────
    print("\n→ Running encoder.forward(post_feat) ...")
    with torch.no_grad():
        layer_outputs = enc(post_feat)

    print(f"\n  encoder returned: {type(layer_outputs).__name__} of len {len(layer_outputs)}")
    for li, t in enumerate(layer_outputs):
        if li < 3 or li > len(layer_outputs) - 3 or li in (len(layer_outputs) // 2,):
            print(_stat_row(f"layer {li}", t))

    # Verification — is shape (B, T, 1024)?
    sample0 = layer_outputs[0]
    if sample0.dim() != 3:
        print(f"\n  ⚠ unexpected dim {sample0.dim()} (expected 3 = B, T, C)")
    last_dim = sample0.shape[-1]
    print(f"\n  ✓ feature dim per layer: {last_dim} (expected {OMARRQ_Multifeature25hz_Encoder.NUM_FEATURES})")
    expected = OMARRQ_Multifeature25hz_Encoder.NUM_FEATURES
    if last_dim != expected:
        print(f"  ✗ MISMATCH: encoder reports {last_dim}, class declares {expected}")
    n_tokens = sample0.shape[1]
    expected_tokens = int(5 * OMARRQ_Multifeature25hz_Encoder.TOKEN_RATE)
    print(f"  ✓ tokens for 5 s @ 25 Hz: {n_tokens} (expected ≈ {expected_tokens})")

    # ── Per-layer statistics: is there real specialisation? ─────────────────
    print(f"\n→ Per-layer statistics ({len(layer_outputs)} layers):")
    print(f"  {'L':>3}  {'mean':>8}  {'std':>6}  {'L2-norm/frame':>13}  {'cos(L0)':>8}")
    l0_flat = layer_outputs[0].reshape(-1, last_dim)
    l0_flat = l0_flat / (l0_flat.norm(dim=-1, keepdim=True) + 1e-8)
    for li, t in enumerate(layer_outputs):
        flat = t.reshape(-1, last_dim)
        nrm = flat.norm(dim=-1).mean().item()
        # cosine to layer 0
        flat_n = flat / (flat.norm(dim=-1, keepdim=True) + 1e-8)
        cos_to_l0 = (flat_n * l0_flat).sum(-1).mean().item()
        print(f"  {li:>3}  {t.float().mean().item():>+8.4f}  "
              f"{t.float().std().item():>6.4f}  {nrm:>13.4f}  {cos_to_l0:>8.4f}")

    # ── Save per-layer mean-pooled embeddings for the anisotropy diagnostic ──
    pooled = torch.stack([t.mean(dim=1) for t in layer_outputs], dim=0)  # (L, B, C)
    args.save_embeddings.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": pooled,         # (L, B, C)
        "raw_layer_outputs": [t for t in layer_outputs],   # frame-level for later
        "clip_paths": [str(p) for p in paths],
        "model_name": "OMARRQ-multifeature25hz",
    }, args.save_embeddings)
    print(f"\n→ Saved layer embeddings to {args.save_embeddings}")
    print("  use this file with anisotropy_diag.py for the isotropy/rank analysis.")


if __name__ == "__main__":
    main()
