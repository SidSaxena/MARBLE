#!/usr/bin/env python3
"""
scripts/diagnostics/test_clamp3_crossmodal_semantic.py
──────────────────────────────────────────
Validate that CLaMP3's symbolic and audio branches actually produce
semantically aligned embeddings (i.e. same-content pairs score higher
than different-content pairs).

The previous smoke test only verified shape and L2-norm correctness on
random inputs — useful but doesn't prove the cross-modal API does
anything meaningful for downstream search.  This script picks K MIDI
files from a VGMIDITVar render, embeds each twice (once symbolic, once
audio), computes the K×K cosine similarity matrix, and asserts that for
each audio row the diagonal entry (its own MIDI) is at or near the top.

Usage
-----
  # After scripts/data/build_vgmiditvar_dataset.py has produced
  # data/VGMIDITVar/VGMIDITVar.jsonl + audio renders:

  uv run python scripts/diagnostics/test_clamp3_crossmodal_semantic.py \\
      --jsonl data/VGMIDITVar/VGMIDITVar.jsonl \\
      --midi-dir data/VGMIDITVar/midi \\
      --num-pairs 5

  # Or specify an explicit list of (midi_path, audio_path) pairs:
  uv run python scripts/diagnostics/test_clamp3_crossmodal_semantic.py \\
      --pairs midi1.mid wav1.wav midi2.mid wav2.wav ...

Pass criterion
--------------
For each audio embedding row, the cosine similarity to its OWN MIDI
should rank at #1 of K (or at worst #2 with a small margin).  In ≥4/5
trials the diagonal must dominate.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio


def _load_pairs(args) -> List[Tuple[Path, Path]]:
    """Build a list of (midi_path, audio_path) tuples from CLI args."""
    if args.pairs:
        if len(args.pairs) % 2 != 0:
            raise ValueError("--pairs requires an even number of values "
                             "(alternating midi then audio).")
        pairs = []
        for i in range(0, len(args.pairs), 2):
            pairs.append((Path(args.pairs[i]), Path(args.pairs[i + 1])))
        return pairs

    if args.jsonl is None or args.midi_dir is None:
        raise ValueError("Either --pairs OR both --jsonl and --midi-dir required.")

    with open(args.jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.split:
        records = [r for r in records if r.get("split") == args.split]

    # Take the first --num-pairs records that have a corresponding MIDI.
    midi_root = Path(args.midi_dir)
    pairs = []
    for rec in records:
        # Try multiple resolution strategies (mirror the datamodule logic).
        if "midi_path" in rec and Path(rec["midi_path"]).exists():
            midi_path = Path(rec["midi_path"])
        else:
            audio_stem = Path(rec["audio_path"]).stem
            split = rec.get("split", "")
            candidate = midi_root / split / f"{audio_stem}.mid"
            if not candidate.exists():
                continue
            midi_path = candidate
        audio_path = Path(rec["audio_path"])
        if not audio_path.exists():
            continue
        pairs.append((midi_path, audio_path))
        if len(pairs) >= args.num_pairs:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--jsonl", type=Path,
                    help="VGMIDITVar JSONL with audio_path / midi_path fields")
    ap.add_argument("--midi-dir", type=Path,
                    help="Root directory containing extracted MIDI files")
    ap.add_argument("--num-pairs", type=int, default=5,
                    help="Number of MIDI/audio pairs to test (default: 5)")
    ap.add_argument("--split", default=None,
                    help="Filter JSONL by split (train/test); default uses all")
    ap.add_argument("--pairs", nargs="+",
                    help="Explicit list of (midi audio) paths to test, "
                         "alternating: midi1 wav1 midi2 wav2 ...")
    args = ap.parse_args()

    pairs = _load_pairs(args)
    if len(pairs) < 2:
        print(f"ERROR: need at least 2 pairs, got {len(pairs)}", file=sys.stderr)
        sys.exit(1)
    print(f"Testing on {len(pairs)} (MIDI, audio) pairs:")
    for i, (m, a) in enumerate(pairs):
        print(f"  [{i}]  midi: {m.name}")
        print(f"       audio: {a.name}")

    # Load encoder once
    print("\nLoading CLaMP3_Symbolic_Encoder ...")
    from marble.encoders.CLaMP3.model import CLaMP3_Symbolic_Encoder
    enc = CLaMP3_Symbolic_Encoder().eval()
    print("  ✓ loaded")

    # ── Build MIDI patches for all pairs ────────────────────────────────────
    print("\nTokenising MIDIs ...")
    from marble.encoders.CLaMP3.midi_util import midi_to_mtf
    from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer
    from marble.encoders.CLaMP3.model import CLaMP3Config

    patchilizer = M3Patchilizer()
    max_patches = CLaMP3Config.PATCH_LENGTH
    patch_size  = CLaMP3Config.PATCH_SIZE
    pad_token   = patchilizer.pad_token_id

    midi_patches: List[torch.Tensor] = []
    for m, _ in pairs:
        mtf = midi_to_mtf(str(m))
        patches_list = patchilizer.encode(mtf, patch_size=patch_size,
                                          add_special_patches=True)
        patches_list = patches_list[: max_patches]
        t = torch.tensor(patches_list, dtype=torch.long)
        if t.size(0) < max_patches:
            pad = torch.full((max_patches - t.size(0), patch_size),
                             pad_token, dtype=torch.long)
            t = torch.cat([t, pad], dim=0)
        midi_patches.append(t)
    midi_batch = torch.stack(midi_patches, dim=0)
    print(f"  MIDI batch: {midi_batch.shape}")

    # ── Load audio for all pairs at 24 kHz mono ─────────────────────────────
    print("\nLoading audio at 24 kHz mono ...")
    audio_wavs: List[torch.Tensor] = []
    target_sr = 24000
    for _, a in pairs:
        wav, sr = torchaudio.load(str(a))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        audio_wavs.append(wav)
    # Pad/truncate to a common length (use shortest for fair comparison)
    min_len = min(w.size(1) for w in audio_wavs)
    audio_batch = torch.stack([w[:, :min_len] for w in audio_wavs], dim=0)
    print(f"  Audio batch: {audio_batch.shape}")

    # ── Compute shared-space embeddings ──────────────────────────────────────
    print("\nComputing embeddings ...")
    with torch.no_grad():
        e_mid = enc.embed_symbolic(midi_batch)        # (K, 768)
        e_aud = enc.embed_audio(audio_batch)           # (K, 768)
    print(f"  symbolic: {e_mid.shape}")
    print(f"  audio:    {e_aud.shape}")

    # ── K×K similarity matrix ────────────────────────────────────────────────
    sim = e_aud @ e_mid.T   # rows=audio, cols=midi; diagonal = matching pairs
    print(f"\nSimilarity matrix (audio_i ↔ midi_j):")
    print("       " + "  ".join(f"midi[{j}]" for j in range(len(pairs))))
    for i in range(len(pairs)):
        row = "  ".join(f"{sim[i, j].item():+.4f}" for j in range(len(pairs)))
        print(f"  aud[{i}]  {row}")

    # ── Pass criterion ───────────────────────────────────────────────────────
    K = len(pairs)
    n_diag_best = 0
    diag_margins = []
    for i in range(K):
        row = sim[i]
        diag = row[i].item()
        rest = torch.cat([row[:i], row[i + 1:]])
        max_off = rest.max().item()
        margin = diag - max_off
        diag_margins.append(margin)
        if margin > 0:
            n_diag_best += 1

    print(f"\nResults:")
    print(f"  Diagonal-best rows: {n_diag_best}/{K}")
    print(f"  Mean diagonal margin: {sum(diag_margins) / K:+.4f}")
    print(f"  Min margin:           {min(diag_margins):+.4f}")
    print(f"  Max margin:           {max(diag_margins):+.4f}")

    pass_threshold = max(int(0.8 * K), K - 1)   # at least 4/5, or K-1
    if n_diag_best >= pass_threshold:
        print(f"\n  ✓ PASS — {n_diag_best}/{K} ≥ {pass_threshold}: "
              f"audio embeddings are closer to their own MIDIs than to others.")
        sys.exit(0)
    else:
        print(f"\n  ✗ FAIL — {n_diag_best}/{K} < {pass_threshold}: "
              f"cross-modal alignment is weaker than expected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
