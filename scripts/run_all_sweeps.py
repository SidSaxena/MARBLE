#!/usr/bin/env python3
"""
scripts/run_all_sweeps.py
─────────────────────────
Orchestrator: runs all MARBLE layer sweeps (OMARRQ, CLaMP3, MERT) sequentially,
ordered fastest → slowest so results appear as soon as possible.

Each sweep calls run_sweep_local.py internally. All sweeps support
resume — already-completed layers (checkpoint exists) are skipped.

Usage
─────
# Run everything (skips completed layers automatically)
python scripts/run_all_sweeps.py

# Run only specific models or tasks
python scripts/run_all_sweeps.py --models MERT
python scripts/run_all_sweeps.py --tasks GS HookTheoryKey
python scripts/run_all_sweeps.py --models CLaMP3 MERT --tasks GS

# Dry-run: print what would be run without running anything
python scripts/run_all_sweeps.py --dry-run

# Override accelerator (Apple Silicon)
python scripts/run_all_sweeps.py --accelerator mps

Speed order (calibrated to OMARRQ × GS = 16 h / 24 layers ≈ 40 min/layer)
───────────────────────────────────────────────────────────────────────────
  ── ZERO-SHOT retrieval (no training, just embed + MAP) ──────────────────
   1. CLaMP3  × Covers80           (13 layers, ~30 min   total — zero-shot)
   2. MERT    × Covers80           (13 layers, ~30 min   total — zero-shot)
   3. OMARRQ  × Covers80           (24 layers, ~1 h      total — zero-shot)
   4. CLaMP3  × SHS100K            (13 layers, ~4–6 h    total — zero-shot, 5K tracks)
   5. MERT    × SHS100K            (13 layers, ~4–6 h    total — zero-shot, 5K tracks)
   6. OMARRQ  × SHS100K            (24 layers, ~8–12 h   total — zero-shot, 5K tracks)

  ── FAST supervised (small dataset, clip-level) ───────────────────────────
   7. OMARRQ  × GS                 (24 layers, ~16 h     — done, skipped)
   8. CLaMP3  × GS                 (13 layers, ~8–10 h)
   9. MERT    × GS                 (13 layers, ~8–10 h)
  10. CLaMP3  × HookTheoryKey      (13 layers, ~8–12 h)
  11. MERT    × HookTheoryKey      (13 layers, ~8–12 h)
  12. OMARRQ  × HookTheoryKey      (24 layers, ~16–22 h)
  13. CLaMP3  × HookTheoryStructure(13 layers, ~8–12 h)
  14. MERT    × HookTheoryStructure(13 layers, ~8–12 h)
  15. OMARRQ  × HookTheoryStructure(24 layers, ~16–22 h)

  ── MEDIUM (frame-level or moderate dataset) ──────────────────────────────
  16. MERT    × GTZANBeatTracking  (13 layers, ~15–20 h — frame-level, 75 Hz)
  17. OMARRQ  × GTZANBeatTracking  (24 layers, ~25–35 h — frame-level, 25 Hz)
  18. MERT    × Chords1217         (13 layers, ~20–30 h — frame-level, 75 Hz)
  19. OMARRQ  × Chords1217         (24 layers, ~35–50 h — frame-level, 25 Hz)

  ── LONG (large dataset, 50K train cap) ───────────────────────────────────
  20. CLaMP3  × NSynth             (13 layers, ~25–35 h)
  21. MERT    × NSynth             (13 layers, ~25–35 h)
  22. OMARRQ  × NSynth             (24 layers, ~50–70 h)

Total wall-time estimate (all 22 sweeps sequential): ~320–460 h ≈ 13–19 days.
Run with --models MERT to run only MERT sweeps in parallel with ongoing OMARRQ/CLaMP3.

Notes
─────
• MERT-v1-95M: 13 layers (0 = CNN feature extractor, 1–12 = transformer), 768-dim.
  Frame-level tasks use 75 Hz token rate (vs OMARRQ 25 Hz / CLaMP3 variable).
• CLaMP3: 13 layers, 768-dim. No BeatTracking or Chords1217 (frame-rate mismatch).
• OMARRQ: 24 Conformer layers, 1024-dim, 25 Hz token rate.
• NSynth train is capped at 50K samples (~289K total) to keep sweep tractable.
• HookTheory data: run `uv run python scripts/download_hooktheory.py` first.
• SHS100K data: run `uv run python scripts/download_shs100k.py` first.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass

# Use the same Python interpreter that is running this script so that the
# correct venv is used on all platforms (important on Windows where "python"
# may not resolve to the venv's interpreter).
PYTHON = sys.executable


# ──────────────────────────────────────────────────────────────────────────────
# Sweep definitions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepDef:
    model:       str   # display name / --model-tag
    task:        str   # display name / --task-tag
    base_config: str   # path to base YAML
    num_layers:  int
    note:        str   # one-line description

SWEEPS: list[SweepDef] = [
    # ══════════════════════════════════════════════════════════════════════════
    # ZERO-SHOT retrieval — no training, just embed + cosine MAP per layer
    # Fastest possible: a few minutes per layer (inference only, no backprop)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Covers80 (~80 songs, 160 clips) — finishes in under an hour total ────
    SweepDef(
        model="CLaMP3",
        task="Covers80",
        base_config="configs/probe.CLaMP3-layers.Covers80.yaml",
        num_layers=13,
        note="Cover retrieval | MAP | zero-shot | 80 songs | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="Covers80",
        base_config="configs/probe.MERT-v1-95M-layers.Covers80.yaml",
        num_layers=13,
        note="Cover retrieval | MAP | zero-shot | 80 songs | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="Covers80",
        base_config="configs/probe.OMARRQ-multifeature25hz.Covers80.yaml",
        num_layers=24,
        note="Cover retrieval | MAP | zero-shot | 80 songs | OMARRQ 24 layers",
    ),

    # ── SHS100K (~5K tracks) — zero-shot, but larger so takes a few hours ────
    SweepDef(
        model="CLaMP3",
        task="SHS100K",
        base_config="configs/probe.CLaMP3-layers.SHS100K.yaml",
        num_layers=13,
        note="Cover retrieval | MAP | zero-shot | 5K tracks | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="SHS100K",
        base_config="configs/probe.MERT-v1-95M-layers.SHS100K.yaml",
        num_layers=13,
        note="Cover retrieval | MAP | zero-shot | 5K tracks | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="SHS100K",
        base_config="configs/probe.OMARRQ-multifeature25hz.SHS100K.yaml",
        num_layers=24,
        note="Cover retrieval | MAP | zero-shot | 5K tracks | OMARRQ 24 layers",
    ),

    # ── VGMIDI-TVar theme/variation retrieval (zero-shot, MIDI-rendered audio) ─
    # Small dataset — fast.  Tests whether the encoders represent intra-piece
    # variation invariance, which is the leitmotif relationship in miniature.
    SweepDef(
        model="CLaMP3",
        task="VGMIDITVar",
        base_config="configs/probe.CLaMP3-layers.VGMIDITVar.yaml",
        num_layers=13,
        note="Theme→variation | MAP | zero-shot | MIDI-rendered | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="VGMIDITVar",
        base_config="configs/probe.MERT-v1-95M-layers.VGMIDITVar.yaml",
        num_layers=13,
        note="Theme→variation | MAP | zero-shot | MIDI-rendered | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="VGMIDITVar",
        base_config="configs/probe.OMARRQ-multifeature25hz.VGMIDITVar.yaml",
        num_layers=24,
        note="Theme→variation | MAP | zero-shot | MIDI-rendered | OMARRQ 24 layers",
    ),
    # CLaMP3 SYMBOLIC path — feeds MIDI directly into CLaMP3's M3 encoder
    # (no MIDI→audio rendering).  Probes a different sub-network than the
    # audio CLaMP3 config above.
    SweepDef(
        model="CLaMP3-symbolic",
        task="VGMIDITVar",
        base_config="configs/probe.CLaMP3-symbolic-layers.VGMIDITVar.yaml",
        num_layers=13,
        note="Theme→variation | MAP | zero-shot | MIDI-native | CLaMP3-symbolic 13 layers",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # FAST supervised — small datasets, clip-level (single prediction per clip)
    # ══════════════════════════════════════════════════════════════════════════

    # ── GS key detection ──────────────────────────────────────────────────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="GS",
        base_config="configs/probe.OMARRQ-multifeature25hz.GS.yaml",
        num_layers=24,
        note="Key detection  | 24 classes | weighted_score | OMARRQ 24 layers",
    ),
    SweepDef(
        model="CLaMP3",
        task="GS",
        base_config="configs/probe.CLaMP3-layers.GS.yaml",
        num_layers=13,
        note="Key detection  | 24 classes | weighted_score | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="GS",
        base_config="configs/probe.MERT-v1-95M-layers.GS.yaml",
        num_layers=13,
        note="Key detection  | 24 classes | weighted_score | MERT 13 layers",
    ),

    # ── HookTheory key estimation ─────────────────────────────────────────────
    SweepDef(
        model="CLaMP3",
        task="HookTheoryKey",
        base_config="configs/probe.CLaMP3-layers.HookTheoryKey.yaml",
        num_layers=13,
        note="Key estimation | 24 classes | weighted_score | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="HookTheoryKey",
        base_config="configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml",
        num_layers=13,
        note="Key estimation | 24 classes | weighted_score | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="HookTheoryKey",
        base_config="configs/probe.OMARRQ-multifeature25hz.HookTheoryKey.yaml",
        num_layers=24,
        note="Key estimation | 24 classes | weighted_score | OMARRQ 24 layers",
    ),

    # ── HookTheory structure classification ──────────────────────────────────
    SweepDef(
        model="CLaMP3",
        task="HookTheoryStructure",
        base_config="configs/probe.CLaMP3-layers.HookTheoryStructure.yaml",
        num_layers=13,
        note="Structure class| 7 classes  | acc metric   | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="HookTheoryStructure",
        base_config="configs/probe.MERT-v1-95M-layers.HookTheoryStructure.yaml",
        num_layers=13,
        note="Structure class| 7 classes  | acc metric   | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="HookTheoryStructure",
        base_config="configs/probe.OMARRQ-multifeature25hz.HookTheoryStructure.yaml",
        num_layers=24,
        note="Structure class| 7 classes  | acc metric   | OMARRQ 24 layers",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # MEDIUM — frame-level tasks (many predictions per clip → slower backprop)
    # CLaMP3 excluded: variable token rate incompatible with frame-level labels
    # ══════════════════════════════════════════════════════════════════════════

    # ── GTZAN beat tracking (frame-level, 25 Hz OMARRQ / 75 Hz MERT) ─────────
    SweepDef(
        model="MERT-v1-95M",
        task="GTZANBeatTracking",
        base_config="configs/probe.MERT-v1-95M-layers.GTZANBeatTracking.yaml",
        num_layers=13,
        note="Beat tracking  | beat_f1   | frame-level 75Hz | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="GTZANBeatTracking",
        base_config="configs/probe.OMARRQ-multifeature25hz.GTZANBeatTracking.yaml",
        num_layers=24,
        note="Beat tracking  | beat_f1   | frame-level 25Hz | OMARRQ 24 layers",
    ),

    # ── Chords1217 chord recognition (frame-level, larger dataset) ───────────
    SweepDef(
        model="MERT-v1-95M",
        task="Chords1217",
        base_config="configs/probe.MERT-v1-95M-layers.Chords1217.yaml",
        num_layers=13,
        note="Chord recog.   | 25 classes | frame-level 75Hz | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="Chords1217",
        base_config="configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml",
        num_layers=24,
        note="Chord recog.   | 25 classes | frame-level 25Hz | OMARRQ 24 layers",
    ),

    # ── HookTheory melody pitch transcription (frame-level) ──────────────────
    # 128 MIDI classes; -1 sentinel for silent frames; metrics: RPA + RCA.
    SweepDef(
        model="MERT-v1-95M",
        task="HookTheoryMelody",
        base_config="configs/probe.MERT-v1-95M-layers.HookTheoryMelody.yaml",
        num_layers=13,
        note="Melody pitch   | 128 MIDI  | frame-level 75Hz | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="HookTheoryMelody",
        base_config="configs/probe.OMARRQ-multifeature25hz.HookTheoryMelody.yaml",
        num_layers=24,
        note="Melody pitch   | 128 MIDI  | frame-level 25Hz | OMARRQ 24 layers",
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # LONG — NSynth pitch classification (50K train cap, still multi-hour/layer)
    # ══════════════════════════════════════════════════════════════════════════
    SweepDef(
        model="CLaMP3",
        task="NSynth",
        base_config="configs/probe.CLaMP3-layers.NSynth.yaml",
        num_layers=13,
        note="Pitch class.   | 88 MIDI   | 50K cap | CLaMP3 13 layers",
    ),
    SweepDef(
        model="MERT-v1-95M",
        task="NSynth",
        base_config="configs/probe.MERT-v1-95M-layers.NSynth.yaml",
        num_layers=13,
        note="Pitch class.   | 88 MIDI   | 50K cap | MERT 13 layers",
    ),
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="NSynth",
        base_config="configs/probe.OMARRQ-multifeature25hz.NSynth.yaml",
        num_layers=24,
        note="Pitch class.   | 88 MIDI   | 50K cap | OMARRQ 24 layers",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _completed_layers(model: str, task: str, num_layers: int) -> list[int]:
    """Return indices of layers that already have a best.ckpt."""
    done = []
    for layer in range(num_layers):
        candidates = list(Path("output").glob(
            f"*{model}*{task}*layer{layer}*/checkpoints/best.ckpt"
        )) + list(Path("output").glob(
            f"*{task}*{model}*layer{layer}*/checkpoints/best.ckpt"
        ))
        if candidates:
            done.append(layer)
    return done


def _data_present(task: str) -> bool:
    """Check that the primary JSONL for a task exists."""
    jsonl_map = {
        "GS":                 "data/GS/GS.train.jsonl",
        "EMO":                "data/EMO/EMO.train.jsonl",
        "Chords1217":         "data/Chords1217/Chords1217.train.jsonl",
        "GTZANBeatTracking":  "data/GTZAN/GTZANBeatTracking.train.jsonl",
        "GTZANGenre":         "data/GTZAN/GTZANGenre.train.jsonl",
        # NSynth: ~19 GB training split required
        "NSynth":             "data/NSynth/NSynth.train.jsonl",
        # Covers80: single evaluation JSONL (no train/val split)
        "Covers80":           "data/Covers80/Covers80.test.jsonl",
        # SHS-100K: community-standard cover retrieval benchmark (test split)
        "SHS100K":            "data/SHS100K/SHS100K.test.jsonl",
        # HookTheory: requires download_hooktheory.py (YouTube audio via yt-dlp)
        "HookTheoryKey":       "data/HookTheory/HookTheoryKey.train.jsonl",
        "HookTheoryStructure": "data/HookTheory/HookTheoryStructure.train.jsonl",
    }
    path = jsonl_map.get(task)
    return path is not None and Path(path).exists()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all MARBLE layer sweeps sequentially."
    )
    parser.add_argument("--models", nargs="*",
                        help="Filter to specific model tags (e.g. OMARRQ CLaMP3)")
    parser.add_argument("--tasks", nargs="*",
                        help="Filter to specific task tags (e.g. GS Chords1217 EMO)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run layers even if checkpoints exist")
    parser.add_argument("--accelerator", default=None,
                        help="Override trainer accelerator (gpu/mps/cpu)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned sweeps without running anything")
    args = parser.parse_args()

    # Filter sweep list
    sweeps = SWEEPS
    if args.models:
        sweeps = [s for s in sweeps if any(m.lower() in s.model.lower() for m in args.models)]
    if args.tasks:
        sweeps = [s for s in sweeps if s.task in args.tasks]

    if not sweeps:
        print("No sweeps match the given filters.", file=sys.stderr)
        sys.exit(1)

    # Print plan
    print(f"\n{'='*70}")
    print(f"  MARBLE Layer Sweep Plan  ({len(sweeps)} sweeps)")
    print(f"{'='*70}")
    total_layers = 0
    for i, s in enumerate(sweeps, 1):
        done = _completed_layers(s.model, s.task, s.num_layers)
        remaining = s.num_layers - len(done)
        data_ok = _data_present(s.task)
        status = "✓ data" if data_ok else "✗ data missing"
        skip_note = f"({len(done)} layers done)" if done else ""
        print(f"  {i}. [{s.model} × {s.task}]  {s.num_layers} layers  {status}  {skip_note}")
        print(f"       {s.note}")
        total_layers += remaining
    print(f"\n  Total remaining layers: {total_layers}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("Dry run — exiting without running.")
        return

    # Check data availability
    missing_data = [s for s in sweeps if not _data_present(s.task)]
    if missing_data:
        print("WARNING: the following sweeps have missing data and will be skipped:")
        for s in missing_data:
            print(f"  {s.model} × {s.task}  →  run data download first")
        print()
        sweeps = [s for s in sweeps if _data_present(s.task)]
        if not sweeps:
            print("No sweeps have data available. Exiting.")
            sys.exit(1)

    # Run each sweep
    sweep_results: dict[str, str] = {}
    t_start = time.time()

    for i, s in enumerate(sweeps, 1):
        print(f"\n{'#'*70}")
        print(f"  Sweep {i}/{len(sweeps)}: {s.model} × {s.task}")
        print(f"  {s.note}")
        print(f"{'#'*70}\n")

        cmd = [
            PYTHON, "scripts/run_sweep_local.py",
            "--base-config", s.base_config,
            "--num-layers",  str(s.num_layers),
            "--model-tag",   s.model,
            "--task-tag",    s.task,
        ]
        if args.no_skip:
            cmd.append("--no-skip")
        if args.accelerator:
            cmd += ["--accelerator", args.accelerator]

        print(f"$ {' '.join(cmd)}\n", flush=True)

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        key = f"{s.model} × {s.task}"
        if result.returncode == 0:
            sweep_results[key] = f"OK  ({elapsed/3600:.1f}h)"
        else:
            sweep_results[key] = f"FAILED (exit {result.returncode})"
            print(f"\n  ⚠ Sweep failed — continuing with next.", file=sys.stderr)

    # Final summary
    total_elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  All sweeps complete  (total: {total_elapsed/3600:.1f}h)")
    print(f"{'='*70}")
    for key, status in sweep_results.items():
        print(f"  {key:<45}  {status}")
    print()


if __name__ == "__main__":
    main()
