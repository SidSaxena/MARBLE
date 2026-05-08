#!/usr/bin/env python3
"""
scripts/run_all_sweeps.py
─────────────────────────
Orchestrator: runs all relevant MARBLE layer sweeps sequentially,
in priority order, writing a master results table at the end.

Each sweep calls run_sweep_local.py internally. All sweeps support
resume — already-completed layers (checkpoint exists) are skipped.

Usage
─────
# Run everything (ordered by priority, skips completed layers)
python scripts/run_all_sweeps.py

# Run only specific models or tasks
python scripts/run_all_sweeps.py --models OMARRQ
python scripts/run_all_sweeps.py --tasks GS Chords1217
python scripts/run_all_sweeps.py --models CLaMP3 --tasks GS EMO

# Dry-run: print what would be run without running anything
python scripts/run_all_sweeps.py --dry-run

# Override accelerator (Apple Silicon)
python scripts/run_all_sweeps.py --accelerator mps

Priority order
──────────────
  1. OMARRQ × GS                  (key detection,    24 layers, ~3–4h)
  2. CLaMP3 × GS                  (key detection,    13 layers, ~1.5h)
  3. OMARRQ × Chords1217          (chord recognition,24 layers, ~8–10h)
  4. OMARRQ × BeatTracking        (beat tracking,    24 layers, ~3–4h)
  5. OMARRQ × NSynth              (pitch class.,     24 layers, ~4–6h)
  6. CLaMP3 × NSynth              (pitch class.,     13 layers, ~2h)
  7. OMARRQ × Covers80            (cover retrieval,  24 layers, ~30min — zero-shot)
  8. CLaMP3 × Covers80            (cover retrieval,  13 layers, ~15min — zero-shot)
  9. OMARRQ × HookTheoryKey       (key estimation,   24 layers, ~4–6h — needs audio DL)
 10. CLaMP3 × HookTheoryKey       (key estimation,   13 layers, ~2h   — needs audio DL)
 11. OMARRQ × HookTheoryStructure (structure class., 24 layers, ~4–6h — needs audio DL)
 12. CLaMP3 × HookTheoryStructure (structure class., 13 layers, ~2h   — needs audio DL)

Total wall-time estimate on RTX 5060 Ti: ~30–40h sequential.
Covers80 is very fast (no training, pure retrieval each layer).
HookTheory tasks require running scripts/download_hooktheory.py first.
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
    # ── Priority 1: key detection (most directly relevant to leitmotif) ──────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="GS",
        base_config="configs/probe.OMARRQ-multifeature25hz.GS.yaml",
        num_layers=24,
        note="Key detection  | 24 classes | weighted_score metric",
    ),
    SweepDef(
        model="CLaMP3",
        task="GS",
        base_config="configs/probe.CLaMP3-layers.GS.yaml",
        num_layers=13,
        note="Key detection  | 24 classes | CLaMP3 comparison",
    ),

    # ── Priority 2: chord recognition (frame-level harmonic identity) ────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="Chords1217",
        base_config="configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml",
        num_layers=24,
        note="Chord recognition | 25 classes | frame-level | needs HF token",
    ),

    # ── Priority 3: beat tracking (rhythmic identity) ────────────────────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="GTZANBeatTracking",
        base_config="configs/probe.OMARRQ-multifeature25hz.GTZANBeatTracking.yaml",
        num_layers=24,
        note="Beat tracking  | beat_f1 metric | frame-level",
    ),

    # ── Priority 4: pitch classification (NSynth) ────────────────────────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="NSynth",
        base_config="configs/probe.OMARRQ-multifeature25hz.NSynth.yaml",
        num_layers=24,
        note="Pitch classification | 88 MIDI classes | acc metric",
    ),
    SweepDef(
        model="CLaMP3",
        task="NSynth",
        base_config="configs/probe.CLaMP3-layers.NSynth.yaml",
        num_layers=13,
        note="Pitch classification | 88 MIDI classes | CLaMP3 comparison",
    ),

    # ── Priority 5: cover-song retrieval (Covers80) ──────────────────────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="Covers80",
        base_config="configs/probe.OMARRQ-multifeature25hz.Covers80.yaml",
        num_layers=24,
        note="Cover-song retrieval | MAP metric | zero-shot (no training)",
    ),
    SweepDef(
        model="CLaMP3",
        task="Covers80",
        base_config="configs/probe.CLaMP3-layers.Covers80.yaml",
        num_layers=13,
        note="Cover-song retrieval | MAP metric | CLaMP3 comparison",
    ),

    # ── Priority 6: HookTheory key estimation (requires audio download) ─────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="HookTheoryKey",
        base_config="configs/probe.OMARRQ-multifeature25hz.HookTheoryKey.yaml",
        num_layers=24,
        note="Key estimation  | 24 classes | weighted_score | run download_hooktheory.py first",
    ),
    SweepDef(
        model="CLaMP3",
        task="HookTheoryKey",
        base_config="configs/probe.CLaMP3-layers.HookTheoryKey.yaml",
        num_layers=13,
        note="Key estimation  | 24 classes | CLaMP3 comparison",
    ),

    # ── Priority 7: HookTheory structure classification ───────────────────────
    SweepDef(
        model="OMARRQ-multifeature25hz",
        task="HookTheoryStructure",
        base_config="configs/probe.OMARRQ-multifeature25hz.HookTheoryStructure.yaml",
        num_layers=24,
        note="Structure class.| 7 classes  | acc metric | run download_hooktheory.py first",
    ),
    SweepDef(
        model="CLaMP3",
        task="HookTheoryStructure",
        base_config="configs/probe.CLaMP3-layers.HookTheoryStructure.yaml",
        num_layers=13,
        note="Structure class.| 7 classes  | CLaMP3 comparison",
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
