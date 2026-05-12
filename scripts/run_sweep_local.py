#!/usr/bin/env python3
"""
scripts/run_sweep_local.py
──────────────────────────
Local (non-Modal) layer sweep runner. Drop-in equivalent of the Modal
`run_sweep` function — generates per-layer configs, runs fit+test for each
layer sequentially, and prints a results summary at the end.

Supports resume: layers whose checkpoint already exists are skipped by
default (disable with --no-skip).

Usage
─────
# Full 24-layer OMARRQ × GiantSteps sweep
python scripts/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 \\
    --model-tag   OMARRQ-multifeature25hz \\
    --task-tag    GS

# Resume an interrupted sweep (skips completed layers automatically)
python scripts/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 --model-tag OMARRQ-multifeature25hz --task-tag GS

# Run only specific layers (e.g. for debugging layer 0 and 12)
python scripts/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 --model-tag OMARRQ-multifeature25hz --task-tag GS \\
    --layers 0 12

# Override accelerator (e.g. for Apple Silicon)
python scripts/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 --model-tag OMARRQ-multifeature25hz --task-tag GS \\
    --accelerator mps
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# Use the same Python interpreter that is running this script so that the
# correct venv is used on all platforms (important on Windows where "python"
# may not resolve to the venv's interpreter).
PYTHON = sys.executable


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kw)


def _extract_test_metrics(stdout: str) -> dict[str, float]:
    """Parse Lightning's test output for key=value metric lines."""
    metrics = {}
    for line in stdout.splitlines():
        # Match lines like:  │  test/weighted_score  │  0.812  │
        # or plain:           test/weighted_score           0.812
        m = re.findall(r"(test/[\w_]+)\s+([0-9]+\.[0-9]+)", line)
        for key, val in m:
            metrics[key] = float(val)
    return metrics


def _format_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "(no test metrics parsed)"
    return "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(metrics.items()))


def _has_test_metrics(summary_path: Path) -> bool:
    """True if a WandB summary file contains at least one ``test/...`` key.

    This is the only reliable completion signal for both supervised and
    zero-shot sweeps:
      - A run killed before/during test will have a wandb-summary.json
        (created at run start) but no `test/...` entries.
      - A run that ran Trainer.test() to completion will have at least one
        `test/<metric>` entry (e.g. `test/weighted_score`, `test/MAP`).
    """
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(k.startswith("test/") for k in data.keys())


def _layer_done(task_tag: str, model_tag: str, layer: int) -> bool:
    """
    Return True if this layer's test stage has already completed successfully.

    Completion is detected by inspecting WandB run summaries inside the
    layer's output directory.  A summary that contains any ``test/...`` key
    proves Trainer.test() finished and logged metrics, which is the only
    state we want to treat as "skippable" on resume.

    Notes
    -----
    * Works identically for supervised (fit + test) and zero-shot
      (max_epochs=0, just test) sweeps.
    * Runs killed mid-fit or mid-test correctly fail the check, so re-running
      the sweep will pick them up again.
    * The previous "wandb/run-*/ directory exists" heuristic was wrong:
      WandB creates that directory at run startup, before test runs.
    """
    patterns = [
        f"*{model_tag}*{task_tag}*layer{layer}",
        f"*{task_tag}*{model_tag}*layer{layer}",
    ]
    for pat in patterns:
        for d in Path("output").glob(pat):
            if not d.is_dir():
                continue
            for summary in d.glob("wandb/run-*/files/wandb-summary.json"):
                if _has_test_metrics(summary):
                    return True
    return False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a MARBLE layer sweep locally (fit + test per layer)."
    )
    parser.add_argument("--base-config", required=True,
                        help="Path to the base YAML config (e.g. configs/probe.OMARRQ-multifeature25hz.GS.yaml)")
    parser.add_argument("--num-layers", type=int, required=True,
                        help="Total number of transformer layers (e.g. 24 for OMARRQ, 13 for CLaMP3)")
    parser.add_argument("--model-tag", required=True,
                        help="Model identifier used in output paths (e.g. OMARRQ-multifeature25hz)")
    parser.add_argument("--task-tag", required=True,
                        help="Task identifier used in output paths (e.g. GS, EMO, Chords1217)")
    parser.add_argument("--layers", type=int, nargs="*",
                        help="Subset of layer indices to run (default: all 0..num_layers-1)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Ignore completion markers; re-run fit+test for every layer.")
    parser.add_argument("--retest", action="store_true",
                        help="For already-completed layers: skip fit but re-run test. "
                             "Useful when you want fresh WandB test logs without retraining. "
                             "Default (without this flag) is to skip both fit and test.")
    parser.add_argument("--accelerator", default=None,
                        help="Override trainer accelerator (gpu/mps/cpu). Defaults to config value.")
    args = parser.parse_args()

    sweep_dir = f"configs/sweeps/{args.model_tag}.{args.task_tag}"

    # ── 1. Generate per-layer configs ────────────────────────────────────────
    _run([
        PYTHON, "scripts/gen_sweep_configs.py",
        "--base-config", args.base_config,
        "--num-layers",  str(args.num_layers),
        "--model-tag",   args.model_tag,
        "--task-tag",    args.task_tag,
        "--out-dir",     sweep_dir,
    ])

    run_layers = args.layers if args.layers is not None else list(range(args.num_layers))
    results: dict[int, dict] = {}
    t_sweep_start = time.time()

    # ── 2. Run each layer ────────────────────────────────────────────────────
    for layer in run_layers:
        cfg = f"{sweep_dir}/sweep.{args.model_tag}.{args.task_tag}.layer{layer}.yaml"

        print(f"\n{'='*60}")
        print(f" Layer {layer}/{args.num_layers - 1}  [{args.model_tag} | {args.task_tag}]")
        print(f"{'='*60}", flush=True)

        t_layer_start = time.time()

        already_done = (not args.no_skip) and _layer_done(args.task_tag, args.model_tag, layer)

        if already_done and not args.retest:
            # ── Fully skip: no fit, no test, no new WandB run ─────────────────
            print(f"  ✓ Already complete — skipping."
                  f"  (--retest to re-run test, --no-skip to redo everything)")
            results[layer] = {"metrics": {}, "elapsed": 0.0, "skipped": True}
            continue

        # ── Fit ───────────────────────────────────────────────────────────────
        if already_done:
            print(f"  ✓ Already complete — skipping fit, re-running test (--retest).")
        else:
            fit_cmd = [PYTHON, "cli.py", "fit", "-c", cfg]
            if args.accelerator:
                fit_cmd += [f"--trainer.accelerator={args.accelerator}"]
            _run(fit_cmd)

        # ── Test ──────────────────────────────────────────────────────────────
        test_cmd = [PYTHON, "cli.py", "test", "-c", cfg]
        if args.accelerator:
            test_cmd += [f"--trainer.accelerator={args.accelerator}"]

        result = subprocess.run(test_cmd, capture_output=True, text=True)
        elapsed = time.time() - t_layer_start

        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr, file=sys.stderr)

        metrics = _extract_test_metrics(result.stdout)
        results[layer] = {"metrics": metrics, "elapsed": elapsed, "stdout": result.stdout}

    # ── 3. Summary table ─────────────────────────────────────────────────────
    total_elapsed = time.time() - t_sweep_start
    print(f"\n{'='*60}")
    print(f" Sweep complete  [{args.model_tag} | {args.task_tag}]")
    print(f" Total wall time: {total_elapsed/60:.1f} min")
    print(f"{'='*60}")
    for layer in run_layers:
        r = results.get(layer, {})
        if r.get("skipped"):
            print(f"  layer {layer:2d}  [------]  (skipped — already complete)")
        else:
            m_str = _format_metrics(r.get("metrics", {}))
            t_str = f"{r.get('elapsed', 0)/60:.1f}m"
            print(f"  layer {layer:2d}  [{t_str:>5}]  {m_str}")

    # ── 4. Best layer ─────────────────────────────────────────────────────────
    # Rank by the first test metric found (works for both weighted_score and r2)
    scored = {
        layer: list(r["metrics"].values())[0]
        for layer, r in results.items()
        if r.get("metrics")
    }
    if scored:
        best_layer = max(scored, key=scored.__getitem__)
        print(f"\n  Best layer: {best_layer}  ({list(results[best_layer]['metrics'].keys())[0]}={scored[best_layer]:.4f})")


if __name__ == "__main__":
    main()
