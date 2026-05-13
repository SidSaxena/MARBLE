#!/usr/bin/env python3
"""
scripts/sweeps/run_sweep_local.py
──────────────────────────
Local (non-Modal) layer sweep runner. Drop-in equivalent of the Modal
`run_sweep` function — generates per-layer configs, runs fit+test for each
layer sequentially, and prints a results summary at the end.

Supports resume: layers whose checkpoint already exists are skipped by
default (disable with --no-skip).

Usage
─────
# Full 24-layer OMARRQ × GiantSteps sweep
python scripts/sweeps/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 \\
    --model-tag   OMARRQ-multifeature25hz \\
    --task-tag    GS

# Resume an interrupted sweep (skips completed layers automatically)
python scripts/sweeps/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 --model-tag OMARRQ-multifeature25hz --task-tag GS

# Run only specific layers (e.g. for debugging layer 0 and 12)
python scripts/sweeps/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 --model-tag OMARRQ-multifeature25hz --task-tag GS \\
    --layers 0 12

# Override accelerator (e.g. for Apple Silicon; auto-applies precision=16-mixed
# since MPS doesn't support bf16-mixed)
python scripts/sweeps/run_sweep_local.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
    --num-layers  24 --model-tag OMARRQ-multifeature25hz --task-tag GS \\
    --accelerator mps

# Parallel: run 2 layers concurrently as separate subprocesses (best on 16 GB
# GPUs where each layer needs ~5–6 GB; auto-halves per-process dataloader
# workers to keep total CPU pressure bounded).
python scripts/sweeps/run_sweep_local.py \\
    --base-config configs/probe.CLaMP3-layers.HookTheoryKey.yaml \\
    --num-layers  13 --model-tag CLaMP3 --task-tag HookTheoryKey \\
    --concurrency 2
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Per-layer worker
# ──────────────────────────────────────────────

def _run_one_layer(
    layer: int,
    args,
    sweep_dir: str,
    stream_fit: bool,
    num_workers_override: int | None,
    precision_override: str | None,
) -> dict:
    """Run fit + test for a single layer. Returns a result dict.

    stream_fit=True  → fit output streams to console as it runs (sequential
                       mode preserves the original behavior bit-for-bit).
    stream_fit=False → fit output is captured and returned in result["log"]
                       (parallel mode — caller prints on completion to avoid
                       interleaved tqdm progress bars from N concurrent fits).

    Test output is always captured (matches the original behavior).
    """
    cfg = f"{sweep_dir}/sweep.{args.model_tag}.{args.task_tag}.layer{layer}.yaml"
    t0 = time.time()

    already_done = (not args.no_skip) and _layer_done(args.task_tag, args.model_tag, layer)

    if already_done and not args.retest:
        return {
            "layer": layer, "skipped": True, "elapsed": 0.0,
            "metrics": {}, "log": "", "test_returncode": 0,
        }

    # CLI overrides shared by fit and test
    common_overrides: list[str] = []
    if args.accelerator:
        common_overrides.append(f"--trainer.accelerator={args.accelerator}")
    if precision_override is not None:
        common_overrides.append(f"--trainer.precision={precision_override}")
    if num_workers_override is not None:
        common_overrides.append(
            f"--data.init_args.num_workers={num_workers_override}"
        )

    log_parts: list[str] = []

    # ── Fit ──────────────────────────────────────────────────────────────────
    if already_done:
        msg = "  ✓ Already complete — skipping fit, re-running test (--retest)."
        if stream_fit:
            print(msg)
        else:
            log_parts.append(msg)
    else:
        fit_cmd = [
            PYTHON, "cli.py", "fit", "-c", cfg,
            f"--trainer.logger.init_args.name=layer-{layer}-fit",
            "--trainer.logger.init_args.job_type=fit",
        ] + common_overrides
        if stream_fit:
            print(f"$ {' '.join(fit_cmd)}", flush=True)
            subprocess.run(fit_cmd, check=True)
        else:
            r = subprocess.run(fit_cmd, capture_output=True, text=True)
            log_parts.append(f"$ {' '.join(fit_cmd)}")
            log_parts.append(r.stdout)
            if r.returncode != 0:
                log_parts.append(f"STDERR:\n{r.stderr}")
                raise subprocess.CalledProcessError(
                    r.returncode, fit_cmd, output=r.stdout, stderr=r.stderr
                )

    # ── Test ─────────────────────────────────────────────────────────────────
    test_cmd = [
        PYTHON, "cli.py", "test", "-c", cfg,
        f"--trainer.logger.init_args.name=layer-{layer}-test",
        "--trainer.logger.init_args.job_type=test",
    ] + common_overrides

    if not stream_fit:
        log_parts.append(f"$ {' '.join(test_cmd)}")
    r = subprocess.run(test_cmd, capture_output=True, text=True)
    log_parts.append(r.stdout)
    if r.returncode != 0:
        # Match original behavior: log stderr but don't raise — test errors
        # are recorded and the sweep continues.
        log_parts.append(f"STDERR:\n{r.stderr}")

    metrics = _extract_test_metrics(r.stdout)
    elapsed = time.time() - t0

    return {
        "layer": layer,
        "skipped": False,
        "elapsed": elapsed,
        "metrics": metrics,
        "log": "\n".join(log_parts),
        "test_returncode": r.returncode,
        "test_stderr": r.stderr if r.returncode != 0 else "",
    }


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
    parser.add_argument("--precision", default=None,
                        help="Override trainer.precision (e.g. 16-mixed, bf16-mixed, 32-true). "
                             "Defaults to config value, except when --accelerator=mps which "
                             "auto-overrides to 16-mixed (MPS does not support bf16-mixed). "
                             "Pass this flag explicitly to override the auto behavior.")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Run N layers in parallel as separate subprocesses (default: 1, "
                             "sequential — bit-identical to the original behavior). "
                             "Each layer needs ~5–6 GB VRAM; for a 16 GB GPU use --concurrency 2.")
    parser.add_argument("--num-workers-per-proc", type=int, default=None,
                        help="Per-subprocess dataloader workers (--data.init_args.num_workers). "
                             "Only injected when --concurrency > 1. Defaults to max(2, 8 // concurrency) "
                             "so the total worker count stays bounded.")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")

    sweep_dir = f"configs/sweeps/{args.model_tag}.{args.task_tag}"

    # ── 1. Generate per-layer configs ────────────────────────────────────────
    _run([
        PYTHON, "scripts/sweeps/gen_sweep_configs.py",
        "--base-config", args.base_config,
        "--num-layers",  str(args.num_layers),
        "--model-tag",   args.model_tag,
        "--task-tag",    args.task_tag,
        "--out-dir",     sweep_dir,
    ])

    run_layers = args.layers if args.layers is not None else list(range(args.num_layers))
    results: dict[int, dict] = {}
    t_sweep_start = time.time()

    # ── 2. Resolve overrides for concurrent runs ─────────────────────────────
    num_workers_override: int | None = None
    if args.concurrency > 1:
        num_workers_override = (
            args.num_workers_per_proc
            if args.num_workers_per_proc is not None
            else max(2, 8 // args.concurrency)
        )

    precision_override: str | None = args.precision
    if precision_override is None and args.accelerator == "mps":
        # MPS doesn't support bf16; auto-fall-back to fp16. Most configs ship
        # with `bf16-mixed`, so without this override every MPS run would die
        # at trainer construction.
        precision_override = "16-mixed"

    # ── 3. Run each layer ────────────────────────────────────────────────────
    if args.concurrency == 1:
        # Sequential — keep original behavior (streaming fit output) so CUDA
        # workflows remain bit-identical to the pre-concurrency version.
        for layer in run_layers:
            print(f"\n{'='*60}")
            print(f" Layer {layer}/{args.num_layers - 1}  [{args.model_tag} | {args.task_tag}]")
            print(f"{'='*60}", flush=True)
            try:
                result = _run_one_layer(
                    layer, args, sweep_dir, stream_fit=True,
                    num_workers_override=num_workers_override,
                    precision_override=precision_override,
                )
            except subprocess.CalledProcessError as e:
                print(f"\n  Layer {layer} fit failed (exit {e.returncode}). "
                      f"Continuing.", file=sys.stderr)
                result = {"layer": layer, "skipped": False, "elapsed": 0.0,
                          "metrics": {}, "log": "", "test_returncode": e.returncode}
            if result.get("skipped"):
                print(f"  ✓ Already complete — skipping."
                      f"  (--retest to re-run test, --no-skip to redo everything)")
            else:
                print(result.get("log", ""))
                if result.get("test_returncode", 0) != 0:
                    print(f"STDERR: {result.get('test_stderr', '')}", file=sys.stderr)
            results[layer] = result
    else:
        # Parallel — N fit+test pairs in flight, output captured per-layer and
        # printed on completion to keep tqdm progress bars from interleaving.
        print(f"\nRunning {len(run_layers)} layers with --concurrency={args.concurrency}"
              f"  (num_workers per proc = {num_workers_override})\n", flush=True)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(
                    _run_one_layer, layer, args, sweep_dir,
                    False, num_workers_override, precision_override,
                ): layer
                for layer in run_layers
            }
            for fut in as_completed(futures):
                layer = futures[fut]
                try:
                    result = fut.result()
                except subprocess.CalledProcessError as e:
                    print(f"\nLayer {layer} fit failed (exit {e.returncode}):\n"
                          f"{e.stderr}", file=sys.stderr)
                    result = {"layer": layer, "skipped": False, "elapsed": 0.0,
                              "metrics": {}, "log": "",
                              "test_returncode": e.returncode}
                print(f"\n{'='*60}")
                if result.get("skipped"):
                    print(f" Layer {layer}  [skipped]  (already complete)")
                else:
                    print(f" Layer {layer}  done  [{result['elapsed']/60:.1f}m]"
                          f"  [{args.model_tag} | {args.task_tag}]")
                print(f"{'='*60}", flush=True)
                if result.get("log"):
                    print(result["log"])
                results[layer] = result

    # ── 4. Summary table ─────────────────────────────────────────────────────
    total_elapsed = time.time() - t_sweep_start
    print(f"\n{'='*60}")
    print(f" Sweep complete  [{args.model_tag} | {args.task_tag}]")
    print(f" Total wall time: {total_elapsed/60:.1f} min  "
          f"(concurrency={args.concurrency})")
    print(f"{'='*60}")
    for layer in run_layers:
        r = results.get(layer, {})
        if r.get("skipped"):
            print(f"  layer {layer:2d}  [------]  (skipped — already complete)")
        else:
            m_str = _format_metrics(r.get("metrics", {}))
            t_str = f"{r.get('elapsed', 0)/60:.1f}m"
            print(f"  layer {layer:2d}  [{t_str:>5}]  {m_str}")

    # ── 5. Best layer ─────────────────────────────────────────────────────────
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
