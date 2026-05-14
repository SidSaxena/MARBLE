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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Serializes prefixed-line writes from multiple concurrent _run_one_layer
# threads so one layer's output doesn't tear another's.
_CONSOLE_LOCK = threading.Lock()

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
    return any(k.startswith("test/") for k in data)


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


def _meanall_config_for(base_config: str) -> Path | None:
    """Find the meanall sibling config for a per-layer base config.

    Pattern: ``probe.<encoder>(-layers)?.<task>.yaml`` →
             ``probe.<encoder>-meanall.<task>.yaml``
    Returns the path if it exists on disk, else None.
    """
    p = Path(base_config)
    parts = p.name.split(".")
    # Expected shape: ['probe', '<encoder>(-layers)?', '<task>', 'yaml']
    if len(parts) != 4 or parts[0] != "probe" or parts[-1] != "yaml":
        return None
    encoder = parts[1].removesuffix("-layers")
    task = parts[2]
    candidate = p.with_name(f"probe.{encoder}-meanall.{task}.yaml")
    return candidate if candidate.exists() else None


def _meanall_done(task_tag: str, model_tag: str) -> bool:
    """Mirror of _layer_done for the meanall run. The meanall config writes
    its output under a path containing ``-meanall``; match that."""
    patterns = [
        f"*{model_tag}-meanall*{task_tag}*",
        f"*{task_tag}*{model_tag}-meanall*",
        f"*{model_tag}*{task_tag}*-meanall*",
        f"*{task_tag}*{model_tag}*-meanall*",
    ]
    for pat in patterns:
        for d in Path("output").glob(pat):
            if not d.is_dir():
                continue
            for summary in d.glob("wandb/run-*/files/wandb-summary.json"):
                if _has_test_metrics(summary):
                    return True
    return False


def _run_meanall_first(args, common_overrides: list[str]) -> None:
    """Run the mean-of-all-layers baseline before the per-layer sweep.

    Why first: early signal — a one-job baseline that calibrates the
    per-layer expectation. If meanall is already near peak, a flat
    layer profile is the prior; if not, the per-layer sweep is doing
    useful work.

    Cost: identical to one per-layer cycle (zero-shot tasks: one quick
    test pass; supervised: one full train+test). No compute savings,
    just ordering for human readability of intermediate logs.

    Failure handling
    ----------------
    Meanall shares the encoder, dataloader, audio-decode pipeline, and
    GPU initialization with every per-layer job. If meanall fails on any
    of those, every per-layer job will hit the same error. By default we
    therefore abort the whole sweep on meanall failure — saves dozens of
    hours of doomed compute on issues like a missing audio codec or a
    bad checkpoint.

    Pass --continue-on-meanall-failure when you have a legitimate reason
    to believe the failure is meanall-specific (e.g. you're iterating on
    just the aggregation logic and known-bad meanall shouldn't block
    progress on per-layer).
    """
    cfg = _meanall_config_for(args.base_config)
    if cfg is None:
        print(
            f"  ! No meanall sibling found for {args.base_config} "
            f"(expected probe.<encoder>-meanall.<task>.yaml). Skipping."
        )
        return
    if (not args.no_skip) and _meanall_done(args.task_tag, args.model_tag):
        print("  ✓ meanall already complete — skipping (–no-skip to redo).")
        return

    print(
        f"\n{'=' * 60}\n meanall (mean-of-all-layers baseline)  "
        f"[{args.model_tag} | {args.task_tag}]\n{'=' * 60}",
        flush=True,
    )

    # Supervised tasks: fit then test.  Zero-shot (max_epochs=0): test only.
    for stage, kind in (("fit", "fit"), ("test", "test")):
        cmd = [
            PYTHON,
            "cli.py",
            stage,
            "-c",
            str(cfg),
            f"--trainer.logger.init_args.name=layer-meanall-{kind}",
            f"--trainer.logger.init_args.job_type={kind}",
        ] + common_overrides
        print(f"$ {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            if args.continue_on_meanall_failure:
                print(
                    f"  ⚠ meanall {stage} failed (exit {rc}); "
                    f"--continue-on-meanall-failure set → proceeding with "
                    f"per-layer sweep anyway.",
                    file=sys.stderr,
                )
                return
            print(
                f"\n  ✗ meanall {stage} failed (exit {rc}). Aborting the "
                f"{args.model_tag} × {args.task_tag} sweep — the per-layer "
                f"jobs share this run's encoder, dataloader, audio decode, "
                f"and GPU init, so they would all hit the same error.\n"
                f"  Fix the root cause (look at the WandB run + stack trace "
                f"above), then re-launch. If you genuinely want to push past "
                f"a meanall failure, pass --continue-on-meanall-failure.",
                file=sys.stderr,
            )
            sys.exit(rc)


# ──────────────────────────────────────────────
# Live-streaming subprocess helper
# ──────────────────────────────────────────────


def _stream_subprocess(
    cmd: list[str],
    layer_idx: int,
    log_file,
    *,
    quiet: bool = False,
) -> tuple[int, str]:
    """Run cmd, tee stdout/stderr to `log_file` and (unless quiet) prefix
    each line with `[L{N}]` and write to the global console under a lock.

    Returns (returncode, full_captured_text). The caller usually only
    needs the captured text for metric extraction (test phase).
    """
    captured: list[str] = []
    prefix = f"[L{layer_idx}] "
    log_file.write(f"$ {' '.join(cmd)}\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge into one stream
        text=True,
        bufsize=1,  # line-buffered
    )
    try:
        for line in proc.stdout:
            captured.append(line)
            log_file.write(line)
            log_file.flush()
            if not quiet:
                with _CONSOLE_LOCK:
                    sys.stdout.write(f"{prefix}{line}")
                    sys.stdout.flush()
    finally:
        rc = proc.wait()
    return rc, "".join(captured)


def _layer_log_path(model_tag: str, task_tag: str, layer: int) -> Path:
    """Predictable, flat per-layer log path so users can `tail -f` easily."""
    d = Path("output") / "logs" / f"{model_tag}.{task_tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"layer-{layer}.log"


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
                       Test is captured-then-printed by the caller.
    stream_fit=False → both fit and test stream LIVE to the console with
                       a `[L{N}]` prefix (under a global lock) AND are
                       tee'd to a per-layer log file at
                       `output/logs/{model}.{task}/layer-{N}.log`. The
                       caller doesn't reprint on completion.
    """
    cfg = f"{sweep_dir}/sweep.{args.model_tag}.{args.task_tag}.layer{layer}.yaml"
    t0 = time.time()

    already_done = (not args.no_skip) and _layer_done(args.task_tag, args.model_tag, layer)

    if already_done and not args.retest:
        return {
            "layer": layer,
            "skipped": True,
            "elapsed": 0.0,
            "metrics": {},
            "log": "",
            "test_returncode": 0,
        }

    # CLI overrides shared by fit and test
    common_overrides: list[str] = []
    if args.accelerator:
        common_overrides.append(f"--trainer.accelerator={args.accelerator}")
    if precision_override is not None:
        common_overrides.append(f"--trainer.precision={precision_override}")
    if num_workers_override is not None:
        common_overrides.append(f"--data.init_args.num_workers={num_workers_override}")

    log_parts: list[str] = []
    log_file = None
    if not stream_fit:
        # NOTE: log_file is intentionally NOT a context manager — its
        # lifetime spans the whole try/finally block below, which closes
        # it explicitly. Wrapping in `with` would close it before the
        # subprocesses finish writing. (ruff SIM115)
        log_file = open(  # noqa: SIM115
            _layer_log_path(args.model_tag, args.task_tag, layer),
            "w",
            encoding="utf-8",
        )

    try:
        # ── Fit ──────────────────────────────────────────────────────────────
        if already_done:
            msg = "  ✓ Already complete — skipping fit, re-running test (--retest)."
            if stream_fit:
                print(msg)
            else:
                log_file.write(msg + "\n")
                log_file.flush()
                with _CONSOLE_LOCK:
                    sys.stdout.write(f"[L{layer}] {msg}\n")
                    sys.stdout.flush()
        else:
            fit_cmd = [
                PYTHON,
                "cli.py",
                "fit",
                "-c",
                cfg,
                f"--trainer.logger.init_args.name=layer-{layer}-fit",
                "--trainer.logger.init_args.job_type=fit",
            ] + common_overrides
            if stream_fit:
                print(f"$ {' '.join(fit_cmd)}", flush=True)
                subprocess.run(fit_cmd, check=True)
            else:
                rc, _ = _stream_subprocess(fit_cmd, layer, log_file)
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, fit_cmd)

        # ── Test ─────────────────────────────────────────────────────────────
        test_cmd = [
            PYTHON,
            "cli.py",
            "test",
            "-c",
            cfg,
            f"--trainer.logger.init_args.name=layer-{layer}-test",
            "--trainer.logger.init_args.job_type=test",
        ] + common_overrides

        if stream_fit:
            # Sequential mode — original behavior: capture, print, parse.
            r = subprocess.run(test_cmd, capture_output=True, text=True)
            log_parts.append(r.stdout)
            if r.returncode != 0:
                log_parts.append(f"STDERR:\n{r.stderr}")
            test_stdout = r.stdout
            test_returncode = r.returncode
            test_stderr = r.stderr if r.returncode != 0 else ""
        else:
            # Parallel mode — live stream, tee to log file.
            test_returncode, test_stdout = _stream_subprocess(test_cmd, layer, log_file)
            test_stderr = ""  # merged into stdout via stderr=STDOUT

        metrics = _extract_test_metrics(test_stdout)
        elapsed = time.time() - t0

        return {
            "layer": layer,
            "skipped": False,
            "elapsed": elapsed,
            "metrics": metrics,
            "log": "\n".join(log_parts),  # only populated in stream_fit=True
            "test_returncode": test_returncode,
            "test_stderr": test_stderr,
        }
    finally:
        if log_file is not None:
            log_file.close()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run a MARBLE layer sweep locally (fit + test per layer)."
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base YAML config (e.g. configs/probe.OMARRQ-multifeature25hz.GS.yaml)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="Total number of transformer layers (e.g. 24 for OMARRQ, 13 for CLaMP3)",
    )
    parser.add_argument(
        "--model-tag",
        required=True,
        help="Model identifier used in output paths (e.g. OMARRQ-multifeature25hz)",
    )
    parser.add_argument(
        "--task-tag",
        required=True,
        help="Task identifier used in output paths (e.g. GS, EMO, Chords1217)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="*",
        help="Subset of layer indices to run (default: all 0..num_layers-1)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Ignore completion markers; re-run fit+test for every layer.",
    )
    parser.add_argument(
        "--retest",
        action="store_true",
        help="For already-completed layers: skip fit but re-run test. "
        "Useful when you want fresh WandB test logs without retraining. "
        "Default (without this flag) is to skip both fit and test.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Override trainer accelerator (gpu/mps/cpu). Defaults to config value.",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="Override trainer.precision (e.g. 16-mixed, bf16-mixed, 32-true). "
        "Defaults to config value, except when --accelerator=mps which "
        "auto-overrides to 16-mixed (MPS does not support bf16-mixed). "
        "Pass this flag explicitly to override the auto behavior.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run N layers in parallel as separate subprocesses (default: 1, "
        "sequential — bit-identical to the original behavior). "
        "Each layer needs ~5–6 GB VRAM; for a 16 GB GPU use --concurrency 2.",
    )
    parser.add_argument(
        "--num-workers-per-proc",
        type=int,
        default=None,
        help="Per-subprocess dataloader workers (--data.init_args.num_workers). "
        "Only injected when --concurrency > 1. Defaults to max(2, 8 // concurrency) "
        "so the total worker count stays bounded.",
    )
    parser.add_argument(
        "--skip-meanall",
        action="store_true",
        help="Don't run the meanall (mean-of-all-layers) baseline before the "
        "per-layer sweep. By default, if a sibling config "
        "`probe.<encoder>-meanall.<task>.yaml` exists, it is run first so "
        "you have an early baseline before launching N per-layer jobs.",
    )
    parser.add_argument(
        "--only-meanall",
        action="store_true",
        help="Run ONLY the meanall baseline (skip the per-layer sweep). Useful "
        "for getting a baseline matrix of (encoder × task) without paying "
        "for the full N-layer sweep — one fit+test per pair instead of N.",
    )
    parser.add_argument(
        "--continue-on-meanall-failure",
        action="store_true",
        help="Don't abort the sweep when the meanall baseline fails. Default "
        "is to bail — meanall shares the encoder + dataloader + audio decode "
        "pipeline with every per-layer job, so a meanall failure almost "
        "always means the per-layer jobs will fail identically. Use this "
        "flag only when you have a specific reason to believe the failure is "
        "meanall-specific.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.only_meanall and args.skip_meanall:
        parser.error("--only-meanall and --skip-meanall are mutually exclusive.")

    sweep_dir = f"configs/sweeps/{args.model_tag}.{args.task_tag}"

    # ── 1. Generate per-layer configs ────────────────────────────────────────
    # Skip generation when --only-meanall is set: the meanall config is a
    # hand-edited sibling config, not generated here, so per-layer YAML
    # generation has no purpose for that mode.
    if not args.only_meanall:
        _run(
            [
                PYTHON,
                "scripts/sweeps/gen_sweep_configs.py",
                "--base-config",
                args.base_config,
                "--num-layers",
                str(args.num_layers),
                "--model-tag",
                args.model_tag,
                "--task-tag",
                args.task_tag,
                "--out-dir",
                sweep_dir,
            ]
        )

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

    # CLI overrides shared by every cli.py invocation (meanall + per-layer).
    common_overrides: list[str] = []
    if args.accelerator:
        common_overrides.append(f"--trainer.accelerator={args.accelerator}")
    if precision_override is not None:
        common_overrides.append(f"--trainer.precision={precision_override}")
    if num_workers_override is not None:
        common_overrides.append(f"--data.init_args.num_workers={num_workers_override}")

    # ── 2b. Meanall baseline (runs FIRST so you get an early reference) ─────
    if not args.skip_meanall:
        _run_meanall_first(args, common_overrides)

    # ── 2c. --only-meanall short-circuit (skip the per-layer sweep entirely)
    if args.only_meanall:
        print(
            f"\n  --only-meanall: skipping per-layer sweep for "
            f"{args.model_tag} × {args.task_tag}.\n"
        )
        return

    # ── 3. Run each layer ────────────────────────────────────────────────────
    if args.concurrency == 1:
        # Sequential — keep original behavior (streaming fit output) so CUDA
        # workflows remain bit-identical to the pre-concurrency version.
        for layer in run_layers:
            print(f"\n{'=' * 60}")
            print(f" Layer {layer}/{args.num_layers - 1}  [{args.model_tag} | {args.task_tag}]")
            print(f"{'=' * 60}", flush=True)
            try:
                result = _run_one_layer(
                    layer,
                    args,
                    sweep_dir,
                    stream_fit=True,
                    num_workers_override=num_workers_override,
                    precision_override=precision_override,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"\n  Layer {layer} fit failed (exit {e.returncode}). Continuing.",
                    file=sys.stderr,
                )
                result = {
                    "layer": layer,
                    "skipped": False,
                    "elapsed": 0.0,
                    "metrics": {},
                    "log": "",
                    "test_returncode": e.returncode,
                }
            if result.get("skipped"):
                print(
                    "  ✓ Already complete — skipping."
                    "  (--retest to re-run test, --no-skip to redo everything)"
                )
            else:
                print(result.get("log", ""))
                if result.get("test_returncode", 0) != 0:
                    print(f"STDERR: {result.get('test_stderr', '')}", file=sys.stderr)
            results[layer] = result
    else:
        # Parallel — N fit+test pairs in flight, output captured per-layer and
        # printed on completion to keep tqdm progress bars from interleaving.
        print(
            f"\nRunning {len(run_layers)} layers with --concurrency={args.concurrency}"
            f"  (num_workers per proc = {num_workers_override})\n",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(
                    _run_one_layer,
                    layer,
                    args,
                    sweep_dir,
                    False,
                    num_workers_override,
                    precision_override,
                ): layer
                for layer in run_layers
            }
            for fut in as_completed(futures):
                layer = futures[fut]
                try:
                    result = fut.result()
                except subprocess.CalledProcessError as e:
                    print(
                        f"\nLayer {layer} fit failed (exit {e.returncode}):\n{e.stderr}",
                        file=sys.stderr,
                    )
                    result = {
                        "layer": layer,
                        "skipped": False,
                        "elapsed": 0.0,
                        "metrics": {},
                        "log": "",
                        "test_returncode": e.returncode,
                    }
                print(f"\n{'=' * 60}")
                if result.get("skipped"):
                    print(f" Layer {layer}  [skipped]  (already complete)")
                else:
                    print(
                        f" Layer {layer}  done  [{result['elapsed'] / 60:.1f}m]"
                        f"  [{args.model_tag} | {args.task_tag}]"
                    )
                print(f"{'=' * 60}", flush=True)
                if result.get("log"):
                    print(result["log"])
                results[layer] = result

    # ── 4. Summary table ─────────────────────────────────────────────────────
    total_elapsed = time.time() - t_sweep_start
    print(f"\n{'=' * 60}")
    print(f" Sweep complete  [{args.model_tag} | {args.task_tag}]")
    print(f" Total wall time: {total_elapsed / 60:.1f} min  (concurrency={args.concurrency})")
    print(f"{'=' * 60}")
    for layer in run_layers:
        r = results.get(layer, {})
        if r.get("skipped"):
            print(f"  layer {layer:2d}  [------]  (skipped — already complete)")
        else:
            m_str = _format_metrics(r.get("metrics", {}))
            t_str = f"{r.get('elapsed', 0) / 60:.1f}m"
            print(f"  layer {layer:2d}  [{t_str:>5}]  {m_str}")

    # ── 5. Best layer ─────────────────────────────────────────────────────────
    # Rank by the first test metric found (works for both weighted_score and r2)
    scored = {
        layer: list(r["metrics"].values())[0] for layer, r in results.items() if r.get("metrics")
    }
    if scored:
        best_layer = max(scored, key=scored.__getitem__)
        print(
            f"\n  Best layer: {best_layer}  ({list(results[best_layer]['metrics'].keys())[0]}={scored[best_layer]:.4f})"
        )


if __name__ == "__main__":
    main()
