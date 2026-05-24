"""A/B test the ``compile_mode`` init arg on any MERT (or other compile-capable)
encoder config.

What it tests
─────────────
Same config, twice. First with ``compile_mode: null`` (eager baseline),
then with ``compile_mode: default`` (torch.compile). Everything else —
data path, prefetch, precompute_labels, batch_size, num_workers — held
identical. Measures per-batch compute time from Lightning's simple
profiler so the comparison is clean (excludes setup overhead and
compile warmup).

Why this matters
────────────────
The compile_mode setting was originally added as an A/B test point in
2 HookTheoryMelody configs, then rolled out to all 68 MERT configs
without a clean measurement of its contribution. This script gives you
that measurement before extending the pattern to MuQ / OMARRQ / etc.

Default protocol
────────────────
- 300 train batches per run (small enough to fit in ~5 min on a
  5060 Ti, large enough to wash out compile warmup which costs the
  first ~50 batches of the with-compile run).
- 1 epoch hard cap, ``limit_val_batches=1`` (one val pass for the
  acc_rpa sanity check; not the focus).
- Profiler ``simple`` to capture per-row timings.
- Logger off (no wandb pollution).
- ``WANDB_MODE=disabled`` set on the environment.

Live observability — what to watch
──────────────────────────────────
1. **tqdm it/s during each run.** The with-compile run starts ~30-50%
   slower for the first 30-50 batches (PyTorch is JIT-compiling kernels).
   After ~50 batches the throughput should climb and stabilise. If it
   stabilises ABOVE the eager run's rate, compile is winning. If it
   stabilises at the same rate, compile isn't helping after warmup. If
   it's slower throughout, compile is actively hurting (rare but
   possible with certain encoder/hardware combos).

2. **The summary block at the end.** Reports per-batch compute time
   from the profiler's ``run_training_batch`` row — this excludes the
   setup overhead and compile warmup, so it's the cleanest "did compile
   actually make the kernels faster" signal. A useful win is ≥ 5 %
   improvement on this number.

3. **val/acc_rpa.** Should be within ~0.5 % between the two runs. A
   larger gap suggests compile introduced numerical drift the model
   is sensitive to (extremely rare for frozen-encoder probes; would
   warrant investigation if it shows up).

Decision rules
──────────────
- WITH-compile per-batch time ≥ 5 % faster: GREEN — keep compile_mode on
  MERT configs, consider rolling out to MuQ next.
- Within ± 2 %: MEH — compile isn't helping after warmup. Probably worth
  removing compile_mode from MERT configs for simplicity (fewer moving
  parts, no risk of compile-cache disk usage on small runs).
- WITH-compile slower: BAD — remove compile_mode, investigate why.

Usage
─────
    # On local 5060 Ti (recommended — free)
    uv run python scripts/diagnostics/smoke_test_compile.py \\
        --config configs/probe.MERT-v1-95M-meanall.HookTheoryMelody.yaml

    # Tighter batch budget if you're really constrained
    uv run python scripts/diagnostics/smoke_test_compile.py \\
        --config <cfg> --batches 200

    # Modal fallback (~$0.30 on A10G)
    MARBLE_IMAGE=audio MARBLE_GPU=A10G \\
        modal run modal_marble.py::run_probe --config <cfg>  # x2 manually
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _write_ab_yaml(base_config: Path, out_path: Path, compile_mode: str | None, batches: int):
    """Write a temp YAML that overrides compile_mode + bounds the run.

    Trainer overrides are minimal:
      - max_epochs=1, limit_train_batches/limit_val_batches: bound runtime
      - profiler=simple: capture per-row timings
      - callbacks=[]: skip ModelCheckpoint (don't pollute production output dir)
      - enable_checkpointing=False, logger=False: skip wandb + ckpt I/O
    """
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    overrides = {
        "trainer": {
            "max_epochs": 1,
            "limit_train_batches": batches,
            "limit_val_batches": 1,
            "profiler": "simple",
            "enable_checkpointing": False,
            "callbacks": [],
            "logger": False,
        },
        "model": {
            "init_args": {
                "encoder": {
                    "init_args": {
                        # null in YAML = None in Python → the encoder's compile gate
                        # falls back to eager
                        "compile_mode": compile_mode
                    }
                }
            }
        },
    }
    _deep_merge(cfg, overrides)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_training(label: str, smoke_yaml: Path) -> tuple[float, str, int]:
    """Stream-capture ``cli.py fit -c smoke_yaml``, return (wall, stdout, rc)."""
    print(f"\n━━━ {label} ━━━", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "cli.py", "fit", "-c", str(smoke_yaml)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        print(line, end="", flush=True)
    proc.wait()
    return time.time() - t0, "".join(captured), proc.returncode


def _parse_profiler(stdout: str) -> dict[str, float | None]:
    """Extract ``Total time (s)`` for the rows that matter for compile A/B."""

    def _row(name_pat: str) -> float | None:
        m = re.search(
            rf"{name_pat}\s*\|\s*[\d.eE+-]+\s*\|\s*[\d.eE+-]+\s*\|\s*([\d.eE+-]+)",
            stdout,
        )
        return float(m.group(1)) if m else None

    return {
        "run_training_batch": _row(r"run_training_batch"),
        "train_dataloader_next": _row(r"\[_TrainingEpochLoop\]\.train_dataloader_next"),
        "training_step": _row(r"\[Strategy\][^\.]+\.training_step"),
    }


def _parse_val_acc(stdout: str) -> float | None:
    """Grab val/acc_rpa from the metric prints (one val pass at epoch end)."""
    # Lightning prints metrics like:  val/acc_rpa=0.3682
    m = re.search(r"val/acc_rpa[=:\s]+([\d.eE+-]+)", stdout)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base config (e.g. configs/probe.MERT-v1-95M-meanall.HookTheoryMelody.yaml)",
    )
    ap.add_argument(
        "--batches",
        type=int,
        default=300,
        help="limit_train_batches per run (default 300 — ~5 min on 5060 Ti)",
    )
    ap.add_argument(
        "--mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Which compile mode to A/B against eager (default: 'default')",
    )
    ap.add_argument(
        "--skip-eager",
        action="store_true",
        help="Skip the baseline (no-compile) run.",
    )
    ap.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip the experiment (with-compile) run.",
    )
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"missing config: {args.config}")

    # Suppress wandb regardless of local credentials state.
    os.environ["WANDB_MODE"] = "disabled"

    smoke_dir = Path("output/.smoke_compile")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    eager_yaml = smoke_dir / "eager.yaml"
    compile_yaml = smoke_dir / "compile.yaml"
    _write_ab_yaml(args.config, eager_yaml, compile_mode=None, batches=args.batches)
    _write_ab_yaml(args.config, compile_yaml, compile_mode=args.mode, batches=args.batches)
    print(f"\nWrote {eager_yaml} and {compile_yaml}")
    print(f"Config: {args.config}   batches: {args.batches}   compile_mode: {args.mode}\n")

    eager_time, eager_out, eager_rc = 0.0, "", -1
    compile_time, compile_out, compile_rc = 0.0, "", -1

    if not args.skip_eager:
        eager_time, eager_out, eager_rc = _run_training(
            "EAGER baseline (compile_mode=null)", eager_yaml
        )
    else:
        print("[skip] eager baseline")

    if not args.skip_compile:
        compile_time, compile_out, compile_rc = _run_training(
            f"COMPILE experiment (compile_mode={args.mode})", compile_yaml
        )
    else:
        print("[skip] compile experiment")

    # Headline: wall-clock it/s INCLUDING compile/setup overhead.
    eager_its_wall = args.batches / eager_time if eager_time > 0 else 0.0
    compile_its_wall = args.batches / compile_time if compile_time > 0 else 0.0

    # Cleaner: per-batch compute time from profiler row, EXCLUDES setup + compile warmup.
    eager_p = _parse_profiler(eager_out)
    compile_p = _parse_profiler(compile_out)

    def _per_batch(profiler_total_s: float | None) -> float | None:
        return (profiler_total_s / args.batches) if profiler_total_s else None

    eager_pb = _per_batch(eager_p["run_training_batch"])
    compile_pb = _per_batch(compile_p["run_training_batch"])

    eager_acc = _parse_val_acc(eager_out)
    compile_acc = _parse_val_acc(compile_out)

    print(f"\n━━━ A/B test result (batches={args.batches}, config={args.config.name}) ━━━")
    if not args.skip_eager:
        print(
            f"  EAGER baseline   :  wall={eager_time:6.1f}s  ≈ {eager_its_wall:.2f} it/s "
            f" |  per-batch compute={eager_pb * 1000:.1f} ms"
            if eager_pb
            else "    eager baseline   :  (profiler parse failed)"
        )
        print(f"    val/acc_rpa = {eager_acc:.4f}" if eager_acc else "    val/acc_rpa = (unparsed)")
        print(f"    rc = {eager_rc}, train_dataloader_next = {eager_p['train_dataloader_next']}s")
    else:
        print("  EAGER baseline   : [skipped]")
    if not args.skip_compile:
        print(
            f"  COMPILE experiment:  wall={compile_time:6.1f}s  ≈ {compile_its_wall:.2f} it/s "
            f" |  per-batch compute={compile_pb * 1000:.1f} ms"
            if compile_pb
            else "    compile experiment: (profiler parse failed)"
        )
        print(
            f"    val/acc_rpa = {compile_acc:.4f}"
            if compile_acc
            else "    val/acc_rpa = (unparsed)"
        )
        print(
            f"    rc = {compile_rc}, train_dataloader_next = {compile_p['train_dataloader_next']}s"
        )
    else:
        print("  COMPILE experiment: [skipped]")

    # Verdict only when both ran AND profiler parsed.
    if not args.skip_eager and not args.skip_compile and eager_pb and compile_pb:
        pct = (eager_pb - compile_pb) / eager_pb * 100
        print(f"\n  Per-batch compute speedup (compile vs eager):  {pct:+.1f}%")
        # Sanity check on val
        if eager_acc and compile_acc:
            drift = abs(eager_acc - compile_acc)
            print(
                f"  val/acc_rpa drift: {drift:.4f} ({'OK' if drift < 0.005 else 'WARN — investigate'})"
            )

        if pct >= 5:
            verdict = "GREEN — compile_mode is a real win, keep on MERT + consider extending to MuQ/OMARRQ"
        elif pct >= -2:
            verdict = (
                "MEH — within noise, compile_mode isn't helping. Consider removing for simplicity"
            )
        else:
            verdict = "BAD — compile_mode is hurting throughput. Remove from MERT configs"
        print(f"  Verdict: {verdict}\n")


if __name__ == "__main__":
    main()
