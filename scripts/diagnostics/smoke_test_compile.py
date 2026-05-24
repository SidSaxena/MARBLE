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


def _swap_to_smoke_jsonls(cfg: dict) -> int:
    """Rewrite every ``jsonl:`` path under ``data.init_args.{train,val,test}.init_args``
    to its smoke variant. Returns the number of paths rewritten.

    Rule: insert ``.smoke`` immediately before the WAV-aware suffix when present,
    otherwise before ``.jsonl``. Matches the manual sed swap used for the MERT
    smoke harness (``HookTheory.train.wav.jsonl`` → ``HookTheory.train.smoke.wav.jsonl``).

    Idempotent: paths that already contain ``.smoke.`` are left alone.
    """
    n_rewrites = 0
    for split in ("train", "val", "test"):
        try:
            block = cfg["data"]["init_args"][split]["init_args"]
        except (KeyError, TypeError):
            continue
        orig = block.get("jsonl")
        if not isinstance(orig, str) or ".smoke." in orig:
            continue
        if orig.endswith(".wav.jsonl"):
            block["jsonl"] = orig.replace(".wav.jsonl", ".smoke.wav.jsonl")
        elif orig.endswith(".jsonl"):
            block["jsonl"] = orig.replace(".jsonl", ".smoke.jsonl")
        else:
            continue
        n_rewrites += 1
    return n_rewrites


def _write_ab_yaml(
    base_config: Path,
    out_path: Path,
    compile_mode: str | None,
    batches: int,
    jsonl_smoke: bool = False,
):
    """Write a temp YAML that overrides compile_mode + bounds the run.

    Trainer overrides are minimal:
      - max_epochs=1, limit_train_batches/limit_val_batches: bound runtime
      - profiler=simple: capture per-row timings
      - callbacks=[]: skip ModelCheckpoint (don't pollute production output dir)
      - enable_checkpointing=False, logger=False: skip wandb + ckpt I/O

    If ``jsonl_smoke=True``, also rewrites the data.init_args jsonl paths to
    their ``.smoke.`` variants (e.g. ``HookTheory.train.wav.jsonl`` →
    ``HookTheory.train.smoke.wav.jsonl``). Used when only a subset of the audio
    corpus is present locally (the typical smoke-on-a-laptop case).
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
    if jsonl_smoke:
        n = _swap_to_smoke_jsonls(cfg)
        if n == 0:
            print(
                f"  ! --jsonl-smoke had no effect on {base_config.name} "
                f"(no eligible jsonl: paths found under data.init_args)",
                file=sys.stderr,
            )
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
        "--modes",
        nargs="+",
        default=["default"],
        choices=["default", "reduce-overhead", "max-autotune"],
        help=(
            "Compile mode(s) to A/B against eager. Pass one or more: "
            "'default reduce-overhead' runs a 3-way comparison "
            "(eager / default / reduce-overhead) in a single invocation."
        ),
    )
    ap.add_argument(
        "--skip-eager",
        action="store_true",
        help="Skip the baseline (no-compile) run. Useful when re-running compile-only experiments.",
    )
    ap.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip ALL compile runs (only run eager). Useful to just measure the eager baseline.",
    )
    ap.add_argument(
        "--jsonl-smoke",
        action="store_true",
        help=(
            "Rewrite data.init_args jsonl paths to their .smoke. variants "
            "(e.g. HookTheory.train.wav.jsonl → HookTheory.train.smoke.wav.jsonl). "
            "Use when only the smoke subset of audio is available locally."
        ),
    )
    args = ap.parse_args()

    if not args.config.exists():
        sys.exit(f"missing config: {args.config}")

    # Suppress wandb regardless of local credentials state.
    os.environ["WANDB_MODE"] = "disabled"

    smoke_dir = Path("output/.smoke_compile")
    smoke_dir.mkdir(parents=True, exist_ok=True)

    # Build the run plan: list of (label, compile_mode) tuples in execution order.
    plan: list[tuple[str, str | None]] = []
    if not args.skip_eager:
        plan.append(("EAGER baseline (compile_mode=null)", None))
    if not args.skip_compile:
        for m in args.modes:
            plan.append((f"COMPILE (compile_mode={m})", m))

    print(f"\nConfig: {args.config}   batches: {args.batches}")
    print(f"Run plan: {[p[0] for p in plan]}\n")

    # Run each variant; collect results keyed by label.
    results: dict[str, dict] = {}  # label → {wall, out, rc, profiler, val_acc, per_batch}
    for label, mode in plan:
        yaml_path = smoke_dir / (f"{mode or 'eager'}.yaml")
        _write_ab_yaml(
            args.config,
            yaml_path,
            compile_mode=mode,
            batches=args.batches,
            jsonl_smoke=args.jsonl_smoke,
        )
        wall, out, rc = _run_training(label, yaml_path)
        prof = _parse_profiler(out)
        rtb = prof["run_training_batch"]
        results[label] = {
            "mode": mode,
            "wall": wall,
            "rc": rc,
            "profiler": prof,
            "val_acc": _parse_val_acc(out),
            "per_batch": (rtb / args.batches) if rtb else None,
            "its_wall": (args.batches / wall) if wall > 0 else 0.0,
        }

    # Pretty-print results table.
    print(f"\n━━━ A/B test result (batches={args.batches}, config={args.config.name}) ━━━")
    for label, r in results.items():
        pb = r["per_batch"]
        pb_str = f"{pb * 1000:.1f} ms" if pb else "(unparsed)"
        acc = r["val_acc"]
        acc_str = f"{acc:.4f}" if acc else "(unparsed)"
        dl = r["profiler"]["train_dataloader_next"]
        dl_str = f"{dl}s" if dl is not None else "(unparsed)"
        print(
            f"  {label:<42}  wall={r['wall']:6.1f}s  ≈ {r['its_wall']:.2f} it/s  "
            f"|  per-batch compute={pb_str}"
        )
        print(f"    val/acc_rpa = {acc_str}    rc = {r['rc']}    train_dataloader_next = {dl_str}")

    # Verdicts: compare each compile run against eager.
    eager_result = next((r for r in results.values() if r["mode"] is None), None)
    if eager_result and eager_result["per_batch"]:
        eager_pb = eager_result["per_batch"]
        eager_acc = eager_result["val_acc"]
        print()
        for label, r in results.items():
            if r["mode"] is None:
                continue
            pb = r["per_batch"]
            if not pb:
                print(f"  {label}: profiler parse failed; can't compute verdict")
                continue
            pct = (eager_pb - pb) / eager_pb * 100
            drift_str = ""
            if eager_acc and r["val_acc"]:
                drift = abs(eager_acc - r["val_acc"])
                drift_str = f"   val/acc drift={drift:.4f} ({'OK' if drift < 0.005 else 'WARN'})"
            print(
                f"  compile_mode={r['mode']:<18}  per-batch speedup vs eager: {pct:+5.1f}%{drift_str}"
            )

        # Overall verdict on the BEST compile mode.
        best = max(
            (r for r in results.values() if r["mode"] is not None and r["per_batch"]),
            key=lambda r: (eager_pb - r["per_batch"]) / eager_pb,
            default=None,
        )
        if best:
            pct = (eager_pb - best["per_batch"]) / eager_pb * 100
            if pct >= 5:
                verdict = (
                    f"GREEN — best mode is '{best['mode']}' at {pct:+.1f}%. "
                    f"Keep on MERT + extend to other compatible encoders."
                )
            elif pct >= -2:
                verdict = (
                    f"MEH — best mode is '{best['mode']}' at {pct:+.1f}% (within noise). "
                    f"Compile isn't meaningfully helping; consider removing for simplicity."
                )
            else:
                verdict = (
                    f"BAD — best mode is '{best['mode']}' at {pct:+.1f}%. "
                    f"Compile is hurting throughput; remove from configs."
                )
            print(f"  Verdict: {verdict}\n")


if __name__ == "__main__":
    main()
