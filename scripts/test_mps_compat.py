#!/usr/bin/env python3
"""scripts/test_mps_compat.py
──────────────────────────────
Validate that MPS (Apple Silicon) can actually run a MARBLE layer-probe
sweep end-to-end. The `--accelerator mps` flag was plumbed through but
never tested.

What this does
--------------
1. Confirms `torch.backends.mps.is_available()` (else: this isn't a
   Mac with Apple Silicon and MPS is irrelevant).
2. Confirms the SHS100K audio referenced in the JSONL is reachable
   (the dataset isn't bundled in the repo).
3. Runs ONE layer of CLaMP3 × SHS100K with `--accelerator mps`
   (which now auto-applies `--precision 16-mixed`). SHS100K is
   zero-shot retrieval (`max_epochs=0`), so test-only — fast and
   cheap for a smoke test.
4. Confirms the run produced a WandB summary with at least one
   `test/*` key — same signal `_layer_done` uses.

Pass criteria
-------------
Run finishes cleanly, WandB summary contains `test/MAP` (or any
`test/*` key), and wall-clock is reasonable (under ~10 min).

Why CLaMP3 × SHS100K (not MERT × SHS100K)
-----------------------------------------
You've already validated MERT × SHS100K on Modal. CLaMP3 is the
next-priority encoder; running it on MPS as the smoke test gives
us a fresh data point on a different architecture without retreading
ground.

Usage
-----
    uv run python scripts/test_mps_compat.py
    uv run python scripts/test_mps_compat.py --layer 6   # different layer
    uv run python scripts/test_mps_compat.py --keep-output  # don't clean
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


CONFIG = "configs/probe.CLaMP3-layers.SHS100K.yaml"
MODEL_TAG = "CLaMP3"
TASK_TAG = "SHS100K"
NUM_LAYERS = 13


def _check_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        print("ERROR: torch not importable", file=sys.stderr)
        return False
    if not hasattr(torch.backends, "mps"):
        print("ERROR: torch doesn't have backends.mps "
              "(too old, or non-Apple build).", file=sys.stderr)
        return False
    if not torch.backends.mps.is_available():
        print("ERROR: torch.backends.mps.is_available() is False.\n"
              "Possible causes:\n"
              "  - Not running on Apple Silicon\n"
              "  - PyTorch built without MPS support\n"
              "  - Running inside a Linux container",
              file=sys.stderr)
        return False
    print("✓ MPS available")
    return True


def _check_audio_reachable(jsonl_path: Path, sample: int = 5) -> bool:
    """Sample N entries from the SHS100K JSONL, confirm files exist."""
    if not jsonl_path.exists():
        print(f"ERROR: JSONL not found: {jsonl_path}", file=sys.stderr)
        return False
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f if line.strip()][:sample]
    missing = [r["audio_path"] for r in records if not Path(r["audio_path"]).exists()]
    if missing:
        print(f"ERROR: audio files unreachable. First missing:\n  {missing[0]}",
              file=sys.stderr)
        print("\nThis script needs SHS100K audio at the paths in the JSONL "
              "(e.g. mount the WD Black drive, or run `download_shs100k.py` "
              "to a local dir and rewrite the JSONL with "
              "`verify_shs100k.py --rewrite --audio-dir <new>`).",
              file=sys.stderr)
        return False
    print(f"✓ First {sample} SHS100K audio files reachable")
    return True


def _wandb_summary_has_test(output_root: Path) -> tuple[bool, list[str]]:
    """Walk output_root for any wandb-summary.json with `test/*` keys."""
    for s in output_root.glob("wandb/run-*/files/wandb-summary.json"):
        try:
            data = json.loads(s.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        test_keys = [k for k in data.keys() if k.startswith("test/")]
        if test_keys:
            return True, test_keys
    return False, []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, default=0,
                    help="Layer index to test (default: 0)")
    ap.add_argument("--keep-output", action="store_true",
                    help="Don't delete the per-layer output dir on success "
                         "(useful for debugging).")
    args = ap.parse_args()

    if not _check_mps_available():
        sys.exit(1)
    if not _check_audio_reachable(Path("data/SHS100K/SHS100K.test.jsonl")):
        sys.exit(1)

    sweep_dir = f"configs/sweeps/{MODEL_TAG}.{TASK_TAG}"
    print(f"\n→ Generating layer-{args.layer} config in {sweep_dir} ...")
    subprocess.run([
        sys.executable, "scripts/gen_sweep_configs.py",
        "--base-config", CONFIG,
        "--num-layers", str(NUM_LAYERS),
        "--model-tag", MODEL_TAG,
        "--task-tag", TASK_TAG,
        "--out-dir", sweep_dir,
        "--layers", str(args.layer),
    ], check=True)

    cfg_path = f"{sweep_dir}/sweep.{MODEL_TAG}.{TASK_TAG}.layer{args.layer}.yaml"

    print(f"\n→ Running cli.py test on MPS (16-mixed) — layer {args.layer} ...")
    t0 = time.time()
    cmd = [
        sys.executable, "cli.py", "test", "-c", cfg_path,
        "--trainer.accelerator=mps",
        "--trainer.precision=16-mixed",
        f"--trainer.logger.init_args.name=mps-smoke-layer-{args.layer}",
        "--trainer.logger.init_args.job_type=mps-smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(f"\n→ Exit code: {result.returncode}  (wall-clock: {elapsed/60:.1f} min)")
    if result.stdout:
        print("\n--- STDOUT ---\n" + result.stdout[-3000:])
    if result.returncode != 0:
        print("\n--- STDERR ---\n" + result.stderr[-3000:], file=sys.stderr)
        print("\n✗ FAIL — cli.py test exited non-zero on MPS.", file=sys.stderr)
        sys.exit(2)

    # ── Confirm the WandB summary actually has test/* keys ──────────────
    output_root = Path(f"output/probe.{TASK_TAG}.{MODEL_TAG}-layers.layer-{args.layer}")
    if not output_root.exists():
        # Fall back to a glob — output dir naming pattern varies by config
        for d in Path("output").glob(f"*{MODEL_TAG}*{TASK_TAG}*layer{args.layer}*"):
            if d.is_dir():
                output_root = d
                break
    ok, test_keys = _wandb_summary_has_test(output_root)
    if not ok:
        print(f"\n✗ FAIL — no wandb-summary.json with `test/*` keys found "
              f"under {output_root}", file=sys.stderr)
        sys.exit(3)

    print(f"\n✓ PASS — MPS ran CLaMP3 × SHS100K layer {args.layer} end-to-end.")
    print(f"  Test metrics found: {', '.join(test_keys)}")
    print(f"  Wall-clock: {elapsed/60:.1f} min")

    if not args.keep_output:
        if output_root.exists():
            shutil.rmtree(output_root)
            print(f"  Cleaned {output_root}")


if __name__ == "__main__":
    main()
