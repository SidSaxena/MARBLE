#!/usr/bin/env python3
"""
wandb_checkpoint_audit.py

Cross-references local checkpoint directories against WandB to determine
which are safe to delete (test phase completed and metrics logged to WandB).

Usage:
    python wandb_checkpoint_audit.py               # show safe-to-delete list
    python wandb_checkpoint_audit.py --delete      # actually delete safe ones
    python wandb_checkpoint_audit.py --encoder MERT-v1-330M  # filter by encoder

Run from the root of your MARBLE repo.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed. Run: pip install wandb")
    sys.exit(1)


def parse_probe_dir(d: Path):
    """
    Parse a probe output directory name into (task, encoder, layer).

    Pattern: probe.{task}.{encoder}.layer{N}
    Meanall: probe.{task}.{encoder}-meanall
    """
    name = d.name
    m = re.match(r"^probe\.(.+)\.(.+)\.layer(\d+)$", name)
    if m:
        return {"task": m.group(1), "encoder": m.group(2), "layer": int(m.group(3)), "meanall": False}
    m = re.match(r"^probe\.(.+)\.(.+)-meanall$", name)
    if m:
        return {"task": m.group(1), "encoder": m.group(2), "layer": None, "meanall": True}
    return None


def has_test_metrics(run) -> bool:
    try:
        return any(k.startswith("test/") for k in run.summary.keys())
    except Exception:
        return False


def normalize(s: str) -> str:
    """Lowercase and strip punctuation for fuzzy matching."""
    return re.sub(r"[\-_\s/]", "", s).lower()


def run_matches(run, encoder: str, task: str, layer_name: str) -> bool:
    """
    True if a WandB run belongs to the given (encoder, task, layer).

    Matching strategy:
      1. run.name must equal layer_name (e.g. "layer-3-test")
      2. run.group must contain both encoder and task (fuzzy, ignores punctuation)
    """
    if run.name != layer_name:
        return False
    group = run.group or ""
    norm_group = normalize(group)
    return normalize(encoder) in norm_group and normalize(task) in norm_group


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(
        description="Audit checkpoints vs WandB; delete completed ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete safe checkpoints")
    parser.add_argument("--encoder", default=None, help="Only check this encoder (substring match)")
    parser.add_argument("--project", default="marble", help="WandB project name (default: marble)")
    args = parser.parse_args()

    output_dir = Path("output")
    if not output_dir.exists():
        print("ERROR: No 'output/' directory found. Run from your MARBLE repo root.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Collect local checkpoint directories
    # ─────────────────────────────────────────────────────────────────────────

    print("Scanning local checkpoints...")
    probe_dirs = []
    for ckpt_dir in sorted(output_dir.glob("*/checkpoints")):
        parsed = parse_probe_dir(ckpt_dir.parent)
        if parsed is None:
            continue
        if args.encoder and args.encoder.lower() not in parsed["encoder"].lower():
            continue
        parsed["ckpt_dir"] = ckpt_dir
        probe_dirs.append(parsed)

    if not probe_dirs:
        print("No checkpoint directories found.")
        sys.exit(0)

    print(f"Found {len(probe_dirs)} checkpoint dirs.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Fetch all completed WandB runs (those with test/* metrics)
    # ─────────────────────────────────────────────────────────────────────────

    api = wandb.Api()
    entity = api.default_entity
    project_path = f"{entity}/{args.project}"

    print(f"Fetching runs from WandB project: {project_path} ...")
    try:
        all_runs = list(api.runs(project_path, per_page=1000))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    completed_runs = [r for r in all_runs if has_test_metrics(r)]
    print(f"Total runs: {len(all_runs)}  |  Completed (have test/* metrics): {len(completed_runs)}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Cross-reference each local dir against WandB
    # ─────────────────────────────────────────────────────────────────────────

    safe = []
    incomplete = []

    for entry in probe_dirs:
        layer_name = "layer-meanall-test" if entry["meanall"] else f"layer-{entry['layer']}-test"
        match = next(
            (r for r in completed_runs if run_matches(r, entry["encoder"], entry["task"], layer_name)),
            None,
        )
        size_mb = dir_size_mb(entry["ckpt_dir"])
        row = {**entry, "size_mb": size_mb, "expected_run": layer_name}
        if match:
            row["wandb_url"] = match.url
            safe.append(row)
        else:
            incomplete.append(row)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Report
    # ─────────────────────────────────────────────────────────────────────────

    print("=" * 72)
    print(f"  SAFE TO DELETE  —  {len(safe)} dirs")
    print(f"  (test completed and logged to WandB)")
    print("=" * 72)
    total_safe_mb = 0.0
    for e in safe:
        print(f"  ✅  {e['ckpt_dir'].parent.name:<58}  {e['size_mb']:>7.0f} MB")
        total_safe_mb += e["size_mb"]
    if safe:
        print(f"\n  Recoverable: {total_safe_mb / 1024:.2f} GB")
    else:
        print("  (none)")

    print()
    print("=" * 72)
    print(f"  INCOMPLETE / NOT IN WANDB  —  {len(incomplete)} dirs")
    print(f"  (no matching completed WandB run — do NOT delete)")
    print("=" * 72)
    for e in incomplete:
        print(f"  ⚠️   {e['ckpt_dir'].parent.name:<58}  {e['size_mb']:>7.0f} MB  (looking for: '{e['expected_run']}')")

    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Delete (if --delete passed)
    # ─────────────────────────────────────────────────────────────────────────

    if not args.delete:
        print("Dry run — nothing deleted.")
        if safe:
            cmd = "python wandb_checkpoint_audit.py --delete"
            if args.encoder:
                cmd += f" --encoder \"{args.encoder}\""
            print(f"To delete the safe ones: {cmd}")
        return

    if not safe:
        print("Nothing to delete.")
        return

    print(f"Deleting {len(safe)} checkpoint directories...")
    freed = 0.0
    for e in safe:
        print(f"  🗑️   rm -rf {e['ckpt_dir']}")
        shutil.rmtree(e["ckpt_dir"])
        freed += e["size_mb"]

    print(f"\n✅  Done. Freed {freed / 1024:.2f} GB.")


if __name__ == "__main__":
    main()
