#!/usr/bin/env python3
"""
scripts/maintenance/wandb_checkpoint_audit.py

Cross-references local checkpoint directories against WandB to determine
which are safe to delete (test phase completed and metrics logged to WandB).

Run from the repo root (paths are cwd-relative — scans ``output/``).

Filters (all substring matches, case-insensitive, AND-combined):
    --encoder <s>   substring of the encoder slug (e.g. ``MuQ``, ``MERT-v1-95M``)
    --task <s>      substring of the task slug    (e.g. ``Covers80``, ``VGMIDITVar``)

Usage:
    # Interactive (recommended): pick which (encoder, task) pairs to
    # delete from a numbered table; safe by default — confirms before
    # deleting:
    uv run python scripts/maintenance/wandb_checkpoint_audit.py -i

    # Audit everything (dry-run with totals):
    uv run python scripts/maintenance/wandb_checkpoint_audit.py

    # Scripted: delete safe checkpoints for one specific (encoder, task) pair:
    uv run python scripts/maintenance/wandb_checkpoint_audit.py \\
        --encoder MERT-v1-330M --task SHS100K --delete

    # Filter to all MERT runs across all tasks:
    uv run python scripts/maintenance/wandb_checkpoint_audit.py --encoder MERT

    # Group the summary by (encoder, task) instead of listing every dir:
    uv run python scripts/maintenance/wandb_checkpoint_audit.py --by-pair

The ``--delete`` flag is opt-in; without it the script is a dry-run.
``--by-pair`` is a presentation toggle (compact summary by encoder×task
totals), independent of ``--delete``. ``-i / --interactive`` overrides
both: it always shows the per-pair table, prompts for a selection,
and confirms before deleting.
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
        return {
            "task": m.group(1),
            "encoder": m.group(2),
            "layer": int(m.group(3)),
            "meanall": False,
        }
    m = re.match(r"^probe\.(.+)\.(.+)-meanall$", name)
    if m:
        return {"task": m.group(1), "encoder": m.group(2), "layer": None, "meanall": True}
    return None


def has_test_metrics(run) -> bool:
    try:
        return any(k.startswith("test/") for k in run.summary)
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


def parse_selection(spec: str, n_items: int) -> set[int]:
    """Parse an interactive selection string into a set of 1-indexed item IDs.

    Accepts:
      - ``"all"`` → every item (1..n_items)
      - ``"1,3,5"`` → discrete IDs
      - ``"1-3"`` → inclusive range
      - ``"1-3,5,7-9"`` → mixed
      - empty / ``"none"`` / ``"0"`` → empty set
      - ``"q"`` → returns ``None`` (signal: cancel)

    Out-of-range IDs are silently dropped (no error, since the prompt
    will show the valid range right above). Whitespace-tolerant.

    Used by the interactive cleanup flow to convert the user's
    free-form selection into a concrete set of pair indices to delete.
    """
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "0"):
        return set()
    if spec in ("q", "quit", "exit", "cancel"):
        return None  # type: ignore[return-value]  # sentinel = cancel
    if spec == "all":
        return set(range(1, n_items + 1))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                if 1 <= i <= n_items:
                    out.add(i)
        else:
            try:
                i = int(part)
            except ValueError:
                continue
            if 1 <= i <= n_items:
                out.add(i)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Audit checkpoints vs WandB; delete completed ones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete safe checkpoints")
    parser.add_argument(
        "--encoder",
        default=None,
        help="Filter to this encoder (case-insensitive substring match on the "
        "encoder slug parsed from the output dir name). Combines with --task "
        "via AND.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Filter to this task (case-insensitive substring match on the "
        "task slug parsed from the output dir name, e.g. 'Covers80', "
        "'VGMIDITVar'). Combines with --encoder via AND.",
    )
    parser.add_argument(
        "--by-pair",
        action="store_true",
        help="Print a compact (encoder × task) summary table instead of "
        "listing every directory. Useful when scoping a bulk cleanup.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive cleanup: show numbered (encoder × task) pairs, "
        "prompt for which to delete (e.g. '1,3,5' or '1-3' or 'all'), "
        "confirm before deleting. Overrides --by-pair (the table is "
        "always shown) and ignores --delete (confirmation is the gate).",
    )
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
        # AND-combined filters: a dir is included only if ALL active filters
        # accept it. ``None`` means "no filter".
        if args.encoder and args.encoder.lower() not in parsed["encoder"].lower():
            continue
        if args.task and args.task.lower() not in parsed["task"].lower():
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
    print(
        f"Total runs: {len(all_runs)}  |  Completed (have test/* metrics): {len(completed_runs)}\n"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Cross-reference each local dir against WandB
    # ─────────────────────────────────────────────────────────────────────────

    safe = []
    incomplete = []

    for entry in probe_dirs:
        layer_name = "layer-meanall-test" if entry["meanall"] else f"layer-{entry['layer']}-test"
        match = next(
            (
                r
                for r in completed_runs
                if run_matches(r, entry["encoder"], entry["task"], layer_name)
            ),
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

    # Always compute pair_stats — used by both --by-pair and --interactive.
    from collections import defaultdict

    pair_stats = defaultdict(lambda: {"safe_n": 0, "safe_mb": 0.0, "inc_n": 0, "inc_mb": 0.0})
    for e in safe:
        pair_stats[(e["encoder"], e["task"])]["safe_n"] += 1
        pair_stats[(e["encoder"], e["task"])]["safe_mb"] += e["size_mb"]
    for e in incomplete:
        pair_stats[(e["encoder"], e["task"])]["inc_n"] += 1
        pair_stats[(e["encoder"], e["task"])]["inc_mb"] += e["size_mb"]

    if args.by_pair or args.interactive:
        # Show the (encoder × task) summary. In interactive mode, prefix each
        # row with a 1-indexed ID so the user can pick by number.
        # Pairs with zero safe dirs are STILL shown (so the user sees the
        # "incomplete only" state) but can't be selected for deletion.
        sorted_pairs = sorted(pair_stats.items())
        print("=" * 92)
        idx_h = "ID" if args.interactive else "  "
        print(f"  {idx_h:>3}  {'Encoder':<28}  {'Task':<22}  {'Safe':>10}  {'Incomplete':>12}")
        print("=" * 92)
        total_safe_mb = 0.0
        for i, ((encoder, task), s) in enumerate(sorted_pairs, start=1):
            safe_str = f"{s['safe_n']:>2} ({s['safe_mb'] / 1024:.2f}G)"
            inc_str = f"{s['inc_n']:>2} ({s['inc_mb'] / 1024:.2f}G)" if s["inc_n"] else "—"
            id_col = f"{i:>3}" if args.interactive else "   "
            print(f"  {id_col}  {encoder:<28}  {task:<22}  {safe_str:>10}  {inc_str:>12}")
            total_safe_mb += s["safe_mb"]
        print("=" * 92)
        print(f"  Total recoverable across all pairs: {total_safe_mb / 1024:.2f} GB")
        print()
    else:
        print("=" * 72)
        print(f"  SAFE TO DELETE  —  {len(safe)} dirs")
        print("  (test completed and logged to WandB)")
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
        print("  (no matching completed WandB run — do NOT delete)")
        print("=" * 72)
        for e in incomplete:
            print(
                f"  ⚠️   {e['ckpt_dir'].parent.name:<58}  {e['size_mb']:>7.0f} MB  (looking for: '{e['expected_run']}')"
            )

        print()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Delete
    # ─────────────────────────────────────────────────────────────────────────

    if args.interactive:
        # Interactive mode: prompt for which pair IDs to delete from the
        # table above, confirm with a summary, then delete. Pair IDs map
        # 1:1 with the `sorted_pairs` enumeration in the report block
        # above (they're the same `sorted(pair_stats.items())`).
        if not safe:
            print("Nothing safe to delete.")
            return
        sorted_pairs = sorted(pair_stats.items())
        print(
            "Select pairs to delete by ID (e.g. '1,3,5' or '1-3' or "
            "'1-3,7' or 'all', empty/'q' to cancel)."
        )
        try:
            selection_raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
        selection = parse_selection(selection_raw, len(sorted_pairs))
        if selection is None:
            print("Cancelled.")
            return
        if not selection:
            print("Nothing selected.")
            return

        # Build the concrete list of (encoder, task) pairs to delete and the
        # set of safe dirs that belong to them.
        chosen_pairs = {sorted_pairs[i - 1][0] for i in selection}
        to_delete = [e for e in safe if (e["encoder"], e["task"]) in chosen_pairs]
        to_delete_mb = sum(e["size_mb"] for e in to_delete)

        print(
            f"\nAbout to delete {len(to_delete)} checkpoint directories "
            f"across {len(chosen_pairs)} pair(s), freeing "
            f"{to_delete_mb / 1024:.2f} GB:"
        )
        for encoder, task in sorted(chosen_pairs):
            s = pair_stats[(encoder, task)]
            print(
                f"  - {encoder:<28}  {task:<22}  "
                f"{s['safe_n']:>2} dirs  ({s['safe_mb'] / 1024:.2f} GB)"
            )
        try:
            confirm = input("\nProceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return
        if confirm not in ("y", "yes"):
            print("Cancelled.")
            return

        print(f"\nDeleting {len(to_delete)} checkpoint directories...")
        freed = 0.0
        for e in to_delete:
            print(f"  🗑️   rm -rf {e['ckpt_dir']}")
            shutil.rmtree(e["ckpt_dir"])
            freed += e["size_mb"]
        print(f"\n✅  Done. Freed {freed / 1024:.2f} GB.")
        return

    if not args.delete:
        print("Dry run — nothing deleted.")
        if safe:
            cmd = "uv run python scripts/maintenance/wandb_checkpoint_audit.py --delete"
            if args.encoder:
                cmd += f' --encoder "{args.encoder}"'
            if args.task:
                cmd += f' --task "{args.task}"'
            print(f"To delete the safe ones: {cmd}")
            print(
                "Or use -i / --interactive to pick which (encoder × task) "
                "pairs to delete from a numbered table."
            )
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
