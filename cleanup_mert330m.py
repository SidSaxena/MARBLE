#!/usr/bin/env python3
"""
cleanup_mert330m.py

Delete MERT-v1-330M checkpoints and embedding cache while preserving logs.

Usage:
    python cleanup_mert330m.py              # dry-run (show what would be deleted)
    python cleanup_mert330m.py --apply      # actually delete
"""

import argparse
import shutil
import sys
from pathlib import Path


def format_size(bytes_val):
    """Convert bytes to human-readable format."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_val < 1024 or unit == "TB":
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Delete MERT-v1-330M checkpoints and embedding cache (preserves logs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cleanup_mert330m.py              # dry-run
    python cleanup_mert330m.py --apply      # actually delete
        """,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )

    args = parser.parse_args()

    encoder_pattern = "MERT-v1-330M"
    output_dir = Path("output")
    cache_root = output_dir / ".emb_cache"

    print("=" * 72)
    print(f"  MERT-v1-330M Cleanup")
    print("=" * 72)
    print(f"Pattern: {encoder_pattern}")
    print(f"Mode:    {'--apply (DESTRUCTIVE)' if args.apply else 'dry-run'}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # Find checkpoints
    # ─────────────────────────────────────────────────────────────────────────

    print("📋 Checkpoints to delete:")
    ckpt_dirs = []
    if output_dir.exists():
        for ckpt_dir in output_dir.glob(f"*{encoder_pattern}*/checkpoints"):
            if ckpt_dir.is_dir():
                size = get_dir_size(ckpt_dir)
                print(f"   • {ckpt_dir.relative_to('.')} ({format_size(size)})")
                ckpt_dirs.append(ckpt_dir)

    if not ckpt_dirs:
        print("   (none found)")

    # ─────────────────────────────────────────────────────────────────────────
    # Find embedding cache
    # ─────────────────────────────────────────────────────────────────────────

    print()
    print("📋 Embedding cache to delete:")
    cache_dirs = []
    if cache_root.exists():
        for cache_dir in cache_root.glob(f"*{encoder_pattern}*"):
            if cache_dir.is_dir():
                size = get_dir_size(cache_dir)
                print(f"   • {cache_dir.relative_to('.')} ({format_size(size)})")
                cache_dirs.append(cache_dir)

    if not cache_dirs:
        print("   (none found)")

    # ─────────────────────────────────────────────────────────────────────────
    # Find logs (for reference only)
    # ─────────────────────────────────────────────────────────────────────────

    print()
    print("📋 Logs (PRESERVING):")
    log_files = []
    log_dir = output_dir / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob(f"*{encoder_pattern}*"):
            if log_file.is_file():
                size = get_dir_size(log_file)
                print(f"   • {log_file.relative_to('.')} ({format_size(size)})")
                log_files.append(log_file)

    if not log_files:
        print("   (none found)")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────────

    total_size = sum(get_dir_size(d) for d in ckpt_dirs) + sum(
        get_dir_size(d) for d in cache_dirs
    )

    print()
    print("=" * 72)
    print(f"Total to delete: {format_size(total_size)}")
    print("=" * 72)
    print()

    if not args.apply:
        print("✓ DRY RUN — nothing deleted")
        print()
        print("To actually delete, run:")
        print("  python cleanup_mert330m.py --apply")
        print()
        return

    # ─────────────────────────────────────────────────────────────────────────
    # DESTRUCTIVE: Actually delete
    # ─────────────────────────────────────────────────────────────────────────

    print("⚠️  DELETING...")
    print()

    deleted_count = 0

    # Delete checkpoints
    for ckpt_dir in ckpt_dirs:
        try:
            print(f"🗑️  rm -rf {ckpt_dir.relative_to('.')}")
            shutil.rmtree(ckpt_dir)
            deleted_count += 1
        except (OSError, PermissionError) as e:
            print(f"   ✗ Error: {e}", file=sys.stderr)

    # Delete cache directories
    for cache_dir in cache_dirs:
        try:
            print(f"🗑️  rm -rf {cache_dir.relative_to('.')}")
            shutil.rmtree(cache_dir)
            deleted_count += 1
        except (OSError, PermissionError) as e:
            print(f"   ✗ Error: {e}", file=sys.stderr)

    print()
    print("=" * 72)
    print("✅ CLEANUP COMPLETE")
    print("=" * 72)
    print(f"Deleted {len(ckpt_dirs)} checkpoint dir(s) + {len(cache_dirs)} cache dir(s)")
    print(f"Freed: {format_size(total_size)}")
    print("Logs preserved at: ./output/logs/")
    print()


if __name__ == "__main__":
    main()
