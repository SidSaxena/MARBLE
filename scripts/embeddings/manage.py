#!/usr/bin/env python3
"""scripts/embeddings/manage.py
─────────────────────────────────
Inspect + clean up the per-clip embedding cache populated by
``CoverRetrievalTask`` (and clip-level supervised tasks once integrated).

Subcommands
-----------

::

    # List every cache directory under output/.emb_cache/ with sizes
    uv run python scripts/embeddings/manage.py list

    # Show the _meta.json for a specific cache directory
    uv run python scripts/embeddings/manage.py info \\
        output/.emb_cache/OMARRQ-multifeature-25hz/SHS100K__a3b8c1d2/

    # Delete all caches for one encoder (regain disk between experiments)
    uv run python scripts/embeddings/manage.py clear OMARRQ-multifeature-25hz

    # Delete one specific cache directory
    uv run python scripts/embeddings/manage.py clear \\
        OMARRQ-multifeature-25hz/SHS100K__a3b8c1d2

All destructive ops dry-run by default — pass ``--apply`` to actually
delete.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from marble.utils.emb_cache import DEFAULT_CACHE_ROOT


def _dir_size_bytes(p: Path) -> int:
    total = 0
    for sub in p.rglob("*"):
        if sub.is_file():
            total += sub.stat().st_size
    return total


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _count_clips(p: Path) -> int:
    return sum(1 for f in p.iterdir() if f.is_file() and f.suffix == ".pt")


# ──────────────────────────────────────────────────────────────────────────
# Subcommands
# ──────────────────────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    root = Path(args.root)
    if not root.exists():
        print(f"No cache root at {root}")
        return
    encoder_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not encoder_dirs:
        print(f"Cache root {root} is empty.")
        return

    header = f"{'cache dir':<60} {'clips':>10} {'size':>12}  meta-summary"
    print(header)
    print("-" * len(header))
    grand_total = 0
    for enc_dir in encoder_dirs:
        # Each <encoder>/ contains <task>__<hash>/ subdirectories
        for cache_dir in sorted(d for d in enc_dir.iterdir() if d.is_dir()):
            n_clips = _count_clips(cache_dir)
            size = _dir_size_bytes(cache_dir)
            grand_total += size
            rel = cache_dir.relative_to(root)
            # Meta summary: pull encoder_model_id + clip_seconds from _meta.json
            meta_summary = ""
            meta_path = cache_dir / "_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    mid = meta.get("encoder_model_id", "?")
                    cs = meta.get("clip_seconds", "?")
                    meta_summary = f"{mid} @ {cs}s"
                except (OSError, json.JSONDecodeError):
                    meta_summary = "(meta unreadable)"
            print(f"{str(rel):<60} {n_clips:>10,} {_human_bytes(size):>12}  {meta_summary}")
    print("-" * len(header))
    print(f"{'TOTAL':<60} {'':<10} {_human_bytes(grand_total):>12}")


def cmd_info(args: argparse.Namespace) -> None:
    target = Path(args.dir)
    if not target.exists() or not target.is_dir():
        print(f"Not a directory: {target}", file=sys.stderr)
        sys.exit(1)
    meta_path = target / "_meta.json"
    if not meta_path.exists():
        print(f"No _meta.json in {target} — this directory may not be a cache.", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    n_clips = _count_clips(target)
    size = _dir_size_bytes(target)
    print(f"Cache directory:  {target}")
    print(f"Clips on disk:    {n_clips:,}")
    print(f"Total size:       {_human_bytes(size)}")
    print("Metadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")


def cmd_clear(args: argparse.Namespace) -> None:
    root = Path(args.root)
    rel = args.target  # may be "<encoder>" or "<encoder>/<task__hash>"
    target = root / rel
    if not target.exists():
        print(f"Nothing to delete at {target}")
        return
    if not target.is_dir():
        print(f"Not a directory: {target}", file=sys.stderr)
        sys.exit(1)

    n_clips = (
        _count_clips(target)
        if (target.parent != root)
        else sum(_count_clips(d) for d in target.iterdir() if d.is_dir())
    )
    size = _dir_size_bytes(target)
    print(f"Target:  {target}")
    print(f"Will delete: {n_clips:,} cached clips, {_human_bytes(size)}")
    if not args.apply:
        print("\n(dry-run — pass --apply to actually delete)")
        return

    shutil.rmtree(target)
    print("✓ deleted.")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--root",
        default=str(DEFAULT_CACHE_ROOT),
        help=f"Cache root directory (default: {DEFAULT_CACHE_ROOT}).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List all cache directories with sizes.")
    sp_list.set_defaults(func=cmd_list)

    sp_info = sub.add_parser("info", help="Show metadata + size for one cache directory.")
    sp_info.add_argument("dir", help="Path to a cache directory (absolute or relative).")
    sp_info.set_defaults(func=cmd_info)

    sp_clear = sub.add_parser("clear", help="Delete cache directories.")
    sp_clear.add_argument(
        "target",
        help="Subdirectory under the cache root to delete. Can be an "
        "encoder slug (e.g. 'OMARRQ-multifeature-25hz' deletes all of "
        "its caches) or '<encoder>/<task__hash>' for one specific cache.",
    )
    sp_clear.add_argument(
        "--apply", action="store_true", help="Actually delete (default: dry-run)."
    )
    sp_clear.set_defaults(func=cmd_clear)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
