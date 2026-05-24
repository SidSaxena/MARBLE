#!/usr/bin/env python3
"""Build smoke-test subsets of one or more JSONLs.

Why
───
Some MARBLE optimizations (e.g. MP3 → WAV conversion) require touching a
chunk of disk and ffmpeg time before you know whether they pay off. A
smoke test on a small deterministic subset of the corpus lets you measure
the perf delta on a few hundred batches without converting the entire
~10 k-file HookTheory MP3 corpus first.

This script trims a JSONL to its first N records and writes the result
as ``<stem>.smoke.jsonl`` alongside the original. Deterministic by
construction — re-running with the same ``--n`` produces a byte-identical
output, so smoke results are reproducible across runs and machines.

Properties
──────────
- **Deterministic**: first N records, no shuffling. ``--seed`` is
  intentionally absent; if you want a different subset, edit the source
  JSONL or use a different --n.
- **Atomic**: writes ``<stem>.smoke.jsonl.tmp`` then renames over the
  destination, so an interrupted run doesn't leave half-written files.
- **No schema munging**: each line is copied byte-for-byte (after the
  initial whitespace strip), so the smoke JSONL has the same record
  structure as the parent and works with every downstream consumer
  (datamodule, cache_audio_info_in_jsonl.py, convert_audio_to_wav.py).

Usage
─────
    uv run python scripts/data/build_smoke_jsonl.py \
        --jsonl data/HookTheory/HookTheory.train.jsonl \
        --jsonl data/HookTheory/HookTheory.val.jsonl \
        --jsonl data/HookTheory/HookTheory.test.jsonl \
        --n 500

    # → writes HookTheory.{train,val,test}.smoke.jsonl in the same dirs
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _smoke_path(jsonl: Path) -> Path:
    """``foo.train.jsonl`` → ``foo.train.smoke.jsonl`` in the same dir.

    Builds on ``Path.stem`` (which drops only the final ``.jsonl``) so
    multi-dot names like ``HookTheory.train.jsonl`` are handled correctly.
    Using ``.with_suffix(".smoke.jsonl")`` on the original would replace
    ``.jsonl`` instead of appending, producing ``HookTheory.train.smoke``
    + losing the proper extension.
    """
    return jsonl.with_name(jsonl.stem + ".smoke.jsonl")


def _subset_one(jsonl: Path, n: int) -> tuple[int, int]:
    """Take the first ``n`` non-blank lines of ``jsonl`` → ``*.smoke.jsonl``.

    Returns ``(n_written, n_total)`` where n_total is the original record
    count (so the caller can report "kept 500 of 8923").
    """
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)

    out = _smoke_path(jsonl)
    tmp = out.with_suffix(out.suffix + ".tmp")

    kept = 0
    total = 0
    with jsonl.open() as src, tmp.open("w") as dst:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            total += 1
            if kept < n:
                # Re-strip + write to canonicalize trailing whitespace; the
                # underlying JSON object is preserved verbatim.
                dst.write(line + "\n")
                kept += 1
        # Keep scanning to count totals even after we've hit N — cheap, lets
        # us report "kept N of TOTAL" which is useful for sanity-checking
        # that N didn't accidentally exceed the source size.
    tmp.replace(out)
    return kept, total


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        required=True,
        help="JSONL to subset. Repeat for multiple splits.",
    )
    ap.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of records to keep from each JSONL (the first N).",
    )
    args = ap.parse_args()

    if args.n <= 0:
        raise SystemExit(f"--n must be > 0, got {args.n}")

    print(f"\nbuild_smoke_jsonl: keeping first {args.n} records of each JSONL", flush=True)
    for jsonl in args.jsonl:
        kept, total = _subset_one(jsonl, args.n)
        out = _smoke_path(jsonl)
        print(f"  {jsonl.name}  → {out.name}  ({kept:,}/{total:,} records)")


if __name__ == "__main__":
    main()
