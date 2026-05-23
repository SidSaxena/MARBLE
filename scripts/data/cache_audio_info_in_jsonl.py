#!/usr/bin/env python3
"""Add num_samples + sample_rate to JSONL records by scanning audio files.

Some datamodules (HookTheoryMelody, specifically) read audio metadata at
init time by calling `torchaudio.info(audio_path)` for every record. On a
local SSD this is fast; on a network-backed Modal volume, each call is a
network round-trip and 11.5 k records take ~16 minutes just to build the
index map — the training loop never gets to start.

This script walks the audio directory once in parallel (thread pool, I/O-
bound work), computes `num_samples` and `sample_rate` per file, and writes
them back into the JSONL records as new fields. The HookTheoryMelody
datamodule (and the others, generally) prefer the cached fields when
present so dataset init becomes instant.

Properties
──────────
- **Idempotent**: records that already have both fields are skipped (no
  re-scan unless `--force`). Re-runnable on partial outputs.
- **Atomic**: writes to `<jsonl>.tmp` then renames over the original, so a
  Ctrl-C mid-write doesn't leave a half-broken file.
- **Tolerates missing audio**: if a YouTube ID's MP3 is missing on disk,
  the record is left unchanged (no fields added). The datamodule still
  has to handle the missing-audio case (which it already does for
  KeyError on misaligned records).
- **Schema-agnostic**: uses `--id-key` and `--audio-suffix` to derive the
  audio path from each record. Default is HookTheory's `hooktheory.id`
  → `<id>.mp3` pattern.

Usage
─────
    # HookTheory (local data dir, the dataset path expected by the datamodule)
    uv run python scripts/data/cache_audio_info_in_jsonl.py \
        --jsonl data/HookTheory/HookTheory.train.jsonl \
        --jsonl data/HookTheory/HookTheory.val.jsonl \
        --jsonl data/HookTheory/HookTheory.test.jsonl \
        --audio-dir data/HookTheory/audio \
        --id-key "youtube.id" \
        --workers 16

    # Re-cache everything from scratch
    uv run python scripts/data/cache_audio_info_in_jsonl.py \
        --jsonl data/HookTheory/HookTheory.train.jsonl \
        --audio-dir data/HookTheory/audio \
        --id-key "youtube.id" \
        --force
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _get_nested(d: dict, dotted_key: str) -> Any:
    """Resolve a dotted key like `youtube.id` from a nested dict."""
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _audio_meta(path: Path) -> tuple[int, int] | None:
    """Return (num_samples, sample_rate) or None if file missing/unreadable."""
    if not path.exists():
        return None
    # torchaudio is the canonical reader the datamodules use. Importing
    # lazily so the script is usable on machines without torchaudio
    # installed (defensive — `torchaudio` is a core marble dep, but if
    # someone runs this in a stripped env it shouldn't import-fail at
    # module load).
    import torchaudio

    try:
        info = torchaudio.info(str(path))
    except RuntimeError as exc:
        print(f"  ! unreadable: {path.name}: {exc}", file=sys.stderr)
        return None
    return int(info.num_frames), int(info.sample_rate)


def cache_for_jsonl(
    jsonl: Path,
    audio_dir: Path,
    id_key: str,
    audio_suffix: str,
    workers: int,
    force: bool,
) -> tuple[int, int, int]:
    """Scan one JSONL, add num_samples + sample_rate where missing.

    Returns (n_records, n_cached_this_run, n_skipped_already_cached).
    """
    if not jsonl.exists():
        print(f"  ! missing JSONL: {jsonl}", file=sys.stderr)
        return 0, 0, 0

    print(f"\n[{jsonl.name}]")
    records: list[dict] = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  {len(records):,} records loaded")

    # Pick records that need scanning
    todo: list[tuple[int, Path]] = []  # (record_idx, audio_path)
    skipped = 0
    for idx, rec in enumerate(records):
        has_ns = isinstance(rec.get("num_samples"), int)
        has_sr = isinstance(rec.get("sample_rate"), int)
        if has_ns and has_sr and not force:
            skipped += 1
            continue
        audio_id = _get_nested(rec, id_key)
        if audio_id is None:
            continue  # malformed record — leave alone
        audio_path = audio_dir / f"{audio_id}{audio_suffix}"
        todo.append((idx, audio_path))

    print(f"  {len(todo):,} need scanning, {skipped:,} already cached")
    if not todo:
        return len(records), 0, skipped

    # Parallel scan
    cached = 0
    missing = 0
    unreadable = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_audio_meta, audio_path): (idx, audio_path) for idx, audio_path in todo
        }
        done = 0
        for fut in as_completed(futures):
            idx, audio_path = futures[fut]
            done += 1
            if done % 500 == 0:
                print(f"  ... {done}/{len(todo)} scanned", flush=True)
            result = fut.result()
            if result is None:
                if audio_path.exists():
                    unreadable += 1
                else:
                    missing += 1
                continue
            num_frames, sample_rate = result
            records[idx]["num_samples"] = num_frames
            records[idx]["sample_rate"] = sample_rate
            cached += 1

    print(f"  cached: {cached}, missing audio: {missing}, unreadable: {unreadable}")

    # Atomic write
    tmp = jsonl.with_suffix(jsonl.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    tmp.replace(jsonl)
    print(f"  wrote → {jsonl}")
    return len(records), cached, skipped


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl",
        type=Path,
        action="append",
        required=True,
        help="JSONL file to patch. Repeat for multiple splits.",
    )
    ap.add_argument(
        "--audio-dir", type=Path, required=True, help="Directory containing the audio files."
    )
    ap.add_argument(
        "--id-key",
        default="youtube.id",
        help="Dotted key in each record that resolves to the "
        "audio file's stem (default: youtube.id for HookTheory).",
    )
    ap.add_argument(
        "--audio-suffix",
        default=".mp3",
        help="File suffix appended to the resolved id (default: .mp3)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel torchaudio.info workers. "
        "Higher = faster on network FS, more CPU "
        "saturation on local FS. (default: 16)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-scan even records that already have cached "
        "fields (use if you suspect stale values).",
    )
    args = ap.parse_args()

    total_records = total_cached = total_skipped = 0
    for jsonl in args.jsonl:
        n, c, s = cache_for_jsonl(
            jsonl=jsonl,
            audio_dir=args.audio_dir,
            id_key=args.id_key,
            audio_suffix=args.audio_suffix,
            workers=args.workers,
            force=args.force,
        )
        total_records += n
        total_cached += c
        total_skipped += s

    print(
        f"\nTotal: {total_records:,} records, "
        f"{total_cached:,} cached this run, "
        f"{total_skipped:,} already cached."
    )


if __name__ == "__main__":
    main()
