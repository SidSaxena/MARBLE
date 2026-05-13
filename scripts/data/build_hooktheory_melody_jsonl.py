#!/usr/bin/env python3
"""scripts/data/build_hooktheory_melody_jsonl.py
─────────────────────────────────────────────
Build `HookTheory.{train,val,test}.jsonl` for the HookTheoryMelody task.

The HookTheoryMelody datamodule expects per-SONG records with the raw
HookTheory annotation tree (beat-aligned melody notes, beat→time
alignment, full-song audio). This is a different schema than the
clip-level `HookTheoryKey.*.jsonl` / `HookTheoryStructure.*.jsonl` that
`download.py` extracts.

Upstream source: `m-a-p/HookTheory` on HuggingFace
   * `Hooktheory.json.gz` (19 MB)  — has 26,175 entries, each with the
     exact schema the datamodule reads (youtube.id, alignment.refined.
     {beats,times}, annotations.{melody,num_beats}).
   * `zips/audio/part_*.tar` (~104 GB total) — full-song audio,
     intentionally skipped by `download.py` for the Key/Structure tasks
     because they only need pre-segmented clips. Melody needs the
     full-song audio because note onsets/offsets are in song-relative
     beats.

What this script does
---------------------
1. Loads `Hooktheory.json.gz` (downloaded automatically from HF if
   missing).
2. Groups songs by their declared `split` field (TRAIN / VALID / TEST).
3. Filters out songs with empty melody annotations (~2,300 of 26,175).
4. Optionally filters out songs whose full audio file is missing
   (when `--filter-by-audio` is set against `--audio-dir`).
5. Writes three JSONLs: `HookTheory.train.jsonl`, `HookTheory.val.jsonl`,
   `HookTheory.test.jsonl`.

Each record IS the upstream per-song dict — the datamodule reads the
fields it needs directly without renaming.

Usage
-----
    # Local: build from cached HF data, skip audio filter
    uv run python scripts/data/build_hooktheory_melody_jsonl.py

    # Build + filter by which full-song audio actually exists on disk
    uv run python scripts/data/build_hooktheory_melody_jsonl.py \\
        --audio-dir data/HookTheory/audio --filter-by-audio

    # Custom output dir (default: data/HookTheory/)
    uv run python scripts/data/build_hooktheory_melody_jsonl.py \\
        --out-dir /mnt/marble-data/HookTheory
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


SOURCE_REPO = "m-a-p/HookTheory"
SOURCE_FILE = "Hooktheory.json.gz"


def _fetch_source(cache_dir: Path | None = None) -> Path:
    """Download `Hooktheory.json.gz` from HF (or return cached path)."""
    from huggingface_hub import hf_hub_download
    kwargs = {"repo_id": SOURCE_REPO, "filename": SOURCE_FILE,
              "repo_type": "dataset"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return Path(hf_hub_download(**kwargs))


def _load_songs(source_path: Path) -> dict:
    with gzip.open(source_path, "rt", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source-json", type=Path, default=None,
                    help="Path to Hooktheory.json.gz. If omitted, fetched "
                         "from HF (m-a-p/HookTheory).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/HookTheory"),
                    help="Where to write HookTheory.{train,val,test}.jsonl "
                         "(default: %(default)s)")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="Directory containing full-song audio "
                         "(<ytid>.mp3). Required if --filter-by-audio.")
    ap.add_argument("--filter-by-audio", action="store_true",
                    help="Drop songs whose <audio-dir>/<ytid>.mp3 is "
                         "missing. Otherwise keep all songs with non-empty "
                         "melody.")
    ap.add_argument("--hf-cache-dir", type=Path, default=None,
                    help="Override HuggingFace cache dir for the auto-"
                         "download. Useful when running inside Modal to "
                         "land the cache on a persistent volume.")
    args = ap.parse_args()

    # ── Load source ──────────────────────────────────────────────────────────
    source = args.source_json or _fetch_source(args.hf_cache_dir)
    print(f"Loading {source} ...", file=sys.stderr)
    songs = _load_songs(source)
    print(f"  {len(songs):,} song entries", file=sys.stderr)

    if args.filter_by_audio and args.audio_dir is None:
        ap.error("--filter-by-audio requires --audio-dir")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Group by split ───────────────────────────────────────────────────────
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    skipped_no_melody = 0
    skipped_no_audio = 0
    skipped_no_youtube = 0

    SPLIT_MAP = {"TRAIN": "train", "VALID": "val", "TEST": "test"}

    for hid, song in songs.items():
        split = SPLIT_MAP.get(song.get("split", ""))
        if split is None:
            continue

        melody = song.get("annotations", {}).get("melody")
        if not melody:
            skipped_no_melody += 1
            continue

        ytid = song.get("youtube", {}).get("id")
        if not ytid:
            skipped_no_youtube += 1
            continue

        if args.filter_by_audio:
            audio_path = args.audio_dir / f"{ytid}.mp3"
            if not audio_path.exists():
                skipped_no_audio += 1
                continue

        by_split[split].append(song)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(" HookTheoryMelody JSONL build summary", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"  Total source songs:    {len(songs):>6,}", file=sys.stderr)
    print(f"  Skipped (no melody):   {skipped_no_melody:>6,}", file=sys.stderr)
    print(f"  Skipped (no ytid):     {skipped_no_youtube:>6,}", file=sys.stderr)
    if args.filter_by_audio:
        print(f"  Skipped (no audio):    {skipped_no_audio:>6,}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"  Output records:", file=sys.stderr)
    for split, recs in by_split.items():
        print(f"    {split:5s}  {len(recs):>6,}", file=sys.stderr)
    print("=" * 64, file=sys.stderr)

    # ── Write JSONLs ─────────────────────────────────────────────────────────
    for split, recs in by_split.items():
        out = args.out_dir / f"HookTheory.{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  wrote {out}  ({len(recs):,} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
