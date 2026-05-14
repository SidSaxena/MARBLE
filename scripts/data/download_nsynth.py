#!/usr/bin/env python3
"""
scripts/data/download_nsynth.py
──────────────────────────
Download the NSynth dataset (pitch classification) from Google Magenta
and generate JSONL metadata files for MARBLE probing.

Dataset:  https://magenta.tensorflow.org/datasets/nsynth
Source:   Google Cloud Storage (public, no sign-in needed)

Splits
------
  nsynth-train   ~289,205 clips  (~19 GB compressed)
  nsynth-valid   ~12,678  clips  (~800 MB)
  nsynth-test    ~4,096   clips  (~400 MB)

Audio format:  WAV, 16 kHz, mono, 64 000 samples (4.00 s)
Pitch range:   MIDI 21–108  (88 classes, A0 – C8)
Instruments:   11 families × 3 sources

Usage
-----
# Download all three splits (default)
python scripts/data/download_nsynth.py

# Download only valid + test (useful for quick evaluation)
python scripts/data/download_nsynth.py --splits valid test

# Custom data directory
python scripts/data/download_nsynth.py --data-dir /mnt/data

# Skip already-downloaded archives
python scripts/data/download_nsynth.py --no-download  # generate JSONL only
"""

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

# ─── URLs (from Google's NSynth page) ────────────────────────────────────────
SPLIT_URLS = {
    "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
    "valid": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "test": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
}

# Approximate uncompressed sizes (for user info only)
SPLIT_SIZES = {
    "train": "~19 GB",
    "valid": "~800 MB",
    "test": "~400 MB",
}

NSYNTH_SR = 16_000
NSYNTH_SAMPLES = 64_000  # 4 s × 16 kHz
NSYNTH_DUR = 4.0


# ─── helpers ─────────────────────────────────────────────────────────────────


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\r  {pct:3d}%  ({downloaded / 1e9:.2f} / {total_size / 1e9:.2f} GB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r  {downloaded / 1e9:.2f} GB downloaded …")
        sys.stdout.flush()


def download_split(split: str, dest_dir: Path) -> Path:
    """Download and extract one NSynth split. Returns the extracted directory."""
    url = SPLIT_URLS[split]
    archive = dest_dir / f"nsynth-{split}.jsonwav.tar.gz"
    split_d = dest_dir / f"nsynth-{split}"

    if split_d.exists() and (split_d / "audio").is_dir():
        print(f"  Already extracted → {split_d}")
        return split_d

    if not archive.exists():
        print(f"\nDownloading {split} split ({SPLIT_SIZES[split]}) …")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, archive, reporthook=_progress_hook)
        print()
    else:
        print(f"  Archive already present: {archive}")

    print("  Extracting …")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)
    print(f"  Extracted → {split_d}")
    return split_d


def generate_jsonl(split_dir: Path, out_jsonl: Path) -> int:
    """
    Parse examples.json from a NSynth split directory and write JSONL.

    NSynth examples.json format:
      {
        "bass_acoustic_000-021-075": {
          "note": 21,
          "velocity": 75,
          "instrument_family_str": "bass",
          "instrument_source_str": "acoustic",
          ...
        },
        ...
      }
    """
    examples_json = split_dir / "examples.json"
    audio_dir = split_dir / "audio"

    if not examples_json.exists():
        raise FileNotFoundError(f"examples.json not found in {split_dir}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio/ directory not found in {split_dir}")

    with open(examples_json) as f:
        examples = json.load(f)

    print(f"  Writing {len(examples)} records → {out_jsonl}")
    skipped = 0
    written = 0
    with open(out_jsonl, "w") as out:
        for key, meta in sorted(examples.items()):
            wav_path = audio_dir / f"{key}.wav"
            if not wav_path.exists():
                skipped += 1
                continue

            note = int(meta["note"])
            if not (21 <= note <= 108):
                skipped += 1
                continue

            record = {
                "audio_path": str(wav_path),
                "note": note,
                "velocity": int(meta.get("velocity", 0)),
                "instrument_family": meta.get("instrument_family_str", ""),
                "instrument_source": meta.get("instrument_source_str", ""),
                "sample_rate": NSYNTH_SR,
                "num_samples": NSYNTH_SAMPLES,
                "channels": 1,
                "duration": NSYNTH_DUR,
            }
            out.write(json.dumps(record) + "\n")
            written += 1

    if skipped:
        print(f"    Skipped {skipped} entries (missing audio or out-of-range note).")
    return written


# ─── main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Download NSynth and generate MARBLE JSONL files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Which splits to download (default: all three).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/NSynth",
        help="Root directory for NSynth data (default: data/NSynth).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading — regenerate JSONL from already-extracted data.",
    )
    args = parser.parse_args()

    dest = Path(args.data_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\nNSynth download  →  {dest.resolve()}")
    print(f"Splits: {args.splits}\n")

    # Map NSynth split name → MARBLE JSONL name (valid → val)
    name_map = {"train": "train", "valid": "val", "test": "test"}

    total_records = 0
    for split in args.splits:
        print(f"── {split} ────────────────────────────────────────────────────")

        if args.no_download:
            split_dir = dest / f"nsynth-{split}"
            if not split_dir.exists():
                print(f"  ERROR: {split_dir} not found.  Run without --no-download first.")
                continue
        else:
            split_dir = download_split(split, dest)

        out_name = f"NSynth.{name_map[split]}.jsonl"
        out_path = dest / out_name
        n = generate_jsonl(split_dir, out_path)
        total_records += n
        print(f"  ✓  {n} records  →  {out_path}\n")

    print(f"Done.  Total records written: {total_records}")
    print("\nNext steps:")
    print("  python scripts/sweeps/run_all_sweeps.py --tasks NSynth")
    print("  # or run a single sweep:")
    print("  python scripts/sweeps/run_sweep_local.py \\")
    print("      --base-config configs/probe.OMARRQ-multifeature25hz.NSynth.yaml \\")
    print("      --num-layers 24 --model-tag OMARRQ-multifeature25hz --task-tag NSynth")


if __name__ == "__main__":
    main()
