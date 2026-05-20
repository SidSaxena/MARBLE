#!/usr/bin/env python3
"""
scripts/data/download_hooktheory.py
──────────────────────────────
Download the HookTheory dataset from HuggingFace (m-a-p/HookTheory).

The dataset is GATED — request access first:
  https://huggingface.co/datasets/m-a-p/HookTheory

What gets downloaded (default, ~4.1 GB):
  • HookTheoryKey.{train,val,test}.jsonl        — annotation files (key labels)
  • HookTheoryStructure.{train,val,test}.jsonl  — annotation files (section labels)
  • hooktheory_clips/*.mp3                      — pre-segmented audio clips

With ``--with-full-audio`` (additional ~104 GB):
  • audio/*.mp3 — full-song mp3s extracted from the tar parts. Required
    for HookTheoryMelody (beat-aligned note onsets need the full song,
    not just the pre-segmented clips). After extraction the tar parts
    are kept for resumability; delete data/HookTheory/_tars/ to reclaim
    ~52 GB once you're sure the extraction succeeded.

Prerequisites
-------------
  1. Request dataset access (one-time, may take a day to approve):
       https://huggingface.co/datasets/m-a-p/HookTheory

  2. Authenticate the HuggingFace CLI (one-time):
       huggingface-cli login

Usage
-----
  # Clips only (HookTheoryKey + HookTheoryStructure):
  uv run python scripts/data/download_hooktheory.py
  uv run python scripts/data/download_hooktheory.py --data-dir /path/to/data

  # Clips + full-song audio (also enables HookTheoryMelody):
  uv run python scripts/data/download_hooktheory.py --with-full-audio

After download, build + run:
  uv run python scripts/data/build_hooktheory_melody_jsonl.py \\
      --audio-dir data/HookTheory/audio --filter-by-audio --out-dir data/HookTheory
  uv run python scripts/sweeps/run_all_sweeps.py --tasks HookTheoryMelody
"""

import argparse
import sys
from pathlib import Path

# Import download.py from the project root. The script lives at
# scripts/data/download_hooktheory.py, so the project root is three
# parents up: parents[0]=data, parents[1]=scripts, parents[2]=root.
# Using parents[2] is more legible than .parent.parent.parent.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from download import download_dataset  # noqa: E402  (sys.path setup above must run first)


def main():
    parser = argparse.ArgumentParser(
        description="Download HookTheory from HuggingFace (m-a-p/HookTheory).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data). Dataset is placed in <data-dir>/HookTheory/.",
    )
    parser.add_argument(
        "--with-full-audio",
        action="store_true",
        help="Also download + extract the full-song audio tars "
        "(~104 GB). Required for HookTheoryMelody; not needed for "
        "HookTheoryKey or HookTheoryStructure.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  HookTheory download  (m-a-p/HookTheory on HuggingFace)")
    print("=" * 60)
    print()
    if args.with_full_audio:
        print("Downloading: pre-segmented clips + JSONL + FULL-SONG AUDIO tars (~108 GB total).")
        print("After download the tars are extracted to audio/<ytid>.mp3 for HookTheoryMelody.")
    else:
        print("Downloading: pre-segmented clips + JSONL annotation files (~4.1 GB).")
        print("(Pass --with-full-audio to additionally fetch the 104 GB full-song audio.)")
    print()

    download_dataset("HookTheory", args.data_dir, with_full_audio=args.with_full_audio)

    print()
    print("All done. Next steps:")
    print("  HookTheoryKey/Structure sweeps:")
    print(
        "    uv run python scripts/sweeps/run_all_sweeps.py "
        "--tasks HookTheoryKey HookTheoryStructure"
    )
    if args.with_full_audio:
        print("  HookTheoryMelody (build JSONL first):")
        print(
            "    uv run python scripts/data/build_hooktheory_melody_jsonl.py \\\n"
            f"        --audio-dir {args.data_dir}/HookTheory/audio "
            "--filter-by-audio \\\n"
            f"        --out-dir {args.data_dir}/HookTheory"
        )
        print("    uv run python scripts/sweeps/run_all_sweeps.py --tasks HookTheoryMelody")


if __name__ == "__main__":
    main()
