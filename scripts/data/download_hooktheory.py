#!/usr/bin/env python3
"""
scripts/data/download_hooktheory.py
──────────────────────────────
Download the HookTheory dataset from HuggingFace (m-a-p/HookTheory).

The dataset is GATED — request access first:
  https://huggingface.co/datasets/m-a-p/HookTheory

What gets downloaded (~4.1 GB total):
  • HookTheoryKey.{train,val,test}.jsonl       — annotation files (key labels)
  • HookTheoryStructure.{train,val,test}.jsonl  — annotation files (section labels)
  • hooktheory_clips/*.mp3                       — pre-segmented audio clips

The 104 GB full-song audio tars are intentionally skipped — the pre-segmented
clips are all that is needed for the MARBLE probing tasks.

Prerequisites
-------------
  1. Request dataset access (one-time, may take a day to approve):
       https://huggingface.co/datasets/m-a-p/HookTheory

  2. Authenticate the HuggingFace CLI (one-time):
       huggingface-cli login

Usage
-----
  uv run python scripts/data/download_hooktheory.py
  uv run python scripts/data/download_hooktheory.py --data-dir /path/to/data

After download, run sweeps with:
  uv run python scripts/sweeps/run_all_sweeps.py --tasks HookTheoryKey HookTheoryStructure
"""

import argparse
import sys
from pathlib import Path

# Import download.py from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from download import download_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download HookTheory from HuggingFace (m-a-p/HookTheory).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory (default: data). Dataset is placed in <data-dir>/HookTheory/.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  HookTheory download  (m-a-p/HookTheory on HuggingFace)")
    print("=" * 60)
    print()
    print("This downloads pre-segmented clips + JSONL annotation files.")
    print("No YouTube downloads or ffmpeg extraction needed.")
    print()

    download_dataset("HookTheory", args.data_dir)

    print()
    print("All done. Start sweeps with:")
    print(
        "  uv run python scripts/sweeps/run_all_sweeps.py --tasks HookTheoryKey HookTheoryStructure"
    )


if __name__ == "__main__":
    main()
