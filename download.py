import argparse
import sys
import os
import time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

__all_datasets__ = [
    "GTZAN",
    "EMO",
    "GS",
    "Chords1217",
    "MTG",
    "MTT",
    "HookTheory",
]

__gated_datasets__ = [
    "Chords1217",
    "HookTheory",
]

def extract_HookTheory(dataset_dir: str):
    """
    Extract pre-segmented clip archives from the HookTheory dataset.

    The HuggingFace repo contains:
      zips/hooktheory_clips/part_XXXXXX.tar  (~4.1 GB, 23 parts)  ← extracted here
      zips/audio/part_XXXXXX.tar             (~104 GB, 79 parts)  ← skipped (not needed)

    After extraction, MP3 clips land in:
      <dataset_dir>/hooktheory_clips/<hooktheory_id>.mp3

    The JSONL annotation files are already present at dataset_dir root
    (snapshot_download placed them there directly):
      HookTheoryKey.{train,val,test}.jsonl
      HookTheoryStructure.{train,val,test}.jsonl

    Uses Python's built-in tarfile module — no bash / system tar required,
    so this works on Windows as well as macOS/Linux.
    """
    import tarfile
    import glob

    clips_dir = os.path.join(dataset_dir, "hooktheory_clips")
    os.makedirs(clips_dir, exist_ok=True)

    pattern = os.path.join(dataset_dir, "zips", "hooktheory_clips", "*.tar")
    tar_paths = sorted(glob.glob(pattern))

    if not tar_paths:
        print(f"  Warning: no clip archives found at {pattern}")
        print("  The download may be incomplete. Re-run: uv run python download.py HookTheory")
        return

    print(f"  Extracting {len(tar_paths)} clip archive(s) → {clips_dir}")
    for tar_path in tar_paths:
        name = os.path.basename(tar_path)
        print(f"    {name} … ", end="", flush=True)
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(clips_dir)
        print("done")

    print("  Clips extracted successfully.")


def download_dataset(dataset_name: str, save_root: str, max_retries: int = 5):
    """
    Download a single dataset from HuggingFace into save_root/<dataset_name>/.

    Uses max_workers=2 to avoid the free-tier 5000-request/5-min rate limit.
    Retries up to max_retries times on 429 errors with exponential back-off.
    Re-running is always safe — already-downloaded files are skipped.

    HookTheory special handling:
      - Skips zips/audio/* (104 GB full-song audio — not needed for probing).
      - Downloads only zips/hooktheory_clips/* (~4.1 GB pre-segmented clips)
        plus the ready-made JSONL annotation files (~7 MB).
      - Extracts clips using Python tarfile (works on Windows).
    """
    repo_id = f"m-a-p/{dataset_name}"
    target_dir = os.path.join(save_root, dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    # HookTheory: skip the 104 GB full-song audio tars — only clips are needed.
    ignore_patterns = ["zips/audio/*"] if dataset_name == "HookTheory" else None
    extra = " (clips + JSONL only, skipping 104 GB full audio)" if ignore_patterns else ""
    print(f"Downloading '{dataset_name}'{extra} → '{target_dir}' …")

    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                max_workers=2,
                ignore_patterns=ignore_patterns,
            )
            break
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < max_retries:
                wait = 60 * attempt
                print(f"\n  Rate-limited (attempt {attempt}/{max_retries}). "
                      f"Waiting {wait}s … (re-running later also works)\n")
                time.sleep(wait)
            else:
                raise

    print(f"'{dataset_name}' downloaded to '{target_dir}'.")

    if dataset_name == "HookTheory":
        extract_HookTheory(target_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache specified Hugging Face datasets (or 'all' for every supported one)."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=f"Name of dataset to download (supported: {', '.join(__all_datasets__)}) or 'all' to download everything"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./data",
        help="Root directory under which to save datasets (default: ./data)"
    )
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        for ds in __all_datasets__:
            if ds in __gated_datasets__:
                bar = "*" * 50
                print(f"{bar}\n[NOTE] Dataset '{ds}' is gated. Visit https://huggingface.co/m-a-p/{ds} to request access.\n{bar}")
            download_dataset(ds, args.save_dir)
    else:
        ds = args.dataset
        if ds not in __all_datasets__:
            print(
                f"Error: Dataset '{ds}' is not supported. Choose from: {', '.join(__all_datasets__)}",
                file=sys.stderr
            )
            sys.exit(1)
        download_dataset(ds, args.save_dir)


if __name__ == "__main__":
    main()
