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

def extract_HookTheory(dataset_dir: str, include_full_audio: bool = False):
    """
    Extract clip archives (and optionally full-song audio) from the
    HookTheory dataset.

    The HuggingFace repo contains:
      zips/hooktheory_clips/part_XXXXXX.tar  (~4.1 GB, 23 parts)
      zips/audio/part_XXXXXX.tar             (~104 GB, 79 parts)

    Clip extraction (always):
      <dataset_dir>/hooktheory_clips/<hooktheory_id>.mp3
        — used by HookTheoryKey + HookTheoryStructure tasks.

    Full-audio extraction (only when include_full_audio=True):
      <dataset_dir>/audio/<youtube_id>.mp3
        — used by HookTheoryMelody (notes are aligned to song-relative
        beats, so per-song full audio is required).
    """
    import tarfile
    import glob

    def _extract(label: str, src_dir: str, dst_dir: str):
        os.makedirs(dst_dir, exist_ok=True)
        pattern = os.path.join(dataset_dir, "zips", src_dir, "*.tar")
        tar_paths = sorted(glob.glob(pattern))
        if not tar_paths:
            print(f"  Warning: no {label} archives found at {pattern}")
            return
        print(f"  Extracting {len(tar_paths)} {label} archive(s) → {dst_dir}")
        for tar_path in tar_paths:
            name = os.path.basename(tar_path)
            print(f"    {name} … ", end="", flush=True)
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(dst_dir)
            print("done")

    _extract("clip", "hooktheory_clips", os.path.join(dataset_dir, "hooktheory_clips"))
    if include_full_audio:
        _extract("full-audio", "audio", os.path.join(dataset_dir, "audio"))


def download_dataset(dataset_name: str, save_root: str,
                     max_retries: int = 5,
                     with_full_audio: bool = False):
    """
    Download a single dataset from HuggingFace into save_root/<dataset_name>/.

    Uses max_workers=2 to avoid the free-tier 5000-request/5-min rate limit.
    Retries up to max_retries times on 429 errors with exponential back-off.
    Re-running is always safe — already-downloaded files are skipped.

    HookTheory special handling:
      - By default, skips zips/audio/* (104 GB full-song audio).
      - Set with_full_audio=True to pull the full audio too — required for
        HookTheoryMelody, optional for Key/Structure.
      - Always extracts the pre-segmented clips (~4.1 GB).
    """
    repo_id = f"m-a-p/{dataset_name}"
    target_dir = os.path.join(save_root, dataset_name)
    os.makedirs(target_dir, exist_ok=True)

    # HookTheory: skip the 104 GB full-song audio tars by default.
    if dataset_name == "HookTheory" and not with_full_audio:
        ignore_patterns = ["zips/audio/*"]
        extra = " (clips + JSONL only, skipping 104 GB full audio)"
    else:
        ignore_patterns = None
        extra = " (with full audio, ~108 GB total)" if dataset_name == "HookTheory" else ""
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
        extract_HookTheory(target_dir, include_full_audio=with_full_audio)


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
