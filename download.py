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

def extract_HookTheory(save_root: str):
    # run data/HookTheoryUpload/extract.sh
    import subprocess
    script_path = os.path.join(save_root, "HookTheory", "extract.sh")
    if not os.path.exists(script_path):
        print(f"Error: Extraction script '{script_path}' does not exist. Please ensure you have the correct dataset structure.")
        sys.exit(1)
    print(f"Running extraction script '{script_path}'...")
    try:
        subprocess.run(["bash", script_path], check=True)
        print("Extraction completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}", file=sys.stderr)
        sys.exit(1)


def download_dataset(dataset_name: str, save_root: str, max_retries: int = 5):
    """
    Download a single dataset from HuggingFace into save_root/<dataset_name>/.

    Uses max_workers=2 to avoid the free-tier 5000-request/5-min rate limit
    that fires when snapshot_download resolves hundreds of files in parallel.
    Retries up to max_retries times on 429 errors with exponential back-off.
    Re-running this command is always safe — already-downloaded files are
    skipped automatically by the HF cache logic.
    """
    repo_id = f"m-a-p/{dataset_name}"
    target_dir = os.path.join(save_root, dataset_name)
    print(f"Starting download of '{dataset_name}' → '{target_dir}' ...")
    os.makedirs(target_dir, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                max_workers=2,   # throttle to avoid HF free-tier rate limit (429)
            )
            break  # success
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < max_retries:
                wait = 60 * attempt   # 60s, 120s, 180s, 240s …
                print(f"\n  Rate-limited by HuggingFace (attempt {attempt}/{max_retries}).")
                print(f"  Waiting {wait}s before retrying … (re-running this command later also works)\n")
                time.sleep(wait)
            else:
                raise

    print(f"Dataset '{dataset_name}' saved to '{target_dir}'.")

    if dataset_name == "HookTheory":
        print("HookTheory dataset requires extraction. Running extraction script...")
        extract_HookTheory(save_root)


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
