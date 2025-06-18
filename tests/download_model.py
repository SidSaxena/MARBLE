import os
from huggingface_hub import Repository
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face Hub.")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The Hugging Face model repository ID (e.g., 'm-a-p/key_sota_20250618')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The local path where the model will be downloaded (e.g., 'output/key_sota_20250618')"
    )

    args = parser.parse_args()

    # Create the target directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Clone the repository to the local target path
    print(f"Cloning Hugging Face model repo: {args.repo_id} into {args.output_dir}...")
    repo = Repository(local_dir=args.output_dir, clone_from=args.repo_id)

    print(f"Model repo cloned to {args.output_dir}")

if __name__ == "__main__":
    main()