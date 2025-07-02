# commit_model.py
from huggingface_hub import upload_folder, create_repo
import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload a model folder to Hugging Face Hub.")
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="The local path to the model files (e.g., 'output/probe.MergeGSHT.MuQ')"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="The name of the model repository on Hugging Face (e.g., 'm-a-p/key_sota_20250618')"
    )
    # Optional: Add an argument to specify the repo as public or private
    parser.add_argument(
        "--private",
        action='store_true',
        help="Set the repository visibility to private. Defaults to public."
    )


    args = parser.parse_args()

    # **Step 1: Create the repository (if it doesn't exist)**
    # The `exist_ok=True` flag prevents an error from being raised if the repo already exists.
    print(f"Ensuring repository '{args.repo_name}' exists on Hugging Face Hub...")
    create_repo(
        repo_id=args.repo_name,
        exist_ok=True,
        private=args.private  # Set repo visibility
    )
    print(f"Repository '{args.repo_name}' is ready.")


    # **Step 2: Upload the folder containing the model**
    print(f"Uploading files from '{args.model_folder}'...")
    upload_folder(
        repo_id=args.repo_name,
        folder_path=args.model_folder
    )
    print(f"Successfully uploaded '{args.model_folder}' to '{args.repo_name}' on Hugging Face.")

if __name__ == "__main__":
    main()