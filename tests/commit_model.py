from huggingface_hub import upload_folder
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

    args = parser.parse_args()

    # Upload the folder containing the model
    upload_folder(
        repo_id=args.repo_name,  # The name of the model repository on Hugging Face
        folder_path=args.model_folder  # The local path to the model files
    )
    print(f"Successfully uploaded '{args.model_folder}' to '{args.repo_name}' on Hugging Face.")

if __name__ == "__main__":
    main()