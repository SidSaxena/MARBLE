"""sota/predict_key.py
─────────────────────────────────────────────────
       Test metric             GS test
─────────────────────────────────────────────────
        test/acc            0.6440397500991821
   test/weighted_score      0.6973509933774834
─────────────────────────────────────────────────
"""
import os
import json
import yaml
from tqdm import tqdm
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import snapshot_download

from marble.core.utils import instantiate_from_config

# 自定义的测试数据集类
class CustomTestDataset(Dataset):
    """
    Custom Dataset for prediction. This dataset is used to load the audio clips without labels
    from the provided TXT file that contains file paths.
    """
    LABEL2IDX = {     # mir_eval format key annotation
        'C major': 0,
        'Db major': 1,
        'D major': 2,
        'Eb major': 3,
        'E major': 4,
        'F major': 5,
        'Gb major': 6,
        'G major': 7,
        'Ab major': 8,
        'A major': 9,
        'Bb major': 10,
        'B major': 11,
        'C minor': 12,
        'Db minor': 13,
        'D minor': 14,
        'Eb minor': 15,
        'E minor': 16,
        'F minor': 17,
        'Gb minor': 18,
        'G minor': 19,
        'Ab minor': 20,
        'A minor': 21,
        'Bb minor': 22,
        'B minor': 23
    }
    IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}

    def __init__(self, sample_rate, channels, clip_seconds, filelist_path, channel_mode="first", min_clip_ratio=1.0):
        self.sample_rate = sample_rate
        self.channels = channels
        self.clip_seconds = clip_seconds
        self.clip_len_target = int(self.clip_seconds * self.sample_rate)
        self.min_clip_ratio = min_clip_ratio
        # Calculate the minimum required number of samples for a valid clip
        self.min_clip_len = int(self.clip_len_target * self.min_clip_ratio)
        self.files = []

        # 读取文件路径列表
        with open(filelist_path, 'r') as f:
            self.files = [line.strip() for line in f.readlines()]

        # 用于存储每个文件的切片信息
        self.index_map = []
        
        # 计算每个文件的切片数量
        for file_path in tqdm(self.files, desc="Processing filelist", ncols=100):
            info = torchaudio.info(file_path)
            num_samples = info.num_frames
            original_sample_rate = info.sample_rate # Get original sample rate
            
            # Adjust num_samples based on target sample rate for slicing logic
            # If original_sample_rate != self.sample_rate, num_samples would be effectively different
            # It's safer to base slicing on the number of frames at the ORIGINAL sample rate
            # and then handle resampling *after* loading the segment.
            # However, for simplicity and to avoid re-calculating offsets based on resampled lengths,
            # we'll still calculate slices based on original num_samples, and then resample each loaded slice.
            # This means clip_len_target conceptually applies to the *target* sample rate.
            # The more robust way for very precise slicing and resampling *before* slicing is more complex,
            # but for fixed-length clips, resampling after loading each chunk is generally acceptable.

            n_full_clips = num_samples // int(self.clip_seconds * original_sample_rate) # Calculate full clips based on original SR
            remaining_samples = num_samples % int(self.clip_seconds * original_sample_rate)

            # Recalculate min_clip_len based on original sample rate for comparison
            min_clip_len_original_sr = int(self.min_clip_ratio * self.clip_seconds * original_sample_rate)

            if remaining_samples >= min_clip_len_original_sr:
                n_slices = n_full_clips + 1
            elif n_full_clips == 0 and remaining_samples > 0 and remaining_samples >= min_clip_len_original_sr:
                n_slices = 1
            else:
                n_slices = n_full_clips

            for slice_idx in range(n_slices):
                self.index_map.append((file_path, slice_idx, original_sample_rate)) # Store original SR

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_path, slice_idx, original_sample_rate = self.index_map[idx]
        
        # Calculate offset and number of frames for the current slice based on original sample rate
        frame_offset_original_sr = slice_idx * int(self.clip_seconds * original_sample_rate)
        
        info = torchaudio.info(file_path)
        total_frames_original_sr = info.num_frames
        
        # Determine the actual number of frames to load for this slice at original SR
        num_frames_to_load_original_sr = min(self.clip_len_target * original_sample_rate // self.sample_rate, 
                                             total_frames_original_sr - frame_offset_original_sr)
        
        # Ensure we don't try to load negative frames
        if num_frames_to_load_original_sr <= 0:
             # This case should ideally not be reached with the current slicing logic,
             # but as a safeguard, return a dummy tensor or handle as an error.
             # For now, let's just make it a very small valid clip
             num_frames_to_load_original_sr = int(self.min_clip_ratio * self.clip_seconds * original_sample_rate)


        waveform, _ = torchaudio.load(file_path, frame_offset=frame_offset_original_sr, num_frames=num_frames_to_load_original_sr)

        # --- 加入重采样逻辑 ---
        if original_sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        # --- 重采样逻辑结束 ---

        # 确保音频的通道符合预期
        if self.channels == 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif waveform.size(0) < self.channels:
            waveform = waveform.repeat(self.channels, 1)

        # Padding is now necessary if the resampled waveform is shorter than the target clip length.
        # This can happen if the original segment, when resampled, becomes shorter than target.
        if waveform.shape[-1] < self.clip_len_target:
            padding_needed = self.clip_len_target - waveform.shape[-1]
            waveform = F.pad(waveform, (0, padding_needed))
        # If the resampled waveform is *longer* than clip_len_target (due to float precision issues with slicing/resampling),
        # we should truncate it. This is less common but good practice.
        elif waveform.shape[-1] > self.clip_len_target:
            waveform = waveform[..., :self.clip_len_target]

        # 返回音频波形和文件路径作为 UID
        return waveform, file_path

# 函数：从 Hugging Face 下载 ckpt
def download_ckpt_from_hf(download_dir):
    """
    Downloads the model from Hugging Face. Uses snapshot_download.
    """
    repo_id = "m-a-p/key_sota_20250618"  # Model repository ID
    if os.path.exists(os.path.join(download_dir, "checkpoints/best.ckpt")):
        return

    print(f"Downloading Hugging Face model repo: {repo_id} to {download_dir} using snapshot_download...")
    # snapshot_download will automatically skip download if files already exist
    # It also handles creating the directory if it doesn't exist
    snapshot_download(repo_id=repo_id, local_dir=download_dir, local_dir_use_symlinks=False)
    print(f"Model repo available at {download_dir}")

# 函数: 加载模型
def load_model(config_path, ckpt_path, device):
    """
    加载模型。如果指定的 ckpt_path 不存在，则从 Hugging Face 下载。
    """
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型配置
    model = instantiate_from_config(config['model'])
    
    # 加载模型检查点
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys in state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys in state_dict: {unexpected}")
    
    model.to(device)
    model.eval()
    return model

# 函数: 预测并保存到 jsonl 文件
def predict_and_save(model, dataloader, output_path, idx2label, device):
    # Dictionary to store logits for each file path
    file_logits = defaultdict(list)

    print("Starting prediction across all slices...")
    for batch in tqdm(dataloader, desc="Predicting slices", ncols=100):
        x, file_paths = batch
        x = x.to(device) # Move input to device
        with torch.no_grad():
            logits = model(x) # Get logits for each slice
            
            # Store logits with their corresponding file path
            for path, logit in zip(file_paths, logits):
                file_logits[path].append(logit.cpu()) # Move logits to CPU to save memory if many slices

    print("Aggregating predictions per file...")
    final_predictions = []
    # Sort file_logits by file_path for consistent output order
    sorted_file_logits = sorted(file_logits.items())

    for file_path, logits_list in tqdm(sorted_file_logits, desc="Aggregating files", ncols=100):
        # Stack all logits for the current file and compute the mean
        stacked_logits = torch.stack(logits_list) # (n_slices, C)
        mean_logit = torch.mean(stacked_logits, dim=0) # (C,)

        # Convert the averaged logits to probabilities and then to the final prediction
        probabilities = F.softmax(mean_logit, dim=-1)
        confidence, pred_idx = torch.max(probabilities, dim=-1)
        
        final_predictions.append({
            "audio_path": file_path,
            "prediction": idx2label[pred_idx.item()],
            "confidence": confidence.item()
        })

    # 将结果保存到 jsonl 文件
    with open(output_path, 'w') as f:
        for pred in final_predictions:
            json.dump(pred, f)
            f.write("\n")
    print(f"Predictions saved to {output_path}")

# 主函数: 实现预测功能
def main(filelist_path, output_path, batch_size, download_dir, min_clip_ratio):
    # 下载模型文件夹并加载配置文件
    download_ckpt_from_hf(download_dir)

    # 模型文件路径
    config_path = os.path.join(download_dir, "config.yaml")
    ckpt_path = os.path.join(download_dir, "checkpoints", "best.ckpt")

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型
    model = load_model(config_path, ckpt_path, device)

    # 加载自定义测试数据集
    test_dataset = CustomTestDataset(
        sample_rate=24000,
        channels=1,
        clip_seconds=15,
        filelist_path=filelist_path,
        channel_mode="mix",
        min_clip_ratio=min_clip_ratio # Example: segments shorter than 0.5 * 15s will be discarded
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # 获取 idx2label 映射
    idx2label = CustomTestDataset.IDX2LABEL

    # 调用预测函数并保存结果
    predict_and_save(model, test_dataloader, output_path, idx2label, device)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Key Prediction Script")
    parser.add_argument('--filelist_path', type=str, required=True, help="Path to the filelist (.txt) containing audio file paths")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the predictions")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for prediction")
    parser.add_argument('--download_dir', type=str, default="output/key_sota_20250618", help="Directory to download the model")
    # Added --min_clip_ratio argument
    parser.add_argument('--min_clip_ratio', type=float, default=0.2, 
                        help="Minimum ratio of clip_seconds for a segment to be included. Segments shorter than this ratio will be discarded.")


    args = parser.parse_args()

    # Pass min_clip_ratio to the main function
    main(args.filelist_path, args.output_path, args.batch_size, args.download_dir, args.min_clip_ratio)