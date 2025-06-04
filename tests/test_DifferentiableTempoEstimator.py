# tests/test_FFTTempoEstimator.py
import os

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import random
import mir_eval

from marble.tasks.GTZANBeatTracking.datamodule import GTZANBeatTrackingAudioTrain
from marble.tasks.GTZANBeatTracking.modules import FFTTempoEstimator

# If running inside "tests" folder, move up one level
pwd = os.getcwd()
if pwd.endswith("tests"):
    os.chdir(os.path.dirname(pwd))

# Instantiate the dataset (adjust paths as needed)
dataset = GTZANBeatTrackingAudioTrain(
    sample_rate=22050,
    channels=1,
    clip_seconds=15.0,
    jsonl="data/GTZAN/GTZANBeatTracking.val.jsonl",
    label_freq=50,
    num_neighbors=2,
    channel_mode="first",
    min_clip_ratio=0.8,
)

def visualize_FFTTempoEstimator():
    """
    Test the FFTTempoEstimator with synthetic data.
    This function synthesizes a clean 2 Hz sine wave, adds noise to the second sample,
    and verifies, visualizes the output probabilities and frequency domain analysis.
    """
    import matplotlib.pyplot as plt
    # Configuration
    fps = 50                 # 50 frames per second
    true_bpm = 120           # target BPM
    true_hz = true_bpm / 60  # Hz
    T = 750                  # Sequence length 750 frames (~4 seconds)
    B = 2                    # batch size 2

    # 1) Synthesize "clean" pulses: 2 Hz sine wave thresholding
    time = torch.arange(T, dtype=torch.float32) / fps            # (T,), seconds
    sine_wave = torch.sin(2 * torch.pi * true_hz * time)         # 2 Hz sine wave
    beat_pattern = (sine_wave > 0.9).float()                     # pulses (T,)

    # 2) Expand to batch and add noise to the second sample
    beat_signals = beat_pattern.unsqueeze(0).repeat(B, 1)        # (B, T)
    noise_level = 0.3
    noise = noise_level * torch.rand_like(beat_signals[1])
    beat_signals[1] = torch.clamp(beat_signals[1] + noise, 0.0, 1.0)

    # 3) Probabilities → Logits
    eps = 1e-4
    beat_logits = torch.log((beat_signals + eps) / (1 - beat_signals + eps))  # (B, T)

    # 4) Instantiate the estimator and forward pass
    estimator = FFTTempoEstimator(label_fps=fps, freq_resolution=4.0, softmax_temp=50.0)
    bpm_pred, beat_probs, mag, freqs_hz, freq_prob = estimator(beat_logits)

    # Print predicted results
    print(f"True BPM: {true_bpm}")
    print(f"Predicted BPMs: {bpm_pred.tolist()}")

    # ===== Figure 1: Time-domain Beat Probabilities =====
    plt.figure()
    plt.plot(time.numpy(), beat_probs[0].detach().numpy(), label='Clean Signal')
    plt.plot(time.numpy(), beat_probs[1].detach().numpy(), label='Noisy Signal', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Beat Probability')
    plt.title('Beat Activation Over Time (Clean vs Noisy)')
    plt.legend()
    plt.show()

    # ===== Figure 2: Frequency Domain Analysis (Noisy Sample) =====
    plt.figure()
    # Only plot the noisy sample
    mag_noisy = mag[1].detach().numpy()
    freq_prob_noisy = freq_prob[1].detach().numpy()
    bpm_axis = (freqs_hz.detach().numpy() * 60.0)  # Convert Hz to BPM
    # magnitude spectrum
    plt.plot(bpm_axis, mag_noisy, label='Magnitude Spectrum')
    # softmax probabilities * peak magnitude (for visualization)
    plt.plot(bpm_axis, freq_prob_noisy * mag_noisy.max(), label='Softmax Probability × Scale', alpha=0.7)
    # predicted BPM
    plt.axvline(x=bpm_pred[1].item(), linestyle='--', label=f'Predicted BPM: {bpm_pred[1].item():.2f}')
    plt.xlabel('BPM')
    plt.ylabel('Magnitude / Scaled Probability')
    plt.title('Frequency Domain Analysis (Noisy Sample)')
    plt.legend()
    plt.show()
    
    
def test_FFTTempoEstimator(dataset, max_samples=50, seed=42):
    """
    随机抽样地遍历 dataset 中的样本，从 beat 激活估计 tempo，并与真实 tempo 做对比。

    Args:
        dataset: 已经初始化好的 GTZANBeatTrackingAudioTrain 实例。
        max_samples (int, optional): 想要随机抽取进行测试的样本数。如果为 None，则遍历整个数据集。
        seed (int, optional): 随机种子，保证可复现。
    """
    # 1) 获取数据集长度
    dataset_size = len(dataset)

    # 2) 确定要测试的样本数量
    if (max_samples is None) or (max_samples >= dataset_size):
        total_samples = dataset_size
    else:
        total_samples = max_samples

    # 3) 通过随机种子保证复现性，然后随机抽取索引
    random.seed(seed)
    if total_samples < dataset_size:
        sampled_indices = random.sample(range(dataset_size), total_samples)
    else:
        sampled_indices = list(range(dataset_size))

    # 4) 获取 label_fps（假设 dataset 中有属性 "label_freq"；若无，可直接写 50）
    label_fps = getattr(dataset, "label_freq", 50)

    # 5) 实例化 FFTTempoEstimator（与前面示例参数保持一致）
    estimator = FFTTempoEstimator(
        label_fps=label_fps,
        freq_resolution=8.0,
        softmax_temp=50.0
    )

    # 6) 用来记录每个样本的绝对误差
    abs_errors = []

    sep_len = 40  # 分隔线长度，这里用 40 个字符
    print(f"开始随机测试，数据集大小 {dataset_size}，随机选取 {total_samples} 个样本（seed={seed}）\n")
    print(f"{'Idx':>3s} | {'SampleIdx':>9s} | {'True BPM':>8s} | {'Pred BPM':>9s} | {'Abs Error':>9s}")
    print("-" * sep_len)

    for rank, idx in enumerate(sampled_indices):
        # 7) 从 dataset 中获取 idx 样本
        _, targets, _ = dataset[idx]

        # 8) 提取 beat 激活 (shape: [label_len]) 和真实 tempo (scalar)
        beat_activation = targets["beat"]        # torch.Tensor, dtype=float32, shape=(label_len,)
        tempo_true = targets["tempo"].item()     # float

        # 9) 将 beat activation 转为 logits，避免 p=0 或 p=1
        eps = 1e-4
        beat_probs = beat_activation.clamp(min=eps, max=1 - eps)
        beat_logits = torch.log(beat_probs / (1.0 - beat_probs))  # shape: (label_len,)

        # 10) 增加 batch 维度并前向
        beat_logits = beat_logits.unsqueeze(0)  # shape: (1, label_len)
        with torch.no_grad():
            bpm_pred_tensor, _, _, _, _ = estimator(beat_logits)
        bpm_pred = bpm_pred_tensor[0].item()

        # 11) 计算绝对误差并记录
        abs_err = abs(bpm_pred - tempo_true)
        abs_errors.append(abs_err)

        # 12) 打印本次随机抽到的样本序号和结果
        print(f"{rank:3d} | {idx:9d} | {tempo_true:8.2f} | {bpm_pred:9.2f} | {abs_err:9.2f}")

    # 13) 计算并打印平均绝对误差（MAE）
    if len(abs_errors) > 0:
        mae = float(np.mean(abs_errors))
        print("\n" + "-" * sep_len)
        print(f"平均绝对误差 (MAE): {mae:.2f} BPM")
    else:
        print("没有计算到任何样本。")

if __name__ == "__main__":
    # 运行可视化函数
    # visualize_FFTTempoEstimator()

    # 测试 FFTTempoEstimator
    test_FFTTempoEstimator(dataset, max_samples=None, seed=42)