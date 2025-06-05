import os
# 如果在 "tests" 目录下运行，则切回上一层
pwd = os.getcwd()
if pwd.endswith("tests"):
    os.chdir(os.path.dirname(pwd))

import time
import torch
import numpy as np
from marble.tasks.Chords1217._datamodule_v1 import Chords1217AudioTrain
from torch.utils.data import DataLoader

def benchmark_get_targets_single_clip(
    jsonl_path: str,
    sample_rate: int,
    channels: int,
    clip_seconds: float,
    label_freq: int,
    mode: str,
    num_files: int = 20
):
    """
    对 Chords1217AudioTrain 在指定 mode 下，测量前 num_files 个文件的第一个 slice (slice_idx=0) 
    调用 get_targets 的总耗时。仅供对比三种实现（python/numpy/numba）哪种最慢。
    """
    print(f"\n--- Benchmark get_targets mode='{mode}' (仅前 {num_files} 个文件，每个文件的第 0 号 slice) ---")
    # 1. 实例化 Dataset（这里只用 Train split）
    dataset = Chords1217AudioTrain(
        jsonl=jsonl_path,
        sample_rate=sample_rate,
        channels=channels,
        clip_seconds=clip_seconds,
        label_freq=label_freq,
        mode=mode
    )

    # 假设每个文件至少有一个 slice，且 orig_sr == sample_rate，orig_clip_frames == clip_seconds * sample_rate
    orig_sr = sample_rate
    orig_clip_frames = int(clip_seconds * sample_rate)

    t0 = time.perf_counter()
    for file_idx in range(min(num_files, len(dataset.chords_meta))):
        # 只测第 0 号 slice 的 get_targets
        _ = dataset.get_targets(
            file_idx=file_idx,
            slice_idx=0,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )
    t1 = time.perf_counter()
    total_time = t1 - t0
    count = min(num_files, len(dataset.chords_meta))
    print(f"Mode='{mode}'，总耗时：{total_time:.4f}s，平均每个文件第 0 号 slice get_targets 用时 {total_time/count:.4f}s")


def benchmark_get_targets_full_epoch(
    jsonl_path: str,
    sample_rate: int,
    channels: int,
    clip_seconds: float,
    label_freq: int,
    mode: str,
    batch_size: int = 8,
    num_workers: int = 4
):
    """
    通过 DataLoader 对整个 train split 迭代一遍，测量整个 epoch 调用 get_targets + collate 的时间。
    """
    print(f"\n--- Benchmark get_targets mode='{mode}' （使用 DataLoader 遍历整个 train split）---")
    train_ds = Chords1217AudioTrain(
        jsonl=jsonl_path,
        sample_rate=sample_rate,
        channels=channels,
        clip_seconds=clip_seconds,
        label_freq=label_freq,
        mode=mode
    )

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    t0 = time.perf_counter()
    for batch in loader:
        # batch 里会调用 get_targets(...) + 其他预处理（如音频 loading、transform 等）
        pass
    t1 = time.perf_counter()
    print(f"Mode='{mode}'，整 train split 迭代一轮 get_targets 耗时：{t1-t0:.4f}s")


def benchmark_load_and_preprocess_single_clip(
    jsonl_path: str,
    sample_rate: int,
    channels: int,
    clip_seconds: float,
    label_freq: int,
    mode: str,
    num_files: int = 20
):
    """
    针对 Chords1217AudioTrain，在指定 mode 下，测量前 num_files 个文件的第一个 slice (slice_idx=0)
    调用 _load_and_preprocess 的总耗时。仅供对比三种实现（python/numpy/numba）在纯加载+预处理上的差异。
    """
    print(f"\n--- Benchmark load_and_preprocess mode='{mode}' (仅前 {num_files} 个文件，第 0 号 slice) ---")
    # 1. 实例化 Dataset（这里只用 Train split）
    dataset = Chords1217AudioTrain(
        jsonl=jsonl_path,
        sample_rate=sample_rate,
        channels=channels,
        clip_seconds=clip_seconds,
        label_freq=label_freq,
        mode=mode
    )

    # 假设 orig_sr == sample_rate，orig_clip_frames == clip_seconds * sample_rate
    orig_sr = sample_rate
    orig_clip_frames = int(clip_seconds * sample_rate)

    t0 = time.perf_counter()
    for file_idx in range(min(num_files, len(dataset.meta))):
        info = dataset.meta[file_idx]
        audio_path = info["audio_path"]
        # 测量 _load_and_preprocess 的耗时
        _ = dataset._load_and_preprocess(
            path=audio_path,
            slice_idx=0,
            orig_sr=orig_sr,
            orig_clip_frames=orig_clip_frames
        )
    t1 = time.perf_counter()
    total_time = t1 - t0
    count = min(num_files, len(dataset.meta))
    print(f"Mode='{mode}'，总耗时：{total_time:.4f}s，平均每个文件第 0 号 slice _load_and_preprocess 用时 {total_time/count:.4f}s")


def benchmark_load_and_preprocess_full_epoch(
    jsonl_path: str,
    sample_rate: int,
    channels: int,
    clip_seconds: float,
    label_freq: int,
    mode: str,
    batch_size: int = 8,
    num_workers: int = 4
):
    """
    通过 DataLoader 遍历整个 train split，但在 collate 阶段只收集 waveform，不调用 get_targets，
    仅测 _load_and_preprocess 的耗时。
    """
    print(f"\n--- Benchmark load_and_preprocess mode='{mode}' （使用 DataLoader 遍历 entire train split，仅 _load_and_preprocess） ---")
    train_ds = Chords1217AudioTrain(
        jsonl=jsonl_path,
        sample_rate=sample_rate,
        channels=channels,
        clip_seconds=clip_seconds,
        label_freq=label_freq,
        mode=mode
    )

    # 定义一个只做加载 + 预处理的 collate_fn
    def collate_waveforms(batch):
        # batch 是 [(waveform, targets, audio_path), ...]
        # waveform 已经包含了 _load_and_preprocess 的结果
        waveforms = [item[0] for item in batch]
        return torch.stack(waveforms, dim=0)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_waveforms,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    t0 = time.perf_counter()
    for _ in loader:
        # 只负责跑 _load_and_preprocess，然后被 collate_fn 收集
        pass
    t1 = time.perf_counter()
    print(f"Mode='{mode}'，整 train split 迭代一轮 _load_and_preprocess 耗时：{t1-t0:.4f}s")


if __name__ == '__main__':
    # ====== 以下是需要根据实际情况修改的参数 ======
    JSONL_PATH    = 'data/Chords1217/Chords1217.train.jsonl'  # 你的 Chords1217 metadata jsonl 文件
    SAMPLE_RATE   = 24000
    CHANNELS      = 1
    CLIP_SECONDS  = 15.0       # 每个 clip 时长
    LABEL_FREQ    = 75         # 每秒多少帧标签（对 get_targets 有用，对 load_and_preprocess 无影响）
    NUM_FILES     = 500         # 单文件测试时用到的文件数量
    BATCH_SIZE    = 64         # DataLoader 的 batch_size
    NUM_WORKERS   = 4          # DataLoader 的 num_workers

    # -------------------------------
    # 1) 先跑 get_targets 的基准测试
    # -------------------------------
    for m in ['python', 'numpy', 'numba']:
        benchmark_get_targets_single_clip(
            jsonl_path=JSONL_PATH,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            clip_seconds=CLIP_SECONDS,
            label_freq=LABEL_FREQ,
            mode=m,
            num_files=NUM_FILES
        )

    # for m in ['python', 'numpy', 'numba']:
    #     benchmark_get_targets_full_epoch(
    #         jsonl_path=JSONL_PATH,
    #         sample_rate=SAMPLE_RATE,
    #         channels=CHANNELS,
    #         clip_seconds=CLIP_SECONDS,
    #         label_freq=LABEL_FREQ,
    #         mode=m,
    #         batch_size=BATCH_SIZE,
    #         num_workers=NUM_WORKERS
    #     )

    # -----------------------------------------
    # 2) 再跑 _load_and_preprocess 的基准测试
    # -----------------------------------------
    for m in ['python', 'numpy', 'numba']:
        benchmark_load_and_preprocess_single_clip(
            jsonl_path=JSONL_PATH,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            clip_seconds=CLIP_SECONDS,
            label_freq=LABEL_FREQ,
            mode=m,
            num_files=NUM_FILES
        )

    # for m in ['python', 'numpy', 'numba']:
    #     benchmark_load_and_preprocess_full_epoch(
    #         jsonl_path=JSONL_PATH,
    #         sample_rate=SAMPLE_RATE,
    #         channels=CHANNELS,
    #         clip_seconds=CLIP_SECONDS,
    #         label_freq=LABEL_FREQ,
    #         mode=m,
    #         batch_size=BATCH_SIZE,
    #         num_workers=NUM_WORKERS
    #     )
