# marble/tasks/GTZANBeatTracking/modules.py
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
import mir_eval

from marble.utils.utils import times_to_mask, mask_to_times


class VanillaTempoEstimator(nn.Module):
    def __init__(self, label_fps: int, threshold: float = 0.7):
        super().__init__()
        self.label_fps = label_fps
        self.threshold = threshold

    def forward(self, beat_logits: torch.Tensor) -> tuple:
        """
        Inputs:
          - beat_logits: Tensor of shape (B, T), where B = batch size, T = number of frames.
                         Its dtype might be torch.float32 or torch.float64.
        Returns:
          - bpm_pred:   Tensor of shape (B,), predicted BPM for each sample (same dtype as input).
        """
        # 1) Convert logits to sigmoid probabilities
        beat_probs = torch.sigmoid(beat_logits)
        beat_probs = (beat_probs > self.threshold).float()  # Apply thresholding
        bpm_pred = beat_probs.sum(dim=1) / (beat_probs.shape[1] / self.label_fps) * 60.0
        return bpm_pred


class VanillaTempoEstimatorWithSSL(nn.Module):
    def __init__(self, label_fps: int, ssl_emb_dim: int, threshold: float = 0.7):
        super().__init__()
        self.label_fps = label_fps
        self.threshold = threshold
        self.ssl_emb_dim = ssl_emb_dim

        # 这里可以直接用一个线性层把 ssl_emb_dim 映射到 1D 的 tempo
        self.ssl_proj = nn.Linear(ssl_emb_dim, 1)

    def forward(self, beat_logits: torch.Tensor, ssl_emb: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          - beat_logits: Tensor of shape (B, T), where B = batch size, T = number of frames.
          - ssl_emb:     Tensor of shape (B, D), SSL embedding.
        Returns:
          - bpm_pred:   Tensor of shape (B,), predicted BPM for each sample.
        """
        # 1) Convert logits to sigmoid probabilities
        beat_probs = torch.sigmoid(beat_logits)  # (B, T)

        # 2) Compute pseudo tempo
        beat_probs = (beat_probs > self.threshold).float()  # Apply thresholding
        bpm_pred = beat_probs.sum(dim=1) / (beat_probs.shape[1] / self.label_fps) * 60.0  # (B,)

        # 3) 用 SSL embedding 做一个线性投影，得到一个额外的 tempo 值
        ssl_tempo = self.ssl_proj(ssl_emb).squeeze(-1)  # (B,)

        # 4) 最终预测：平均两个 tempo
        bpm_pred = (bpm_pred + ssl_tempo) / 2.0  # (B,)
        return bpm_pred


class FFTTempoEstimator(nn.Module):
    """
    Differentiable tempo estimator that uses FFT to extract the fundamental frequency
    from a beat activation sequence and convert it to BPM.
    """
    def __init__(self, label_fps: int, freq_resolution: float = 1.0, softmax_temp: float = 50.0):
        super().__init__()
        self.label_fps = label_fps
        self.freq_resolution = freq_resolution
        self.softmax_temp = softmax_temp

    def forward(self, beat_logits: torch.Tensor) -> torch.Tensor:
        # 1) logits → probabilities
        beat_probs = torch.sigmoid(beat_logits)   # dtype: float32/float64 (or bfloat16 if AMP)
        # 2) subtract mean
        mean_per_sample = beat_probs.mean(dim=1, keepdim=True)
        beat_centered = beat_probs - mean_per_sample

        # Save original dtype so we can cast back at the end if needed
        orig_dtype = beat_centered.dtype

        # If we’re in bfloat16, cast to float32 for FFT
        if orig_dtype == torch.bfloat16:
            beat_centered = beat_centered.to(torch.float32)

        B, T = beat_centered.shape
        N_fft = int(T * self.freq_resolution) if (self.freq_resolution != 1.0) else T

        # 3) real FFT in float32 (now supported)
        fft_out = torch.fft.rfft(beat_centered, n=N_fft, dim=1)  # (B, Nf), complex64
        mag = torch.abs(fft_out)                                 # (B, Nf), float32

        # 4) build freqs_hz in same dtype as mag (float32)
        Nf = mag.shape[-1]
        device = mag.device
        dtype = mag.dtype  # float32

        freq_idxs = torch.arange(Nf, device=device, dtype=dtype)
        freqs_hz = freq_idxs * (float(self.label_fps) / float(N_fft))  # (Nf,), float32

        # 5) softmax over scaled magnitudes
        mag_scaled = mag * self.softmax_temp    # (B, Nf), float32
        freq_prob  = torch.nn.functional.softmax(mag_scaled, dim=1)  # (B, Nf), float32

        # 6) predicted freq and convert to BPM
        freq_hz_pred = torch.matmul(freq_prob, freqs_hz)  # (B,), float32
        bpm_pred     = freq_hz_pred * 60.0                # (B,), float32

        # 7) if original was bfloat16, cast result back
        if orig_dtype == torch.bfloat16:
            bpm_pred = bpm_pred.to(torch.bfloat16)

        return bpm_pred


class FFTTempoEstimatorWithSSLFiLM(nn.Module):
    def __init__(
        self,
        label_fps: int,
        ssl_emb_dim: int,
        freq_resolution: float = 1.0,
        softmax_temp: float = 50.0,
        use_dynamic_filter: bool = True
    ):
        super().__init__()
        self.label_fps = label_fps
        self.freq_resolution = freq_resolution
        self.softmax_temp = softmax_temp
        self.use_dynamic_filter = use_dynamic_filter

        # lazy init：我们暂不在 __init__ 硬编码 Nf，因为 Nf 取决于输入长度 T。
        self.dynamic_filter_net = None  # 稍后根据 Nf 和 ssl_emb_dim 来初始化

        # 如果想让 filter 具有“偏置”（相当于 global filter_weights），
        # 可以额外准备一个 (Nf,) 的偏置向量：
        self.global_filter_bias = None  # 同样 lazy init

        self.ssl_emb_dim = ssl_emb_dim

    def _lazy_init_filter(self, Nf: int, device: torch.device, dtype: torch.dtype):
        """
        在第一次 forward 时，根据 Nf 初始化 dynamic_filter_net 和 global_filter_bias
        """
        # ① 初始化全局 bias（相当于原先的 self.filter_weights）
        init_bias = torch.zeros(Nf, dtype=dtype, device=device)
        self.global_filter_bias = nn.Parameter(init_bias)  # (Nf,)

        # ② 初始化一个 MLP，把 ssl_emb_dim -> Nf
        #    这里举例一层线性 + GELU + 线性，再加上 global_bias
        self.dynamic_filter_net = nn.Sequential(
            nn.Linear(self.ssl_emb_dim, Nf * 2),  # 先扩到 2*Nf
            nn.GELU(),
            nn.Linear(Nf * 2, Nf)                  # 最终映射到 (Nf,)
        )

    def forward(self, beat_logits: torch.Tensor, ssl_emb: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          - beat_logits: Tensor (B, T)
          - ssl_emb:      Tensor (B, D)  # time-avg pooled 后的 SSL embedding
        Returns:
          - bpm_pred:     Tensor (B,)
        """
        B, T = beat_logits.shape

        # 1) beat_logits → beat_probs → 去 DC
        beat_probs = torch.sigmoid(beat_logits)              # (B, T)
        mean_per_sample = beat_probs.mean(dim=1, keepdim=True)# (B, 1)
        beat_centered = beat_probs - mean_per_sample          # (B, T)

        # 2) 决定 N_fft, 做 rFFT
        N_fft = int(T * self.freq_resolution) if self.freq_resolution != 1.0 else T
        fft_out = torch.fft.rfft(beat_centered, n=N_fft, dim=1)  # (B, Nf), complex
        mag = torch.abs(fft_out)                                 # (B, Nf)
        Nf = mag.shape[-1]

        # 3) lazy init：第一次进来时，根据 Nf 初始化 dynamic_filter_net & global_filter_bias
        if self.use_dynamic_filter and self.dynamic_filter_net is None:
            self._lazy_init_filter(Nf, device=mag.device, dtype=mag.dtype)

        # 4) 用 ssl_emb 生成 (B, Nf) 的权重矩阵
        if self.use_dynamic_filter:
            # dynamic_filter_net(ssl_emb) → (B, Nf)
            w_dyn = self.dynamic_filter_net(ssl_emb)         # (B, Nf)
            # 加上 global bias → (B, Nf)
            w_total = w_dyn + self.global_filter_bias.unsqueeze(0)
            # 再做 sigmoid，把值约束到 (0,1)
            w_total = torch.sigmoid(w_total)                 # (B, Nf)
            # 最后给 mag 乘上这个可学习权重
            mag = mag * w_total                              # (B, Nf)
        # 如果你不想用 dynamic_filter，则 mag 保持原样

        # 5) 构建 freqs_hz
        dtype = mag.dtype
        device = mag.device
        freq_idxs = torch.arange(Nf, device=device, dtype=dtype)  # (Nf,)
        freqs_hz = freq_idxs * (float(self.label_fps) / float(N_fft))  # (Nf,)

        # 6) Softmax 之前 * temperature
        mag_scaled = mag * self.softmax_temp  # (B, Nf)
        freq_prob = F.softmax(mag_scaled, dim=1)  # (B, Nf)

        # 7) 频率加权求和 → freq_hz_pred
        freq_hz_pred = torch.matmul(freq_prob, freqs_hz)  # (B,), 单位 Hz

        # 8) 转换 BPM
        bpm_pred = freq_hz_pred * 60.0  # (B,)
        return bpm_pred


class FFTTempoEstimatorWithSSLDiscriminator(nn.Module):
    def __init__(
        self,
        label_fps: int,
        ssl_emb_dim: int,
        freq_resolution: float = 1.0,
        softmax_temp: float = 50.0,
        topk: int = 3
    ):
        super().__init__()
        self.label_fps = label_fps
        self.freq_resolution = freq_resolution
        self.softmax_temp = softmax_temp
        self.topk = topk
        self.ssl_emb_dim = ssl_emb_dim

        # 我们把 topk_bpms, topk_vals, ssl_emb 都拼成一个向量后，让一个 MLP 预测 topk 个 logits
        # 输入维度 = topk (BPMS) + topk (mag) + ssl_emb_dim
        in_dim = topk + topk + ssl_emb_dim
        self.discriminator = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, topk)  # 输出 topk 个 logits
        )

    def forward(self, beat_logits: torch.Tensor, ssl_emb: torch.Tensor) -> torch.Tensor:
        B, T = beat_logits.shape

        # 1) time-domain → beat_probs → 去 DC
        beat_probs = torch.sigmoid(beat_logits)                # (B, T)
        mean_per_sample = beat_probs.mean(dim=1, keepdim=True) # (B, 1)
        beat_centered = beat_probs - mean_per_sample           # (B, T)

        # 2) FFT → mag
        N_fft = int(T * self.freq_resolution) if self.freq_resolution != 1.0 else T
        fft_out = torch.fft.rfft(beat_centered, n=N_fft, dim=1)  # (B, Nf)
        mag = torch.abs(fft_out)                                 # (B, Nf)
        Nf = mag.shape[-1]

        # 3) 构 freqs_hz
        dtype = mag.dtype
        device = mag.device
        freq_idxs = torch.arange(Nf, device=device, dtype=dtype)  # (Nf,)
        freqs_hz = freq_idxs * (float(self.label_fps) / float(N_fft))  # (Nf,)

        # 4) top-k 峰
        topk_vals, topk_idxs = torch.topk(mag, self.topk, dim=1)  # (B, topk)
        topk_freqs = freqs_hz[topk_idxs]                          # (B, topk)
        topk_bpms = topk_freqs * 60.0                              # (B, topk)

        # 5) 构造 Discriminator 输入：把 BPM 做归一化 / 缩放，幅度也做归一化
        inp_bpms = topk_bpms / 100.0             # (B, topk)
        inp_mags = topk_vals / (topk_vals.max(dim=1, keepdim=True)[0] + 1e-8)  # (B, topk)，归一到 [0,1]
        # ssl_emb 本身维度可能很大，投影一下或直接拼接都行。这里直接拼接，但你也可以先做个线性层。
        inp = torch.cat([inp_bpms, inp_mags, ssl_emb], dim=1)  # (B, topk+topk+ssl_emb_dim)

        # 6) 用 MLP 打分
        logits = self.discriminator(inp)         # (B, topk)
        cand_prob = F.softmax(logits, dim=1)     # (B, topk)

        # 7) 预测 BPM = 加权平均
        bpm_pred = torch.sum(cand_prob * topk_bpms, dim=1)  # (B,)
        return bpm_pred
