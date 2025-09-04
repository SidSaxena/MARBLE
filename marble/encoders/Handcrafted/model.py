# marble/encoders/Handcrafted/model.py
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torchaudio
import torchaudio.prototype.transforms as T

from marble.core.base_encoder import BaseEncoder


class _LibrosaFeatureEncoder(BaseEncoder):
    """
    Base class for handcrafted feature encoders implemented with torch/torchaudio.
    Subclasses implement `_compute_features(y, sr, hop_length)` that returns [D, T] torch tensor.
    """

    NAME: str = "LibrosaFeature"
    SAMPLING_RATE: int = 22050
    HOP_LENGTH: int = 512
    TOKEN_RATE: float = SAMPLING_RATE / HOP_LENGTH  # ~43 fps

    NUM_FEATURES: int = -1  # feature dimension D
    AGG_FEATURES: int = -1  # aggregated dimension 6*D

    def __init__(self, keep_time_dim: bool = True) -> None:
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        self.keep_time_dim = keep_time_dim

    # --- To be implemented by subclass ---
    def _compute_features(self, y: torch.Tensor, sr: int, hop_length: int) -> torch.Tensor:
        """
        Args:
            y: 1-D waveform tensor [num_samples], float32/-1..1 preferred.
        Returns:
            feats: [D, T] torch.Tensor (float32)
        """
        raise NotImplementedError

    # --- Utility methods ---
    @staticmethod
    def _normalize_audio(y: torch.Tensor) -> torch.Tensor:
        """Peak-normalize the waveform to [-1, 1] using torch tensors."""
        if not torch.is_floating_point(y):
            y = y.to(torch.float32)
        maxv = torch.max(torch.abs(y))
        if torch.gt(maxv, 0):
            y = y / maxv
        return y

    @staticmethod
    def _pad_stack_time(feat_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Pad feature sequences [T_i, D] to the max time length and stack into [B, T_max, D].
        """
        if len(feat_list) == 0:
            return torch.empty(0)
        device = feat_list[0].device
        dtype = feat_list[0].dtype
        T_max = max(f.shape[0] for f in feat_list)
        D = feat_list[0].shape[1]
        out = torch.zeros((len(feat_list), T_max, D), device=device, dtype=dtype)
        for i, f in enumerate(feat_list):
            T_i = f.shape[0]
            out[i, :T_i] = f
        return out

    @staticmethod
    def _aggregate_six_moments(feats_DT: torch.Tensor) -> torch.Tensor:
        """
        Given feats [D, T], compute aggregated statistics:
        for diff order i in {0,1,2}: concat mean(D) and std(D) across time -> [6*D].
        """
        D, T = feats_DT.shape
        stats: List[torch.Tensor] = []

        # i = 0: original
        f0 = feats_DT
        stats.append(f0.mean(dim=1))            # [D]
        stats.append(f0.std(dim=1, unbiased=False))

        # i = 1
        if T >= 2:
            f1 = torch.diff(feats_DT, n=1, dim=1)
            stats.append(f1.mean(dim=1))
            stats.append(f1.std(dim=1, unbiased=False))
        else:
            zeros = torch.zeros(D, device=feats_DT.device, dtype=feats_DT.dtype)
            stats.extend([zeros, zeros])

        # i = 2
        if T >= 3:
            f2 = torch.diff(feats_DT, n=2, dim=1)
            stats.append(f2.mean(dim=1))
            stats.append(f2.std(dim=1, unbiased=False))
        else:
            zeros = torch.zeros(D, device=feats_DT.device, dtype=feats_DT.dtype)
            stats.extend([zeros, zeros])

        moments = torch.cat(stats, dim=0)  # [6*D]
        return moments

    @staticmethod
    def _maybe_resample(y: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
        if sr_in == sr_out:
            return y
        # torchaudio expects [channel, time] or [time]; we have [time]
        return torchaudio.functional.resample(y, orig_freq=sr_in, new_freq=sr_out)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict[str, Tuple[torch.Tensor, ...]]:
        """
        Args:
            x: (B, num_samples) or (B, 1, num_samples) waveform tensor in [-1, 1].
            kwargs may include:
              - sampling_rate / sr: if not 22050, resample to 22050.
        Returns:
            dict(hidden_states=(...)):
              - If keep_time_dim=True:
                  (framewise, aggregated)
                    framewise: (B, T, D)
                    aggregated: (B, 1, 6*D)
              - If keep_time_dim=False:
                  (aggregated,)
        """
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        assert x.dim() == 2, f"Expected waveform shape (B, num_samples), got {x.shape}"
        B = x.shape[0]
        device = x.device
        dtype = torch.float32

        # sampling rate handling
        sr_in: Optional[int] = kwargs.get("sampling_rate") or kwargs.get("sr")
        if sr_in is None:
            sr_in = self.SAMPLING_RATE  # assume already at target if not provided

        frame_feats: List[torch.Tensor] = []
        agg_feats: List[torch.Tensor] = []

        for b in range(B):
            y = x[b].to(dtype)

            # resample if needed
            y = self._maybe_resample(y, sr_in, self.SAMPLING_RATE)

            # normalize
            y = self._normalize_audio(y)

            # Framewise features [D, T] torch
            feats_DT = self._compute_features(y, self.SAMPLING_RATE, self.HOP_LENGTH)
            assert isinstance(feats_DT, torch.Tensor), "Subclass must return torch.Tensor"
            assert feats_DT.ndim == 2 and feats_DT.shape[0] == self.NUM_FEATURES, \
                f"Expected [D, T] with D={self.NUM_FEATURES}, got {tuple(feats_DT.shape)}"
            feats_DT = feats_DT.to(device=device, dtype=dtype)

            # Aggregated statistics [6*D]
            moments = self._aggregate_six_moments(feats_DT)  # [6*D]
            assert moments.shape[0] == self.AGG_FEATURES

            # collect
            frame_feats.append(feats_DT.T)             # [T, D]
            agg_feats.append(moments.unsqueeze(0))     # [1, 6*D]

        # Pad and stack
        framewise = self._pad_stack_time(frame_feats) if self.keep_time_dim else None
        aggregated = torch.stack(agg_feats, dim=0).to(device=device, dtype=dtype)  # (B, 1, 6*D)

        if self.keep_time_dim:
            return (framewise, aggregated)
        else:
            return (aggregated,)


# -------- Implementations --------

class ChromaTA_Encoder(_LibrosaFeatureEncoder):
    """Chroma using torchaudio.prototype.transforms.ChromaSpectrogram (torch-only)."""
    NAME = "ChromaTAProto"
    NUM_FEATURES = 12
    AGG_FEATURES = 6 * NUM_FEATURES

    def __init__(self, keep_time_dim: bool = True,
                 n_fft: int = 4096, win_length: Optional[int] = None,
                 hop_length: Optional[int] = None, power: float = 2.0,
                 n_chroma: int = 12, tuning: float = 0.0, ctroct: float = 5.0,
                 octwidth: float = 2.0, norm: int = 2, base_c: bool = True):
        super().__init__(keep_time_dim=keep_time_dim)
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or self.HOP_LENGTH
        self.power = power
        self.chroma = T.ChromaSpectrogram(
            sample_rate=self.SAMPLING_RATE,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power,
            n_chroma=n_chroma,
            tuning=tuning,
            ctroct=ctroct,
            octwidth=octwidth,
            norm=norm,
            base_c=base_c
        )

    def _compute_features(self, y: torch.Tensor, sr: int, hop_length: int) -> torch.Tensor:
        # y: [time], -> chroma: [n_chroma, time]
        # ChromaSpectrogram expects [time] or [batch, time]; we give [time]
        chroma = self.chroma(y)  # [1, n_chroma, time] or [n_chroma, time] depending on version
        if chroma.dim() == 3:
            chroma = chroma.squeeze(0)
        # l1 norm
        chroma_sum = chroma.sum(dim=0, keepdim=True)  # [1, time]
        chroma_sum = torch.clamp(chroma_sum, min=1e-6)
        chroma = chroma / chroma_sum
        return chroma.to(torch.float32)


class MFCC_Encoder(_LibrosaFeatureEncoder):
    """MFCC using torchaudio.transforms.MFCC (torch-only)."""
    NAME = "MFCC"
    NUM_FEATURES = 20
    AGG_FEATURES = 6 * NUM_FEATURES

    def __init__(self, keep_time_dim: bool = True, n_mfcc: int = 20,
                 n_mels: int = 40, n_fft: int = 2048, hop_length: Optional[int] = None):
        super().__init__(keep_time_dim=keep_time_dim)
        self.NUM_FEATURES = n_mfcc  # keep attribute consistent if user overrides
        self.AGG_FEATURES = 6 * self.NUM_FEATURES
        self.hop_length = hop_length or self.HOP_LENGTH
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=self.SAMPLING_RATE,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": self.hop_length,
                "n_mels": n_mels,
                "center": True,
                "power": 2.0,
            },
        )

    def _compute_features(self, y: torch.Tensor, sr: int, hop_length: int) -> torch.Tensor:
        # torchaudio MFCC returns [n_mfcc, time]
        feats = self.mfcc(y)  # [n_mfcc, time]
        return feats.to(torch.float32)


# optinal: slow ver librosa cpu ver.
try:
    import librosa  # 可选依赖
    class Chroma_Encoder(_LibrosaFeatureEncoder):
        """Handcrafted encoder using librosa.feature.chroma_cqt (仅此处使用 numpy，外部仍是 torch)."""
        NAME = "Chroma"
        NUM_FEATURES = 12
        AGG_FEATURES = 6 * NUM_FEATURES

        def _compute_features(self, y: torch.Tensor, sr: int, hop_length: int) -> torch.Tensor:
            # librosa 需要 numpy，这里仅在局部转换，随后立刻转回 torch
            y_np = y.detach().cpu().numpy()
            feats_np = librosa.feature.chroma_cqt(y=y_np, sr=sr, hop_length=hop_length).astype("float32", copy=False)
            # l1 norm
            feats_sum = feats_np.sum(axis=0, keepdims=True)  # [1, T]
            feats_sum = np.maximum(feats_sum, 1e-6)
            feats_np = feats_np / feats_sum
            return torch.from_numpy(feats_np)
except Exception:
    # 如果没有 librosa，就跳过该类
    pass


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sec = 10
    B = 4
    wav = torch.randn(B, _LibrosaFeatureEncoder.SAMPLING_RATE * sec, dtype=torch.float32, device=device)

    chroma = ChromaTA_Encoder(keep_time_dim=True).to(device)
    mfcc = MFCC_Encoder(keep_time_dim=False).to(device)

    with torch.no_grad():
        out_c = chroma(wav, sampling_rate=_LibrosaFeatureEncoder.SAMPLING_RATE)
        out_m = mfcc(wav, sampling_rate=_LibrosaFeatureEncoder.SAMPLING_RATE)

    # out_c: (framewise, aggregated); out_m: (aggregated,)
    print("[Chroma keep_time_dim=True] framewise:", out_c[0].shape, " aggregated:", out_c[1].shape)
    print("[MFCC   keep_time_dim=False]:", out_m[0].shape)
