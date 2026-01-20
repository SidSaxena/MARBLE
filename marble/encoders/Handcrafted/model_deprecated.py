# marble/encoders/Handcrafted/model.py
from typing import Dict, Tuple, List
import numpy as np
import torch
import librosa
import torchaudio.prototype.transforms as T

from marble.core.base_encoder import BaseEncoder


class _LibrosaFeatureEncoder(BaseEncoder):
    """
    Base class for handcrafted feature encoders using librosa.
    Subclasses implement `_compute_features(y, sr, hop_length)` that returns [D, T] features.
    """

    NAME: str = "LibrosaFeature"
    SAMPLING_RATE: int = 22050
    HOP_LENGTH: int = 512
    TOKEN_RATE: float = SAMPLING_RATE / HOP_LENGTH  # ~43 fps

    NUM_FEATURES: int = -1  # feature dimension D
    AGG_FEATURES: int = -1  # aggregated dimension 6*D

    def __init__(self, keep_time_dim=True) -> None:
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        self.keep_time_dim = keep_time_dim

    # --- To be implemented by subclass ---
    def _compute_features(self, y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        raise NotImplementedError

    # --- Utility methods ---
    @staticmethod
    def _normalize_audio(y: np.ndarray) -> np.ndarray:
        """Peak-normalize the waveform to [-1, 1]."""
        y = y.astype(np.float32, copy=False)
        maxv = np.abs(y).max()
        if maxv > 0:
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
            T = f.shape[0]
            out[i, :T] = f
        return out

    def forward(
        self,
        x: torch.Tensor,
        *args,
        **kwargs
    ) -> Dict[str, Tuple[torch.Tensor, ...]]:
        """
        Args:
            x: (B, num_samples) waveform tensor in [-1, 1].
            keep_time_dim: if True (default), return both framewise and aggregated features;
                           if False, return only the aggregated vector.
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
        # assert x.dim() == 2, f"Expected waveform shape (B, num_samples), got {x.shape}"
        if x.dim() == 3:
            x = x.squeeze(1)
        assert x.dim() == 2, f"Expected waveform shape (B, num_samples), got {x.shape}"
        B = x.shape[0]
        device = x.device

        frame_feats: List[torch.Tensor] = []
        agg_feats: List[torch.Tensor] = []

        # Move to CPU numpy for librosa processing
        x_np = x.detach().cpu().numpy()

        for b in range(B):
            y = x_np[b]
            y = self._normalize_audio(y)

            # Framewise features [D, T]
            feats = self._compute_features(y, self.SAMPLING_RATE, self.HOP_LENGTH).astype(np.float32, copy=False)
            assert feats.ndim == 2 and feats.shape[0] == self.NUM_FEATURES

            # Aggregated statistics (diff up to 2nd order, mean + std)
            moments = []
            for i in range(3):
                f = np.diff(feats, n=i, axis=1)  # i=0 keeps the original
                moments.append(f.mean(axis=1))
                moments.append(f.std(axis=1))
            moments = np.concatenate(moments, axis=0).astype(np.float32, copy=False)
            assert moments.shape[0] == self.AGG_FEATURES

            frame_feats.append(torch.from_numpy(feats.T))     # [T, D]
            agg_feats.append(torch.from_numpy(moments)[None]) # [1, 6*D]

        # Pad and stack
        frame_feats = [f.to(device=device, dtype=torch.float32) for f in frame_feats]
        agg_feats = [f.to(device=device, dtype=torch.float32) for f in agg_feats]

        framewise = self._pad_stack_time(frame_feats)  # (B, T_max, D)
        aggregated = torch.stack(agg_feats, dim=0)     # (B, 1, 6*D)


        if self.keep_time_dim:
            return (framewise,)
        else:
            return (aggregated,)


class Chroma_Encoder(_LibrosaFeatureEncoder):
    """Handcrafted encoder using chroma_cqt."""
    NAME = "Chroma"
    NUM_FEATURES = 12
    AGG_FEATURES = 6 * NUM_FEATURES

    def _compute_features(self, y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        feats = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        return feats.astype(np.float32, copy=False)


class MFCC_Encoder(_LibrosaFeatureEncoder):
    """Handcrafted encoder using MFCC."""
    NAME = "MFCC"
    NUM_FEATURES = 20
    AGG_FEATURES = 6 * NUM_FEATURES

    def _compute_features(self, y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
        feats = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=self.NUM_FEATURES)
        return feats.astype(np.float32, copy=False)


if __name__ == "__main__":
    # Simple test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sec = 10
    B = 4
    wav = torch.randn(B, _LibrosaFeatureEncoder.SAMPLING_RATE * sec, dtype=torch.float32).to(device)

    chroma = Chroma_Encoder(keep_time_dim=False)
    mfcc = MFCC_Encoder(keep_time_dim=False)

    with torch.no_grad():
        out_c = chroma(wav)
        out_m = mfcc(wav)

    print("[Chroma keep_time_dim=True] :", out_c[0].shape)
    print("[MFCC   keep_time_dim=False]:", out_m[0].shape)
