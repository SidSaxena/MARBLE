# marble/modules/transforms.py
import random
import re
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange, reduce

from marble.core.base_transform import BaseAudioTransform, BaseEmbTransform


############################## Audio Transforms ##############################
class AudioTransformDataset(torch.utils.data.Dataset):
    """Sequentially apply BaseAudioTransform instances on raw waveforms."""

    def __init__(self, base_dataset, transforms: list[BaseAudioTransform]):
        self.base = base_dataset
        self.transforms = transforms
        # assume base_dataset has sample_rate attribute
        self.sample_rate = getattr(base_dataset, "sample_rate", None)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base[idx] returns at minimum 3 elements:
        #   waveform: Tensor of shape [C, T] (or [1, T] for mono)
        #   label: any (e.g. int)
        #   path: str
        # Optional extras (e.g. clip_id for the embedding cache) come
        # after path and are passed through unchanged.
        items = self.base[idx]
        waveform, label, path, *extras = items

        # ensure waveform is [C, T]
        assert waveform.ndim == 2 and waveform.shape[0] > 0, (
            f"Expected waveform shape [C, T], got {waveform.shape}"
        )

        sample = {
            "input_features": waveform,  # Tensor [C, T]
            "sampling_rate": self.sample_rate,  # int
        }

        # apply each transform in sequence
        for t in self.transforms:
            sample = t(sample)

        # final waveform
        final_input = sample["input_features"]  # Tensor [C, T] or [T] (for mert)
        return (final_input, label, path, *extras)


class AudioLayerNorm(BaseAudioTransform):
    """
    Normalize each channel to zero‐mean, unit‐variance over time.

    Args:
        eps (float): to avoid div by zero.
        affine (bool): if True, learn scale & bias per channel.
    """

    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            # gamma, beta: each [1, 1] (broadcast to [C, T])
            self.gamma = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.zeros(1, 1))

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # w: [C, T]
        w = sample["input_features"]
        mean = w.mean(dim=-1, keepdim=True)  # [C, 1]
        std = w.std(dim=-1, keepdim=True)  # [C, 1]
        # normalized: [C, T]
        w_norm = (w - mean) / (std + self.eps)
        if self.affine:
            # broadcast gamma, beta to [C, T]
            w_norm = w_norm * self.gamma + self.beta
        sample["input_features"] = w_norm  # [C, T]
        return sample


class RandomCrop(BaseAudioTransform):
    def __init__(self, crop_size: int):
        """
        Args:
            crop_size (int): target length in samples (T_out).
        """
        super().__init__()
        self.crop_size = crop_size

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # waveform: [C, T]
        waveform = sample["input_features"]
        C, T = waveform.shape
        if self.crop_size >= T:
            pad = self.crop_size - T
            # pad to [C, crop_size]
            waveform = F.pad(waveform, (0, pad))
        else:
            start = random.randint(0, T - self.crop_size)
            # crop to [C, crop_size]
            waveform = waveform[:, start : start + self.crop_size]
        sample["input_features"] = waveform  # [C, crop_size]
        return sample


class AddNoise(BaseAudioTransform):
    """
    Adds random Gaussian noise to the waveform based on a random SNR."""

    def __init__(self, snr_min: float = 5.0, snr_max: float = 20.0):
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, sample):
        # waveform: [C, T]
        waveform = sample["input_features"]
        # 随机采样一个 SNR
        snr = torch.empty(1).uniform_(self.snr_min, self.snr_max).item()  # scalar
        rms = waveform.pow(2).mean().sqrt()  # scalar
        # noise: [C, T]
        noise_std = rms / (10 ** (snr / 20))
        noise = torch.randn_like(waveform) * noise_std
        sample["input_features"] = waveform + noise
        return sample


class Resample(BaseAudioTransform):
    def __init__(self, orig_freq: int, new_freq: int):
        """
        Args:
            orig_freq (int): original sampling rate.
            new_freq  (int): desired sampling rate.
        """
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(orig_freq, new_freq)

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # input waveform: [C, T]
        out = self.resampler(sample["input_features"])
        # output waveform: [C, T_new]
        sample["input_features"] = out
        return sample


class Spectrogram(BaseAudioTransform):
    def __init__(
        self,
        n_fft: int = 400,
        win_length: int | None = None,
        hop_length: int | None = None,
        power: float = 2.0,
    ):
        """
        Args:
            n_fft (int): FFT window size.
            win_length (int): window length.
            hop_length (int): hop length between frames.
            power (float): exponent for magnitude.
        """
        super().__init__()
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length or (win_length or n_fft) // 2,
            power=power,
        )

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # input waveform: [C, T]
        S = self.spec(sample["input_features"])
        # spectrogram: [C, F, T']
        sample["input_features"] = S
        return sample


class MelSpectrogram(BaseAudioTransform):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 400,
        n_mels: int = 80,
        win_length: int | None = None,
        hop_length: int | None = None,
    ):
        """
        Args:
            sample_rate (int): sampling rate.
            n_fft (int): FFT window size.
            n_mels (int): number of Mel bins.
            win_length (int): window length.
            hop_length (int): hop between frames.
        """
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length or n_fft,
            hop_length=hop_length or (win_length or n_fft) // 2,
            n_mels=n_mels,
        )

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # input waveform: [C, T]
        M = self.melspec(sample["input_features"])
        # mel spectrogram: [C, n_mels, T']
        sample["input_features"] = M
        return sample


############################## Embedding Transforms ##############################


class LayerSelector(BaseEmbTransform):
    """
    Selects a subset of hidden-state layers and optionally aggregates them.

    Accepts an integer list ``[0, 1, 5]`` OR string ranges ``["0..23"]``.

    Modes
    -----
    ``mode="select"`` (default, BACKWARD-COMPATIBLE):
        Returns a 4D tensor ``(B, len(layers), T, C)`` — same as before.
        With ``layers=[N]`` this is the standard per-layer sweep.

    ``mode="mean"``:
        Returns ``(B, 1, T, C)`` — the element-wise mean of the selected
        layers. Matches the upstream OMAR-RQ probing convention
        (``omar_rq/probe/data/nsynth_pitch.py:133`` uses ``mean(dim=0)``
        over the layer axis).

    ``mode="sum"``:
        Returns ``(B, 1, T, C)`` — element-wise sum.

    ``mode="concat"``:
        Returns ``(B, 1, T, C * len(layers))`` — concatenated along the
        feature dim. The probe head's ``in_dim`` must match.
    """

    RANGE_RE = re.compile(r"^(\d+)\.\.(\d+)$")
    _VALID_MODES = ("select", "mean", "sum", "concat")

    def __init__(
        self,
        layers: Sequence[int | str],
        mode: str = "select",
    ):
        super().__init__()
        self.layers = self._parse_layers(layers)
        if mode not in self._VALID_MODES:
            raise ValueError(f"Unknown mode {mode!r}; valid: {self._VALID_MODES}")
        self.mode = mode
        print(f"LayerSelector initialized with layers: {self.layers}  (mode={self.mode})")

    def _parse_layers(self, layers):
        parsed = []
        for x in layers:
            if isinstance(x, str):
                m = self.RANGE_RE.match(x.strip())
                if m:
                    start, end = map(int, m.groups())
                    if end < start:
                        raise ValueError(f"Range end ({end}) < start ({start})")
                    parsed.extend(range(start, end + 1))
                else:
                    parsed.append(int(x))
            else:
                parsed.append(int(x))
        return parsed

    def forward(self, hidden_states: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
        selected = [hidden_states[i] for i in self.layers]
        stacked = torch.stack(selected, dim=1)  # (B, L, T, C)
        assert stacked.ndim == 4, f"Expected 4D tensor after stacking, got {stacked.ndim}D"

        if self.mode == "select":
            return stacked
        if self.mode == "mean":
            return stacked.mean(dim=1, keepdim=True)  # (B, 1, T, C)
        if self.mode == "sum":
            return stacked.sum(dim=1, keepdim=True)  # (B, 1, T, C)
        if self.mode == "concat":
            # (B, L, T, C) → (B, 1, T, L*C)
            b, l, t, c = stacked.shape
            return stacked.permute(0, 2, 1, 3).reshape(b, 1, t, l * c)
        raise ValueError(f"Unknown mode {self.mode!r}")


class LayerSoftmaxSum(BaseEmbTransform):
    """
    SUPERB-style learned layer aggregation: softmax-normalised scalar gates.

    ``out = Σ_l softmax(α)_l · h_l`` with ``α ∈ R^L`` learnable (init 0 →
    uniform weights = meanall at step 0). This is the s3prl/SUPERB
    "featurizer" convention — unlike :class:`LayerWeightedSum` below (an
    unnormalised 1×1 Conv1d with bias, never used by any config), the
    softmax gates are positive and sum to 1, so ``layer_weights()`` reads
    directly as "how much each layer contributes" and can be logged per
    epoch (see ``marble.modules.callbacks.LogLayerWeightsCallback``) for
    the SUPERB-style layer-contribution chart.

    Supervised probes only: the gates need a training signal, so this does
    not apply to zero-shot retrieval tasks (there ``meanall`` is the
    unsupervised aggregate). For multi-head runs use
    ``PerLayerHeads(include_weighted=True)`` instead — there the weighted
    head rides alongside the per-layer heads in one run.

    ``normalize=True`` (default) applies a NON-learnable LayerNorm across the
    hidden dim to each layer before the weighted sum — the "normalized
    benchmarking" fix of Feng et al., "A Large-Scale Evaluation of Speech
    Foundation Models" (IEEE/ACM TASLP 2024, arXiv:2404.09385): per-layer
    feature norms differ wildly across depth (the last layer is often
    tiny-scaled), so without normalization the learned gates jointly encode
    scale-compensation AND informativeness and CANNOT be read as layer
    contributions. With it, task performance is essentially unchanged and the
    gates become interpretable. Keep it on for anything thesis-facing;
    ``normalize=False`` reproduces the raw SUPERB featurizer.
    """

    def __init__(
        self,
        num_layers: int,
        normalize: bool = True,
        learnable: bool = True,
        init_weights: list[float] | None = None,
    ):
        """
        Args:
            num_layers: number of layers in the incoming (B, L, T, H) stack.
            normalize: Feng et al. LayerNorm-before-mix (see class docstring).
            learnable: when False the gates are a fixed buffer (no gradient,
                not in the optimizer) — used to TRANSFER gates learned on one
                corpus to another (e.g. HookTheory gates applied to MedleyDB
                with only the probe head training).
            init_weights: optional post-softmax contribution weights, one per
                layer, aligned with the order the preceding LayerSelector
                emits (it preserves its ``layers`` list order). Values must be
                positive; they are renormalised to sum to 1 and stored as log
                weights, so ``softmax`` reproduces them exactly. Default (None)
                keeps the SUPERB convention: zeros → uniform mix at step 0.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"LayerSoftmaxSum needs num_layers >= 2, got {num_layers}")
        self.num_layers = int(num_layers)
        self.normalize = bool(normalize)
        self.learnable = bool(learnable)
        if init_weights is not None:
            if len(init_weights) != self.num_layers:
                raise ValueError(
                    f"init_weights has {len(init_weights)} entries but num_layers="
                    f"{self.num_layers}; they must align 1:1 with the LayerSelector order"
                )
            w = torch.tensor([float(v) for v in init_weights], dtype=torch.float32)
            if (w <= 0).any():
                raise ValueError(f"init_weights must be strictly positive, got {init_weights}")
            gate0 = torch.log(w / w.sum())  # softmax(log w̄) == w̄ exactly
        else:
            if not self.learnable:
                raise ValueError(
                    "learnable=False without init_weights would freeze a uniform mix "
                    "— that is LayerSelector(mode='mean'); pass the weights explicitly"
                )
            gate0 = torch.zeros(self.num_layers)
        if self.learnable:
            self.layer_gate = nn.Parameter(gate0)
        else:
            self.register_buffer("layer_gate", gate0)

    def layer_weights(self) -> torch.Tensor:
        """Softmax-normalised per-layer contribution weights (detached, CPU)."""
        return torch.softmax(self.layer_gate.detach(), dim=0).cpu()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer-stacked tensor (B, L, T, H) — or the encoder's
                layer tuple, stacked here for parity with the other
                aggregating transforms.
        Returns:
            Tensor: (B, 1, T, H) softmax-weighted sum over layers.
        """
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        if x.size(1) != self.num_layers:
            raise ValueError(
                f"LayerSoftmaxSum was built for {self.num_layers} layers but got L={x.size(1)}"
            )
        if self.normalize:
            # Non-learnable (no affine): a learnable per-layer affine would
            # reintroduce exactly the scale freedom the LayerNorm removes.
            x = F.layer_norm(x, (x.size(-1),))
        w = torch.softmax(self.layer_gate, dim=0)
        return (x * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)


class LayerWeightedSum(BaseEmbTransform):
    """
    Learns a weighted sum over L layers via a 1×1 Conv1d.

    NOTE: legacy/unused — no config references this. It is NOT the SUPERB
    featurizer (weights are unnormalised and there is a bias term, so they
    do not read as layer contributions). Prefer :class:`LayerSoftmaxSum`.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Weighted sum over layers, of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        x_flat = rearrange(x, "b l t h -> b l (t h)")
        y = self.conv(x_flat)
        return rearrange(y, "b 1 (t h) -> b 1 t h", h=x.size(-1))


class MLPReduce(BaseEmbTransform):
    """
    Flattens layers & hidden dims and reduces via an MLP.
    """

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(num_layers * hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Reduced representation of shape
                (batch_size, 1, seq_len, hidden_size).
        """
        if isinstance(x, tuple):
            x = torch.stack(x, dim=1)
        xt = rearrange(x, "b l t h -> (b t) (l h)")
        y = self.fc(xt)
        return rearrange(y, "(b t) h -> b 1 t h", t=x.size(2))


class TimeAdaptivePool(BaseEmbTransform):
    """
    Applies adaptive average pooling over time to a fixed length.
    """

    def __init__(self, target_frames: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_frames)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐pooled tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        x2 = rearrange(x, "b l t h -> (b l) h t")
        y = self.pool(x2)
        return rearrange(y, "(b l) h t -> b l t h", b=x.size(0), l=x.size(1))


class LinearInterpolation(BaseEmbTransform):
    """
    Linearly resamples the time axis to a fixed number of frames.
    """

    def __init__(self, target_frames: int, align_corners: bool = False):
        super().__init__()
        self.target_frames = target_frames
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer-stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time-resampled tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        b, l, t, h = x.shape
        # Treat hidden_size as channels for 1D interpolation over time
        x2 = rearrange(x, "b l t h -> (b l) h t")  # (B*L, H, T)
        y = F.interpolate(
            x2, size=self.target_frames, mode="linear", align_corners=self.align_corners
        )
        return rearrange(y, "(b l) h t -> b l t h", b=b, l=l)


class TimeAvgPool(BaseEmbTransform):
    """
    Computes simple average pooling over the time dimension.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Time‐averaged tensor of shape
                (batch_size, num_layers, 1, hidden_size).
        """
        return reduce(x, "b l t h -> b l 1 h", "mean")


class TimeInterpolation(BaseEmbTransform):
    """
    Interpolates the time dimension to a new fixed length.
    """

    def __init__(self, target_frames: int, mode: str = "linear", align_corners: bool = False):
        super().__init__()
        self.target_frames = target_frames
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Layer‐stacked tensor of shape
                (batch_size, num_layers, seq_len, hidden_size).
        Returns:
            Tensor: Interpolated tensor of shape
                (batch_size, num_layers, target_frames, hidden_size).
        """
        x2 = rearrange(x, "b l t h -> (b l) h t")
        y = F.interpolate(
            x2,
            size=self.target_frames,
            mode=self.mode,
            align_corners=self.align_corners
            if self.mode in ("linear", "bilinear", "trilinear")
            else None,
        )
        return rearrange(y, "(b l) h t -> b l t h", b=x.size(0), l=x.size(1))
