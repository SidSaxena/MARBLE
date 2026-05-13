# marble/encoders/OMAR_RQ/model.py
"""
OMAR-RQ encoder wrapping the `omar_rq` package.

Only the `multifeature-25hz-fsq` variant is exposed here — it is the
best-performing OMAR-RQ model (pitch .940, chord .749, beat .855).

Architecture (from config.gin):
  - 24 Conformer layers (depth=24), 1024-dim features (embed_dim=1024)
  - 24 kHz input, 25 Hz token rate (patch_size=960 samples @ 24kHz)

Install:
  pip install git+https://github.com/MTG/omar-rq.git

The encoder returns a tuple of 24 tensors (one per layer), each shaped
(B, T_tokens, 1024), compatible with MARBLE's LayerSelector.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from marble.core.base_encoder import BaseEncoder
from marble.core.base_transform import BaseAudioTransform


class OMARRQ_FeatureExtractor(BaseAudioTransform):
    """
    Audio pre-processing transform for OMAR-RQ.

    Squeezes the channel dimension from (1, T) → (T,) so that the model
    receives a flat mono waveform, as expected by `omar_rq`.

    The transform follows MARBLE's BaseAudioTransform contract: it takes
    a dict with at least ``"waveform": Tensor[C, T]`` and returns the
    same dict with the waveform modified in-place.
    """

    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sample: dict containing ``"input_features"`` of shape ``(C, T)``
                    (C should be 1 for mono).  MARBLE's AudioTransformDataset
                    passes data under the ``"input_features"`` key.

        Returns:
            sample: same dict with ``"input_features"`` reshaped to ``(T,)``.
        """
        waveform = sample["input_features"]
        if waveform.dim() == 2:
            # (C, T) → (T,)  — squeeze the channel dimension
            waveform = waveform.squeeze(0)
        sample["input_features"] = waveform
        return sample


class OMARRQ_Multifeature25hz_Encoder(BaseEncoder):
    """
    OMAR-RQ multifeature-25hz-fsq encoder.

    Wraps the ``omar_rq`` package to expose per-layer Conformer
    representations.  Only the ``multifeature-25hz-fsq`` checkpoint is
    supported here.

    Returns
    -------
    tuple of 24 ``torch.Tensor``
        One tensor per Conformer layer, each of shape ``(B, T_tokens, 1024)``.
        Compatible with ``marble.modules.transforms.LayerSelector``.
    """

    NAME = "OMAR-RQ-multifeature-25hz"
    HUGGINGFACE_MODEL_NAME = "mtg-upf/omar-rq-multifeature-25hz-fsq"
    SAMPLING_RATE = 24000   # model trained at 24 kHz (config.gin: new_freq=24000)
    TOKEN_RATE = 25.0       # patch_size=960 @ 24kHz → 960/24000 = 40ms = 25 Hz
    NUM_FEATURES = 1024     # config.gin: embed_dim=1024
    N_TRANSFORMER_LAYERS = 24  # config.gin: depth=24

    def __init__(
        self,
        model_id: Optional[str] = None,
        train_mode: str = "freeze",
    ) -> None:
        super().__init__()

        hf_id = model_id if model_id is not None else self.HUGGINGFACE_MODEL_NAME

        try:
            from omar_rq import get_model  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The `omar_rq` package is required for OMARRQ_Multifeature25hz_Encoder. "
                "Install it with:\n"
                "  pip install git+https://github.com/MTG/omar-rq.git"
            ) from exc

        self.model = get_model(model_id=hf_id, device="cpu")

        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            raise NotImplementedError(
                f"train_mode='{train_mode}' is not supported for "
                "OMARRQ_Multifeature25hz_Encoder. Only 'freeze' is available."
            )

    # Upstream documents "up to 30 s" of audio per forward. Past that the
    # vit_tokenization assertions are silent on what happens — guard here.
    MAX_INPUT_SECONDS = 30.0

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        input_len: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Raw waveform of shape ``(B, T)`` or ``(B, 1, T)`` at 24 kHz.
            Pre-conditioned to mono (1 channel) by the dataloader's
            ``OMARRQ_FeatureExtractor``.
        input_len : torch.Tensor, optional
            Not used; kept for API compatibility.

        Returns
        -------
        tuple of torch.Tensor
            24-element tuple; each element has shape ``(B, T_tokens, 1024)``.
            T_tokens = ⌊T / 960⌋ since the model patches at 40 ms (25 Hz).
        """
        # Squeeze optional channel dim: (B, 1, T) → (B, T)
        if x.dim() == 3:
            x = x.squeeze(1)

        # Guard against >30 s inputs (upstream silently misbehaves).
        # T = x.shape[-1] samples at 24 kHz.
        max_samples = int(self.SAMPLING_RATE * self.MAX_INPUT_SECONDS)
        if x.shape[-1] > max_samples:
            raise ValueError(
                f"OMAR-RQ accepts up to {self.MAX_INPUT_SECONDS:.0f} s; "
                f"got {x.shape[-1] / self.SAMPLING_RATE:.1f} s "
                f"({x.shape[-1]} samples)."
            )

        # extract_embeddings returns a tensor of shape (L, B, T_tokens, C)
        # where L = N_TRANSFORMER_LAYERS = 24, C = 1024.
        # The upstream API takes a `set` (it coerces lists, but spec is a set).
        # Layers are returned in INSERTION ORDER (conformer.py:447) so output
        # index 0 == first conformer block, index 23 == last.
        layer_indices = set(range(self.N_TRANSFORMER_LAYERS))
        embeddings = self.model.extract_embeddings(x, layers=layer_indices)
        # embeddings: (L=24, B, T_tokens, C=1024)

        return tuple(embeddings[i] for i in range(self.N_TRANSFORMER_LAYERS))
