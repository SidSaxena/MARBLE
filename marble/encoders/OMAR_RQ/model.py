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

import torch

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

    def forward(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
    # Default is the NON-FSQ variant. Variant comparison on VGMIDITVar
    # (2026-05-14, see docs/data/omar_rq_audit.md) found the -fsq variant
    # gets ~2.3× worse retrieval MAP than non-fsq, while supervised-task
    # numbers in the paper are nearly identical. Non-fsq is the right
    # default for both retrieval and classification probes; pass
    # model_id="mtg-upf/omar-rq-multifeature-25hz-fsq" explicitly to
    # restore the original variant for direct comparison.
    HUGGINGFACE_MODEL_NAME = "mtg-upf/omar-rq-multifeature-25hz"
    SAMPLING_RATE = 24000  # model trained at 24 kHz (config.gin: new_freq=24000)
    TOKEN_RATE = 25.0  # patch_size=960 @ 24kHz → 960/24000 = 40ms = 25 Hz
    NUM_FEATURES = 1024  # config.gin: embed_dim=1024
    N_TRANSFORMER_LAYERS = 24  # config.gin: depth=24

    def __init__(
        self,
        model_id: str | None = None,
        train_mode: str = "freeze",
        compile_mode: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_id : str | None
            HuggingFace model id (defaults to ``HUGGINGFACE_MODEL_NAME``).
        train_mode : str
            Only ``"freeze"`` is supported.
        compile_mode : str | None
            If set, wraps the underlying ``self.model`` with
            ``torch.compile(mode=compile_mode)``. Recommended values: ``"default"``
            (safe with drop_last=False; no CUDA Graphs) or ``"reduce-overhead"``
            (uses CUDA Graphs — faster but recompiles on shape mismatch).
            Capability-gated on Triton + CUDA; falls back to eager with a
            warning if either is missing. ~30-90s first-forward cost; cached
            in ``~/.cache/torch_inductor`` across processes.
        """
        super().__init__()
        # Stashed for the train() override below — Lightning's per-epoch
        # self.train() call propagates recursively to children and would
        # otherwise undo the .eval() applied here for train_mode='freeze'.
        self._marble_train_mode = train_mode

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

        # Read actual metadata from the loaded module instead of trusting
        # the class-level defaults. Lets a single class wrap ANY OMAR-RQ
        # variant (multifeature, multifeature-25hz, multifeature-25hz-fsq, ...)
        # which differ in sample rate, token rate, and possibly layer count.
        self.sampling_rate = int(self.model.sr)
        self.token_rate = float(self.model.eps)
        self.num_features = int(self.model.net.embed_dim)
        self.n_transformer_layers = len(self.model.net.layers)

        # Sanity-warn when the loaded variant differs from the class defaults
        # so a misconfigured probe head (in_dim mismatch) is loud upfront.
        if self.num_features != self.NUM_FEATURES:
            print(
                f"  ! OMAR-RQ variant feature dim = {self.num_features} "
                f"(class default {self.NUM_FEATURES}); update probe in_dim"
            )
        if abs(self.token_rate - self.TOKEN_RATE) > 0.1:
            print(
                f"  ! OMAR-RQ variant token rate = {self.token_rate:.2f} Hz "
                f"(class default {self.TOKEN_RATE} Hz); update fps/label_freq"
            )

        if train_mode == "freeze":
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            raise NotImplementedError(
                f"train_mode='{train_mode}' is not supported for "
                "OMARRQ_Multifeature25hz_Encoder. Only 'freeze' is available."
            )

        # Optional torch.compile wrap. Same capability gate as MERT — Triton +
        # CUDA required; falls back to eager with a warning otherwise. The
        # OMAR-RQ upstream forward is a clean pass-through to extract_embeddings
        # (no Python-side data-dependent control flow), so compile is expected
        # to work cleanly with both 'default' and 'reduce-overhead' modes.
        if compile_mode is not None:
            skip_reason = None
            try:
                import triton  # noqa: F401 — capability check
            except ImportError:
                skip_reason = (
                    "Triton not installed (no official Windows build; "
                    "install on Linux/Mac for compile support)"
                )
            if skip_reason is None and not torch.cuda.is_available():
                skip_reason = "no CUDA device available (compile gives little benefit on CPU/MPS)"
            if skip_reason is not None:
                print(
                    f"[OMARRQ-multifeature-25hz] torch.compile(mode={compile_mode!r}) requested "
                    f"but skipped — {skip_reason}. Falling back to eager."
                )
            else:
                try:
                    self.model = torch.compile(self.model, mode=compile_mode)
                    print(
                        f"[OMARRQ-multifeature-25hz] torch.compile(mode={compile_mode!r}) applied. "
                        f"First forward will trigger compilation (30-90s typically)."
                    )
                except Exception as e:  # pragma: no cover — defensive
                    print(
                        f"[OMARRQ-multifeature-25hz] torch.compile(mode={compile_mode!r}) failed "
                        f"at wrap time: {type(e).__name__}: {e}. Falling back to eager."
                    )

    # Upstream documents "up to 30 s" of audio per forward. Past that the
    # vit_tokenization assertions are silent on what happens — guard here.
    MAX_INPUT_SECONDS = 30.0

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        input_len: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
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
        max_samples = int(self.sampling_rate * self.MAX_INPUT_SECONDS)
        if x.shape[-1] > max_samples:
            raise ValueError(
                f"OMAR-RQ accepts up to {self.MAX_INPUT_SECONDS:.0f} s; "
                f"got {x.shape[-1] / self.sampling_rate:.1f} s "
                f"({x.shape[-1]} samples)."
            )

        # extract_embeddings returns (L, B, T_tokens, C). The upstream API
        # takes a `set` (it coerces lists, but spec is a set). Layers are
        # returned in INSERTION ORDER (conformer.py:447) so output index 0
        # is the first conformer block, n_layers-1 is the last.
        layer_indices = set(range(self.n_transformer_layers))
        embeddings = self.model.extract_embeddings(x, layers=layer_indices)

        return tuple(embeddings[i] for i in range(self.n_transformer_layers))

    def train(self, mode: bool = True):
        # Re-apply .eval() to the frozen submodule after the recursive
        # propagation from the parent LightningModule's train() call.
        # OMAR-RQ only supports train_mode='freeze' today, but the guard
        # keeps the override correct if other modes are added later.
        super().train(mode)
        if self._marble_train_mode == "freeze":
            self.model.eval()
        return self
