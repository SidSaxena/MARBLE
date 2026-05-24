# marble/encoders/MuQ/model.py
from typing import Sequence, Dict, Optional, Union, Tuple, List

import torch

from marble.encoders.MuQ.muq import MuQ
from marble.core.base_encoder import BaseEncoder


class MuQ_Encoder(BaseEncoder):
    """
    A Hugging Face HuBERT-based wrapper with optional LoRA adapters, full fine-tuning, or freezing.
    """

    NAME = "MuQ"
    HUGGINGFACE_MODEL_NAME = "OpenMuQ/MuQ-large-msd-iter"
    TOKEN_RATE = 25  # Number of feature frames per second of audio
    SAMPLING_RATE = 24000  # Audio sampling rate expected by the model
    NUM_FEATURES = 1024  # Hidden dimension of the HuBERT model
    N_TRANSFORMER_LAYERS = 12  # Number of transformer layers in the backbone

    def __init__(
        self,
        pre_trained_folder: str = None,
        train_mode: str = "freeze",  # one of ["freeze", "full", "lora"]
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Sequence[str] = ["q_proj", "v_proj"],
        compile_mode: str | None = None,
    ) -> None:
        """
        Initialize the MERT HuBERT encoder.

        Args:
            pre_trained_folder (str, optional): Path or HF identifier of the pretrained model.
            train_mode (str): "freeze" to freeze base parameters, "full" for full fine-tuning,
                              or "lora" to freeze base and add LoRA adapters.
            lora_r (int): LoRA adapter rank (only if train_mode="lora").
            lora_alpha (int): LoRA scaling alpha (only if train_mode="lora").
            lora_dropout (float): Dropout probability for LoRA adapters.
            compile_mode (str | None): If set, wraps ``self.model`` with
                ``torch.compile(mode=compile_mode)``. Recommended values:
                ``"default"`` (safe with drop_last=False) or ``"reduce-overhead"``
                (CUDA Graphs; faster but recompiles on shape mismatch).
                Capability-gated on Triton + CUDA; falls back to eager otherwise.
                Note: MuQ's Conformer has an attention_mask-dependent code path
                (muq_model.py); for consistent compilation, the encoder must
                always be called with the same mask presence (None vs tensor).
                The MARBLE inference path always passes ``attention_mask=None``,
                so this is satisfied by default — but be aware if you override.
                Only applied when train_mode="freeze"; otherwise ignored.
        """
        super().__init__()
        self.sample_rate = self.SAMPLING_RATE
        # Stashed for the train() override below — Lightning's per-epoch
        # self.train() call propagates recursively to children and would
        # otherwise undo the .eval() applied here for train_mode='freeze'.
        self._marble_train_mode = train_mode

        # Load the core MusicHuBERT model
        self.model = MuQ.from_pretrained(
            pre_trained_folder or self.HUGGINGFACE_MODEL_NAME
        )


        # Configure which parameters to train
        if train_mode == "freeze":
            # Freeze all backbone parameters
            for param in self.model.parameters():
                param.requires_grad = False

        elif train_mode == "lora":
            # Freeze backbone and add LoRA adapters
            from peft import get_peft_model, LoraConfig, TaskType

            for param in self.model.parameters():
                param.requires_grad = False

            peft_config = LoraConfig(
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)

        elif train_mode == "full":
            # Enable training of all parameters
            for param in self.model.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

        # Set model to train or eval mode
        if train_mode in ["lora", "full"]:
            self.model.train()
        else:
            self.model.eval()

        # Optional torch.compile wrap. Same capability gate as MERT/OMARRQ.
        # Only applied when frozen — compile gives little benefit during fine-
        # tuning since the gradient path is also compiled and recompiled per
        # autograd step. MuQ's attention_mask-dependent branching in
        # muq_model.py:211 means we want consistent mask presence at trace
        # time; MARBLE's inference path always passes attention_mask=None
        # which satisfies that constraint.
        if compile_mode is not None and train_mode == "freeze":
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
                    f"[MuQ] torch.compile(mode={compile_mode!r}) requested "
                    f"but skipped — {skip_reason}. Falling back to eager."
                )
            else:
                try:
                    self.model = torch.compile(self.model, mode=compile_mode)
                    print(
                        f"[MuQ] torch.compile(mode={compile_mode!r}) applied. "
                        f"First forward will trigger compilation (30-90s typically)."
                    )
                except Exception as e:  # pragma: no cover — defensive
                    print(
                        f"[MuQ] torch.compile(mode={compile_mode!r}) failed at "
                        f"wrap time: {type(e).__name__}: {e}. Falling back to eager."
                    )
        elif compile_mode is not None:
            print(
                f"[MuQ] compile_mode={compile_mode!r} ignored — only applied "
                f"when train_mode='freeze' (got {train_mode!r})."
            )

    def forward(
        self,
        x: torch.Tensor,
        *args,
        output_hidden_states: bool = True,
        **kwargs
    ) -> dict:
        """
        Perform a forward pass through the HuBERT encoder.

        Args:
            x (torch.Tensor): Waveform tensor, shape (batch_size, num_samples), values in [-1, 1].
            output_hidden_states (bool): If True, return all intermediate hidden states.
            *args, **kwargs: Additional arguments passed to the underlying model.

        Returns:
            hidden_states (tuple of torch.FloatTensor, optional): All layer outputs
                  if output_hidden_states=True; each is (batch_size, seq_len, NUM_FEATURES).
        """
        # Ensure input dtype matches model parameters (fp16 vs fp32)
        model_dtype = next(self.model.parameters()).dtype
        x = x.to(device=self.model.device, dtype=model_dtype)

        outputs = self.model(
            x=x,
            output_hidden_states=output_hidden_states
        )

        return outputs.hidden_states

    def train(self, mode: bool = True):
        # Re-apply .eval() to the frozen submodule after the recursive
        # propagation from the parent LightningModule's train() call.
        # Without this, dropout/BatchNorm in self.model would run in
        # train mode every epoch despite the .eval() in __init__.
        super().train(mode)
        if self._marble_train_mode == "freeze":
            self.model.eval()
        return self


if __name__ == "__main__":
    device = 'cuda'
    # fake wav for testing
    wav = torch.randn(4, 24000 * 10)  # 10 seconds of audio at 24kHz
    wavs = torch.tensor(wav).to(device)

    # This will automatically fetch the checkpoint from huggingface
    muq = MuQ_Encoder()
    muq = muq.to(device).eval()

    with torch.no_grad():
        output = muq(wavs, output_hidden_states=True)

    print('Total number of layers: ', len(output))
    print('Output shape of each layer: ', [layer.shape for layer in output])
