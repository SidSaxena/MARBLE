# tasks/gtzan_genre/decoder.py
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from marble.core.base_decoder import BaseDecoder
from marble.core.utils import instantiate_from_config


class MLPDecoderKeepTime(BaseDecoder):
    """
    MLP Decoder that collapses the 'layer' dimension (L) but preserves the 'time' dimension (T).
    Takes input tensors of shape [B, L, T, H], where:
      - B is batch size
      - L is the number of layers/features to pool over
      - T is the sequence length or time dimension
      - H is the embedding dimension

    This decoder mean-pools across L only, producing an intermediate tensor of shape [B, T, H].
    Then it applies a stack of Linear, activation, and Dropout layers to each time step independently,
    yielding an output of shape [B, T, out_dim].
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10,
        hidden_layers: list = [512],  # noqa: B006 — pre-existing config-facing default
        activation_fn: dict | None = None,  # e.g. {"class_path": "torch.nn.ReLU"}
        dropout: float = 0.5,
    ):
        super().__init__(in_dim, out_dim)

        layers = []
        prev_dim = in_dim

        # Build a sequence of Linear → Activation → Dropout layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_fn is not None:
                act = instantiate_from_config(activation_fn)
                layers.append(act)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final projection from last hidden_dim to out_dim
        layers.append(nn.Linear(prev_dim, out_dim))

        # Combine into a single nn.Sequential. This will operate over the last dimension H.
        self.net = nn.Sequential(*layers)

    def forward(self, emb, *_):
        """
        Forward pass of the MLPDecoder.

        Args:
            emb (torch.Tensor): Input tensor of shape [B, L, T, H].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, out_dim].
        """
        # Ensure we have a 4D tensor: [B, L, T, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"

        # Mean-pool across the layer dimension (L), but keep the time dimension (T):
        # Resulting shape: [B, T, H]
        emb = reduce(emb, "b l t h -> b t h", "mean")

        # Pass through the MLP. nn.Linear layers automatically apply to the last dimension,
        # so feeding a [B, T, H] tensor yields [B, T, out_dim].
        return self.net(emb)


class PerLayerHeads(BaseDecoder):
    """
    K parallel per-layer probe heads over a layer-stacked embedding — one
    training run sweeps every encoder layer at once.

    Motivation
    ----------
    The per-layer probing protocol re-runs training once per encoder layer
    (13-24 runs per encoder x fold) even though the frozen-encoder forward is
    shared and the runs differ ONLY in which layer feeds the head. This
    decoder holds ``num_layers`` independent heads, each structurally
    IDENTICAL to the single-layer ``MLPDecoderKeepTime`` configuration, so a
    single run trains all layers simultaneously off the shared forward (or
    the shared (L, T, H) frame cache). Keeping the probe architecture uniform
    across layers is load-bearing for the comparability of layer-wise
    results: probe-head choice can reorder layer/model rankings, so it must
    be held fixed across everything being compared (Zaiem et al. 2023,
    "Speech Self-Supervised Representation Benchmarking: Are We Doing It
    Right?", Interspeech — their benchmark conclusions flip under different
    probe heads).

    Input contract
    --------------
    ``emb``: [B, L, T, H] with ``L == num_layers`` — i.e. the config's
    ``LayerSelector`` must select ALL encoder layers (``layers: ["0..L-1"]``),
    NOT a single layer. Head k consumes ``emb[:, k:k+1, :, :]`` — exactly the
    [B, 1, T, H] slice a single-layer run's decoder receives after
    ``LayerSelector(layers=[k])`` — so each head's forward math is identical
    to the corresponding single-layer run's.

    ``include_meanall=True`` appends one extra head consuming
    ``emb.mean(dim=1, keepdim=True)``. That equals what a "meanall" run's
    ``MLPDecoderKeepTime`` computes (LayerSelector over all layers + the
    decoder's internal mean over L), so the meanall baseline rides along in
    the same run too.

    ``include_weighted=True`` appends a SUPERB-style learned weighted-sum
    head: softmax gates ``w = softmax(layer_gate)`` (init 0 → uniform =
    meanall at step 0) mix the L layers, and a further identical
    ``MLPDecoderKeepTime`` head consumes the mix. Each layer is passed
    through a NON-learnable LayerNorm before mixing (Feng et al., TASLP
    2024, arXiv:2404.09385 "normalized benchmarking"): per-layer feature
    scales differ wildly with depth, so un-normalized gates confound
    scale-compensation with informativeness and cannot be read as layer
    contributions. ``layer_weights()`` exposes the softmax gates for
    per-epoch logging (``LogLayerWeightsCallback``). NOTE the per-layer
    heads still receive RAW (un-normalized) slices — their anchor
    comparability to single-layer runs is untouched; the LayerNorm applies
    only inside the weighted head's mix. The gate + weighted head are
    parameter-disjoint from every other head, so adding them leaves all
    other heads' training trajectories bitwise unchanged (tested).
    Interpretation caveats (log both, cite in thesis): learned gates
    correlate only weakly with true per-layer probe performance (Spearman
    ρ≈0.37–0.49, Feng et al.) — the per-layer heads' curves are the
    ground-truth contribution measure; the gates are a secondary panel.

    Output
    ------
    [B, K, T, out_dim] with K = num_layers (+1 if include_meanall). Head k's
    logits sit at index k along dim 1; the meanall head (when present) is
    LAST. ``self.head_names`` gives per-index names ("l0".."l{L-1}",
    "meanall") used by ``ProbeAudioTaskMultiHead`` for per-head metric keys
    and by ``PerHeadBestCheckpoint`` for per-head snapshot files.

    Because the heads are parameter-disjoint, the summed multi-head loss
    produces gradients for head k identical to training head k alone — see
    ``ProbeAudioTaskMultiHead._shared_step`` for the full update-equivalence
    invariant and ``tests/test_multihead_probe.py`` for the proof.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10,
        num_layers: int = 13,
        hidden_layers: list | None = None,  # default [512] — mirrors MLPDecoderKeepTime
        activation_fn: dict | None = None,  # e.g. {"class_path": "torch.nn.ReLU"}
        dropout: float = 0.5,
        include_meanall: bool = False,
        include_weighted: bool = False,
    ):
        super().__init__(in_dim, out_dim)
        if num_layers < 1:
            raise ValueError(f"PerLayerHeads needs num_layers >= 1, got {num_layers}")
        self.num_layers = int(num_layers)
        self.include_meanall = bool(include_meanall)
        self.include_weighted = bool(include_weighted)
        if self.include_weighted:
            # SUPERB featurizer gates (softmax applied in forward). Zeros →
            # uniform mix at init, i.e. the weighted head starts as meanall.
            self.layer_gate = nn.Parameter(torch.zeros(self.num_layers))
        # None-default dance instead of a mutable default arg; [512] mirrors
        # MLPDecoderKeepTime's default so PerLayerHeads with no overrides
        # builds the exact same head a default single-layer run would.
        hidden_layers = [512] if hidden_layers is None else list(hidden_layers)

        def _per_head_activation():
            # activation_fn may arrive as a raw config dict (the jsonargparse /
            # LightningCLI path — MLPDecoderKeepTime then instantiates its OWN
            # module from it, one per head) or as an already-instantiated
            # module (the instantiate_recursive path resolves nested
            # class_path dicts before we see them; instantiate_from_config is
            # idempotent on instances). An instance MUST be deep-copied per
            # head so parameterised activations (e.g. PReLU) are never
            # silently weight-tied across heads.
            if activation_fn is None or isinstance(activation_fn, dict):
                return activation_fn
            return copy.deepcopy(activation_fn)

        # Head k is a full MLPDecoderKeepTime — the SAME class with the SAME
        # init_args a single-layer run's decoder gets, not a re-implementation,
        # so any future change to the single-head architecture automatically
        # applies here and the architectures cannot drift apart.
        n_heads = (
            self.num_layers
            + (1 if self.include_meanall else 0)
            + (1 if self.include_weighted else 0)
        )
        self.heads = nn.ModuleList(
            MLPDecoderKeepTime(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_layers=hidden_layers,
                activation_fn=_per_head_activation(),
                dropout=dropout,
            )
            for _ in range(n_heads)
        )
        self.head_names = (
            [f"l{k}" for k in range(self.num_layers)]
            + (["meanall"] if self.include_meanall else [])
            + (["weighted"] if self.include_weighted else [])
        )

    def layer_weights(self) -> torch.Tensor:
        """Softmax-normalised gates of the weighted head (detached, CPU).

        Only meaningful with ``include_weighted=True``; raises otherwise so a
        caller can't silently log garbage."""
        if not self.include_weighted:
            raise RuntimeError("layer_weights() requires include_weighted=True")
        return torch.softmax(self.layer_gate.detach(), dim=0).cpu()

    def forward(self, emb, *_):
        """
        Args:
            emb (torch.Tensor): Input tensor of shape [B, L, T, H] with
                L == num_layers (ALL encoder layers, in order).

        Returns:
            torch.Tensor: Stacked per-head logits [B, K, T, out_dim],
                K = num_layers (+1 meanall head last, if configured).
        """
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        if emb.size(1) != self.num_layers:
            raise ValueError(
                f"PerLayerHeads was built for {self.num_layers} layers but the "
                f"embedding has L={emb.size(1)}. The config's LayerSelector must "
                f'select ALL encoder layers (layers: ["0..{self.num_layers - 1}"]), '
                f"not a subset."
            )
        # Head k sees the [B, 1, T, H] slice for layer k — MLPDecoderKeepTime
        # then mean-pools the singleton L axis away (identity: mean over one
        # element), exactly as in a LayerSelector(layers=[k]) single-layer run.
        outs = [self.heads[k](emb[:, k : k + 1, :, :]) for k in range(self.num_layers)]
        idx = self.num_layers
        if self.include_meanall:
            # Mean over L then the head's internal mean over the singleton L
            # axis == a plain MLPDecoderKeepTime's single mean over all L —
            # bitwise the same meanall computation as a dedicated meanall run.
            outs.append(self.heads[idx](emb.mean(dim=1, keepdim=True)))
            idx += 1
        if self.include_weighted:
            # Per-layer non-learnable LayerNorm BEFORE the softmax mix (Feng
            # et al. TASLP 2024) — see class docstring. Applies only to the
            # weighted head's input; per-layer heads above see raw slices.
            normed = torch.nn.functional.layer_norm(emb, (emb.size(-1),))
            w = torch.softmax(self.layer_gate, dim=0)
            mixed = (normed * w.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
            outs.append(self.heads[idx](mixed))
            idx += 1
        return torch.stack(outs, dim=1)  # [B, K, T, out_dim]


class MLPDecoder(BaseDecoder):
    """
    MLP Decoder with customizable layers, optional activation functions, and dropout.
    Supports input tensors of shape [B, L, T, H], where H is the embedding dimension.
    Uses einops for pooling operations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10,
        hidden_layers: list = [512],  # noqa: B006 — pre-existing config-facing default
        activation_fn: dict | None = None,  # e.g. {"class_path": "torch.nn.ReLU"}
        dropout: float = 0.5,
    ):
        super().__init__(in_dim, out_dim)

        layers = []
        prev_dim = in_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_fn is not None:
                activation_fn = instantiate_from_config(activation_fn)
                layers.append(activation_fn)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb, *_):
        # emb: [B, L, T, H] -> mean-pool across L and T -> [B, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        emb = reduce(emb, "b l t h -> b h", "mean")
        return self.net(emb)


class LinearDecoder(BaseDecoder):
    """
    Linear Decoder supporting input tensors of shape [B, L, T, H].
    Uses einops for pooling operations.
    """

    def __init__(self, in_dim: int, out_dim: int = 10):
        super().__init__(in_dim, out_dim)
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, emb, *_):
        # emb: [B, L, T, H] -> mean-pool across L and T -> [B, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        emb = reduce(emb, "b l t h -> b h", "mean")
        return self.net(emb)


class LSTMDecoder(BaseDecoder):
    """
    LSTM Decoder for 4D sequence data.
    Supports input tensors of shape [B, L, T, H].
    Uses einops for reshaping.
    """

    def __init__(self, in_dim: int, out_dim: int = 10, hidden_size: int = 128, num_layers: int = 2):
        super().__init__(in_dim, out_dim)
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_dim)

    def forward(self, emb, *_):
        # emb: [B, L, T, H] -> flatten to [B, L*T, H]
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"
        emb_flat = rearrange(emb, "b l t h -> b (l t) h")
        lstm_out, _ = self.lstm(emb_flat)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


class BiLSTMMLPDecoderKeepTime(BaseDecoder):
    """
    BiLSTM + MLP Decoder that collapses the 'layer' dimension (L) but preserves the 'time' dimension (T).

    Input:  emb of shape [B, L, T, H]
    Steps:
      1) Mean-pool across L -> [B, T, H]
      2) BiLSTM over T -> [B, T, D], D = lstm_hidden * (2 if bidirectional else 1)
      3) Per-time-step MLP -> [B, T, out_dim]

    Optional:
      - Provide 'lengths' (LongTensor[B]) to pack/pad variable-length sequences.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 10,
        # LSTM settings
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        lstm_dropout: float = 0.0,  # only active if lstm_layers > 1
        # MLP head settings
        mlp_hidden_layers: list[int] = [512],  # noqa: B006 — pre-existing config-facing default
        activation_fn: dict
        | None = None,  # e.g. {"class_path": "torch.nn.ReLU", "init_args": {"inplace": True}}
        mlp_dropout: float = 0.5,
    ):
        super().__init__(in_dim, out_dim)

        self.bidirectional = bidirectional

        # LSTM expects input size = H (after pooling L)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,  # [B, T, H]
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Build MLP head applied to last dim (per time step)
        layers = []
        prev_dim = lstm_out_dim
        for hidden_dim in mlp_hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation_fn is not None:
                act = instantiate_from_config(activation_fn)
                layers.append(act)
            if mlp_dropout > 0.0:
                layers.append(nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        emb: torch.Tensor,
        lengths: torch.Tensor | None = None,
        *_,
    ) -> torch.Tensor:
        """
        Args:
            emb: Tensor [B, L, T, H]
            lengths: Optional LongTensor [B], true lengths along T (before any padding)

        Returns:
            Tensor [B, T, out_dim]
        """
        assert emb.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {emb.dim()}D tensor"

        # 1) pool over L -> [B, T, H]
        x = reduce(emb, "b l t h -> b t h", "mean")

        # 2) BiLSTM over time (support var-length with pack/pad)
        if lengths is not None:
            # Ensure lengths on CPU for packing; enforce descending not required with enforce_sorted=False
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            x_lstm, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T_max, D]
            # Note: pad_packed_sequence pads to max length in batch; positions beyond each length are padded with 0
        else:
            x_lstm, _ = self.lstm(x)  # [B, T, D]

        # 3) MLP per time step -> [B, T, out_dim]
        out = self.mlp(x_lstm)

        # If lengths given, out already zero-padded beyond valid timesteps
        return out


# Attempt to import Flash Attention implementation; fallback to native PyTorch scaled_dot_product_attention
try:
    from flash_attn.modules.mha import FlashMHA
except ImportError:
    FlashMHA = None

if FlashMHA is None:

    class FlashMHA(nn.Module):
        """
        Fallback multi-head attention using PyTorch's scaled_dot_product_attention.
        """

        def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            assert self.head_dim * num_heads == embed_dim, (
                "embed_dim must be divisible by num_heads"
            )
            self.dropout = dropout
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, query, key, value, attn_mask=None):
            B, Tq, D = query.shape
            _, Tk, _ = key.shape
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            q = rearrange(q, "b t (h d) -> b h t d", h=self.num_heads)
            k = rearrange(k, "b t (h d) -> b h t d", h=self.num_heads)
            v = rearrange(v, "b t (h d) -> b h t d", h=self.num_heads)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p=self.dropout, is_causal=False
            )
            out = rearrange(out, "b h t d -> b t (h d)")
            return self.out_proj(out)


class FlashTransformerDecoderLayer(nn.Module):
    """
    Single layer of Transformer decoder using Flash Attention (or fallback), supports 4D sequence inputs.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = FlashMHA(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = FlashMHA(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x,  # [B, seq_len, H]
        memory,  # [B, mem_seq_len, H]
        tgt_mask=None,
        memory_mask=None,
    ):
        residual = x
        x2 = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = residual + self.dropout1(x2)
        x = self.norm1(x)
        residual = x
        x2 = self.cross_attn(x, memory, memory, attn_mask=memory_mask)
        x = residual + self.dropout2(x2)
        x = self.norm2(x)
        residual = x
        x2 = self.linear2(self.dropout3(F.relu(self.linear1(x))))
        x = residual + self.dropout3(x2)
        x = self.norm3(x)
        return x


class TransformerDecoder(BaseDecoder):
    """
    Transformer Decoder with Flash Attention (or fallback), supports input tensors of shape [B, L, T, H].
    Utilizes einops for reshaping operations.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_hidden_dim: int = 2048,
        max_seq_len: int = 500,
        dropout: float = 0.1,
    ):
        super().__init__(in_dim, out_dim)
        self.embed_dim = in_dim
        self.pos_emb = nn.Embedding(max_seq_len, in_dim)
        self.layers = nn.ModuleList(
            [
                FlashTransformerDecoderLayer(
                    embed_dim=in_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        tgt: torch.Tensor,  # [B, L, T, H]
        memory: torch.Tensor,  # [B, Lm, Tm, H]
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ):
        assert tgt.dim() == 4, f"Expected 4D tensor [B, L, T, H], got {tgt.dim()}D tensor"
        assert memory.dim() == 4, f"Expected 4D tensor [B, Lm, Tm, H], got {memory.dim()}D tensor"
        B, L, T, H = tgt.shape
        # Flatten target and memory sequences
        seq_len = L * T
        tgt_flat = rearrange(tgt, "b l t h -> b (l t) h")
        pos_ids = torch.arange(seq_len, device=tgt.device).unsqueeze(0).expand(B, seq_len)
        x = tgt_flat + self.pos_emb(pos_ids)
        memory_flat = rearrange(memory, "b lm tm h -> b (lm tm) h")
        for layer in self.layers:
            x = layer(x, memory_flat, tgt_mask=tgt_mask, memory_mask=memory_mask)
        logits_flat = self.fc_out(x)
        # Reshape back to [B, L, T, out_dim]
        return rearrange(logits_flat, "b (l t) d -> b l t d", l=L, t=T)
