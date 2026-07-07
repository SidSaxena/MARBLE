# File: marble/tasks/HookTheoryMelody/probe.py
"""
HookTheory melody-transcription probe.

Task
----
Per-frame multiclass classification of MIDI pitch.  Each clip is decoded
into a sequence of frame-level pitch labels at ``label_freq`` Hz.

Label space
-----------
* Valid frames carry the MIDI pitch number (0–127) for the active note.
* Silent / unvoiced frames carry the sentinel ``-1`` (NOT predicted; loss
  and accuracy mask them out).

Choosing the number of output classes
-------------------------------------
Empirically the HookTheory melodies sit in MIDI ~36–95 (C2–B6), but we
output 128 classes so the model can address any MIDI value without
out-of-range indexing.  Unused indices simply receive no gradient.

Architecture
------------
    encoder → LayerSelector → MLPDecoderKeepTime (in_dim → 128)
    (NO TimeAvgPool — frame-level task preserves the time dimension)

Loss & metrics
--------------
* ``MelodyCrossEntropyLoss``  : masked CE, ``ignore_index = -1`` (silent frames)
* ``RawPitchAccuracy``        : top-1 accuracy on labelled (non-silent) frames
* ``RawChromaAccuracy``       : same, but predictions and targets are taken
  modulo 12 so octave errors are forgiven.  Standard MIR melody metric.

Both metrics tolerate small (≤5 frame) time-axis mismatches between
encoder output and labels, because the encoder's token rate and the
configured ``label_freq`` rarely line up to the sample.
"""

import torch
import torch.nn as nn
from torchmetrics import Metric, MetricCollection

from marble.core.base_task import BaseTask, _unpack_batch

# ──────────────────────────────────────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────────────────────────────────────


class ProbeAudioTask(BaseTask):
    """Frame-level melody pitch-classification probe.

    Inherits training_step / validation_step from BaseTask (which already
    handle the 3-tuple or 4-tuple batch shape via _unpack_batch and pass
    clip_ids through to forward for cache lookups). Overrides test_step
    only to skip the loss computation; the frame-level metrics use the
    raw logits directly.
    """

    def test_step(self, batch, batch_idx: int):
        # Use the same unpack helper the parent class uses so this works
        # whether the datamodule emits the legacy 3-tuple
        # (waveform, labels, path) or the new 4-tuple
        # (waveform, labels, path, clip_id). clip_ids are forwarded to
        # `self(...)` so the embedding cache (when enabled with
        # cache_pool_time=False) can short-circuit the encoder pass.
        from marble.core.base_task import _unpack_batch

        x, y, _paths, clip_ids = _unpack_batch(batch)
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(logits, y)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


class ProbeAudioTaskMultiHead(BaseTask):
    """Multi-head parallel layer probe: ONE training run trains a per-layer
    head for EVERY encoder layer simultaneously off the shared frozen-encoder
    forward, replacing the 13-24 per-layer runs of a conventional sweep.

    Wiring (all pieces additive — the single-head path is untouched):

    * decoder   — exactly one :class:`marble.modules.decoders.PerLayerHeads`
      (K structurally-identical ``MLPDecoderKeepTime`` heads, optional
      meanall head). Forward returns stacked logits ``(B, K, T, C)``.
    * transforms — ``LayerSelector`` must select ALL layers
      (``layers: ["0..L-1"]``) so the embedding reaches the decoder as
      ``(B, L, T, H)``. Fully compatible with the frame-level ``(L, T, H)``
      embedding cache (``cache_embeddings=true, cache_pool_time=false``):
      the cached tuple feeds all heads, no layer is dropped anywhere.
    * loss      — exactly one loss fn (e.g. ``MelodyCrossEntropyLoss``),
      applied per head against the SHARED labels and summed (see the
      update-equivalence invariant in ``_shared_step``).
    * metrics   — the config declares each metric ONCE (same YAML shape as
      the single-head configs); this class clones the collection per head
      and logs ``{split}/{name}_l{k}`` / ``{split}/{name}_meanall``, plus
      the aggregate ``{split}/{primary_metric}_best`` (max over heads) as
      the checkpoint/scheduler monitor.
    * per-head best weights — pair with
      :class:`marble.modules.callbacks.PerHeadBestCheckpoint`, which
      snapshots each head at ITS OWN best val epoch and restores all heads
      at test start (reproducing the single-head "test the best.ckpt"
      protocol per layer).

    Deliberate protocol deviations vs. independent single-head runs (full
    discussion in ``docs/multihead_probe_validation.md``): fixed
    ``max_epochs`` with no per-layer early stopping (harmless — each head is
    restored to its own best epoch at test time), a shared LR schedule
    (``ReduceLROnPlateau`` keyed to ``val/{primary}_best`` couples LR drops
    across heads), one wandb run logging all layers (summary scripts read
    ``test/{metric}_l{k}`` keys instead of per-run summaries), and per-head
    init/dropout RNG draws that differ from a solo run's (seed-level noise,
    not bias).

    Generic beyond melody: nothing here is task-specific — any frame-level
    probe whose loss/metrics consume ``(B, T, C)`` logits can adopt it
    (HookTheoryMelody itself is the intended second user; MedleyDBMelody
    re-exports this class as its single source of truth).
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        emb_transforms: list[nn.Module] | None = None,
        decoders: list[nn.Module] | None = None,
        losses: list[nn.Module] | None = None,
        metrics: dict[str, dict[str, nn.Module]] | None = None,
        sample_rate: int | None = None,
        use_ema: bool = False,
        cache_embeddings: bool = False,
        cache_pool_time: bool = True,
        strip_frozen_encoder_from_ckpt: bool = True,
        primary_metric: str = "acc_rpa",
        **kwargs,
    ):
        # metrics=None to the base: BaseTask would build ONE collection per
        # split and _shared_step would feed it the stacked (B, K, T, C)
        # logits — meaningless. We intercept the config and build K clones
        # per split below instead.
        super().__init__(
            encoder=encoder,
            emb_transforms=emb_transforms,
            decoders=decoders,
            losses=losses,
            metrics=None,
            sample_rate=sample_rate,
            use_ema=use_ema,
            cache_embeddings=cache_embeddings,
            cache_pool_time=cache_pool_time,
            strip_frozen_encoder_from_ckpt=strip_frozen_encoder_from_ckpt,
            **kwargs,
        )

        # ── validate the multi-head wiring up front (config errors here are
        # much cheaper than a silently-wrong 40-epoch run) ────────────────────
        if len(self.decoders) != 1:
            raise ValueError(
                f"ProbeAudioTaskMultiHead expects exactly ONE decoder (a "
                f"PerLayerHeads); got {len(self.decoders)}."
            )
        dec = self.decoders[0]
        if not (hasattr(dec, "heads") and hasattr(dec, "head_names")):
            raise ValueError(
                f"ProbeAudioTaskMultiHead's decoder must expose .heads and "
                f".head_names (see marble.modules.decoders.PerLayerHeads); got "
                f"{type(dec).__name__}."
            )
        if len(self.loss_fns) != 1:
            raise ValueError(
                f"ProbeAudioTaskMultiHead expects exactly ONE loss fn (applied "
                f"per head and summed); got {len(self.loss_fns)}."
            )
        self.head_names: list[str] = list(dec.head_names)
        self.primary_metric = str(primary_metric)

        # ── per-head metric collections ──────────────────────────────────────
        # One clone of the configured collection per head per split — K cheap
        # accumulators (RPA/RCA are two long counters each). clone() is a deep
        # copy, so head states are fully independent; prefix/postfix bake the
        # final key: e.g. prefix "val/" + name "acc_rpa" + postfix "_l3"
        # → "val/acc_rpa_l3".
        if metrics:
            for split in ("train", "val", "test"):
                split_cfg = metrics.get(split)
                if not split_cfg:
                    continue
                if self.primary_metric not in split_cfg:
                    raise ValueError(
                        f"primary_metric {self.primary_metric!r} missing from the "
                        f"{split!r} metrics config (keys: {sorted(split_cfg)}). The "
                        f"{{split}}/{self.primary_metric}_best aggregate (checkpoint "
                        f"monitor) needs it in every configured split."
                    )
                # compute_groups=False is REQUIRED for correctness here, not an
                # optimisation toggle. torchmetrics' default (True) groups metrics
                # whose accumulator states coincide on the FIRST update batch and
                # then feeds only the group leader, aliasing the others to its
                # value. RawPitchAccuracy and RawChromaAccuracy share an identical
                # (correct, total) state, so any head whose first batch happens to
                # have zero right-chroma-wrong-octave frames gets its RPA and RCA
                # silently collapsed to one number (the chroma value) for the whole
                # epoch — stochastic per head, and it corrupted the HTM test report
                # for a subset of heads before this was caught. Keep grouping OFF.
                base_mc = MetricCollection(dict(split_cfg), compute_groups=False)
                heads_mc = nn.ModuleList(
                    base_mc.clone(prefix=f"{split}/", postfix=f"_{hn}") for hn in self.head_names
                )
                setattr(self, f"{split}_head_metrics", heads_mc)

    # ── train / val ──────────────────────────────────────────────────────────

    def _shared_step(self, batch, batch_idx: int, split: str) -> torch.Tensor:
        """Multi-head analogue of BaseTask._shared_step: same unpack/log
        structure, but the loss is a per-head sum and the metrics are K
        per-head collections."""
        x, y, _paths, clip_ids = _unpack_batch(batch)
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        # Single PerLayerHeads decoder → BaseTask.forward returns its stacked
        # (B, K, T, C) tensor directly (not a list).
        assert isinstance(logits, torch.Tensor) and logits.dim() == 4, (
            f"expected stacked (B, K, T, C) logits from PerLayerHeads, got {type(logits).__name__}"
        )
        assert logits.size(1) == len(self.head_names), (
            f"head-count mismatch: logits K={logits.size(1)} vs {len(self.head_names)} head names"
        )
        bs = x.size(0)
        loss_fn = self.loss_fns[0]

        # ── multi-head loss: SUM of per-head losses on the SHARED labels ────
        # Update-equivalence invariant: the K heads are PARAMETER-DISJOINT
        # (the encoder is frozen; no module is shared across heads), so
        # ∂(Σ_k loss_k)/∂θ_k = ∂loss_k/∂θ_k, and because Adam's state
        # (m, v, step) is per-parameter, the summed loss yields exactly the
        # parameter updates K independent single-head runs would produce
        # UNDER A COMMON, HEAD-INDEPENDENT LR SEQUENCE — proven bitwise
        # (constant LR, dropout=0) in
        # tests/test_multihead_probe.py::test_update_equivalence.
        # SCOPE (adversarial-review finding, 2026-07-06): the invariant
        # breaks under any cross-head coupling — global gradient clipping
        # (norm over ALL heads' grads; no probe config sets
        # gradient_clip_val) or an LR schedule driven by a shared quantity.
        # The shipped multihead config DOES use ReduceLROnPlateau on
        # val/acc_rpa_best (the best head's metric), so every head follows
        # ONE common LR schedule instead of its own adaptive one: an
        # identical-protocol-across-layers deviation from per-run schedules.
        # Multi-head results are therefore a VALIDATED APPROXIMATION of
        # independent runs (fold-0 anchor comparison in
        # docs/multihead_probe_validation.md), not a bitwise reproduction —
        # phrase them as such anywhere thesis-facing.
        # (No 1/K averaging: scaling the sum would scale every head's
        # gradient and change the effective LR vs the single-head runs.)
        head_losses = [loss_fn(logits[:, k], y) for k in range(logits.size(1))]
        loss = sum(head_losses)
        self.log(
            f"{split}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )

        # Per-head metrics: log the Metric OBJECTS so Lightning computes each
        # ONCE at epoch end from global accumulated state — identical
        # rationale to BaseTask._shared_step (see the comment there).
        # prog_bar off: K heads x M metrics would swamp the bar; the
        # {split}/{primary}_best aggregate (epoch end) is the summary.
        heads_mc = getattr(self, f"{split}_head_metrics", None)
        if heads_mc is not None:
            for k, mc in enumerate(heads_mc):
                mc.update(logits[:, k], y)
                self.log_dict(
                    mc,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bs,
                )
        return loss

    # ── epoch-end aggregates ─────────────────────────────────────────────────

    def _head_primary_values(self, split: str) -> list[torch.Tensor] | None:
        """Compute the primary metric of every head for ``split``, in head
        order. Safe to call from ``on_{validation,test}_epoch_end``: Lightning
        invokes those BEFORE the logger connector computes/resets the
        object-logged metrics, so state is intact — and torchmetrics caches
        ``compute()`` results, so Lightning's own epoch-end compute reuses
        this one instead of re-syncing."""
        heads_mc = getattr(self, f"{split}_head_metrics", None)
        if heads_mc is None:
            return None
        values = []
        for name, mc in zip(self.head_names, heads_mc, strict=True):
            out = mc.compute()  # keys carry prefix/postfix, e.g. "val/acc_rpa_l3"
            values.append(out[f"{split}/{self.primary_metric}_{name}"])
        return values

    def _log_head_best(self, split: str) -> None:
        """Log ``{split}/{primary}_best`` (max over heads — the checkpoint /
        LR-scheduler monitor) and ``..._best_head`` (argmax head index; the
        meanall head, when present, is the last index)."""
        values = self._head_primary_values(split)
        if not values:
            return
        stacked = torch.stack([v.float() for v in values])
        best_idx = int(torch.argmax(stacked).item())
        # sync_dist=False: torchmetrics compute() already dist-synced the
        # accumulator states, so every rank logs the identical global value.
        self.log(f"{split}/{self.primary_metric}_best", stacked[best_idx], prog_bar=True)
        self.log(f"{split}/{self.primary_metric}_best_head", float(best_idx))

    def on_validation_epoch_end(self) -> None:
        self._log_head_best("val")

    def on_test_epoch_end(self) -> None:
        self._log_head_best("test")

    # ── test ─────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx: int):
        """Multi-head analogue of ProbeAudioTask.test_step — skips the loss,
        updates the per-head test collections.

        One deliberate difference from the single-head test path: that path
        logs ``mc(logits, y)`` per batch (Lightning then batch-size-weight-
        averages the per-batch values), while here the metric OBJECTS are
        logged, so the epoch value is the exact corpus-level ratio
        Σcorrect/Σtotal. For RPA/RCA the two differ at the ~1e-3 level (well
        inside the ±0.01 run-noise band used for validation) and the object
        form is the more correct aggregation — see
        docs/multihead_probe_validation.md.
        """
        x, y, _paths, clip_ids = _unpack_batch(batch)
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        heads_mc = getattr(self, "test_head_metrics", None)
        if heads_mc is not None:
            for k, mc in enumerate(heads_mc):
                mc.update(logits[:, k], y)
                self.log_dict(
                    mc,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=x.size(0),
                )


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _crop_to_min_t(logits: torch.Tensor, targets: torch.Tensor, tol: int):
    """Align logits/targets along the time axis (within ``tol`` frames)."""
    B, T_l, C = logits.shape
    B_t, T_t = targets.shape
    if B_t != B:
        raise ValueError(f"Batch mismatch: logits B={B} targets B={B_t}")
    diff = abs(T_l - T_t)
    if diff > tol:
        raise ValueError(f"Time dim mismatch too large: |{T_l} - {T_t}| = {diff} > tol ({tol})")
    T_min = min(T_l, T_t)
    if T_l != T_min:
        logits = logits[:, :T_min, :]
    if T_t != T_min:
        targets = targets[:, :T_min]
    return logits, targets


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────


class MelodyCrossEntropyLoss(nn.Module):
    """Masked cross-entropy for frame-level pitch labels.

    Mirrors ``ChordCrossEntropyLoss`` from the Chords1217 probe: crops the
    time axis to the common minimum (within ``time_dim_mismatch_tol``
    frames) and ignores any target equal to ``ignore_index`` (default -1).
    """

    def __init__(self, time_dim_mismatch_tol: int = 5, ignore_index: int = -1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.time_dim_mismatch_tol = time_dim_mismatch_tol
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = _crop_to_min_t(logits, targets, self.time_dim_mismatch_tol)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        valid_mask = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            # All frames in the batch are silent (target == ignore_index).
            # Common in melody data: a 15-s clip drawn from an instrumental
            # section of the song has no labelled frames at all.
            #
            # We cannot ``return torch.tensor(0.0, device=logits.device)``
            # here — that creates a fresh leaf tensor with no grad_fn,
            # detached from the autograd graph, and ``loss.backward()``
            # crashes with "element 0 of tensors does not require grad".
            # Multiplying ``logits`` by zero produces the same numerical
            # value (0) BUT keeps the graph wired up so backward succeeds
            # with zero gradient — effectively a no-op optimizer step for
            # this batch.
            return logits.sum() * 0.0
        return self.ce(flat_logits[valid_mask], flat_targets[valid_mask])


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


class RawPitchAccuracy(Metric):
    """Frame-level top-1 pitch accuracy on labelled frames.

    Equivalent to the Raw Pitch Accuracy (RPA) used in melody-extraction
    literature when the underlying voicing is taken as ground truth.
    Ignored positions (``targets == ignore_index``) do not contribute to
    either the correct or total counts.
    """

    def __init__(
        self,
        time_dim_mismatch_tol: int = 5,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.time_dim_mismatch_tol = time_dim_mismatch_tol
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits, targets = _crop_to_min_t(logits, targets, self.time_dim_mismatch_tol)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        valid_mask = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            return
        preds = torch.argmax(flat_logits[valid_mask], dim=-1)
        self.correct += (preds == flat_targets[valid_mask]).sum()
        self.total += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()


class RawChromaAccuracy(Metric):
    """Frame-level chroma accuracy on labelled frames.

    Computed modulo 12 — predictions that hit the correct pitch class
    (regardless of octave) are counted as correct.  This is the standard
    Raw Chroma Accuracy (RCA) from mir_eval melody metrics.
    """

    def __init__(
        self,
        time_dim_mismatch_tol: int = 5,
        ignore_index: int = -1,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.time_dim_mismatch_tol = time_dim_mismatch_tol
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits, targets = _crop_to_min_t(logits, targets, self.time_dim_mismatch_tol)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        valid_mask = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            return
        preds = torch.argmax(flat_logits[valid_mask], dim=-1)
        # Chroma equivalence: pitch class only.
        self.correct += ((preds % 12) == (flat_targets[valid_mask] % 12)).sum()
        self.total += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()
