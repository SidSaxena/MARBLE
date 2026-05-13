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

from marble.core.base_task import BaseTask


# ──────────────────────────────────────────────────────────────────────────────
# Task
# ──────────────────────────────────────────────────────────────────────────────

class ProbeAudioTask(BaseTask):
    """Frame-level melody pitch-classification probe."""

    def test_step(self, batch, batch_idx: int):
        x, y, paths = batch
        logits = self(x)
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(logits, y)
            self.log_dict(metrics_out, prog_bar=True, on_step=False,
                          on_epoch=True, sync_dist=True)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _crop_to_min_t(logits: torch.Tensor, targets: torch.Tensor, tol: int):
    """Align logits/targets along the time axis (within ``tol`` frames)."""
    B, T_l, C = logits.shape
    B_t, T_t = targets.shape
    if B != B_t:
        raise ValueError(f"Batch mismatch: logits B={B} targets B={B_t}")
    diff = abs(T_l - T_t)
    if diff > tol:
        raise ValueError(
            f"Time dim mismatch too large: |{T_l} - {T_t}| = {diff} > tol ({tol})"
        )
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
        flat_logits  = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        valid_mask   = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
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
        self.add_state("total",   default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits, targets = _crop_to_min_t(logits, targets, self.time_dim_mismatch_tol)
        flat_logits  = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        valid_mask   = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            return
        preds = torch.argmax(flat_logits[valid_mask], dim=-1)
        self.correct += (preds == flat_targets[valid_mask]).sum()
        self.total   += valid_mask.sum()

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
        self.add_state("total",   default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        logits, targets = _crop_to_min_t(logits, targets, self.time_dim_mismatch_tol)
        flat_logits  = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
        valid_mask   = flat_targets != self.ignore_index
        if valid_mask.sum() == 0:
            return
        preds = torch.argmax(flat_logits[valid_mask], dim=-1)
        # Chroma equivalence: pitch class only.
        self.correct += ((preds % 12) == (flat_targets[valid_mask] % 12)).sum()
        self.total   += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.tensor(0.0, device=self.correct.device)
        return self.correct.float() / self.total.float()
