# marble/tasks/GTZANBeatTracking/metrics.py
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
import mir_eval

from marble.utils.utils import times_to_mask, mask_to_times


class TimeEventFMeasure(torchmetrics.Metric):
    """
    A Time‐Event F‐Measure for event‐based tasks (beat/downbeat).

    - Binarizes any nonzero entries in est_mask/ref_mask (threshold = 0.5 by default).
    - Sorts, deduplicates, filters negative times, then calls mir_eval.beat.validate.
    - If validation fails, skip matching but still count events.
    - If both predictions and references are empty, returns F1=1.0.
    """

    def __init__(
        self,
        label_freq: int,
        tol: float = 0.07,
        threshold: float = 0.5,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if not isinstance(label_freq, int) or label_freq <= 0:
            raise ValueError(f"label_freq must be a positive integer, got {label_freq}")
        self.label_freq = label_freq
        self.tol = tol
        self.threshold = threshold

        # States: matched count, estimated count, reference count
        self.add_state("matching", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("estimate", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("reference", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, est_mask: torch.Tensor, ref_mask: torch.Tensor):
        """
        Update counts for one batch of estimated/reference masks.

        Args:
            est_mask: (B, T) tensor. Any nonzero entry is a predicted event.
            ref_mask: (B, T) tensor. Any nonzero entry is a reference event.
        """
        # 1) Shape check
        if est_mask.ndim != 2 or ref_mask.ndim != 2:
            raise ValueError(f"est_mask and ref_mask must be 2D, got {est_mask.shape}, {ref_mask.shape}")

        est_np = (est_mask.detach().cpu().float().numpy() > self.threshold).astype(np.int32)
        ref_np = (ref_mask.detach().cpu().float().numpy() > self.threshold).astype(np.int32)

        batch_size, T_est = est_np.shape
        batch_size_ref, T_ref = ref_np.shape
        if batch_size != batch_size_ref:
            raise ValueError(f"est_mask and ref_mask batch dim must match, got {est_np.shape} vs {ref_np.shape}")
        if T_est != T_ref:
            # tolerate different lengths
            assert abs(T_est - T_ref) <= 1, \
                f"est_mask and ref_mask must have same time dimension, got {T_est} vs {T_ref}"

        for b in range(batch_size):
            # 2) Convert masks → times via our utility
            ref_times = mask_to_times(ref_np[b], fps=self.label_freq)  # np.ndarray of times
            est_times = mask_to_times(est_np[b], fps=self.label_freq)

            # 3) Validate & match events
            valid_for_matching = True
            if (ref_times.size > 0) and (est_times.size > 0):
                try:
                    mir_eval.beat.validate(ref_times, est_times)
                except AssertionError:
                    warnings.warn(
                        f"Sample {b}: ref_times or est_times failed mir_eval.beat.validate. "
                        "Skipping event matching but still counting events."
                    )
                    valid_for_matching = False

            if valid_for_matching and (ref_times.size > 0) and (est_times.size > 0):
                pairs = mir_eval.util.match_events(ref_times, est_times, self.tol)
                n_match = len(pairs)
            else:
                n_match = 0

            # 4) Accumulate counts
            self.matching += int(n_match)
            self.estimate += int(est_times.size)
            self.reference += int(ref_times.size)

    def compute(self):
        """
        Compute final F1:

        - If no predictions and no references: return 1.0
        - If one side is empty and the other isn’t: return 0.0
        - Else compute precision/recall via match counts, then f1.
        """
        est = int(self.estimate)
        ref = int(self.reference)
        match = int(self.matching)

        if est == 0 and ref == 0:
            return torch.tensor(1.0, dtype=torch.float32)
        if est == 0 or ref == 0:
            return torch.tensor(0.0, dtype=torch.float32)

        prec = float(match) / float(est)
        rec = float(match) / float(ref)
        if prec == 0.0 and rec == 0.0:
            return torch.tensor(0.0, dtype=torch.float32)

        f1_value = mir_eval.util.f_measure(prec, rec)
        return torch.tensor(f1_value, dtype=torch.float32)


class TempoMAE(torchmetrics.Metric):
    """
    Compute Mean Absolute Error for tempo estimation.
    Expects:
        est_tempo: (B,) predicted tempo values (in BPM)
        ref_tempo: (B,) ground‐truth tempo values (in BPM)
    """
    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("abs_error_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, est_tempo: torch.Tensor, ref_tempo: torch.Tensor):
        """
        est_tempo: (B,) float tensor of predicted BPM
        ref_tempo: (B,) float tensor of ground truth BPM
        """
        if est_tempo.ndim != 1 or ref_tempo.ndim != 1:
            raise ValueError(f"est_tempo and ref_tempo must be 1D, got {est_tempo.shape}, {ref_tempo.shape}")

        diff = torch.abs(est_tempo.detach() - ref_tempo.detach())
        self.abs_error_sum += diff.sum()
        self.count += diff.numel()

    def compute(self):
        if int(self.count) == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        return (self.abs_error_sum / self.count).to(dtype=torch.float32)


class TempoAccuracy(torchmetrics.Metric):
    """
    Computes Acc1 and an extended Acc2 for tempo estimation.

    - Acc1:   |pred - ref| / ref <= tol
    - Acc2-ext: Any of:
            1) Exact:         |pred - ref| / ref <= tol
            2) Double‐time:   |2*pred - ref| / ref <= tol
            3) Half‐time:     |(pred / 2) - ref| / ref <= tol
            4) Triple‐time:   |3*pred - ref| / ref <= tol
            5) One‐third:     |(pred / 3) - ref| / ref <= tol

    States:
        correct1 (LongTensor): number of predictions satisfying Acc1
        correct2 (LongTensor): number of predictions satisfying any of the above Acc2‐ext conditions
        total    (LongTensor): total number of predictions seen

    Args:
        tol (float): relative‐error threshold, default 0.04 (±4%).
        dist_sync_on_step (bool): if True, synchronize states across processes at each step.
    """
    def __init__(self, tol: float = 0.04, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.tol = tol

        self.add_state("correct1", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("correct2", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total",    default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, pred_tempo: torch.Tensor, ref_tempo: torch.Tensor):
        """
        Update counts with a batch of predictions and references.

        Args:
            pred_tempo (Tensor): predicted BPMs, shape (B,) or (B, 1), float.
            ref_tempo  (Tensor): reference/true BPMs, shape (B,) or (B, 1), float.
        """
        # Flatten to 1D
        pred = pred_tempo.view(-1).float()
        ref  = ref_tempo.view(-1).float()
        if pred.shape != ref.shape:
            raise ValueError(f"pred_tempo and ref_tempo must have same shape, got {pred.shape} vs {ref.shape}")

        batch_size = pred.shape[0]
        eps = 1e-8  # avoid division by zero

        # Exact‐match relative error:
        rel_err = torch.abs(pred - ref) / (ref + eps)
        mask_exact = rel_err <= self.tol

        # Double‐time relative error: |2*pred - ref| / ref
        rel_err_dbl = torch.abs(pred * 2.0 - ref) / (ref + eps)
        mask_dbl = rel_err_dbl <= self.tol

        # Half‐time relative error: |(pred / 2) - ref| / ref
        rel_err_half = torch.abs(pred / 2.0 - ref) / (ref + eps)
        mask_half = rel_err_half <= self.tol

        # Triple‐time relative error: |3*pred - ref| / ref
        rel_err_triple = torch.abs(pred * 3.0 - ref) / (ref + eps)
        mask_triple = rel_err_triple <= self.tol

        # One‐third‐time relative error: |(pred / 3) - ref| / ref
        rel_err_third = torch.abs(pred / 3.0 - ref) / (ref + eps)
        mask_third = rel_err_third <= self.tol

        # Acc1 = exactly within tol
        mask1 = mask_exact

        # Acc2-ext = any of (exact, dbl, half, triple, one‐third)
        mask2 = mask_exact | mask_dbl | mask_half | mask_triple | mask_third

        self.correct1 += mask1.sum().to(torch.long)
        self.correct2 += mask2.sum().to(torch.long)
        self.total    += batch_size

    def compute(self):
        """
        Compute final Acc1 and Acc2-ext as floats in [0, 1].

        Returns:
            dict with keys "acc1" and "acc2" (extended).
        """
        total = int(self.total)
        if total == 0:
            return {"acc1": torch.tensor(0.0), "acc2": torch.tensor(0.0)}

        acc1 = self.correct1.float() / total
        acc2 = self.correct2.float() / total
        return {"acc1": acc1, "acc2": acc2}
