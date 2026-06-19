# marble/tasks/VGMLoopFrames/probe.py
#
# Frame-level VGM loop-structure probes.
#
# The datamodule emits targets as a dict:
#   {"boundary": (L,) float32,  "function": (L,) int64}
#
# BaseTask._shared_step passes ``(logits, y)`` directly to each loss_fn,
# which breaks for dict-valued y.  ProbeAudioTask here overrides _shared_step
# to extract the relevant key before computing loss and metrics.
#
# Two variants are registered:
#   VGMLoopBoundaryProbe  — target key "boundary", expects (B, L, 1) logits
#   VGMLoopFunctionProbe  — target key "function", expects (B, L, 3) logits
#
# Both inherit from a shared _VGMLoopFramesProbeBase that handles dict-y
# unpacking.  The actual class imported by the YAML as ProbeAudioTask is
# also aliased so configs can use
#   class_path: marble.tasks.VGMLoopFrames.probe.ProbeAudioTask
# with a ``target_key`` init_arg to select the head.
#
# Spec frame-level metrics (boundary-F1 @ ±0.5/±3 s, seam recall, function
# pairwise-F) are computed on the TEST split only, per-clip, and logged
# alongside the torchmetrics configured in YAML.  They are ported from
# msa_compare.vgm.eval and live in frame_metrics.py.  The whole block is wrapped
# in try/except so a metric hiccup can never break a run.

import logging

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask, _unpack_batch
from marble.core.utils import instantiate_from_config

from . import frame_metrics as fm

logger = logging.getLogger(__name__)


class _VGMLoopFramesProbeBase(BaseTask):
    """
    Shared base for frame-level VGM loop probes.

    Adds ``target_key`` to select which sub-tensor of the targets dict to
    use for loss and metrics.  All other behaviour (encoder, transforms,
    decoders, EMA, logging) is inherited from BaseTask.

    Parameters
    ----------
    target_key : str
        Key to extract from the per-batch ``targets`` dict.
        ``"boundary"`` → float32 heatmap; ``"function"`` → int64 class map.
    label_freq : int
        Frame rate (Hz) of the target tensors — MUST match the datamodule's
        ``label_freq`` (default 25).  Used to convert frame indices to seconds
        for the seconds-based spec metrics on the test split.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        metrics: dict[str, dict[str, dict]],
        target_key: str,
        label_freq: int = 25,
        cache_embeddings: bool = False,
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        metric_maps = {
            split: {
                name: instantiate_from_config(cfg) for name, cfg in metrics.get(split, {}).items()
            }
            for split in ("train", "val", "test")
        }

        self.target_key = target_key
        self.label_freq = int(label_freq)

        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics=metric_maps,
            sample_rate=sample_rate,
            use_ema=use_ema,
            cache_embeddings=cache_embeddings,
        )

    # ── override _shared_step to handle dict-valued y ─────────────────────────

    def _shared_step(self, batch, batch_idx: int, split: str) -> torch.Tensor:
        x, targets, _paths, clip_ids = _unpack_batch(batch)

        # Extract the scalar/1D target that this head cares about
        if isinstance(targets, dict):
            y = targets[self.target_key]
        else:
            y = targets

        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)

        # logits from MLPDecoderKeepTime: (B, L, out_dim)
        # For boundary (out_dim=1): squeeze to (B, L); for function keep (B, L, 3)
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)  # (B, L)

        bs = x.size(0)

        # CrossEntropyLoss requires the class dim at index 1: (N, C, ...).
        # When logits is (B, L, C), flatten to (B*L, C) and y to (B*L,) before
        # the CE call.  The boundary BCE path keeps (B, L)/(B, L) as-is.
        if logits.dim() == 3:
            B, L, C = logits.shape
            loss_logits = logits.reshape(B * L, C)
            loss_y = y.reshape(B * L)
        else:
            loss_logits = logits
            loss_y = y

        losses = [fn(loss_logits, loss_y) for fn in self.loss_fns]
        loss = sum(losses)
        self.log(
            f"{split}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs,
        )

        mc: MetricCollection = getattr(self, f"{split}_metrics", None)
        if mc is not None:
            # For CrossEntropyLoss the metric expects (B*L, C) preds and (B*L,) labels.
            # For BCEWithLogitsLoss the metric expects (B*L,) preds and (B*L,) labels.
            if logits.dim() == 3:
                # (B, L, C) → (B*L, C) and (B, L) → (B*L,)
                B, L, C = logits.shape
                flat_logits = logits.reshape(B * L, C)
                flat_y = y.reshape(B * L)
            else:
                # (B, L) boundary case
                flat_logits = logits.reshape(-1)
                flat_y = y.reshape(-1)
            metrics_out = mc(flat_logits, flat_y)
            self.log_dict(
                metrics_out,
                prog_bar=(split == "val"),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=bs,
            )

        # ── spec frame-level metrics (test split only, per-clip) ──────────────
        # Defensive: never let a metric error break the run.
        if split == "test":
            try:
                self._log_spec_frame_metrics(logits, y, bs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("VGMLoopFrames spec metric logging failed: %r", exc)

        return loss

    # ── spec metric hook (overridden per head) ────────────────────────────────

    def _log_spec_frame_metrics(self, logits: torch.Tensor, y: torch.Tensor, bs: int) -> None:
        """No-op on the base; boundary / function heads override this."""
        return None

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx: int):
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx: int):
        self._shared_step(batch, batch_idx, "test")


class VGMLoopBoundaryProbe(_VGMLoopFramesProbeBase):
    """
    T1 boundary-detection probe.

    Decoder: (B, L, 1) logits → squeezed to (B, L).
    Loss:    BCEWithLogitsLoss vs (B, L) float32 heatmap.
    Metric:  configured in YAML (e.g. BinaryAUROC or BinaryF1Score), plus
             spec boundary-F1 @ ±0.5/±3 s + seam recall logged on test.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("target_key", "boundary")
        super().__init__(**kwargs)

    def _log_spec_frame_metrics(self, logits: torch.Tensor, y: torch.Tensor, bs: int) -> None:
        # logits/y are (B, L).  Peak-pick predicted heatmap (after sigmoid) and
        # the GT heatmap per-clip, convert frames→seconds via label_freq, then
        # boundary-F1 @ ±0.5/±3 s.  GT seam is approximated as the LATEST GT
        # boundary (loop seam comes after the intro_end boundary); see report
        # caveat — the heatmap alone does not tag which peak is the seam.
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()  # (B, L)
        gt = y.detach().float().cpu().numpy()  # (B, L)
        B = probs.shape[0]

        agg: dict[str, list[float]] = {}
        for b in range(B):
            est_frames = fm.peak_pick_boundaries(probs[b], threshold=0.5)
            gt_frames = fm.peak_pick_boundaries(gt[b], threshold=0.5)
            est_sec = fm.frames_to_seconds(est_frames, self.label_freq)
            gt_sec = fm.frames_to_seconds(gt_frames, self.label_freq)

            bm = fm.boundary_metrics(gt_sec, est_sec)
            for k in ("f_0_5", "f_3_0", "p_0_5", "p_3_0", "r_0_5", "r_3_0"):
                agg.setdefault(f"boundary_{k}", []).append(bm[k])

            # Seam recall: only meaningful when there is a GT seam to hit.
            if gt_sec:
                seam = max(gt_sec)  # latest boundary ≈ loop seam (see caveat)
                sr = fm.seam_recall(seam, est_sec)
                for w in ("0_5", "3_0"):
                    agg.setdefault(f"seam_recall_{w}", []).append(sr[f"recall_{w}"])

        for name, vals in agg.items():
            if vals:
                self.log(
                    f"test/{name}",
                    float(sum(vals) / len(vals)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bs,
                )


class VGMLoopFunctionProbe(_VGMLoopFramesProbeBase):
    """
    T3 per-frame function-class probe.

    Decoder: (B, L, 3) logits.
    Loss:    CrossEntropyLoss — expects (B*L, 3) logits vs (B*L,) int64.
    Metric:  MulticlassAccuracy num_classes=3 (configured in YAML), plus spec
             pairwise-F + frame-label agreement + single-segment fraction (test).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("target_key", "function")
        super().__init__(**kwargs)

    def _log_spec_frame_metrics(self, logits: torch.Tensor, y: torch.Tensor, bs: int) -> None:
        # logits (B, L, 3) → argmax over classes → (B, L) predicted class map.
        preds = logits.detach().argmax(dim=-1).cpu().numpy()  # (B, L)
        gt = y.detach().cpu().numpy()  # (B, L)
        B = preds.shape[0]

        pf_vals: list[float] = []
        agr_vals: list[float] = []
        single_flags: list[float] = []
        for b in range(B):
            m = fm.function_frame_metrics(gt[b], preds[b])
            pf_vals.append(m["pairwise_f"])
            agr_vals.append(m["frame_label_agreement"])
            single_flags.append(1.0 if fm.is_single_segment(gt[b]) else 0.0)

        logged = {
            "test/function_pairwise_f": pf_vals,
            "test/function_frame_label_agreement": agr_vals,
            "test/function_single_segment_frac": single_flags,
        }
        for name, vals in logged.items():
            if vals:
                self.log(
                    name,
                    float(sum(vals) / len(vals)),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=bs,
                )


# Generic alias so YAML can use class_path: marble.tasks.VGMLoopFrames.probe.ProbeAudioTask
# with target_key as an init_arg.
ProbeAudioTask = _VGMLoopFramesProbeBase
