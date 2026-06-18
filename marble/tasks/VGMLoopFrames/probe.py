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

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask, _unpack_batch
from marble.core.utils import instantiate_from_config


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

        losses = [fn(logits, y) for fn in self.loss_fns]
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

        return loss

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
    Metric:  configured in YAML (e.g. BinaryAUROC or BinaryF1Score).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("target_key", "boundary")
        super().__init__(**kwargs)


class VGMLoopFunctionProbe(_VGMLoopFramesProbeBase):
    """
    T3 per-frame function-class probe.

    Decoder: (B, L, 3) logits.
    Loss:    CrossEntropyLoss — expects (B*L, 3) logits vs (B*L,) int64.
    Metric:  MulticlassAccuracy num_classes=3 (configured in YAML).
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("target_key", "function")
        super().__init__(**kwargs)


# Generic alias so YAML can use class_path: marble.tasks.VGMLoopFrames.probe.ProbeAudioTask
# with target_key as an init_arg.
ProbeAudioTask = _VGMLoopFramesProbeBase
