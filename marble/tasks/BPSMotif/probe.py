# marble/tasks/BPSMotif/probe.py
"""
BPS-Motif probe tasks.

Two parallel tasks share this module:

* :class:`BPSMotifMNIDTask` — clip-level binary classification on
  motif-window vs sampled non-motif-window MIDI slices. Inherits
  fit/test/metric/cache plumbing from :class:`BaseTask`; the
  per-file aggregation in BaseTask is a no-op here since each window
  is already a separate file.

* :class:`BPSMotifRetrievalTask` — within-piece within-letter motif
  retrieval. Trivial subclass of :class:`CoverRetrievalTask`: the only
  difference is that the datamodule encodes ``(piece_id, motif_letter)``
  jointly into ``work_id`` so the standard MAP scoring counts
  "same piece + same letter" occurrences as relevant — exactly the
  within-movement motif identity we want, because motif letters in this
  dataset are movement-local. No method overrides needed; this class
  exists only to give the WandB run a distinct ``task`` tag and to give
  the LightningCLI a stable import path.
"""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.Covers80.probe import CoverRetrievalTask


class BPSMotifMNIDTask(BaseTask):
    """Binary motif-window classification probe on BPS-Motif.

    See :class:`marble.core.base_task.BaseTask` for the underlying
    fit/test/metric machinery. This subclass exists to:

    * Build encoder / transforms / decoders / losses / metrics from the
      YAML config (the standard MARBLE wiring pattern).
    * Hold a stable import path so the LightningCLI ``class_path`` is
      ``marble.tasks.BPSMotif.probe.BPSMotifMNIDTask``.
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
        cache_embeddings: bool = False,
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]
        metric_maps = {
            split: {name: instantiate_from_config(cfg) for name, cfg in metrics[split].items()}
            for split in ("train", "val", "test")
        }
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

    # ── per-window test aggregation ────────────────────────────────────────
    #
    # BPS-Motif windows are 1:1 with files (each motif occurrence / sampled
    # negative is its own per-window MIDI), so there is no per-file
    # aggregation step like HookTheoryKey's per-slice→per-song majority
    # vote. We rely on BaseTask's default test_step + on_test_epoch_end,
    # which compute the standard torchmetrics (F1, accuracy, precision,
    # recall via the metric collection defined in the YAML).

    def on_test_start(self) -> None:
        self._test_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        # 4-tuple (patches, label, midi_path, clip_id).
        if isinstance(batch, (tuple, list)) and len(batch) >= 4:
            x, labels, _paths, clip_ids = batch[0], batch[1], batch[2], batch[3]
        else:
            x, labels, _paths = batch
            clip_ids = None
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        for logit, lb in zip(logits, labels, strict=False):
            self._test_outputs.append({"logit": logit, "label": lb})

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        batched_logits = torch.stack([e["logit"] for e in self._test_outputs])
        batched_labels = torch.stack([e["label"] for e in self._test_outputs])
        mc: MetricCollection | None = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(
                metrics_out,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )


class BPSMotifRetrievalTask(CoverRetrievalTask):
    """Within-piece within-letter motif retrieval on BPS-Motif.

    Subclass only for the import path. All MAP / centering / aggregation
    logic is inherited from :class:`CoverRetrievalTask`. The datamodule
    encodes (piece_id, motif_letter) jointly into ``work_id`` so the
    standard ``_compute_map`` scores same-piece-same-letter occurrences
    as relevant — which IS within-piece within-letter retrieval, because
    BPS-Motif's letters are movement-local.
    """

    pass
