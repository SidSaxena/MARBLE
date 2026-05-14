# marble/tasks/GS/probe.py

import mir_eval
import numpy as np
import torch
from torchmetrics import Metric, MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.GS.datamodule import _GSAudioBase


class ProbeAudioTask(BaseTask):
    """
    GS genre probe task. Inherits training/val logic, multi-head,
    losses, metrics and EMA support from BaseTask.
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
        # 1) build all submodules from your YAML configs
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        # metrics comes in as nested dict: { split: { name: cfg, … }, … }
        metric_maps = {
            split: {name: instantiate_from_config(cfg) for name, cfg in metrics[split].items()}
            for split in ("train", "val", "test")
        }

        # 2) hand everything off to BaseTask
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

    def on_test_start(self) -> None:
        # Initialize storage for per-slice test outputs
        self._test_file_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        # 4-tuple (waveform, label, path, clip_id) when caching is on;
        # fall back to legacy 3-tuple shape for older datamodules.
        if isinstance(batch, (tuple, list)) and len(batch) >= 4:
            x, labels, ori_uids, clip_ids = batch[0], batch[1], batch[2], batch[3]
        else:
            x, labels, ori_uids = batch
            clip_ids = None
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)

        for uid, logit, lb in zip(ori_uids, logits, labels, strict=False):
            self._test_file_outputs.append(
                {
                    "uid": uid,
                    "logit": logit,
                    "label": lb,
                }
            )

    def on_test_epoch_end(self) -> None:
        # Aggregate per-file predictions
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            uid = entry["uid"]
            info = file_dict.setdefault(uid, {"logits": [], "label": entry["label"]})
            info["logits"].append(entry["logit"])

        # aggregate logits and compute file-level metrics
        print(f"Aggregating {len(file_dict)} files with per-slice outputs")
        batched_logits = []
        batched_labels = []
        for _uid, info in file_dict.items():
            arr = torch.stack(info["logits"])  # (n_slices, C)
            mean_logit = arr.mean(dim=0)  # (C,)
            batched_logits.append(mean_logit)
            batched_labels.append(info["label"])
        batched_logits = torch.stack(batched_logits)
        batched_labels = torch.stack(batched_labels)
        # compute metrics
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


class KeyWeightedScore(Metric):
    """
    Key weighted score for key estimation using logits.
    Uses mir_eval to compute the weighted score for key estimation.
    """

    IDX2LABEL = _GSAudioBase.IDX2LABEL
    LABEL2IDX = _GSAudioBase.LABEL2IDX

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Update the metric with predictions and ground truth labels.
        """
        device = preds.device  # Use the device of preds to ensure consistency
        self.preds.append(preds.to(device))
        self.labels.append(labels.to(device))

    def compute(self):
        """
        Compute the weighted score using mir_eval without the need for aggregation.
        Assumes the inputs have already been aggregated.
        """
        preds = torch.cat(self.preds, dim=0)
        labels = torch.cat(self.labels, dim=0)

        # Convert logits to predicted labels
        preds_label = torch.argmax(preds, dim=-1)

        # Convert to numpy for mir_eval
        preds_label = preds_label.cpu().numpy()
        labels = labels.cpu().numpy()

        # Calculate the weighted score
        scores = [
            mir_eval.key.weighted_score(self.IDX2LABEL[ref_key], self.IDX2LABEL[est_key])
            for ref_key, est_key in zip(labels, preds_label, strict=False)
        ]
        return torch.tensor(np.mean(scores), device=preds.device)
