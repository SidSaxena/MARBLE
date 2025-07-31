# marble/tasks/MTGTop50/probe.py
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import mir_eval
import torchmetrics
from torchmetrics import Metric, MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.MTGTop50.datamodule import _MTGTop50AudioBase


class ProbeAudioTask(BaseTask):
    """
    MTGTop50 probe task. Inherits training/val logic, multi-head,
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
    ):
        # 1) build all submodules from your YAML configs
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

        # metrics comes in as nested dict: { split: { name: cfg, … }, … }
        metric_maps = {
            split: {
                name: instantiate_from_config(cfg)
                for name, cfg in metrics[split].items()
            }
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
        )
        
    def _shared_step(self, batch, batch_idx: int, split: str) -> torch.Tensor:
        """
        Common logic for train/val:
          - unpack batch
          - forward
          - sum all loss_fns
          - log loss and metrics
        """
        x, y, uids_or_paths = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        # compute and log loss
        losses = [fn(logits, y.to(logits.dtype)) for fn in self.loss_fns]
        loss = sum(losses)
        self.log(f"{split}/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # compute and log metrics
        mc: MetricCollection = getattr(self, f"{split}_metrics", None)
        if mc is not None:
            metrics_out = mc(probs, y)
            self.log_dict(metrics_out, prog_bar=(split == "val"), on_step=False, on_epoch=True, sync_dist=True)

        return loss
        
    def on_test_start(self) -> None:
        # Initialize storage for per-slice test outputs
        self._test_file_outputs: list[dict] = []
    
    def test_step(self, batch, batch_idx):
        x, labels, ori_uids = batch
        logits = self(x)
        probs = torch.sigmoid(logits)

        for uid, prob, lb in zip(ori_uids, probs, labels):
            self._test_file_outputs.append({
                "uid": uid,
                "prob": prob,
                "label": lb,
            })

    def on_test_epoch_end(self) -> None:
        # Aggregate per-file predictions
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            uid = entry["uid"]
            info = file_dict.setdefault(uid, {"probs": [], "label": entry["label"]})
            info["probs"].append(entry["prob"])

        # aggregate probs and compute file-level metrics
        print(f"Aggregating {len(file_dict)} files with per-slice outputs")
        batched_probs = []
        batched_labels = []
        for uid, info in file_dict.items():
            arr = torch.stack(info["probs"])      # (n_slices, C)
            mean_prob = arr.mean(dim=0)             # (C,)
            batched_probs.append(mean_prob)
            batched_labels.append(info["label"])
        batched_probs = torch.stack(batched_probs)
        batched_labels = torch.stack(batched_labels)
        # compute metrics
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_probs, batched_labels)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
