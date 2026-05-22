# marble/tasks/SuperMarioStructure/probe.py
"""
SuperMarioStructure probe — Super Mario VGM functional structure.

Verbatim clone of marble/tasks/HXMSA/probe.py with one edit: docstring
references SuperMario. The 4-tuple unpacking, per-uid logit aggregation
(slices within a segment averaged, segments evaluated independently),
and BaseTask integration are all identical.

Metric configuration is driven by the YAML config — configs include
both Accuracy (overall) and MulticlassF1Score with average='macro'
(imbalance-aware: VGM tracks are dominated by `loop` segments, so the
6-class distribution is similarly long-tailed to HXMSA's 13-class).
"""

import json
from pathlib import Path

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config

# Matches the 6-class inventory in datamodule.py (LABEL2IDX).
_IDX2LABEL = ["bridge", "intro", "linear", "loop", "outro", "stinger"]


class ProbeAudioTask(BaseTask):
    """
    SuperMarioStructure probe task. Inherits training/val logic,
    multi-head, losses, metrics, and EMA support from BaseTask.
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

    def on_test_start(self) -> None:
        self._test_file_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        # 4-tuple (waveform, label, ori_uid, clip_id) when caching is on;
        # 3-tuple fallback for older datamodules.
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
        # Aggregate per-segment predictions: average slice-level logits per
        # ori_uid (each ori_uid corresponds to one SuperMario functional segment).
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            uid = entry["uid"]
            info = file_dict.setdefault(uid, {"logits": [], "label": entry["label"]})
            info["logits"].append(entry["logit"])

        print(f"Aggregating {len(file_dict)} segments with per-slice outputs")
        batched_logits = []
        batched_labels = []
        for _uid, info in file_dict.items():
            arr = torch.stack(info["logits"])  # (n_slices, C)
            mean_logit = arr.mean(dim=0)  # (C,)
            batched_logits.append(mean_logit)
            batched_labels.append(info["label"])
        batched_logits = torch.stack(batched_logits)
        batched_labels = torch.stack(batched_labels)
        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(metrics_out, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # Dump per-class metrics + confusion matrix + per-sample predictions
        # so the layer-sweep produces enough data for a full per-class analysis
        # without needing to refit. Lands next to the checkpoint dir as
        # `test_predictions.json` (one file per sweep run).
        y_pred = batched_logits.argmax(dim=-1).cpu().numpy()
        y_true = batched_labels.cpu().numpy()
        K = batched_logits.shape[-1]
        per_class = []
        confusion = [[0] * K for _ in range(K)]
        for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=False):
            confusion[t][p] += 1
        for c in range(K):
            tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == c and p != c)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            support = int((y_true == c).sum())
            per_class.append(
                {
                    "label": _IDX2LABEL[c] if c < len(_IDX2LABEL) else str(c),
                    "support": support,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            )

        predictions = [
            {
                "uid": uid,
                "true": _IDX2LABEL[t] if t < len(_IDX2LABEL) else int(t),
                "pred": _IDX2LABEL[p] if p < len(_IDX2LABEL) else int(p),
                "logits": [round(float(x), 4) for x in logit.tolist()],
            }
            for uid, t, p, logit in zip(
                list(file_dict.keys()),
                y_true.tolist(),
                y_pred.tolist(),
                batched_logits.cpu(),
                strict=False,
            )
        ]

        # Resolve output dir from the trainer's first ModelCheckpoint callback
        # (the dirpath we set in the config); fall back to logger.save_dir.
        out_dir = None
        for cb in self.trainer.callbacks:
            if hasattr(cb, "dirpath") and cb.dirpath:
                out_dir = Path(cb.dirpath).parent
                break
        if out_dir is None and self.trainer.logger is not None:
            sd = getattr(self.trainer.logger, "save_dir", None)
            if sd:
                out_dir = Path(sd)
        if out_dir is None:
            out_dir = Path("output")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "test_predictions.json"
        with open(out_path, "w") as f:
            json.dump(
                {
                    "per_class": per_class,
                    "confusion": confusion,
                    "predictions": predictions,
                    "labels": _IDX2LABEL,
                },
                f,
                indent=2,
            )
        print(f"Per-class metrics + predictions dumped to {out_path}")
