# marble/tasks/NSynth/probe.py
"""
NSynth pitch-classification probe.

Architecture:  encoder → LayerSelector → TimeAvgPool → MLPDecoder (88 classes)

Test aggregation: for the same audio file, mean-pool logits across slices
(NSynth clips are fixed-length so there is always exactly 1 slice per file,
but aggregation is kept for robustness).

Metric: top-1 accuracy (val/acc, test/acc).
"""

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeAudioTask(BaseTask):
    """
    Pitch-classification probe for NSynth (88 MIDI pitch classes: A0–C8).
    Follows the same structure as the GS and EMO probes for consistency.
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
        enc      = instantiate_from_config(encoder)
        tfs      = [instantiate_from_config(c) for c in emb_transforms]
        decs     = [instantiate_from_config(c) for c in decoders]
        loss_fns = [instantiate_from_config(c) for c in losses]

        metric_maps = {
            split: {
                name: instantiate_from_config(cfg)
                for name, cfg in metrics[split].items()
            }
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
        )

    # ── test: aggregate per audio file then compute metrics ──────────────────

    def on_test_start(self) -> None:
        self._test_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        x, labels, paths = batch
        logits = self(x)
        for path, logit, lb in zip(paths, logits, labels):
            self._test_outputs.append({
                "path":  path,
                "logit": logit.detach(),
                "label": lb,
            })

    def on_test_epoch_end(self) -> None:
        # Aggregate slices per file (mean logit)
        file_dict: dict[str, dict] = {}
        for entry in self._test_outputs:
            info = file_dict.setdefault(
                entry["path"],
                {"logits": [], "label": entry["label"]},
            )
            info["logits"].append(entry["logit"])

        batched_logits = []
        batched_labels = []
        for info in file_dict.values():
            arr = torch.stack(info["logits"])   # (n_slices, 88)
            batched_logits.append(arr.mean(0))  # (88,)
            batched_labels.append(info["label"])

        logits = torch.stack(batched_logits)    # (N, 88)
        labels = torch.stack(batched_labels)    # (N,)

        mc: MetricCollection = getattr(self, "test_metrics", None)
        if mc is not None:
            out = mc(logits, labels)
            self.log_dict(
                out,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        print(f"\nNSynth test: aggregated {len(file_dict)} files.")
