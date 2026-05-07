# marble/tasks/LeitmotifDetection/probe.py
"""
Probe task for leitmotif / theme detection.

Adds file-level majority-vote (averaged-probability) aggregation on top of
the standard clip-level classification loop from BaseTask.

During ``test_step`` each clip's softmax probability vector is accumulated
per audio file (keyed by ``audio_path``).  In ``on_test_epoch_end`` the
per-clip probs are averaged → argmax → file-level prediction, and a
``test/file_acc`` metric is logged.
"""

import torch

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeLeitmotifTask(BaseTask):
    """
    Probe task for leitmotif detection with file-level aggregation.

    Parameters
    ----------
    sample_rate : int
    use_ema : bool
    encoder : dict
    emb_transforms : list[dict]
    decoders : list[dict]
    losses : list[dict]
    metrics : dict[str, dict[str, dict]]
        Nested dict ``{split: {name: config, …}, …}`` for train/val/test.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list,
        decoders: list,
        losses: list,
        metrics: dict,
    ) -> None:
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]

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

    # ------------------------------------------------------------------
    # Test loop with file-level aggregation
    # ------------------------------------------------------------------

    def on_test_start(self) -> None:
        """Initialise storage for per-clip outputs."""
        self._test_file_outputs: list = []

    def test_step(self, batch, batch_idx):
        """
        Collect per-clip softmax probabilities, keyed by ``audio_path``.

        Batch shape: ``(waveform, label, audio_path)`` as returned by the
        dataset's ``__getitem__``.
        """
        x, labels, file_paths = batch
        logits = self(x)
        probs = torch.softmax(logits, dim=1).cpu()

        for fp, prob, lb in zip(file_paths, probs, labels.cpu()):
            self._test_file_outputs.append(
                {
                    "file_path": fp,
                    "prob": prob.numpy(),
                    "label": int(lb),
                }
            )

    def on_test_epoch_end(self) -> None:
        """
        Average clip-level probs per file → argmax → file-level accuracy.
        """
        file_dict: dict = {}
        for entry in self._test_file_outputs:
            fp = entry["file_path"]
            info = file_dict.setdefault(fp, {"probs": [], "label": entry["label"]})
            info["probs"].append(entry["prob"])

        total, correct = 0, 0
        for fp, info in file_dict.items():
            arr = torch.tensor(info["probs"])       # (n_clips, C)
            mean_prob = arr.mean(dim=0)              # (C,)
            pred = int(mean_prob.argmax().item())
            total += 1
            correct += int(pred == info["label"])

        file_acc = correct / total if total > 0 else 0.0
        self.log(
            "test/file_acc",
            file_acc,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
