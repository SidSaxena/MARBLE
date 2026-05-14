# marble/tasks/GTZANGenre/probe.py
import torch

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config


class ProbeAudioTask(BaseTask):
    """
    GTZAN genre probe task.  Inherits training/val logic, multi‐head,
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
            x, labels, file_paths, clip_ids = batch[0], batch[1], batch[2], batch[3]
        else:
            x, labels, file_paths = batch
            clip_ids = None
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        probs = torch.softmax(logits, dim=1).cpu()

        # Store per-slice probabilities and labels for file-level aggregation
        for fp, prob, lb in zip(file_paths, probs, labels.cpu(), strict=False):
            self._test_file_outputs.append(
                {
                    "file_path": fp,
                    "prob": prob.numpy(),
                    "label": int(lb),
                }
            )

    def on_test_epoch_end(self) -> None:
        # Aggregate per-file predictions
        file_dict: dict[str, dict] = {}
        for entry in self._test_file_outputs:
            fp = entry["file_path"]
            info = file_dict.setdefault(fp, {"probs": [], "label": entry["label"]})
            info["probs"].append(entry["prob"])

        total, correct = 0, 0
        for _fp, info in file_dict.items():
            arr = torch.tensor(info["probs"])  # (n_slices, C)
            mean_prob = arr.mean(dim=0)  # (C,)
            pred = int(mean_prob.argmax().item())
            total += 1
            correct += int(pred == info["label"])

        file_acc = correct / total if total > 0 else 0.0
        # Log file-level accuracy with sync across devices
        self.log("test/file_acc", file_acc, prog_bar=True, on_epoch=True, sync_dist=True)
