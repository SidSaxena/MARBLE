"""BaseTask logs val/train metrics as GLOBAL epoch values (Metric objects),
not per-batch averages. Regression guard for the bug where ranking metrics
(AUROC / AveragePrecision) collapsed on single-class validation batches:
mean-averaging per-batch values gave e.g. val/auc_roc ≈ 0.5 instead of the
true global ≈ 0.99. Runs a real Trainer.validate through _shared_step.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, F1Score

from marble.core.base_task import BaseTask


class _IdentityTask(BaseTask):
    """forward returns the input unchanged — inputs are already (B, 2) logits."""

    def forward(self, x, clip_ids=None):
        return x


def _data():
    # Single-class batches (the failure mode): batch0 all class 0, batch1 all
    # class 1, batches 2-3 mixed. No shuffle. Highly separable GLOBALLY.
    torch.manual_seed(0)
    ys = torch.cat(
        [torch.zeros(16), torch.ones(16), torch.tensor([0, 1] * 8), torch.tensor([1, 0] * 8)]
    ).long()
    logit1 = ys.float() * 2.5 + 0.5 * torch.randn(64)
    X = torch.stack([-logit1, logit1], dim=1)  # (64, 2)
    ds = list(zip(X, ys, ["p"] * 64))  # (x_i, y_i, path) — BaseTask 3-tuple
    return DataLoader(ds, batch_size=16, shuffle=False), X, ys


def test_val_metrics_are_global_not_batch_averaged():
    loader, X, ys = _data()
    true_auc = float(AUROC(task="multiclass", num_classes=2, average="macro")(X, ys))
    assert true_auc > 0.9, "fixture should be globally separable"

    model = _IdentityTask(
        encoder=nn.Linear(2, 2),
        decoders=[nn.Identity()],
        losses=[nn.CrossEntropyLoss()],
        metrics={
            "val": {
                "auc_roc": AUROC(task="multiclass", num_classes=2, average="macro"),
                "f1": F1Score(task="multiclass", num_classes=2, average="macro"),
            }
        },
        cache_embeddings=False,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.validate(model, dataloaders=loader, verbose=False)
    got = float(trainer.callback_metrics["val/auc_roc"])
    # GLOBAL AUROC ≈ 0.99. The old per-batch-averaged path produced ≈ 0.5.
    assert abs(got - true_auc) < 1e-3, f"val/auc_roc={got:.4f} != global {true_auc:.4f}"
    assert got > 0.9
