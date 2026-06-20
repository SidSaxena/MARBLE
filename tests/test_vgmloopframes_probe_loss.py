"""tests/test_vgmloopframes_probe_loss.py

Focused regression test: _shared_step loss call must not crash for the
VGMLoopFunctionProbe when logits=(B,L,3) and targets=(B,L) int64.

We test the loss computation in isolation (no encoder, no Lightning trainer)
by calling CrossEntropyLoss directly with the shapes that _shared_step
produces — both before and after the flatten fix.
"""

import pytest
import torch
import torch.nn as nn


def _ce_loss_flat(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The fixed path: flatten (B,L,C)→(B*L,C) and (B,L)→(B*L,) before CE."""
    B, L, C = logits.shape
    return nn.CrossEntropyLoss()(logits.reshape(B * L, C), targets.reshape(B * L))


def test_ce_loss_with_blc_logits_crashes_without_flatten():
    """Confirm the un-fixed call raises RuntimeError (documents the original bug)."""
    logits = torch.randn(2, 50, 3)
    targets = torch.randint(0, 3, (2, 50))
    with pytest.raises(RuntimeError):
        nn.CrossEntropyLoss()(logits, targets)


def test_ce_loss_flat_does_not_crash():
    """The flattened path must produce a finite scalar without raising."""
    logits = torch.randn(2, 50, 3)
    targets = torch.randint(0, 3, (2, 50))
    loss = _ce_loss_flat(logits, targets)
    assert loss.ndim == 0, "loss must be a scalar"
    assert torch.isfinite(loss), "loss must be finite"


def test_ce_loss_flat_single_batch():
    """Works for batch size 1 (edge case)."""
    logits = torch.randn(1, 25, 3)
    targets = torch.randint(0, 3, (1, 25))
    loss = _ce_loss_flat(logits, targets)
    assert torch.isfinite(loss)


def test_bce_path_unchanged():
    """
    Boundary BCEWithLogitsLoss path: logits (B,L) and targets (B,L) float32.
    This path must NOT be flattened in the loss call — verify it works as-is.
    """
    logits = torch.randn(2, 50)
    targets = torch.rand(2, 50)
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
