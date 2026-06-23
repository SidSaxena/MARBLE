"""
tests/test_supermario_balanced.py

Unit tests for the class-balanced additions to
``marble.tasks.SuperMarioStructure.probe.ProbeAudioTask``:

  - ``_compute_inverse_freq_sqrt_weights`` — math correctness on a
    hand-built JSONL with known per-class counts.
  - ``_resolve_class_weights`` — explicit list / "auto" / None / bad
    input handling.
  - ``_reweight_ce_loss`` — surgical replacement of the first
    CE/NLL loss with a re-weighted one; non-CE losses untouched.

End-to-end probe construction is out of scope (would force the
CLaMP3 encoder import); these tests target the pure-Python helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from marble.tasks.SuperMarioStructure.probe import (
    ProbeAudioTask,
    _compute_inverse_freq_sqrt_weights,
    _IDX2LABEL,
)


def _write_jsonl(path: Path, label_counts: dict[str, int]) -> None:
    """Build a SuperMarioStructure-shaped JSONL with the requested
    per-label record counts."""
    with open(path, "w") as f:
        uid = 0
        for label, n in label_counts.items():
            for _ in range(n):
                rec = {
                    "ori_uid": f"u{uid:04d}",
                    "label": label,
                    "midi_path": f"data/fake/{uid:04d}.mid",
                }
                f.write(json.dumps(rec) + "\n")
                uid += 1


# ─────────────────────────────────────────────────────────────────────
# _compute_inverse_freq_sqrt_weights
# ─────────────────────────────────────────────────────────────────────


def test_inverse_freq_sqrt_basic_balance(tmp_path: Path):
    """Balanced corpus (equal class counts) → all weights equal to sqrt(N_classes)."""
    n_per_class = 10
    _write_jsonl(
        tmp_path / "train.jsonl",
        {label: n_per_class for label in _IDX2LABEL},
    )
    w = _compute_inverse_freq_sqrt_weights(tmp_path / "train.jsonl")
    expected = math.sqrt(len(_IDX2LABEL) * n_per_class / n_per_class)
    assert w.shape == (len(_IDX2LABEL),)
    assert torch.allclose(w, torch.full((6,), expected), rtol=1e-5)


def test_inverse_freq_sqrt_real_skew(tmp_path: Path):
    """SuperMarioStructure-like distribution: loop dominant, others
    sparse. Weights should be ordered correctly (most-frequent class
    gets the smallest weight)."""
    counts = {
        "loop": 1200,
        "intro": 350,
        "linear": 160,
        "stinger": 60,
        "outro": 50,
        "bridge": 40,
    }
    _write_jsonl(tmp_path / "train.jsonl", counts)
    w = _compute_inverse_freq_sqrt_weights(tmp_path / "train.jsonl")
    w_by_label = dict(zip(_IDX2LABEL, w.tolist(), strict=True))
    # loop is the most frequent → smallest weight.
    assert w_by_label["loop"] < w_by_label["intro"]
    assert w_by_label["intro"] < w_by_label["linear"]
    assert w_by_label["bridge"] > w_by_label["loop"]
    # Verify the exact ratio: weight ratio between two classes equals
    # sqrt(N_b / N_a).
    ratio = w_by_label["bridge"] / w_by_label["loop"]
    expected_ratio = math.sqrt(counts["loop"] / counts["bridge"])
    assert ratio == pytest.approx(expected_ratio, rel=1e-4)


def test_inverse_freq_sqrt_zero_count_class(tmp_path: Path):
    """A class with zero training examples gets weight 1.0 (avoids inf)."""
    counts = {label: 0 for label in _IDX2LABEL}
    counts["loop"] = 100
    _write_jsonl(tmp_path / "train.jsonl", counts)
    w = _compute_inverse_freq_sqrt_weights(tmp_path / "train.jsonl")
    w_by_label = dict(zip(_IDX2LABEL, w.tolist(), strict=True))
    for label in _IDX2LABEL:
        if label == "loop":
            assert w_by_label[label] == pytest.approx(1.0)  # sqrt(100 / 100)
        else:
            assert w_by_label[label] == 1.0


def test_inverse_freq_sqrt_empty_jsonl_raises(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="no labelled records"):
        _compute_inverse_freq_sqrt_weights(p)


def test_inverse_freq_sqrt_ignores_unknown_labels(tmp_path: Path):
    """Records with labels outside the canonical 6 are silently skipped
    (build script would've already warned)."""
    p = tmp_path / "train.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"ori_uid": "u0", "label": "loop", "midi_path": "x"}) + "\n")
        f.write(json.dumps({"ori_uid": "u1", "label": "transition", "midi_path": "x"}) + "\n")
    w = _compute_inverse_freq_sqrt_weights(p)
    # Only 1 valid record → N_total=1, N_loop=1 → weight=sqrt(1)=1; others 1.0.
    w_by_label = dict(zip(_IDX2LABEL, w.tolist(), strict=True))
    assert w_by_label["loop"] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────
# _resolve_class_weights
# ─────────────────────────────────────────────────────────────────────


def test_resolve_class_weights_none_returns_none():
    assert ProbeAudioTask._resolve_class_weights(None, None) is None
    assert ProbeAudioTask._resolve_class_weights(None, "ignored") is None


def test_resolve_class_weights_explicit_list():
    spec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    out = ProbeAudioTask._resolve_class_weights(spec, train_jsonl=None)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    # float32 round-trip — compare with tolerance, not equality.
    assert torch.allclose(out, torch.tensor(spec, dtype=torch.float32))


def test_resolve_class_weights_explicit_list_wrong_length_raises():
    with pytest.raises(ValueError, match="6 entries"):
        ProbeAudioTask._resolve_class_weights([0.5, 0.5], train_jsonl=None)


def test_resolve_class_weights_auto_requires_jsonl():
    with pytest.raises(ValueError, match="requires train_jsonl"):
        ProbeAudioTask._resolve_class_weights("auto", train_jsonl=None)


def test_resolve_class_weights_auto_loads_from_jsonl(tmp_path: Path):
    _write_jsonl(tmp_path / "train.jsonl", {label: 10 for label in _IDX2LABEL})
    out = ProbeAudioTask._resolve_class_weights("auto", str(tmp_path / "train.jsonl"))
    assert out is not None
    assert out.shape == (len(_IDX2LABEL),)


def test_resolve_class_weights_bad_str_raises():
    with pytest.raises(ValueError, match="must be 'auto'"):
        ProbeAudioTask._resolve_class_weights("uniform", train_jsonl=None)


# ─────────────────────────────────────────────────────────────────────
# _reweight_ce_loss
# ─────────────────────────────────────────────────────────────────────


def test_reweight_ce_replaces_first_ce_loss():
    losses = [torch.nn.CrossEntropyLoss(reduction="mean")]
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ProbeAudioTask._reweight_ce_loss(losses, weights)
    assert isinstance(losses[0], torch.nn.CrossEntropyLoss)
    assert torch.allclose(losses[0].weight, weights)
    # Other CE kwargs preserved.
    assert losses[0].reduction == "mean"


def test_reweight_ce_preserves_loss_kwargs():
    losses = [
        torch.nn.CrossEntropyLoss(
            reduction="sum", ignore_index=-100, label_smoothing=0.1
        )
    ]
    weights = torch.ones(6)
    ProbeAudioTask._reweight_ce_loss(losses, weights)
    assert losses[0].reduction == "sum"
    assert losses[0].ignore_index == -100
    assert losses[0].label_smoothing == pytest.approx(0.1)


def test_reweight_ce_handles_nll():
    losses = [torch.nn.NLLLoss(reduction="mean")]
    weights = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ProbeAudioTask._reweight_ce_loss(losses, weights)
    assert isinstance(losses[0], torch.nn.NLLLoss)
    assert torch.allclose(losses[0].weight, weights)


def test_reweight_ce_leaves_non_ce_losses_untouched():
    """A losses list with a non-CE entry (e.g. MSELoss) and no CE/NLL
    logs a warning and doesn't change anything."""
    mse = torch.nn.MSELoss()
    losses = [mse]
    weights = torch.ones(6)
    ProbeAudioTask._reweight_ce_loss(losses, weights)
    assert losses[0] is mse  # same object, no replacement


def test_reweight_ce_only_replaces_first_ce_loss():
    """Multiple CE losses — only the first one is reweighted. Documented
    behaviour: auxiliary CE objectives keep their original weighting."""
    losses = [
        torch.nn.CrossEntropyLoss(reduction="mean"),
        torch.nn.CrossEntropyLoss(reduction="sum"),  # auxiliary, unchanged
    ]
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ProbeAudioTask._reweight_ce_loss(losses, weights)
    assert losses[0].weight is not None
    assert losses[1].weight is None
