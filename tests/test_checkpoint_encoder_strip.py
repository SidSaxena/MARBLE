"""BaseTask drops a FROZEN encoder's weights from checkpoints (huge + reloaded
fresh at init) and re-injects them on load so a strict load_state_dict still
matches — bit-identically. Profiled on the real CLaMP3 task: 2210 MB -> 0.87 MB,
0 tensors differ. These are the fast, encoder-free regression guards.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from marble.core.base_task import BaseTask


class _TinyTask(BaseTask):
    """Minimal concrete BaseTask for exercising the checkpoint hooks."""


def _build(*, frozen: bool, strip: bool) -> _TinyTask:
    # Deterministic construction so a fresh build re-creates the SAME encoder
    # weights — mirroring the real encoder, which reloads identical pretrained
    # weights from its own checkpoint every time.
    torch.manual_seed(0)
    enc = nn.Linear(4, 4)
    if frozen:
        for p in enc.parameters():
            p.requires_grad_(False)
    return _TinyTask(
        encoder=enc,
        decoders=[nn.Linear(4, 2)],
        strip_frozen_encoder_from_ckpt=strip,
    )


def test_on_save_strips_frozen_encoder_keeps_decoder():
    t = _build(frozen=True, strip=True)
    ck = {"state_dict": dict(t.state_dict())}
    t.on_save_checkpoint(ck)
    assert not any(k.startswith("encoder.") for k in ck["state_dict"])
    assert any(k.startswith("decoders.") for k in ck["state_dict"])


def test_strip_then_reinject_strict_load_is_bit_identical():
    t = _build(frozen=True, strip=True)
    with torch.no_grad():  # give the decoder a non-trivial "trained" state
        for p in t.decoders.parameters():
            p.add_(torch.randn_like(p))
    ck = {"state_dict": dict(t.state_dict())}
    t.on_save_checkpoint(ck)  # strips encoder.*

    t2 = _build(frozen=True, strip=True)  # fresh module (encoder re-constructed)
    missing = [k for k in t2.state_dict() if k not in ck["state_dict"]]
    assert missing and all(k.startswith("encoder.") for k in missing)
    t2.on_load_checkpoint(ck)  # re-inject the live encoder weights
    t2.load_state_dict(ck["state_dict"])  # strict — must match

    ref, got = t.state_dict(), t2.state_dict()
    assert set(ref) == set(got)
    assert all(torch.equal(ref[k], got[k]) for k in ref)


def test_trainable_encoder_is_not_stripped():
    # A fine-tuned (grad-enabled) encoder IS the result — must be kept.
    t = _build(frozen=False, strip=True)
    ck = {"state_dict": dict(t.state_dict())}
    t.on_save_checkpoint(ck)
    assert any(k.startswith("encoder.") for k in ck["state_dict"])


def test_strip_disabled_keeps_frozen_encoder():
    t = _build(frozen=True, strip=False)
    ck = {"state_dict": dict(t.state_dict())}
    t.on_save_checkpoint(ck)
    assert any(k.startswith("encoder.") for k in ck["state_dict"])


def test_full_checkpoint_load_is_untouched_by_reinject():
    # Back-compat: a full (un-stripped) checkpoint already has encoder keys;
    # on_load_checkpoint must leave it alone.
    t = _build(frozen=True, strip=False)
    ck = {"state_dict": dict(t.state_dict())}
    before = dict(ck["state_dict"])
    t.on_load_checkpoint(ck)
    assert ck["state_dict"].keys() == before.keys()
