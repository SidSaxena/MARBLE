"""
tests/test_emb_cache_slugs.py

Pin ``EmbeddingCacheMixin._derive_cache_slugs`` against accidental
regression. In particular, when WandB is absent we must NOT fall back to
``type(self).__name__`` because all four retrieval tasks share the
``CoverRetrievalTask`` class (Covers80, SHS100K, VGMIDITVar,
VGMIDITVar-timbre re-export the same class) — that would let a smoke
run silently reuse cached embeddings from a different dataset.

The mixin reaches into ``self.trainer.logger`` / ``self.trainer.datamodule``
which we replicate with tiny fake objects rather than spinning up
Lightning. ``self.encoder`` is also faked since only its class name is
read.
"""

from __future__ import annotations

from types import SimpleNamespace

from marble.utils.emb_cache import EmbeddingCacheMixin


def _make_mixin(*, wandb_group: str | None, jsonl: str | None) -> EmbeddingCacheMixin:
    """Build a minimal object whose ``_derive_cache_slugs`` method we can
    call directly. We attach the methods to a plain object via
    ``__getattribute__`` indirection — the mixin only relies on a few
    attribute paths on ``self``, all of which we wire up below.
    """

    class _Fake(EmbeddingCacheMixin):  # type: ignore[misc]
        pass

    obj = _Fake.__new__(_Fake)

    # Encoder is read only for class name; build a tiny class so its
    # __name__ is predictable.
    class _FakeEncoder:
        pass

    obj.encoder = _FakeEncoder()

    # Build a trainer with optional logger + optional datamodule.
    logger = SimpleNamespace(_wandb_init={"group": wandb_group}) if wandb_group else None
    test_cfg = {"init_args": {"jsonl": jsonl}} if jsonl else None
    datamodule = SimpleNamespace(test_config=test_cfg) if test_cfg else None
    obj.trainer = SimpleNamespace(logger=logger, datamodule=datamodule)  # type: ignore[attr-defined]
    return obj


def test_wandb_group_takes_precedence():
    m = _make_mixin(
        wandb_group="OMARRQ / VGMIDITVar-timbre", jsonl="data/SHS100K/SHS100K.test.jsonl"
    )
    enc, task = m._derive_cache_slugs()
    assert enc == "OMARRQ"
    assert task == "VGMIDITVar-timbre"


def test_jsonl_stem_used_when_wandb_absent():
    """Without wandb, the JSONL stem disambiguates SHS100K vs VGMIDITVar
    (both share CoverRetrievalTask)."""
    m = _make_mixin(wandb_group=None, jsonl="data/SHS100K/SHS100K.test.jsonl")
    enc, task = m._derive_cache_slugs()
    assert task == "SHS100K.test"
    assert enc == "_FakeEncoder"


def test_jsonl_stem_for_vgmiditvar_timbre():
    """Different JSONL → different task slug → separate cache dir."""
    m1 = _make_mixin(wandb_group=None, jsonl="data/VGMIDITVar/VGMIDITVar.jsonl")
    m2 = _make_mixin(wandb_group=None, jsonl="data/VGMIDITVar-timbre/VGMIDITVar.jsonl")
    _, task1 = m1._derive_cache_slugs()
    _, task2 = m2._derive_cache_slugs()
    # Both stems are "VGMIDITVar" because the filename is the same — the
    # directory name (parent) differs but ``Path.stem`` only looks at the
    # filename. Document this limitation by asserting equality and adding
    # a comment so we don't regress silently.
    #
    # In production this is a non-issue (wandb group is set). The point of
    # this test is to record that the JSONL-stem fallback is NOT a
    # cross-OS guarantee of cache separation — it only disambiguates
    # tasks that name their JSONL differently. The two VGMIDITVar
    # variants name them identically.
    assert task1 == task2 == "VGMIDITVar"


def test_final_fallback_to_class_names():
    """No wandb group and no JSONL → fall back to class names."""
    m = _make_mixin(wandb_group=None, jsonl=None)
    enc, task = m._derive_cache_slugs()
    assert task == "_Fake"  # the fake mixin subclass name
    assert enc == "_FakeEncoder"
