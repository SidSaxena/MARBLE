"""Regression test: frozen encoder submodules stay in eval after parent .train().

Bug history: all v2 encoders call ``self.model.eval()`` in ``__init__`` when
``train_mode='freeze'``. But ``nn.Module.train()`` propagates recursively
to children, and Lightning calls ``.train()`` on the parent LightningModule
at the start of every epoch — silently undoing the frozen-state eval.

The fix overrides ``train()`` on each encoder to re-apply ``.eval()`` to
the frozen submodule(s) after super().train() propagation. This test
verifies that, for every encoder, calling ``.train()`` on a freeze-mode
instance leaves the inner submodule in eval mode.

Tests are wrapped in try/except so missing model weights (e.g. an encoder
not yet downloaded) yields SKIP rather than FAIL — the test is meant to
run on dev machines with weights pre-cached, not as a network-dependent
CI test.

Run manually:
    uv run python tests/test_encoder_eval_under_train_propagation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class SkipTest(Exception):
    """Raise to mark a test as skipped (missing weights / optional dep)."""


def _assert_frozen_stays_eval(encoder, name: str, extra_submodules=()):
    """Common assertion harness: parent.train(True) must leave self.model
    (and any extra frozen submodules) in eval mode."""
    encoder.train()  # explicit train(True) — mimics Lightning's per-epoch call
    assert encoder.model.training is False, (
        f"{name}.model.training is True after .train() — fix didn't take effect"
    )
    for sub_name in extra_submodules:
        sub = getattr(encoder, sub_name)
        assert sub.training is False, (
            f"{name}.{sub_name}.training is True after .train() — extra frozen "
            "submodule should also stay in eval"
        )
    # And eval() — sanity check that the override still respects an explicit
    # eval() call (a no-op for the frozen submodule, but shouldn't blow up).
    encoder.eval()
    assert encoder.model.training is False
    for sub_name in extra_submodules:
        assert getattr(encoder, sub_name).training is False


def test_muq_freeze_stays_eval():
    try:
        from marble.encoders.MuQ.model import MuQ_Encoder

        m = MuQ_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"MuQ_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "MuQ_Encoder")


def test_mert_95m_freeze_stays_eval():
    try:
        from marble.encoders.MERT.model import MERT_v1_95M_Encoder

        m = MERT_v1_95M_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"MERT_v1_95M_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "MERT_v1_95M_Encoder")


def test_mert_330m_freeze_stays_eval():
    """Subclass inherits the train() override — verify it works through MRO."""
    try:
        from marble.encoders.MERT.model import MERT_v1_330M_Encoder

        m = MERT_v1_330M_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"MERT_v1_330M_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "MERT_v1_330M_Encoder")


def test_omar_rq_freeze_stays_eval():
    try:
        from marble.encoders.OMAR_RQ.model import OMARRQ_Multifeature25hz_Encoder

        m = OMARRQ_Multifeature25hz_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(
            f"OMARRQ_Multifeature25hz_Encoder unavailable: {type(e).__name__}: {e}"
        ) from e
    _assert_frozen_stays_eval(m, "OMARRQ_Multifeature25hz_Encoder")


def test_clamp3_freeze_stays_eval():
    """CLaMP3 has TWO frozen submodules: self.model AND self.mert_encoder."""
    try:
        from marble.encoders.CLaMP3.model import CLaMP3_Encoder

        m = CLaMP3_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"CLaMP3_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "CLaMP3_Encoder", extra_submodules=("mert_encoder",))
    # Also confirm the inner MERT's own override kept its inner model eval.
    assert m.mert_encoder.model.training is False, (
        "CLaMP3.mert_encoder.model.training is True — inner MERT's override "
        "didn't keep its frozen submodule in eval"
    )


def test_clamp3_symbolic_freeze_stays_eval():
    """CLaMP3_Symbolic_Encoder is a subclass — verify inheritance works."""
    try:
        from marble.encoders.CLaMP3.model import CLaMP3_Symbolic_Encoder

        m = CLaMP3_Symbolic_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"CLaMP3_Symbolic_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "CLaMP3_Symbolic_Encoder", extra_submodules=("mert_encoder",))


def test_dasheng_freeze_stays_eval():
    try:
        from marble.encoders.DaSheng.model import DaSheng_Encoder

        # Use the smallest variant to keep the test light.
        m = DaSheng_Encoder(model_size="base", train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"DaSheng_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "DaSheng_Encoder")


def test_muqmulan_freeze_stays_eval():
    try:
        from marble.encoders.MuQMuLan.model import MuQMuLan_Encoder

        m = MuQMuLan_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"MuQMuLan_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "MuQMuLan_Encoder")


def test_musicfm_freeze_stays_eval():
    try:
        from marble.encoders.MusicFM.model import MusicFM_Encoder

        m = MusicFM_Encoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"MusicFM_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "MusicFM_Encoder")


def test_qwen2_audio_freeze_stays_eval():
    try:
        from marble.encoders.Qwen2AudioInstructEncoder.model import (
            Qwen2AudioInstructEncoder,
        )

        m = Qwen2AudioInstructEncoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"Qwen2AudioInstructEncoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "Qwen2AudioInstructEncoder")


def test_qwen25_omni_freeze_stays_eval():
    try:
        from marble.encoders.Qwen2_5OmniEncoder.model import Qwen2_5OmniEncoder

        m = Qwen2_5OmniEncoder(train_mode="freeze")
    except Exception as e:
        raise SkipTest(f"Qwen2_5OmniEncoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "Qwen2_5OmniEncoder")


def test_xcodec_always_frozen_stays_eval():
    """Xcodec has no train_mode kwarg — it's unconditionally frozen.
    The override forces .eval() with no train_mode check."""
    try:
        from marble.encoders.Xcodec.model import Xcodec_Encoder

        m = Xcodec_Encoder()
    except Exception as e:
        raise SkipTest(f"Xcodec_Encoder unavailable: {type(e).__name__}: {e}") from e
    _assert_frozen_stays_eval(m, "Xcodec_Encoder")


def test_non_freeze_modes_still_train():
    """Sanity check the inverse: train_mode='full' or 'lora' should leave
    self.model in TRAIN mode after .train() — the override must not over-
    fire and pin everything to eval."""
    try:
        from marble.encoders.MuQ.model import MuQ_Encoder

        m = MuQ_Encoder(train_mode="full")
    except Exception as e:
        raise SkipTest(f"MuQ_Encoder (full) unavailable: {type(e).__name__}: {e}") from e
    m.train()
    assert m.model.training is True, (
        "MuQ_Encoder(train_mode='full').train() left self.model in eval mode "
        "— the freeze guard is misfiring"
    )
    m.eval()
    assert m.model.training is False, (
        "MuQ_Encoder(train_mode='full').eval() didn't put self.model in eval"
    )


if __name__ == "__main__":
    tests = [
        test_muq_freeze_stays_eval,
        test_mert_95m_freeze_stays_eval,
        test_mert_330m_freeze_stays_eval,
        test_omar_rq_freeze_stays_eval,
        test_clamp3_freeze_stays_eval,
        test_clamp3_symbolic_freeze_stays_eval,
        test_dasheng_freeze_stays_eval,
        test_muqmulan_freeze_stays_eval,
        test_musicfm_freeze_stays_eval,
        test_qwen2_audio_freeze_stays_eval,
        test_qwen25_omni_freeze_stays_eval,
        test_xcodec_always_frozen_stays_eval,
        test_non_freeze_modes_still_train,
    ]
    passed = failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
            passed += 1
        except SkipTest as e:
            print(f"  SKIP {t.__name__}: {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (of {len(tests)})")
    if passed == 0:
        print("ERROR: zero tests actually ran — every encoder was unavailable.")
        sys.exit(2)
    if failed:
        sys.exit(1)
