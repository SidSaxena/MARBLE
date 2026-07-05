"""Round-trip tests for the frame-level embedding cache (pool_time=False).

The existing emb_cache shape contract (pool_time=True, stores (L, H) per
slice) is exercised by the live sweeps — every cache-enabled config
relies on it. This file tests the new pool_time=False path that stores
(L, T, H), required for keep-time / frame-level probes
(HookTheoryMelody, GTZANBeatTracking, Chords1217, LibriSpeechASR).

Run manually:
    uv run python tests/test_emb_cache_frames.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

# Make the project importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marble.utils.emb_cache import (  # noqa: E402
    EmbeddingCache,
    encoder_tuple_to_pooled,
    encoder_tuple_to_stacked_frames,
    stacked_frames_to_layer_tuple,
    stacked_to_layer_tuple,
)


def test_pool_time_true_round_trip_unchanged():
    """The (L, H) path stores fp16 on disk (default) and upcasts to fp32 on
    load: shape unchanged, values preserved within fp16 rounding."""
    with tempfile.TemporaryDirectory() as td:
        cache = EmbeddingCache(
            encoder_slug="TestEnc",
            task_name="TestTask",
            config_hash="abc12345",
            root=Path(td),
            pool_time=True,
        )
        assert cache.pool_time is True
        clip_ids = ["c0", "c1", "c2"]
        # encoder produced (B, T, H) per layer; pooled to (B, L, H)
        layer_outputs = tuple(torch.randn(3, 5, 8) for _ in range(4))
        pooled = encoder_tuple_to_pooled(layer_outputs)  # (3, 4, 8)
        assert pooled.shape == (3, 4, 8)
        cache.put_batch(clip_ids, pooled)
        assert cache.has_all(clip_ids)
        # Storage contract: fp16 on disk, fp32 on load.
        raw = torch.load(cache.path_for("c0"), weights_only=True)["embedding"]
        assert raw.dtype == torch.float16
        loaded = cache.get_batch(clip_ids)  # (3, 4, 8)
        assert loaded.shape == (3, 4, 8)
        assert loaded.dtype == torch.float32
        # Value preserved within fp16 rounding.
        torch.testing.assert_close(loaded, pooled, atol=5e-3, rtol=1e-2)
        # round-trip via the layer-tuple helper
        tup = stacked_to_layer_tuple(loaded)
        assert len(tup) == 4
        assert all(t.shape == (3, 1, 8) for t in tup), "T must collapse to 1 on hit"


def test_pool_time_false_round_trip_preserves_time():
    """Frame-level: (L, T, H) round-trip retains the time axis."""
    with tempfile.TemporaryDirectory() as td:
        cache = EmbeddingCache(
            encoder_slug="TestEnc",
            task_name="TestTaskFrames",
            config_hash="abc12345",
            root=Path(td),
            pool_time=False,
        )
        assert cache.pool_time is False
        clip_ids = ["c0", "c1"]
        # 2 clips, 4 layers, 7 frames, 8 hidden
        layer_outputs = tuple(torch.randn(2, 7, 8) for _ in range(4))
        stacked = encoder_tuple_to_stacked_frames(layer_outputs)  # (2, 4, 7, 8)
        assert stacked.shape == (2, 4, 7, 8)
        cache.put_batch(clip_ids, stacked)
        assert cache.has_all(clip_ids)
        loaded = cache.get_batch(clip_ids)
        assert loaded.shape == (2, 4, 7, 8), f"got {tuple(loaded.shape)}"
        assert loaded.dtype == torch.float32  # upcast on load
        torch.testing.assert_close(loaded, stacked, atol=5e-3, rtol=1e-2)
        # Layer-tuple unpack: each layer (B, T, H)
        tup = stacked_frames_to_layer_tuple(loaded)
        assert len(tup) == 4
        for i, layer_tensor in enumerate(tup):
            assert layer_tensor.shape == (2, 7, 8), f"layer {i}: {tuple(layer_tensor.shape)}"
            torch.testing.assert_close(layer_tensor, layer_outputs[i], atol=5e-3, rtol=1e-2)


def test_pool_time_false_rejects_2d_put():
    """If a frame-level cache receives (L, H) by mistake, it must complain
    loudly — silent shape-mismatch on hit would silently degrade training."""
    with tempfile.TemporaryDirectory() as td:
        cache = EmbeddingCache(
            encoder_slug="TestEnc",
            task_name="TestRejects",
            config_hash="abc12345",
            root=Path(td),
            pool_time=False,
        )
        try:
            cache.put("c0", torch.randn(4, 8))  # (L, H) — wrong shape
        except ValueError as e:
            assert "L, T, H" in str(e), f"error msg didn't mention expected shape: {e}"
            return
        raise AssertionError("Expected ValueError for 2D put on frame-level cache")


def test_pool_time_true_rejects_3d_put():
    """Symmetric: clip-level cache rejects (L, T, H) input."""
    with tempfile.TemporaryDirectory() as td:
        cache = EmbeddingCache(
            encoder_slug="TestEnc",
            task_name="TestRejects2",
            config_hash="abc12345",
            root=Path(td),
            pool_time=True,
        )
        try:
            cache.put("c0", torch.randn(4, 7, 8))  # (L, T, H) — wrong shape
        except ValueError as e:
            assert "L, H" in str(e), f"error msg didn't mention expected shape: {e}"
            return
        raise AssertionError("Expected ValueError for 3D put on clip-level cache")


def test_config_hash_distinguishes_pool_time():
    """The config hash must include pool_time so the two cache variants
    can't accidentally share a directory and corrupt each other's
    shape expectations."""
    from marble.utils.emb_cache import compute_config_hash

    h_pool = compute_config_hash(
        encoder_model_id="enc-x",
        sample_rate=24000,
        clip_seconds=15.0,
        pipeline_signature="sig",
        pool_time=True,
    )
    h_frame = compute_config_hash(
        encoder_model_id="enc-x",
        sample_rate=24000,
        clip_seconds=15.0,
        pipeline_signature="sig",
        pool_time=False,
    )
    assert h_pool != h_frame, f"config hash collides between pool_time True/False: both = {h_pool}"


def test_frame_cache_meta_records_pool_time():
    """The _meta.json should record pool_time so a human inspecting the
    cache dir can tell which shape is stored."""
    import json

    with tempfile.TemporaryDirectory() as td:
        cache = EmbeddingCache(
            encoder_slug="TestEnc",
            task_name="TestMeta",
            config_hash="abc12345",
            root=Path(td),
            pool_time=False,
            metadata={"sample_rate": 24000},
        )
        meta_path = cache.dir / "_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta.get("pool_time") is False
        assert meta.get("sample_rate") == 24000


if __name__ == "__main__":
    tests = [
        test_pool_time_true_round_trip_unchanged,
        test_pool_time_false_round_trip_preserves_time,
        test_pool_time_false_rejects_2d_put,
        test_pool_time_true_rejects_3d_put,
        test_config_hash_distinguishes_pool_time,
        test_frame_cache_meta_records_pool_time,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
        except Exception as e:
            print(f"  FAIL {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")
