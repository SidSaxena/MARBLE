"""RAM-memoization layer of EmbeddingCache (bounded LRU, main-process only)."""

import pickle

import torch

from marble.utils.emb_cache import EmbeddingCache


def _cache(tmp_path, cap=None):
    kw = dict(encoder_slug="e", task_name="t", config_hash="h", root=tmp_path)
    if cap is not None:
        kw["ram_cache_bytes"] = cap
    return EmbeddingCache(**kw)


def test_ram_memoization_returns_identical(tmp_path):
    c = _cache(tmp_path)
    t = torch.randn(13, 1024)
    c.put("a", t)
    first = c.get("a")  # disk -> populates RAM
    second = c.get("a")  # RAM hit
    assert second is first  # served from RAM, not re-loaded
    assert first.dtype == torch.float32  # upcast from fp16 on load
    torch.testing.assert_close(first, t, atol=5e-3, rtol=1e-2)  # within fp16 rounding


def test_ram_cap_evicts_lru(tmp_path):
    t = torch.randn(13, 1024)
    nbytes = t.element_size() * t.nelement()
    c = _cache(tmp_path, cap=nbytes + nbytes // 2)  # room for ~1 item
    for i in range(3):
        c.put(f"c{i}", torch.randn(13, 1024))
    for i in range(3):
        c.get(f"c{i}")  # each load evicts the older one
    assert list(c._ram.keys()) == ["c2"]


def test_ram_disabled_with_zero_cap(tmp_path):
    c = _cache(tmp_path, cap=0)
    c.put("a", torch.randn(2, 4))
    c.get("a")
    assert len(c._ram) == 0  # nothing memoized


def test_getstate_drops_ram_for_workers(tmp_path):
    c = _cache(tmp_path)
    c.put("a", torch.randn(2, 4))
    c.get("a")
    assert len(c._ram) == 1
    state = c.__getstate__()
    assert "_ram" not in state and "_lock" not in state
    c2 = pickle.loads(pickle.dumps(c))  # simulate DataLoader worker spawn
    assert len(c2._ram) == 0
