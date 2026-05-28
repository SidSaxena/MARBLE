"""
tests/test_retrieval_metrics_streaming.py

Tests for the streaming (GPU-chunked) variants of the retrieval metric
functions. Two scopes:

* **device="cpu"**: exercises the streaming dispatch path on every test
  environment, with no GPU dependency. Asserts numerical equivalence
  with the materialised-sim path. This is the primary correctness gate.

* **device="cuda"**: gated by ``torch.cuda.is_available()``. Asserts the
  GPU path produces metrics within a small tolerance of CPU
  (argsort tie-breaking on GPU can scramble equal-valued ranks, so the
  test asserts ``approx`` not exact).

We deliberately run the same fixture through both paths and compare,
rather than asserting against hand-computed values — the existing
test_retrieval_metrics.py already pins the absolute numbers.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import pytest
import torch

from marble.utils.retrieval_metrics import (
    _iter_row_orders,
    _iter_row_orders_streaming,
    compute_perpair_map_all,
    compute_perpair_map_all_streaming,
    compute_retrieval_metrics,
    compute_retrieval_metrics_streaming,
)

cuda_available = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available — GPU-path test skipped",
)


# ──────────────────────────────────────────────────────────────────────
# _iter_row_orders_streaming — exhaustive equivalence on CPU device
# ──────────────────────────────────────────────────────────────────────


def _materialised_orders(sim: torch.Tensor) -> torch.Tensor:
    """Reference: stack all yields from the CPU-materialised iter."""
    return torch.stack([o.clone() for _, o in _iter_row_orders(sim)])


def _streaming_orders(embs: torch.Tensor, *, batch: int, device: str) -> torch.Tensor:
    """Stack all yields from the streaming iter."""
    return torch.stack(
        [o.clone() for _, o in _iter_row_orders_streaming(embs, batch=batch, device=device)]
    )


def test_iter_streaming_cpu_matches_materialised_when_no_ties():
    """On embeddings designed to have no sim ties, CPU streaming output
    must exactly match the materialised-sim baseline. Picks an N small
    enough that ties are astronomically unlikely with random gaussian
    embeddings."""
    torch.manual_seed(1234)
    N, H = 50, 32
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    sim = embs @ embs.T

    ref = _materialised_orders(sim)
    out = _streaming_orders(embs, batch=16, device="cpu")
    assert torch.equal(ref, out), (
        f"streaming-CPU orders differ from materialised: {(ref != out).sum().item()} mismatches"
    )


def test_iter_streaming_batch_invariance():
    """Streaming output must be identical regardless of batch size — the
    chunking math must not introduce off-by-one errors at chunk
    boundaries."""
    torch.manual_seed(5678)
    N, H = 100, 16
    embs = torch.randn(N, H)
    out_b1 = _streaming_orders(embs, batch=1, device="cpu")
    out_b13 = _streaming_orders(embs, batch=13, device="cpu")  # awkward divisor
    out_bN = _streaming_orders(embs, batch=N, device="cpu")
    assert torch.equal(out_b1, out_b13), "batch=1 vs batch=13 mismatch"
    assert torch.equal(out_b1, out_bN), "batch=1 vs batch=N mismatch"


def test_iter_streaming_handles_n_zero():
    """N=0 embedding tensor: iterator yields nothing, no crash."""
    embs = torch.empty(0, 32)
    n = sum(1 for _ in _iter_row_orders_streaming(embs, batch=8, device="cpu"))
    assert n == 0


def test_iter_streaming_self_excluded():
    """Self-index never appears in any row's order (the ``-inf`` mask +
    ``[:N-1]`` slice contract holds in the streaming path)."""
    torch.manual_seed(42)
    N = 20
    embs = torch.randn(N, 16)
    for i, order_i in _iter_row_orders_streaming(embs, batch=7, device="cpu"):
        assert order_i.shape == (N - 1,)
        assert i not in order_i.tolist(), f"row {i} contains self"


# ──────────────────────────────────────────────────────────────────────
# compute_retrieval_metrics_streaming — vs materialised at modest N
# ──────────────────────────────────────────────────────────────────────


def test_compute_metrics_streaming_cpu_matches_materialised():
    """All metric values from the streaming path must equal the
    materialised path on the same input. CPU device → no argsort
    tie-difference, so we use exact equality."""
    torch.manual_seed(7)
    N, H = 200, 64
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 12, (N,))

    kwargs = dict(
        recall_ks=(1, 5, 10),
        hit_ks=(1, 5),
        include_r_precision=True,
        include_median_rank=True,
        include_map=True,
        map_at_ks=(1, 5),
        include_mrr=True,
    )
    sim = embs @ embs.T
    ref = compute_retrieval_metrics(sim, work_ids, **kwargs)
    out = compute_retrieval_metrics_streaming(embs, work_ids, device="cpu", batch=37, **kwargs)
    assert set(ref) == set(out)
    for key in ref:
        if math.isnan(ref[key]):
            assert math.isnan(out[key]), f"key {key} ref=nan, streaming={out[key]}"
        else:
            assert ref[key] == pytest.approx(out[key], abs=1e-9), (
                f"key {key}: ref={ref[key]} vs streaming={out[key]}"
            )


def test_compute_metrics_streaming_handles_empty_corpus():
    """N=0 returns the same all-NaN/zero dict as the materialised path."""
    embs = torch.empty(0, 16)
    work_ids = torch.empty(0, dtype=torch.long)
    sim = torch.empty(0, 0)
    kwargs = dict(
        recall_ks=(5,),
        hit_ks=(),
        include_r_precision=True,
        include_median_rank=True,
        include_map=True,
        include_mrr=True,
    )
    ref = compute_retrieval_metrics(sim, work_ids, **kwargs)
    out = compute_retrieval_metrics_streaming(embs, work_ids, device="cpu", **kwargs)
    assert set(ref) == set(out)
    for key in ref:
        if math.isnan(ref[key]):
            assert math.isnan(out[key])
        else:
            assert ref[key] == out[key]


# ──────────────────────────────────────────────────────────────────────
# compute_perpair_map_all_streaming — same comparison
# ──────────────────────────────────────────────────────────────────────


def test_perpair_streaming_cpu_matches_materialised():
    """Per-cell (map, n_queries) from streaming match the materialised
    path on a fixture with multiple works × multiple conditions."""
    torch.manual_seed(29)
    N, H = 40, 16
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 5, (N,)).tolist()
    conditions = [i % 3 for i in range(N)]

    sim = embs @ embs.T
    ref = compute_perpair_map_all(sim, work_ids, conditions)
    out = compute_perpair_map_all_streaming(embs, work_ids, conditions, device="cpu", batch=8)
    assert set(ref) == set(out)
    for cell in ref:
        ap_ref, n_ref = ref[cell]
        ap_out, n_out = out[cell]
        assert n_ref == n_out, f"cell {cell}: n_queries differ {n_ref} vs {n_out}"
        assert ap_ref == pytest.approx(ap_out, abs=1e-9), (
            f"cell {cell}: ap differ {ap_ref} vs {ap_out}"
        )


# ──────────────────────────────────────────────────────────────────────
# GPU-only tests — gated by cuda_available
# ──────────────────────────────────────────────────────────────────────


@cuda_available
def test_iter_streaming_cuda_approximates_cpu():
    """GPU argsort can break ties differently from CPU. For random
    embeddings the tie rate is tiny — assert the index disagreement
    rate is below 1% so the downstream MAP delta stays negligible."""
    torch.manual_seed(2024)
    N, H = 200, 64
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)

    cpu_orders = _streaming_orders(embs, batch=64, device="cpu")
    gpu_orders = _streaming_orders(embs, batch=64, device="cuda")
    n_diff = (cpu_orders != gpu_orders).sum().item()
    total = cpu_orders.numel()
    frac = n_diff / total
    # Under no-tie conditions (random gaussian embeddings) this should
    # be exactly 0; allow up to 1% to absorb rare ties.
    assert frac < 0.01, f"GPU disagreement rate {frac:.4f} > 0.01 ({n_diff}/{total})"


@cuda_available
def test_compute_metrics_streaming_cuda_within_tolerance():
    """End-to-end metric values from GPU within 1e-4 of CPU."""
    torch.manual_seed(7)
    N, H = 200, 64
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 12, (N,))

    kwargs = dict(
        recall_ks=(1, 5, 10),
        hit_ks=(1, 5),
        include_r_precision=True,
        include_median_rank=True,
        include_map=True,
        map_at_ks=(1,),
        include_mrr=True,
    )
    sim = embs @ embs.T
    ref = compute_retrieval_metrics(sim, work_ids, **kwargs)
    out = compute_retrieval_metrics_streaming(embs, work_ids, device="cuda", batch=64, **kwargs)
    for key in ref:
        if math.isnan(ref[key]):
            assert math.isnan(out[key])
        else:
            assert ref[key] == pytest.approx(out[key], abs=1e-4), (
                f"GPU key {key}: ref={ref[key]} vs gpu={out[key]} (delta > 1e-4)"
            )


@cuda_available
def test_perpair_streaming_cuda_within_tolerance():
    torch.manual_seed(29)
    N, H = 40, 16
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 5, (N,)).tolist()
    conditions = [i % 3 for i in range(N)]

    sim = embs @ embs.T
    ref = compute_perpair_map_all(sim, work_ids, conditions)
    out = compute_perpair_map_all_streaming(embs, work_ids, conditions, device="cuda", batch=8)
    for cell in ref:
        ap_ref, n_ref = ref[cell]
        ap_out, n_out = out[cell]
        # n_queries is purely a count — must agree exactly.
        assert n_ref == n_out, f"cell {cell}: n_queries differ"
        # ap may differ in the 4th decimal due to argsort tie-breaking.
        assert ap_ref == pytest.approx(ap_out, abs=1e-4), (
            f"GPU perpair cell {cell}: ref={ap_ref} vs gpu={ap_out}"
        )
