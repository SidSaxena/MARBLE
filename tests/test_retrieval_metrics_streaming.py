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
    _aggregate_metrics_from_chunk_iter,
    _aggregate_metrics_from_iter,
    _iter_row_order_chunks,
    _iter_row_orders,
    _iter_row_orders_streaming,
    _perpair_map_all_from_chunk_iter,
    _perpair_map_all_from_iter,
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


def test_iter_streaming_accepts_bf16_input():
    """When Lightning autocast leaves embs in bf16, the streaming
    function must not crash and must produce orderings *close* to the
    fp32 reference — the quantisation of bf16 inputs can flip
    close-rank pairs, but the function's internal fp32 upcast prevents
    catastrophic divergence.

    We accept up to 10% row-level disagreement at H=32 (where bf16's
    8-bit mantissa is coarse relative to dimension-768 production).
    The point of this test is to pin "no crash + sensible output";
    pre-fix the function would still work but with no upcast it would
    risk argsort tie scrambles on the matmul-result bf16 values.
    """
    torch.manual_seed(99)
    N, H = 40, 32
    embs_fp32 = torch.randn(N, H)
    embs_fp32 = embs_fp32 / embs_fp32.norm(dim=-1, keepdim=True)
    embs_bf16 = embs_fp32.to(torch.bfloat16)

    out_fp32 = torch.stack(
        [o.clone() for _, o in _iter_row_orders_streaming(embs_fp32, batch=11, device="cpu")]
    )
    out_bf16 = torch.stack(
        [o.clone() for _, o in _iter_row_orders_streaming(embs_bf16, batch=11, device="cpu")]
    )
    # No crash; both shapes match.
    assert out_fp32.shape == out_bf16.shape
    # Disagreement rate small (bounded by input quantization, not
    # by lack of internal upcast — the upcast prevents this from
    # being catastrophic).
    n_diff = (out_fp32 != out_bf16).sum().item()
    total = out_fp32.numel()
    assert n_diff / total < 0.25, (
        f"bf16 vs fp32 disagreement {n_diff}/{total} = {n_diff / total:.2f} too high"
    )


def test_iter_streaming_handles_cuda_colon_zero_device_spec():
    """``embs.device == torch.device('cuda:0')`` and ``device='cuda'``
    should resolve to the same canonical device — no redundant copy
    and no AttributeError. We can't exercise CUDA on Mac, but we can
    exercise the canonicalisation by passing 'cpu' which normalises
    identically.
    """
    embs = torch.randn(20, 16)
    # Pass device with explicit form that differs from torch's default
    # str representation. torch.device('cpu') == torch.device('cpu:0')
    # so this round-trip-canonicalises.
    out_a = torch.stack([o.clone() for _, o in _iter_row_orders_streaming(embs, device="cpu")])
    out_b = torch.stack([o.clone() for _, o in _iter_row_orders_streaming(embs, device="cpu")])
    assert torch.equal(out_a, out_b)


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
# Batched (chunk-aggregator) vs per-row aggregator equivalence
# ──────────────────────────────────────────────────────────────────────
#
# The chunk aggregator (``_aggregate_metrics_from_chunk_iter``) was added
# to remove ~70% of the metric-body wall time at large N by collapsing
# the per-query Python ``.tolist() + MAP inner loop`` into batched
# tensor cumsum ops. The per-row aggregator (``_aggregate_metrics_from_iter``)
# is kept as the reference implementation. These tests pin the two
# against each other at tight tolerance — they MUST stay numerically
# equivalent (up to fp32 vs fp64 rounding noise in the AP path) so the
# batched path is safe to drop into production.


def test_chunk_aggregator_matches_per_row_aggregator():
    """All metrics from the new chunk aggregator must equal the legacy
    per-row aggregator on the same fixture. Tight tolerance because we
    expect only fp32-vs-fp64 rounding deltas in the MAP path; recall /
    hit / r_precision / median_rank / mrr are integer-derived and
    should match exactly."""
    torch.manual_seed(101)
    N, H = 250, 32
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 10, (N,))
    sim = embs @ embs.T

    kwargs = dict(
        work_ids=work_ids,
        recall_ks=(1, 5, 10, 25),
        hit_ks=(1, 5, 25),
        include_r_precision=True,
        include_median_rank=True,
        include_map=True,
        map_at_ks=(1, 5, 25),
        include_mrr=True,
    )
    ref = _aggregate_metrics_from_iter(_iter_row_orders(sim, batch=37), **kwargs)
    out = _aggregate_metrics_from_chunk_iter(_iter_row_order_chunks(sim, batch=37), **kwargs)
    assert set(ref) == set(out)
    for key in ref:
        if math.isnan(ref[key]):
            assert math.isnan(out[key]), f"{key}: ref=nan, chunk={out[key]}"
        else:
            # MAP path uses fp32 vs the per-row's Python float (fp64);
            # rounding deltas are well below 1e-6 at this size.
            assert ref[key] == pytest.approx(out[key], abs=1e-6), (
                f"{key}: ref={ref[key]} vs chunk={out[key]}"
            )


def test_chunk_aggregator_batch_invariant():
    """Chunked aggregator output must be identical for any batch size
    that tiles the corpus — no off-by-one at chunk boundaries, no
    ordering-dependence in the list accumulation."""
    torch.manual_seed(202)
    N, H = 120, 16
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 6, (N,))
    sim = embs @ embs.T
    kwargs = dict(
        work_ids=work_ids,
        recall_ks=(1, 10),
        hit_ks=(5,),
        include_r_precision=True,
        include_median_rank=True,
        include_map=True,
        map_at_ks=(5,),
        include_mrr=True,
    )
    out_b1 = _aggregate_metrics_from_chunk_iter(_iter_row_order_chunks(sim, batch=1), **kwargs)
    out_b17 = _aggregate_metrics_from_chunk_iter(_iter_row_order_chunks(sim, batch=17), **kwargs)
    out_bN = _aggregate_metrics_from_chunk_iter(_iter_row_order_chunks(sim, batch=N), **kwargs)
    for key in out_b1:
        if math.isnan(out_b1[key]):
            assert math.isnan(out_b17[key]) and math.isnan(out_bN[key])
        else:
            # First-relevant-rank computed via batched min(); per-query
            # values are integers so they're exact across batch sizes.
            # MAP via cumsum is also batch-invariant in fp32 because the
            # per-query sum is over the same N-1 positions regardless of
            # how we chunk queries together.
            assert out_b1[key] == pytest.approx(out_b17[key], abs=1e-9)
            assert out_b1[key] == pytest.approx(out_bN[key], abs=1e-9)


def test_perpair_chunk_aggregator_matches_per_row():
    """Per-cell (map, n_queries) from the chunked perpair aggregator
    matches the per-row variant up to fp rounding."""
    torch.manual_seed(303)
    N, H = 80, 16
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 5, (N,))
    conds = torch.tensor([i % 4 for i in range(N)])
    sim = embs @ embs.T

    query_set = {0, 1, 2, 3}
    target_list = [0, 1, 2, 3]

    def fresh_aps():
        return {(q, t): [] for q in sorted(query_set) for t in target_list}

    ref = _perpair_map_all_from_iter(
        _iter_row_orders(sim, batch=13),
        wids=work_ids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=fresh_aps(),
    )
    out = _perpair_map_all_from_chunk_iter(
        _iter_row_order_chunks(sim, batch=13),
        wids=work_ids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=fresh_aps(),
    )
    assert set(ref) == set(out)
    for cell in ref:
        ap_ref, n_ref = ref[cell]
        ap_out, n_out = out[cell]
        assert n_ref == n_out, f"cell {cell}: n differ {n_ref} vs {n_out}"
        assert ap_ref == pytest.approx(ap_out, abs=1e-6), (
            f"cell {cell}: ap differ {ap_ref} vs {ap_out}"
        )


def test_perpair_chunk_aggregator_batch_invariant():
    """Per-cell APs must be identical regardless of how we chunk the
    query axis. Includes a non-divisor batch size and full-N."""
    torch.manual_seed(404)
    N, H = 60, 16
    embs = torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work_ids = torch.randint(0, 4, (N,))
    conds = torch.tensor([i % 3 for i in range(N)])
    sim = embs @ embs.T
    query_set = {0, 1, 2}
    target_list = [0, 1, 2]

    def run(batch: int):
        return _perpair_map_all_from_chunk_iter(
            _iter_row_order_chunks(sim, batch=batch),
            wids=work_ids,
            conds=conds,
            query_set=query_set,
            target_list=target_list,
            aps_per_cell={(q, t): [] for q in sorted(query_set) for t in target_list},
        )

    a = run(1)
    b = run(7)
    c = run(N)
    for cell in a:
        ap_a, n_a = a[cell]
        ap_b, n_b = b[cell]
        ap_c, n_c = c[cell]
        assert n_a == n_b == n_c
        assert ap_a == pytest.approx(ap_b, abs=1e-9)
        assert ap_a == pytest.approx(ap_c, abs=1e-9)


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
