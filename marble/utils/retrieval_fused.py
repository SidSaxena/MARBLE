"""Fused single-pass retrieval evaluation (GPU-resident aggregation).

``CoverRetrievalTask.on_test_epoch_end`` historically ran up to four separate
full ``N×N`` streaming passes over the *same* centered embeddings — base
metrics, the per-condition MAP grid, the variation-controlled grid, and the
score-distribution dump — each recomputing the sim + argsort and shipping
every ``(B, N-1)`` order chunk to the CPU (~843 MB/chunk → ~85 GB PCIe per
pass at N≈103k), where all aggregation ran memory-bandwidth-bound at ~60 GB/s.

:func:`fused_retrieval_pass` computes **one** chunked sim + argsort on the
requested device and feeds every requested aggregator *on that device*
(the aggregators' per-chunk math is pure gather/cumsum/scatter tensor work;
only per-query scalars cross back to host). At N≈103k this turns a ~50 min
CPU-bound metric stage into a few minutes of GPU work.

Numerical equivalence with the individual passes is pinned by
``tests/test_retrieval_fused.py`` (CPU) and was validated at full scale
against the MuQ-L11 VGMIDITVar-timbre audit numbers.
"""

from __future__ import annotations

import torch

from marble.utils.retrieval_metrics import (
    BaseMetricChunkAggregator,
    PerPairGridChunkAggregator,
    _iter_sim_order_chunks,
)
from marble.utils.retrieval_scores import RetrievalScoreAccumulator


def fused_retrieval_pass(
    embs: torch.Tensor,
    work_ids: torch.Tensor,
    *,
    conditions: torch.Tensor | list[int] | None = None,
    variations: torch.Tensor | list[int] | None = None,
    base_kwargs: dict | None = None,
    with_grid: bool = True,
    with_varctl: bool = True,
    with_scores: bool = True,
    score_n_bins: int = 50,
    device: str = "cuda",
    batch: int = 1024,
) -> dict:
    """One streaming pass over ``embs`` computing every requested aggregate.

    Args:
        embs: ``(N, H)`` L2-normalised embeddings (any device; moved once).
        work_ids: ``(N,)`` group labels.
        conditions: ``(N,)`` per-item condition labels (e.g. gm_program).
            ``None`` disables the grids and per-cell scores.
        variations: ``(N,)`` within-work variation ids. ``None`` disables the
            varctl grid and the twin split in the score dump.
        base_kwargs: kwargs for the base metric suite — same keys as
            ``compute_retrieval_metrics`` (recall_ks, hit_ks,
            include_r_precision, include_median_rank, include_map,
            map_at_ks, include_mrr). ``None`` → skip base metrics.
        with_grid / with_varctl / with_scores: individual toggles (each also
            requires the data they need — conditions / variations).
        score_n_bins: histogram resolution for the score dump.
        device: compute device for sim/argsort AND aggregation.
        batch: query rows per chunk.

    Returns dict with keys:
        ``base``        — metric dict (or ``None``),
        ``grid``        — ``{(q, t): (map, n)}`` confounded grid (or ``None``),
        ``grid_varctl`` — same with same-(work, variation) twins masked
                          from gallery+relevance (or ``None``),
        ``scores``      — ``RetrievalScoreAccumulator.result()`` (or ``None``).
    """
    work_t = work_ids if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
    cond_t = None
    if conditions is not None:
        cond_t = conditions if isinstance(conditions, torch.Tensor) else torch.tensor(conditions)
    var_t = None
    if variations is not None:
        var_t = variations if isinstance(variations, torch.Tensor) else torch.tensor(variations)

    base_agg = BaseMetricChunkAggregator(work_t, **base_kwargs) if base_kwargs else None

    grid_agg = None
    varctl_agg = None
    if cond_t is not None:
        unique_conds = sorted({int(c) for c in torch.unique(cond_t).tolist() if int(c) != -1})
        if unique_conds and with_grid:
            grid_agg = PerPairGridChunkAggregator(
                wids=work_t,
                conds=cond_t,
                query_set=set(unique_conds),
                target_list=unique_conds,
                aps_per_cell={(q, t): [] for q in unique_conds for t in unique_conds},
            )
        if unique_conds and with_varctl and var_t is not None:
            varctl_agg = PerPairGridChunkAggregator(
                wids=work_t,
                conds=cond_t,
                query_set=set(unique_conds),
                target_list=unique_conds,
                aps_per_cell={(q, t): [] for q in unique_conds for t in unique_conds},
                vars=var_t,
            )

    score_acc = None
    if with_scores:
        score_acc = RetrievalScoreAccumulator(work_t, cond_t, var_t, n_bins=score_n_bins)

    want_sim = score_acc is not None
    for start, sim_chunk, order_chunk in _iter_sim_order_chunks(
        embs, batch=batch, device=device, keep_on_device=True, want_sim=want_sim
    ):
        if base_agg is not None:
            base_agg.update(start, order_chunk)
        if grid_agg is not None:
            grid_agg.update(start, order_chunk)
        if varctl_agg is not None:
            varctl_agg.update(start, order_chunk)
        if score_acc is not None:
            b = order_chunk.size(0)
            score_acc.update(torch.arange(start, start + b), sim_chunk)

    return {
        "base": base_agg.result() if base_agg is not None else None,
        "grid": grid_agg.result() if grid_agg is not None else None,
        "grid_varctl": varctl_agg.result() if varctl_agg is not None else None,
        "scores": score_acc.result() if score_acc is not None else None,
    }
