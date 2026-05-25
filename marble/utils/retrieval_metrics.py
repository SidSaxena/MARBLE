"""
marble/utils/retrieval_metrics.py

Pure-function retrieval metrics for ``CoverRetrievalTask`` and any other
probe that ranks files by cosine similarity over a labelled corpus.

All functions operate on tensor inputs only — no Lightning coupling,
no I/O, no plotting. Tests in ``tests/test_retrieval_metrics.py``.

Conventions
-----------
- ``sim``: ``(N, N)`` float tensor of pairwise similarities. The
  diagonal must be set to a sentinel BEFORE calling these functions if
  the caller wants self-similarity excluded; we follow the existing
  convention from ``CoverRetrievalTask._compute_map`` and exclude self
  in-line (``sims_i[i] = -2``) — that means callers can pass the raw
  (or centered) similarity matrix and we do the self-mask here.
- ``work_ids``: ``(N,)`` int / long tensor of group labels. Two items
  are considered "relevant to each other" iff their work_ids match.
- All functions return Python ``float``. ``NaN`` is returned if no
  query has ≥1 relevant other-item (degenerate case — e.g. a corpus
  where every work_id is unique).

Why these specific metrics
--------------------------
For leitmotif-discovery work, the operationally-meaningful question is
"at a feasible review-budget K, will I see the relevant items?" That
makes Recall@K and Hit Rate@K the headline metrics. MAP (already
logged) captures rank quality within the budget; median rank captures
the typical depth needed to surface ANY positive; R-Precision
calibrates K to per-query "natural" budget (= number of relevant
items). Together they form a complete ranking-evaluation suite for
binary relevance. NDCG@K was considered and declined — for binary
relevance it's mathematically close to MAP@K with a different rank
discount, and we don't have graded relevance to justify the extra
metric.
"""

from __future__ import annotations

import torch

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _ranking_order(sim: torch.Tensor) -> torch.Tensor:
    """Return ``(N, N-1)`` long tensor where row i lists OTHER-item
    indices (self excluded) sorted by descending similarity to query i.

    Self-exclusion matters: a naive ``argsort`` over ``(N, N)`` leaves
    the self-row at rank N with sim ≈ 1.0, and ``work_ids[self] ==
    work_ids[self]`` is always True, which would (a) inflate
    ``n_relevant`` by 1, (b) put a spurious True at rank N. We force
    self to the bottom with ``-inf`` then drop the last column.

    Note: this differs subtly from ``CoverRetrievalTask._compute_map``,
    which sets ``sim[i,i] = -2.0`` (NOT -inf) and keeps the self item
    in ``is_rel``. The off-by-one effect is small (~1/N) on the
    reported MAP for large corpora but visible on small unit tests.
    Kept as-is in ``_compute_map`` for backward compatibility with
    existing wandb numbers; the new metrics below use proper
    self-exclusion.
    """
    sim = sim.clone()
    N = sim.size(0)
    sim[range(N), range(N)] = float("-inf")  # force self to the bottom
    order = sim.argsort(descending=True, dim=-1)
    return order[:, :-1]  # drop the last column (always self)


# ──────────────────────────────────────────────────────────────────────
# Public metrics
# ──────────────────────────────────────────────────────────────────────


def recall_at_k(sim: torch.Tensor, work_ids: torch.Tensor, k: int) -> float:
    """Recall@K — average over queries of (# relevant in top-K) / (# relevant total).

    For queries with zero other-relevant items, contribute nothing
    (skipped — they're degenerate). If ALL queries are degenerate
    returns NaN.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.
        k: rank cutoff (1 ≤ k ≤ N − 1). Caller should skip metric
           entirely when k ≥ N.

    Returns:
        Average recall as a Python float in [0, 1].
    """
    N = work_ids.size(0)
    if k <= 0 or k >= N:
        raise ValueError(f"k={k} must be in [1, N-1] where N={N}")
    order = _ranking_order(sim)
    recalls: list[float] = []
    for i in range(N):
        is_rel = work_ids[order[i]] == work_ids[i]
        n_relevant = int(is_rel.sum().item())
        if n_relevant == 0:
            continue
        hits_at_k = int(is_rel[:k].sum().item())
        recalls.append(hits_at_k / n_relevant)
    return float(sum(recalls) / len(recalls)) if recalls else float("nan")


def hit_rate_at_k(sim: torch.Tensor, work_ids: torch.Tensor, k: int) -> float:
    """Hit Rate@K — fraction of queries with ≥1 relevant in the top-K.

    Binary per query; lower variance than Recall@K. "Did the system
    surface at least one true positive in the review budget?"

    Same args / return shape as :func:`recall_at_k`.
    """
    N = work_ids.size(0)
    if k <= 0 or k >= N:
        raise ValueError(f"k={k} must be in [1, N-1] where N={N}")
    order = _ranking_order(sim)
    hits: list[float] = []
    for i in range(N):
        is_rel = work_ids[order[i]] == work_ids[i]
        if int(is_rel.sum().item()) == 0:
            continue
        hits.append(1.0 if bool(is_rel[:k].any().item()) else 0.0)
    return float(sum(hits) / len(hits)) if hits else float("nan")


def median_rank_first_hit(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
    """Median rank (1-indexed) of the first relevant item per query.

    Diagnostic — "how deep into the ranking before I see *anything*?"
    Queries with zero relevant items are skipped.

    Returns:
        Median 1-indexed rank as float (use ``int(round(.))`` if a
        rank integer is desired downstream). NaN if no query has a hit.
    """
    N = work_ids.size(0)
    order = _ranking_order(sim)
    first_ranks: list[float] = []
    for i in range(N):
        is_rel = work_ids[order[i]] == work_ids[i]
        if int(is_rel.sum().item()) == 0:
            continue
        # 1-indexed rank of the first True in is_rel
        first_idx = int(is_rel.nonzero(as_tuple=True)[0][0].item())
        first_ranks.append(float(first_idx + 1))
    if not first_ranks:
        return float("nan")
    # NB: torch.Tensor.median() returns the lower-middle value for
    # even-length sequences (not the average of the two middles).
    # Use quantile(0.5) for the standard numpy/IR-style median.
    return float(torch.tensor(first_ranks, dtype=torch.float).quantile(0.5).item())


def r_precision(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
    """R-Precision — precision at K = (number of relevant items for this query).

    Self-calibrating K per query: each query's natural cutoff is the
    size of its relevant set. Standard in IR literature; sidesteps the
    "which K?" question entirely.

    Queries with zero relevant items are skipped.
    """
    N = work_ids.size(0)
    order = _ranking_order(sim)
    rps: list[float] = []
    for i in range(N):
        is_rel = work_ids[order[i]] == work_ids[i]
        r = int(is_rel.sum().item())
        if r == 0:
            continue
        # Precision at rank R = (# relevant in top R) / R
        hits = int(is_rel[:r].sum().item())
        rps.append(hits / r)
    return float(sum(rps) / len(rps)) if rps else float("nan")
