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

import math
from collections.abc import Iterator

import numpy as np
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

    As of audit-cleanup-2 (commit ac121f0), ``CoverRetrievalTask._compute_map``,
    ``_map_at_k``, and ``_mrr`` use this same ``-inf`` + last-column-drop
    pattern. Earlier those used a ``-2.0`` finite sentinel that left
    self at rank N with ``is_rel == True``, inflating ``n_relevant`` by 1
    and adding a spurious hit. The bias was ``1/(n_true+1)`` per query —
    not ``~1/N`` as previously documented here — i.e. ~50% for Covers80/
    SHS100K (1 relevant per query) and ~12.5% for VGMIDITVar-timbre
    (7 relevants). See ``docs/benchmarking_methodology.md`` for the
    pre-fix vs post-fix comparability rules.

    .. warning::
       This function materialises the full ``(N, N-1)`` int64 ranking
       tensor (~84 GB for VGMIDITVar-timbre at N=102 960). It is kept
       for tests and for small corpora (Covers80, SHS100K). Production
       paths on large corpora must use :func:`compute_retrieval_metrics`
       or iterate :func:`_iter_row_orders` directly — both sort in
       bounded-memory row batches.
    """
    rows = [order_i.clone() for _, order_i in _iter_row_orders(sim)]
    return torch.stack(rows, dim=0) if rows else torch.empty(0, 0, dtype=torch.long)


def _iter_row_orders(sim: torch.Tensor, *, batch: int = 2048) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(i, row_order)`` for each query row, self excluded.

    Each ``row_order`` is a ``(N-1,)`` long tensor giving the indices of
    OTHER items sorted by descending similarity to query ``i``. Self is
    forced to the bottom with ``-inf`` and then dropped — the same
    pattern ``CoverRetrievalTask._compute_map`` uses (audit-2 #6).

    Memory: rows are sorted in chunks of ``batch`` at a time. Each chunk
    allocates a ``(batch, N)`` float copy (~412 MB at batch=2048,
    N=102 960) plus a ``(batch, N)`` int64 argsort output (~1.7 GB).
    Both are released between chunks. This bounds peak working memory
    irrespective of N — the alternative ``(N, N)`` argsort output that
    :func:`_ranking_order` (and the original ``order = …`` precompute in
    ``probe.py``) produces is **84 GB at N=102 960** and cannot fit in
    RAM on a 32-GB-class machine.

    The yielded tensor is a view into the active chunk; consume it in
    the loop body before the next iteration. The standard
    ``for i, order_i in _iter_row_orders(sim): …`` pattern is safe.

    Args:
        sim: ``(N, N)`` similarity matrix on CPU (or any device — the
            chunk is moved nowhere).
        batch: chunk size in rows. Default 2048 keeps peak working
            memory ≲ 2 GB at N≈100 k.
    """
    N = sim.size(0)
    if N == 0:
        return
    for start in range(0, N, batch):
        end = min(start + batch, N)
        chunk = sim[start:end].clone()
        row_idx = torch.arange(end - start)
        col_idx = torch.arange(start, end)
        chunk[row_idx, col_idx] = float("-inf")
        # (end-start, N) int64 — released when chunk goes out of scope.
        order_chunk = chunk.argsort(descending=True, dim=-1)[:, : N - 1]
        for j in range(end - start):
            yield start + j, order_chunk[j]
        del chunk, order_chunk


def _iter_row_orders_streaming(
    embs: torch.Tensor,
    *,
    batch: int = 1024,
    device: str = "cuda",
) -> Iterator[tuple[int, torch.Tensor]]:
    """Like :func:`_iter_row_orders` but never materialises the full
    (N, N) similarity matrix.

    Computes sim row-chunks on demand on ``device`` via
    ``embs[start:end] @ embs.T``, masks self-diagonals to ``-inf``,
    argsorts on ``device``, then transfers the (batch, N-1) int64
    order tensor back to CPU and yields per-query rows.

    Designed for hardware where the full ``(N, N)`` sim doesn't fit in
    RAM but ``(N, H)`` embeddings do — e.g. VGMIDITVar-timbre at
    N=102 960 produces a 42 GB sim that overflows into Windows pagefile
    on a 32 GB-RAM machine. With this generator only the per-file
    embeddings (~316 MB at H=768) live on GPU + a ~3 GB transient
    chunk per iteration; the (N, N) sim never exists anywhere.

    Args:
        embs: ``(N, H)`` per-file embeddings on CPU (or already on the
            target device — we move/clone as needed). Float32.
        batch: chunk size in rows. Default 1024 keeps peak GPU memory
            below ~3 GB at H=1024, N=100k.
        device: where to run matmul + argsort. ``"cuda"`` is the win
            case; ``"cpu"`` is a fallback that's still useful because
            it avoids the full-sim materialisation (lower peak RAM
            than the materialised-sim path, at the cost of recomputing
            chunks instead of writing once).

    Yields:
        ``(i, order_i)`` where ``order_i`` is a ``(N-1,)`` int64 CPU
        tensor — the ranking of OTHER items by descending similarity
        to query ``i``, self excluded.

    Note:
        Argsort tie-breaking on GPU and CPU can differ for items with
        equal similarity. For real embeddings this is rare (close ties
        are coincidental, not structural) and the MAP delta is < 1e-4.
        Tests assert tolerance, not exact equality.
    """
    N = embs.size(0)
    if N == 0:
        return
    embs_dev = embs.to(device) if str(embs.device) != device else embs
    for start in range(0, N, batch):
        end = min(start + batch, N)
        # Compute (B, N) sim chunk on-device. Float32 throughout —
        # bf16 would risk argsort tie-instability and the matmul work
        # at this batch is sub-second on a modern GPU at fp32 anyway.
        chunk = embs_dev[start:end] @ embs_dev.T
        # Mask self-diagonals with -inf so the self position sorts to
        # the bottom and we can drop it via the [:N-1] slice below.
        row_idx = torch.arange(end - start, device=chunk.device)
        col_idx = torch.arange(start, end, device=chunk.device)
        chunk[row_idx, col_idx] = float("-inf")
        # (B, N) int64 → (B, N-1) after dropping the self slot.
        order_chunk = chunk.argsort(descending=True, dim=-1)[:, : N - 1]
        # Bring indices back to CPU so the caller's metric loop body
        # (work_ids indexing, .sum().item(), python list ops) doesn't
        # have to context-switch. Transfer of (B, N-1) int64 is
        # ~845 MB at B=1024, N=100k — ~50 ms over PCIe 4.0 x16.
        order_chunk_cpu = order_chunk.cpu()
        del chunk, order_chunk
        for j in range(end - start):
            yield start + j, order_chunk_cpu[j]
        del order_chunk_cpu


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

    Note:
        Each call iterates :func:`_iter_row_orders` once. When computing
        multiple metrics on the same ``sim`` use
        :func:`compute_retrieval_metrics` to amortise the row sorts
        across all metrics in a single pass.
    """
    N = work_ids.size(0)
    if k <= 0 or k >= N:
        raise ValueError(f"k={k} must be in [1, N-1] where N={N}")
    recalls: list[float] = []
    for i, order_i in _iter_row_orders(sim):
        is_rel = work_ids[order_i] == work_ids[i]
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

    Same args / return shape as :func:`recall_at_k`. See
    :func:`compute_retrieval_metrics` to share row sorts across metrics.
    """
    N = work_ids.size(0)
    if k <= 0 or k >= N:
        raise ValueError(f"k={k} must be in [1, N-1] where N={N}")
    hits: list[float] = []
    for i, order_i in _iter_row_orders(sim):
        is_rel = work_ids[order_i] == work_ids[i]
        if int(is_rel.sum().item()) == 0:
            continue
        hits.append(1.0 if bool(is_rel[:k].any().item()) else 0.0)
    return float(sum(hits) / len(hits)) if hits else float("nan")


def median_rank_first_hit(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
    """Median rank (1-indexed) of the first relevant item per query.

    Diagnostic — "how deep into the ranking before I see *anything*?"
    Queries with zero relevant items are skipped.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.

    Returns:
        Median 1-indexed rank as float (use ``int(round(.))`` if a
        rank integer is desired downstream). NaN if no query has a hit.
    """
    first_ranks: list[float] = []
    for i, order_i in _iter_row_orders(sim):
        is_rel = work_ids[order_i] == work_ids[i]
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

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.

    Queries with zero relevant items are skipped.
    """
    rps: list[float] = []
    for i, order_i in _iter_row_orders(sim):
        is_rel = work_ids[order_i] == work_ids[i]
        r = int(is_rel.sum().item())
        if r == 0:
            continue
        # Precision at rank R = (# relevant in top R) / R
        hits = int(is_rel[:r].sum().item())
        rps.append(hits / r)
    return float(sum(rps) / len(rps)) if rps else float("nan")


# ──────────────────────────────────────────────────────────────────────
# Single-pass bundle (use this from probe.py to avoid N argsort passes)
# ──────────────────────────────────────────────────────────────────────


def compute_retrieval_metrics(
    sim: torch.Tensor,
    work_ids: torch.Tensor,
    *,
    recall_ks: list[int] | tuple[int, ...] = (10,),
    hit_ks: list[int] | tuple[int, ...] = (),
    include_r_precision: bool = True,
    include_median_rank: bool = True,
    include_map: bool = False,
    map_at_ks: list[int] | tuple[int, ...] = (),
    include_mrr: bool = False,
    batch: int = 2048,
) -> dict[str, float]:
    """Compute the full retrieval-metric suite in a single batched pass.

    Runs :func:`_iter_row_orders` exactly once and aggregates every
    requested metric from the same per-query rankings. Memory is bounded
    by ``batch`` (see :func:`_iter_row_orders`) so this is the only
    metric-suite entry point that is safe for N ≳ 25 000.

    Callers like ``CoverRetrievalTask.on_test_epoch_end`` should prefer
    this over invoking ``recall_at_k`` / ``r_precision`` /
    ``median_rank_first_hit`` / ``hit_rate_at_k`` and the probe's
    ``_compute_map`` / ``_map_at_k`` / ``_mrr`` independently — each
    re-runs the row sorts. Folding the MAP family in here turns a
    20-minute extended pass over VGMIDITVar-timbre into ~7 min.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.
        recall_ks: iterable of K values for ``recall@K``. K ≥ N is
            silently dropped.
        hit_ks: iterable of K values for ``hit_rate@K``. K ≥ N is
            silently dropped.
        include_r_precision: emit ``r_precision`` key.
        include_median_rank: emit ``median_rank`` key.
        include_map: emit ``map`` key — full Mean Average Precision
            over all ranks, numerically identical to
            ``CoverRetrievalTask._compute_map``.
        map_at_ks: iterable of K values for ``map@K``. K ≥ N is
            silently dropped. Matches ``_map_at_k``'s non-standard
            normalisation: divides AP@K by total ``n_relevant`` (not
            ``min(K, n_relevant)``) — kept for back-compat with the
            existing ``test/map@1`` wandb key.
        include_mrr: emit ``mrr`` key — mean reciprocal rank of first
            relevant. Matches ``_mrr``.
        batch: forwarded to :func:`_iter_row_orders`.

    Returns:
        Dict with keys ``recall@K`` / ``hit_rate@K`` / ``r_precision`` /
        ``median_rank`` / ``map`` / ``map@K`` / ``mrr`` according to
        which kwargs were enabled. Degenerate metrics (all queries
        skipped) return ``float('nan')`` for recall/hit/r_precision/
        median_rank and ``0.0`` for map/map@K/mrr — matching the
        probe's historical static-method return values.
    """
    return _aggregate_metrics_from_iter(
        _iter_row_orders(sim, batch=batch),
        work_ids=work_ids,
        recall_ks=recall_ks,
        hit_ks=hit_ks,
        include_r_precision=include_r_precision,
        include_median_rank=include_median_rank,
        include_map=include_map,
        map_at_ks=map_at_ks,
        include_mrr=include_mrr,
    )


def _aggregate_metrics_from_iter(
    order_iter: Iterator[tuple[int, torch.Tensor]],
    *,
    work_ids: torch.Tensor,
    recall_ks: list[int] | tuple[int, ...],
    hit_ks: list[int] | tuple[int, ...],
    include_r_precision: bool,
    include_median_rank: bool,
    include_map: bool,
    map_at_ks: list[int] | tuple[int, ...],
    include_mrr: bool,
) -> dict[str, float]:
    """Consume a row-order iterator (CPU-materialised or GPU-streaming)
    and aggregate every requested metric in a single pass.

    Body shared by :func:`compute_retrieval_metrics` and
    :func:`compute_retrieval_metrics_streaming`. The only difference
    between those two is which iterator they pass — the metric
    aggregation is identical, including edge cases for N=0 and
    all-degenerate corpora.
    """
    N = work_ids.size(0)
    recall_ks_eff = [int(k) for k in recall_ks if 0 < int(k) < N]
    hit_ks_eff = [int(k) for k in hit_ks if 0 < int(k) < N]
    map_ks_eff = [int(k) for k in map_at_ks if 0 < int(k) < N]

    if N == 0:
        # Degenerate: every requested metric is undefined.
        out: dict[str, float] = {f"recall@{k}": float("nan") for k in recall_ks_eff}
        out.update({f"hit_rate@{k}": float("nan") for k in hit_ks_eff})
        if include_r_precision:
            out["r_precision"] = float("nan")
        if include_median_rank:
            out["median_rank"] = float("nan")
        if include_map:
            out["map"] = 0.0  # matches _compute_map's empty-aps return
        for k in map_ks_eff:
            out[f"map@{k}"] = 0.0  # matches _map_at_k's empty-aps return
        if include_mrr:
            out["mrr"] = 0.0  # matches _mrr's empty-recip return
        return out

    # Vectorised n_relevant precompute: one O(N) scatter-add pass instead of
    # N Python ``is_rel.sum().item()`` calls in the inner loop. Each query's
    # n_relevant is "count of items with my work_id, minus self".
    _, inverse = work_ids.unique(return_inverse=True)
    work_counts = torch.zeros(int(inverse.max().item()) + 1, dtype=torch.long).scatter_add_(
        0, inverse, torch.ones_like(inverse, dtype=torch.long)
    )
    n_rel_all = work_counts[inverse] - 1  # exclude self from each query's count

    recalls: dict[int, list[float]] = {k: [] for k in recall_ks_eff}
    hits: dict[int, list[float]] = {k: [] for k in hit_ks_eff}
    rps: list[float] = []
    first_ranks: list[float] = []
    aps: list[float] = []
    aps_at_k: dict[int, list[float]] = {k: [] for k in map_ks_eff}
    recip_ranks: list[float] = []

    need_is_rel_list = bool(
        recall_ks_eff or hit_ks_eff or include_r_precision or include_map or map_ks_eff
    )
    max_map_k = max(map_ks_eff) if map_ks_eff else 0

    for i, order_i in order_iter:
        n_rel = int(n_rel_all[i].item())
        if n_rel == 0:
            continue
        is_rel = work_ids[order_i] == work_ids[i]
        is_rel_list = is_rel.tolist() if need_is_rel_list else None

        for k in recall_ks_eff:
            recalls[k].append(sum(is_rel_list[:k]) / n_rel)
        for k in hit_ks_eff:
            hits[k].append(1.0 if any(is_rel_list[:k]) else 0.0)
        if include_r_precision:
            rps.append(sum(is_rel_list[:n_rel]) / n_rel)

        if include_median_rank or include_mrr:
            # 1-indexed rank of the first True in is_rel
            first_idx = int(is_rel.nonzero(as_tuple=True)[0][0].item())
            if include_median_rank:
                first_ranks.append(float(first_idx + 1))
            if include_mrr:
                recip_ranks.append(1.0 / (first_idx + 1))

        # MAP family — single per-query walk over the ranked is_rel.
        # AP = (1/n_rel) * sum_{rank where rel} (hits_so_far / rank).
        # Compute full-rank AP (for ``map``) and per-K AP (for
        # ``map@K``) in the same pass. ``map@K`` uses the probe's
        # historical normalisation: divides by total n_relevant, not
        # min(K, n_relevant).
        if include_map or map_ks_eff:
            hits_so_far = 0
            ap_full = 0.0
            ap_at: dict[int, float] = dict.fromkeys(map_ks_eff, 0.0)
            cap = (N - 1) if include_map else max_map_k
            for rank_idx, rel in enumerate(is_rel_list[:cap]):
                if not rel:
                    continue
                hits_so_far += 1
                rank = rank_idx + 1
                if include_map:
                    ap_full += hits_so_far / rank
                for k in map_ks_eff:
                    if rank <= k:
                        ap_at[k] += hits_so_far / rank
            if include_map:
                aps.append(ap_full / n_rel)
            for k in map_ks_eff:
                aps_at_k[k].append(ap_at[k] / n_rel)

    out: dict[str, float] = {}
    for k in recall_ks_eff:
        vs = recalls[k]
        out[f"recall@{k}"] = float(sum(vs) / len(vs)) if vs else float("nan")
    for k in hit_ks_eff:
        vs = hits[k]
        out[f"hit_rate@{k}"] = float(sum(vs) / len(vs)) if vs else float("nan")
    if include_r_precision:
        out["r_precision"] = float(sum(rps) / len(rps)) if rps else float("nan")
    if include_median_rank:
        if first_ranks:
            out["median_rank"] = float(
                torch.tensor(first_ranks, dtype=torch.float).quantile(0.5).item()
            )
        else:
            out["median_rank"] = float("nan")
    if include_map:
        out["map"] = float(sum(aps) / len(aps)) if aps else 0.0
    for k in map_ks_eff:
        vs = aps_at_k[k]
        out[f"map@{k}"] = float(sum(vs) / len(vs)) if vs else 0.0
    if include_mrr:
        out["mrr"] = float(sum(recip_ranks) / len(recip_ranks)) if recip_ranks else 0.0
    return out


def compute_retrieval_metrics_streaming(
    embs: torch.Tensor,
    work_ids: torch.Tensor,
    *,
    recall_ks: list[int] | tuple[int, ...] = (10,),
    hit_ks: list[int] | tuple[int, ...] = (),
    include_r_precision: bool = True,
    include_median_rank: bool = True,
    include_map: bool = False,
    map_at_ks: list[int] | tuple[int, ...] = (),
    include_mrr: bool = False,
    device: str = "cuda",
    batch: int = 1024,
) -> dict[str, float]:
    """GPU-chunked variant of :func:`compute_retrieval_metrics`.

    Identical metric output (within argsort tie-breaking tolerance,
    typically < 1e-4 MAP delta on real embeddings). Computes
    ``sim = embs @ embs.T`` row-chunks on demand on ``device`` rather
    than materialising the full ``(N, N)`` matrix — required when
    N is large enough that ``embs @ embs.T`` overflows the available
    RAM (e.g. VGMIDITVar-timbre at N=102 960 → 42 GB).

    Args:
        embs: ``(N, H)`` per-file L2-normalised embeddings, fp32 on CPU.
            Will be moved to ``device`` internally (~316–422 MB).
        work_ids: ``(N,)`` group labels on CPU.
        device: ``"cuda"`` (recommended) or ``"cpu"`` (still useful as a
            low-memory CPU fallback that avoids the full-sim allocation).
        batch: chunk size in rows. Default 1024 keeps peak GPU memory
            ≲ 3 GB at H=1024, N=100k.

    All other kwargs match :func:`compute_retrieval_metrics`.

    Returns:
        Same dict shape as :func:`compute_retrieval_metrics`.
    """
    return _aggregate_metrics_from_iter(
        _iter_row_orders_streaming(embs, batch=batch, device=device),
        work_ids=work_ids,
        recall_ks=recall_ks,
        hit_ks=hit_ks,
        include_r_precision=include_r_precision,
        include_median_rank=include_median_rank,
        include_map=include_map,
        map_at_ks=map_at_ks,
        include_mrr=include_mrr,
    )


# ──────────────────────────────────────────────────────────────────────
# Per-condition (cross-instrument / cross-soundfont) MAP
# ──────────────────────────────────────────────────────────────────────


def compute_perpair_map(
    sim: torch.Tensor,
    work_ids: list[int] | torch.Tensor,
    conditions: list[int] | torch.Tensor,
    query_condition: int | None,
    target_condition: int | None,
) -> tuple[float, int]:
    """Per-(query_condition, target_condition) MAP for cross-condition retrieval.

    Compute MAP restricted to queries whose ``conditions[i] ==
    query_condition`` against candidates whose ``conditions[j] ==
    target_condition``. ``None`` for either means "any condition".

    The "condition" axis is dataset-specific:
      - VGMIDITVar-timbre: ``conditions = gm_program`` (GM instrument
        code, e.g. 0=piano, 24=guitar, 48=strings, 60=horn, 73=flute,
        80=lead-square, 89=warm-pad). Diagonal cells (same instrument)
        measure within-timbre retrieval; off-diagonal cells measure
        cross-instrument retrieval — the operationally-relevant
        leitmotif/motif-discovery metric.
      - Legacy multisf builds: ``conditions = soundfont_id`` (which
        SoundFont rendered this audio). Same intuition: off-diagonal
        = cross-soundfont retrieval.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels (same as :func:`recall_at_k`).
        conditions: ``(N,)`` per-item condition labels (gm_program OR
                    soundfont_id OR similar).
        query_condition: filter queries to this condition value.
                        ``None`` accepts any query.
        target_condition: filter candidates to this condition value.
                         ``None`` accepts any candidate.

    Returns:
        ``(map_value, n_queries)`` — float MAP and the number of
        queries that contributed to it. ``map_value`` is 0.0 if no
        queries match (callers should also check ``n_queries`` to
        distinguish "empty cell" from "all-wrong cell").
    """
    n = sim.size(0)
    wids = work_ids if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
    prog_t = conditions if isinstance(conditions, torch.Tensor) else torch.tensor(conditions)

    query_mask = (
        torch.ones(n, dtype=torch.bool) if query_condition is None else (prog_t == query_condition)
    )
    target_mask = (
        torch.ones(n, dtype=torch.bool)
        if target_condition is None
        else (prog_t == target_condition)
    )

    aps: list[float] = []
    for i in range(n):
        if not query_mask[i]:
            continue
        # Allowed candidates: target_mask, excluding self
        allowed = target_mask.clone()
        allowed[i] = False
        if allowed.sum() == 0:
            continue
        sims_i = sim[i].clone()
        # ``-inf`` (not ``-2.0``) for non-allowed items — same class of bug
        # as audit-2 #6 fix in ``_compute_map``. Real cosine sims can be
        # below -2.0 on un-normalised embeddings; a finite sentinel would
        # rank genuine target items below the non-target "wall", silently
        # truncating the ranking.
        sims_i[~allowed] = float("-inf")
        order = sims_i.argsort(descending=True)
        order = order[: int(allowed.sum())]
        is_rel = (wids[order] == wids[i]) & allowed[order]
        n_rel = int(is_rel.sum().item())
        if n_rel == 0:
            continue
        hits = 0
        ap = 0.0
        for rank, rel in enumerate(is_rel.tolist(), start=1):
            if rel:
                hits += 1
                ap += hits / rank
        ap /= n_rel
        aps.append(ap)
    return (float(torch.tensor(aps).mean().item()) if aps else 0.0, len(aps))


def compute_perpair_map_all(
    sim: torch.Tensor,
    work_ids: list[int] | torch.Tensor,
    conditions: list[int] | torch.Tensor,
    *,
    query_conds: list[int] | None = None,
    target_conds: list[int] | None = None,
    batch: int = 2048,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Compute all (query_condition × target_condition) MAP cells in one pass.

    Equivalent to calling :func:`compute_perpair_map` for every ``(q, t)``
    pair, but shares the per-query argsort across cells — for
    VGMIDITVar-timbre with 8 GM programs that cuts wall-time from
    O(64·N·argsort) to O(N·argsort + 64·N·Python). Empirically ~4 h →
    ~3 min on N=102 960.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.
        conditions: ``(N,)`` per-item condition labels.
        query_conds: condition values to enumerate as query slots.
            ``None`` → all observed condition values (sentinel ``-1``
            included only if it actually appears in ``conditions``;
            callers should filter beforehand if they want it dropped).
        target_conds: condition values to enumerate as target slots.
            ``None`` → same as ``query_conds``.
        batch: forwarded to :func:`_iter_row_orders`.

    Returns:
        ``dict[(q_cond, t_cond)] -> (map_value, n_queries)``. Cells with
        zero queries contributing return ``(0.0, 0)`` — same convention
        as :func:`compute_perpair_map`. Numerical equivalence with
        per-cell ``compute_perpair_map`` calls is asserted in tests.
    """
    n = sim.size(0)
    wids = work_ids if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
    conds = conditions if isinstance(conditions, torch.Tensor) else torch.tensor(conditions)

    if query_conds is None:
        query_conds = sorted({int(c) for c in conds.tolist()})
    if target_conds is None:
        target_conds = list(query_conds)
    query_set = set(query_conds)
    target_list = list(target_conds)

    aps_per_cell: dict[tuple[int, int], list[float]] = {
        (q, t): [] for q in query_conds for t in target_list
    }

    if n == 0:
        return {k: (0.0, 0) for k in aps_per_cell}

    return _perpair_map_all_from_iter(
        _iter_row_orders(sim, batch=batch),
        wids=wids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=aps_per_cell,
    )


def _perpair_map_all_from_iter(
    order_iter: Iterator[tuple[int, torch.Tensor]],
    *,
    wids: torch.Tensor,
    conds: torch.Tensor,
    query_set: set[int],
    target_list: list[int],
    aps_per_cell: dict[tuple[int, int], list[float]],
) -> dict[tuple[int, int], tuple[float, int]]:
    """Body shared by :func:`compute_perpair_map_all` and
    :func:`compute_perpair_map_all_streaming` — both consume an
    order-iterator and aggregate per-cell APs identically."""
    conds_list = conds.tolist()
    for i, order_i in order_iter:
        q_cond = conds_list[i]
        if q_cond not in query_set:
            continue
        cond_in_order = conds[order_i]  # (N-1,)
        rel_in_order = wids[order_i] == wids[i]  # (N-1,)
        for t_cond in target_list:
            mask = cond_in_order == t_cond
            if not bool(mask.any().item()):
                continue
            sub_rel = rel_in_order[mask]
            n_rel = int(sub_rel.sum().item())
            if n_rel == 0:
                continue
            hits = 0
            ap = 0.0
            for rank, rel in enumerate(sub_rel.tolist(), start=1):
                if rel:
                    hits += 1
                    ap += hits / rank
            ap /= n_rel
            aps_per_cell[(q_cond, t_cond)].append(ap)

    return {
        (q, t): (float(sum(aps) / len(aps)) if aps else 0.0, len(aps))
        for (q, t), aps in aps_per_cell.items()
    }


def compute_perpair_map_all_streaming(
    embs: torch.Tensor,
    work_ids: list[int] | torch.Tensor,
    conditions: list[int] | torch.Tensor,
    *,
    query_conds: list[int] | None = None,
    target_conds: list[int] | None = None,
    device: str = "cuda",
    batch: int = 1024,
) -> dict[tuple[int, int], tuple[float, int]]:
    """GPU-chunked variant of :func:`compute_perpair_map_all`.

    Drop-in replacement that takes per-file embeddings (``(N, H)``)
    instead of a pre-materialised similarity matrix. Computes sim
    row-chunks on demand on ``device`` via
    :func:`_iter_row_orders_streaming`. The aggregation body is
    identical (shared via :func:`_perpair_map_all_from_iter`) so
    numerical output is the same up to argsort tie-breaking.

    For VGMIDITVar-timbre (N=102 960, 8 GM programs): drops from
    ~3 min on CPU with the materialised-sim path (using
    ~84 GB peak via the 42 GB sim_c + chunk transients) to ~30–60 s on
    GPU with no full-sim allocation at all.

    Args:
        embs: ``(N, H)`` per-file embeddings. The caller is responsible
            for centering + L2-normalising before this call (matching
            the live probe's use of ``embs_c``).
        work_ids: ``(N,)`` group labels.
        conditions: ``(N,)`` per-item condition labels.
        query_conds, target_conds: same as
            :func:`compute_perpair_map_all`.
        device: ``"cuda"`` or ``"cpu"``.
        batch: chunk size in rows.

    Returns:
        Same shape as :func:`compute_perpair_map_all`.
    """
    n = embs.size(0)
    wids = work_ids if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
    conds = conditions if isinstance(conditions, torch.Tensor) else torch.tensor(conditions)

    if query_conds is None:
        query_conds = sorted({int(c) for c in conds.tolist()})
    if target_conds is None:
        target_conds = list(query_conds)
    query_set = set(query_conds)
    target_list = list(target_conds)

    aps_per_cell: dict[tuple[int, int], list[float]] = {
        (q, t): [] for q in query_conds for t in target_list
    }

    if n == 0:
        return {k: (0.0, 0) for k in aps_per_cell}

    return _perpair_map_all_from_iter(
        _iter_row_orders_streaming(embs, batch=batch, device=device),
        wids=wids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=aps_per_cell,
    )


# ──────────────────────────────────────────────────────────────────────
# Anisotropy diagnostics
# ──────────────────────────────────────────────────────────────────────


def anisotropy_metrics(
    embs: torch.Tensor,
    n_pairs: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Embedding-space anisotropy diagnostics for a (N, H) matrix.

    Lifted from ``scripts/diagnostics/anisotropy_diag.py:_isotropy_metrics``
    so the same numbers that motivated MARBLE's centered-MAP variant are
    available as live wandb metrics rather than post-hoc diagnostics.

    Returns a dict with four numbers, all computed on L2-normalised input:
      - ``mean_vec_norm`` (float in [0, 1]): norm of the corpus mean.
        Near 0 ⇒ isotropic; near 1 ⇒ all embeddings live in a thin cone.
        OMARRQ historically registers ~0.5 on retrieval tasks; that
        anisotropy is what motivates ``test/map_centered``.
      - ``avg_pair_cos`` (float in [-1, 1]): mean cosine similarity over
        ``n_pairs`` random off-diagonal pairs. Should be ≈ 0 for isotropic
        embeddings; > 0 indicates the cone-effect inflates all similarities.
      - ``top1_sv_share`` (float in [0, 1]): leading singular value² as a
        fraction of total variance (post-centering). > 0.3 = significant
        rank collapse on the top direction.
      - ``effective_rank`` (float in [1, min(N, H)]): exp(entropy of
        normalised SV² spectrum). Lower ⇒ more anisotropic / lower-rank.
        Noisy at small N (e.g. Covers80 with N=160).

    Notes
    -----
    - SVD is capped at 4096 samples for speed (matches the offline script);
      larger corpora subsample randomly with ``seed`` for reproducibility.
    - The function operates on whatever scale the caller passes in. When
      called from ``CoverRetrievalTask.on_test_epoch_end`` the input is
      already L2-normalised per-file mean-pooled embeddings — so the SVD
      below measures variance of unit vectors on the sphere, which is
      arguably MORE relevant for cosine retrieval than the raw-embedding
      SVD used by ``scripts/diagnostics/anisotropy_diag.py``. The two
      numbers are mathematically distinct; ``mean_vec_norm`` and
      ``avg_pair_cos`` are scale-invariant and agree across both inputs.
    - Returns NaN-filled dict on degenerate input (N < 2) — single-point
      corpora can't define pairwise cosine or SVD; the probe still logs
      these values and downstream wandb filters can drop them.
    """
    e = embs.detach().cpu().float().numpy()
    n, c = e.shape
    # Degenerate input — no pairs to sample, SVD ill-defined.
    if n < 2:
        return {
            "mean_vec_norm": float("nan"),
            "avg_pair_cos": float("nan"),
            "top1_sv_share": float("nan"),
            "effective_rank": float("nan"),
        }
    rng = np.random.default_rng(seed)

    # L2-normalise for cosine-based stats
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    normed = e / np.clip(norms, 1e-8, None)

    # avg pair cosine — sample off-diagonal pairs with replacement
    n_pairs_eff = min(n_pairs, n * (n - 1) // 2)
    a = rng.choice(n, size=n_pairs_eff, replace=True)
    b = rng.choice(n, size=n_pairs_eff, replace=True)
    same = a == b
    b[same] = (b[same] + 1) % n
    sims = (normed[a] * normed[b]).sum(axis=1)

    # mean-vector norm — cone effect proxy
    mean_vec = normed.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))

    # Rank collapse via SVD of centered embeddings (sample-capped for cost)
    n_svd = min(n, 4096)
    sub_idx = rng.choice(n, size=n_svd, replace=False) if n_svd < n else np.arange(n)
    sub = e[sub_idx] - e[sub_idx].mean(axis=0, keepdims=True)
    try:
        sv = np.linalg.svd(sub, compute_uv=False)
    except np.linalg.LinAlgError:
        sv = np.zeros(min(n_svd, c))
    sv2 = sv * sv
    if sv2.sum() > 0:
        share = sv2 / sv2.sum()
        top1 = float(share[0])
        # Entropy in nats → effective rank = exp(H)
        h_entropy = -float(np.sum(share * np.log(share + 1e-12)))
        eff_rank = float(math.exp(h_entropy))
    else:
        top1 = float("nan")
        eff_rank = float("nan")

    return {
        "mean_vec_norm": mean_norm,
        "avg_pair_cos": float(sims.mean()),
        "top1_sv_share": top1,
        "effective_rank": eff_rank,
    }
