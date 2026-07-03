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
    # ``torch.device`` round-trip canonicalises both forms ('cuda',
    # 'cuda:0') so we don't redundantly copy when embs is already on
    # the target device under a slightly different string spelling.
    target = torch.device(device)
    embs_dev = embs if embs.device == target else embs.to(target)
    # Force fp32 for the matmul: Lightning's bf16-mixed autocast can
    # leave per-clip embeddings in bf16, but argsort on bf16 has
    # higher tie rates (1/128 vs 1/2^23 mantissa precision) which
    # would cause GPU vs CPU rank disagreement and slightly different
    # MAP. Upcast once (no-op when embs is already fp32) so the
    # streaming path is precision-stable irrespective of the upstream
    # autocast policy.
    if embs_dev.dtype != torch.float32:
        embs_dev = embs_dev.float()
    # Defensive: this function runs at test time so gradient tracking
    # is wasteful + could pull autograd state into the matmul. Lightning's
    # trainer.test() already uses inference_mode but external callers
    # (scripts, recon) might not.
    with torch.no_grad():
        for start in range(0, N, batch):
            end = min(start + batch, N)
            # Compute (B, N) sim chunk on-device.
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
    return _aggregate_metrics_from_chunk_iter(
        _iter_row_order_chunks(sim, batch=batch),
        work_ids=work_ids,
        recall_ks=recall_ks,
        hit_ks=hit_ks,
        include_r_precision=include_r_precision,
        include_median_rank=include_median_rank,
        include_map=include_map,
        map_at_ks=map_at_ks,
        include_mrr=include_mrr,
    )


def _iter_row_order_chunks(
    sim: torch.Tensor, *, batch: int = 2048
) -> Iterator[tuple[int, torch.Tensor]]:
    """Chunked variant of :func:`_iter_row_orders` that yields whole
    chunks ``(start, order_chunk)`` instead of per-row ``(i, order_i)``.

    Lets the metric aggregator process ``B`` queries' rankings in a
    single batched tensor pass instead of Python-looping per query —
    the per-row iterator pays ~10 ms of ``.tolist()`` + Python MAP
    inner-loop overhead per query, which at N=102 960 totals ~10 min
    of pure Python. Yielding chunks lets the aggregator collapse that
    into ~2 min of vectorised tensor ops.

    See :func:`_aggregate_metrics_from_chunk_iter` for the consumer.
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
        order_chunk = chunk.argsort(descending=True, dim=-1)[:, : N - 1]
        del chunk
        yield start, order_chunk


def _iter_sim_order_chunks(
    embs: torch.Tensor,
    *,
    batch: int = 1024,
    device: str = "cuda",
    keep_on_device: bool = False,
    want_sim: bool = False,
) -> Iterator[tuple[int, torch.Tensor | None, torch.Tensor]]:
    """Chunked streaming similarity/ordering iterator.

    Yields ``(start, sim_chunk_or_None, order_chunk)`` per batch of query
    rows. ``sim_chunk`` (``(B, N)``, self set to ``-inf``) is only yielded
    when ``want_sim=True`` (e.g. for score-distribution accumulation —
    self positions are excluded by the consumer's masks, so the ``-inf``
    poisoning is harmless there). With ``keep_on_device=True`` both
    tensors stay on ``device`` so downstream aggregation runs there too —
    this avoids the ~843 MB/chunk order-matrix PCIe copy that made the
    CPU-aggregation path memory-bandwidth-bound.
    """
    N = embs.size(0)
    if N == 0:
        return
    target = torch.device(device)
    embs_dev = embs if embs.device == target else embs.to(target)
    if embs_dev.dtype != torch.float32:
        embs_dev = embs_dev.float()
    with torch.no_grad():
        for start in range(0, N, batch):
            end = min(start + batch, N)
            chunk = embs_dev[start:end] @ embs_dev.T
            row_idx = torch.arange(end - start, device=chunk.device)
            col_idx = torch.arange(start, end, device=chunk.device)
            chunk[row_idx, col_idx] = float("-inf")
            order_chunk = chunk.argsort(descending=True, dim=-1)[:, : N - 1]
            if not keep_on_device:
                order_chunk = order_chunk.cpu()
            sim_out = chunk if want_sim else None
            if want_sim and not keep_on_device:
                sim_out = chunk.cpu()
            yield start, sim_out, order_chunk
            del chunk, order_chunk, sim_out


def _iter_row_order_chunks_streaming(
    embs: torch.Tensor,
    *,
    batch: int = 1024,
    device: str = "cuda",
) -> Iterator[tuple[int, torch.Tensor]]:
    """Chunked GPU-streaming variant. Yields ``(start, order_chunk_cpu)``
    per batch. The CPU-side aggregator then does ALL metric work for the
    chunk as one tensor batch — see :func:`_aggregate_metrics_from_chunk_iter`.

    Thin wrapper over :func:`_iter_sim_order_chunks` (order-only, CPU
    chunks) — kept for existing callers and CPU-equivalence tests.
    """
    for start, _sim, order_chunk in _iter_sim_order_chunks(
        embs, batch=batch, device=device, keep_on_device=False, want_sim=False
    ):
        yield start, order_chunk


class BaseMetricChunkAggregator:
    """Stateful, device-generic form of the vectorised metric aggregator.

    Consumes chunks of order indices (shape ``(B, N-1)`` int64, CPU **or**
    CUDA) via :meth:`update` and produces the metric dict via
    :meth:`result`. All per-chunk math runs on the chunk's device — feed
    it CUDA chunks (see ``_iter_sim_order_chunks(keep_on_device=True)``)
    and the gathers/cumsums execute on GPU; only the per-query scalar
    results (``(B,)``-sized) cross back to host per chunk.

    :func:`_aggregate_metrics_from_chunk_iter` is the thin functional
    wrapper (kept for existing callers/tests).
    """

    def __init__(
        self,
        work_ids: torch.Tensor,
        *,
        recall_ks: list[int] | tuple[int, ...],
        hit_ks: list[int] | tuple[int, ...],
        include_r_precision: bool,
        include_median_rank: bool,
        include_map: bool,
        map_at_ks: list[int] | tuple[int, ...],
        include_mrr: bool,
    ):
        work_ids = work_ids if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
        self.N = int(work_ids.size(0))
        N = self.N
        self.recall_ks_eff = [int(k) for k in recall_ks if 0 < int(k) < N]
        self.hit_ks_eff = [int(k) for k in hit_ks if 0 < int(k) < N]
        self.map_ks_eff = [int(k) for k in map_at_ks if 0 < int(k) < N]
        self.include_r_precision = include_r_precision
        self.include_median_rank = include_median_rank
        self.include_map = include_map
        self.include_mrr = include_mrr
        self.need_cumsum = include_map or bool(self.map_ks_eff)

        self._work_ids = work_ids
        self._n_rel_all: torch.Tensor | None = None
        if N > 0:
            # Vectorised n_relevant precompute (same as the per-row aggregator).
            _, inverse = work_ids.unique(return_inverse=True)
            work_counts = torch.zeros(int(inverse.max().item()) + 1, dtype=torch.long).scatter_add_(
                0, inverse, torch.ones_like(inverse, dtype=torch.long)
            )
            self._n_rel_all = work_counts[inverse] - 1  # (N,) long
        self._dev: torch.device | None = None
        self._arange_n_minus_1: torch.Tensor | None = None
        self._ranks_float: torch.Tensor | None = None

        # Per-metric running accumulators — per-query scalars, host-side.
        self.recalls: dict[int, list[float]] = {k: [] for k in self.recall_ks_eff}
        self.hits: dict[int, list[float]] = {k: [] for k in self.hit_ks_eff}
        self.rps: list[float] = []
        self.first_ranks: list[float] = []
        self.aps: list[float] = []
        self.aps_at_k: dict[int, list[float]] = {k: [] for k in self.map_ks_eff}
        self.recip_ranks: list[float] = []

    def _to_device(self, dev: torch.device) -> None:
        if self._dev == dev:
            return
        self._dev = dev
        self._work_ids = self._work_ids.to(dev)
        if self._n_rel_all is not None:
            self._n_rel_all = self._n_rel_all.to(dev)
        self._arange_n_minus_1 = torch.arange(self.N - 1, device=dev)
        self._ranks_float = (self._arange_n_minus_1 + 1).float()  # 1-indexed ranks

    def update(self, start: int, order_chunk: torch.Tensor) -> None:
        N = self.N
        self._to_device(order_chunk.device)
        work_ids = self._work_ids
        n_rel_all = self._n_rel_all
        arange_n_minus_1 = self._arange_n_minus_1
        ranks_float = self._ranks_float

        B = order_chunk.size(0)
        end = start + B
        query_ids = torch.arange(start, end, device=order_chunk.device)  # (B,)

        # is_rel_chunk[i, j] = True iff candidate at rank j for query (start+i) is same-work
        wids_at_order = work_ids[order_chunk]  # (B, N-1)
        wids_at_query = work_ids[query_ids].unsqueeze(-1)  # (B, 1)
        is_rel_chunk = wids_at_order == wids_at_query  # (B, N-1) bool

        n_rel_chunk = n_rel_all[query_ids]  # (B,) long
        # Mask out queries with no relevants — they're skipped per metric.
        valid = n_rel_chunk > 0  # (B,) bool
        # Use clamp(1) so the division below doesn't NaN/inf; invalid
        # rows are filtered out before the final extend()s.
        n_rel_safe = n_rel_chunk.clamp(min=1).float()  # (B,)

        # ── Recall@K ────────────────────────────────────────────────
        for k in self.recall_ks_eff:
            recall_k = is_rel_chunk[:, :k].sum(dim=-1).float() / n_rel_safe
            self.recalls[k].extend(recall_k[valid].tolist())

        # ── Hit@K ───────────────────────────────────────────────────
        for k in self.hit_ks_eff:
            hit_k = is_rel_chunk[:, :k].any(dim=-1).float()
            self.hits[k].extend(hit_k[valid].tolist())

        # ── R-Precision: precision at K = n_rel (per query) ────────
        if self.include_r_precision:
            # mask[i, j] = j < n_rel_chunk[i] — take only the first n_rel
            # positions of the ranking for each query.
            rp_mask = arange_n_minus_1.unsqueeze(0) < n_rel_chunk.unsqueeze(-1)  # (B, N-1)
            rp = (is_rel_chunk & rp_mask).sum(dim=-1).float() / n_rel_safe
            self.rps.extend(rp[valid].tolist())

        # ── MAP / MAP@K (cumsum-based AP per query) ────────────────
        if self.need_cumsum:
            # hits_so_far[i, j] = number of True in is_rel_chunk[i, :j+1]
            hits_so_far = is_rel_chunk.long().cumsum(dim=-1)  # (B, N-1) long
            # per_rank[i, j] = (1 if is_rel else 0) * hits_so_far / rank
            per_rank = is_rel_chunk.float() * hits_so_far.float() / ranks_float.unsqueeze(0)
            if self.include_map:
                ap_full = per_rank.sum(dim=-1) / n_rel_safe  # (B,)
                self.aps.extend(ap_full[valid].tolist())
            for k in self.map_ks_eff:
                ap_k = per_rank[:, :k].sum(dim=-1) / n_rel_safe
                self.aps_at_k[k].extend(ap_k[valid].tolist())

        # ── First-relevant rank (used for median_rank + MRR) ───────
        if self.include_median_rank or self.include_mrr:
            # For each row find the smallest j where is_rel_chunk[i, j] is True.
            # Sentinel = N (out of range) for rows with no True — but those
            # rows are excluded by ``valid`` so they don't reach the lists.
            sentinel = arange_n_minus_1.unsqueeze(0).expand(B, -1)  # (B, N-1)
            where_true = torch.where(is_rel_chunk, sentinel, torch.full_like(sentinel, N))
            first_idx = where_true.min(dim=-1).values  # (B,) — < N when valid
            first_idx_plus_one = (first_idx + 1).float()
            if self.include_median_rank:
                self.first_ranks.extend(first_idx_plus_one[valid].tolist())
            if self.include_mrr:
                self.recip_ranks.extend((1.0 / first_idx_plus_one)[valid].tolist())

    def result(self) -> dict[str, float]:
        # Final reductions (identical to the per-row aggregator's output).
        out: dict[str, float] = {}
        for k in self.recall_ks_eff:
            vs = self.recalls[k]
            out[f"recall@{k}"] = float(sum(vs) / len(vs)) if vs else float("nan")
        for k in self.hit_ks_eff:
            vs = self.hits[k]
            out[f"hit_rate@{k}"] = float(sum(vs) / len(vs)) if vs else float("nan")
        if self.include_r_precision:
            out["r_precision"] = float(sum(self.rps) / len(self.rps)) if self.rps else float("nan")
        if self.include_median_rank:
            if self.first_ranks:
                out["median_rank"] = float(
                    torch.tensor(self.first_ranks, dtype=torch.float).quantile(0.5).item()
                )
            else:
                out["median_rank"] = float("nan")
        if self.include_map:
            out["map"] = float(sum(self.aps) / len(self.aps)) if self.aps else 0.0
        for k in self.map_ks_eff:
            vs = self.aps_at_k[k]
            out[f"map@{k}"] = float(sum(vs) / len(vs)) if vs else 0.0
        if self.include_mrr:
            out["mrr"] = (
                float(sum(self.recip_ranks) / len(self.recip_ranks)) if self.recip_ranks else 0.0
            )
        return out


def _aggregate_metrics_from_chunk_iter(
    chunk_iter: Iterator[tuple[int, torch.Tensor]],
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
    """Functional wrapper over :class:`BaseMetricChunkAggregator`.

    Consumes chunks of order indices (shape ``(B, N-1)`` int64) and
    computes every requested metric in batched tensor ops, no Python
    per-query loop. Used by :func:`compute_retrieval_metrics` and
    :func:`compute_retrieval_metrics_streaming`. Output is numerically
    equivalent to :func:`_aggregate_metrics_from_iter` (the per-row
    variant) up to floating-point rounding noise (verified by tests
    at ~1e-7 abs tolerance).

    Why chunked: a profile at 1k queries showed the per-row
    ``is_rel.tolist()`` + Python MAP inner loop = ~6 s / 1k queries.
    The chunked vectorised path = ~1.6 s / 1k queries → ~3.8× speedup
    on the metric body. At N=102 960 that saves ~6-8 min per pass.
    """
    agg = BaseMetricChunkAggregator(
        work_ids,
        recall_ks=recall_ks,
        hit_ks=hit_ks,
        include_r_precision=include_r_precision,
        include_median_rank=include_median_rank,
        include_map=include_map,
        map_at_ks=map_at_ks,
        include_mrr=include_mrr,
    )
    for start, order_chunk in chunk_iter:
        agg.update(start, order_chunk)
    return agg.result()


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

    Reference (per-row) implementation. Kept as the equivalence baseline
    for :func:`_aggregate_metrics_from_chunk_iter` — the chunked path
    that production code routes through. Both must produce the same
    metric dict (within fp32 vs fp64 rounding tolerance on the AP path)
    — tests in ``tests/test_retrieval_metrics_streaming.py`` pin the
    equivalence at abs=1e-6.

    Not on the hot path. Retained for two reasons: (1) per-row logic is
    easier to read and audit than the cumsum-based chunked variant when
    debugging a metric anomaly, (2) future metrics whose chunked form
    is awkward (e.g. ones that need per-query early-exit) can fall back
    to this aggregator without re-implementing the iterator-consumer
    scaffold.
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
    return _aggregate_metrics_from_chunk_iter(
        _iter_row_order_chunks_streaming(embs, batch=batch, device=device),
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
# Same-group-restricted MAP (hard-distractor / length-stratified)
# ──────────────────────────────────────────────────────────────────────


def compute_masked_map(
    sim: torch.Tensor,
    work_ids: torch.Tensor,
    *,
    gallery_groups: torch.Tensor | None = None,
    query_subset: torch.Tensor | None = None,
) -> float:
    """Standard query-weighted MAP with an optional hard-distractor mask
    and/or query subset.

    This is the *same* MAP as :func:`compute_retrieval_metrics`'s ``map``
    (query-equal-weighted Average Precision, self excluded via the ``-inf``
    sentinel + last-column drop) — but with two extra knobs:

    * ``gallery_groups`` (``(N,)`` int): when given, every gallery item whose
      group differs from the query's group is masked out of that query's
      ranking with ``-inf`` BEFORE the argsort, exactly like the self-mask.
      This restricts each query's gallery to its OWN ``gallery_groups`` value
      — e.g. the same tune family — so the MAP measures discrimination
      *within* that group rather than against the easy other-group negatives.
      ``None`` ⇒ full gallery (identical to standard MAP).

    * ``query_subset`` (``(N,)`` bool): when given, only queries where the
      mask is True contribute their AP to the mean. ``None`` ⇒ all queries.
      Relevance (which gallery items count as hits) is unaffected — only the
      set of queries whose AP is averaged changes. Used for length-stratified
      MAP (short vs long query subsets).

    Both knobs compose: a length-stratified same-family MAP is achievable by
    passing both. Returns ``float('nan')`` if no valid query contributes
    (degenerate — e.g. a singleton family, or an empty subset).

    Implementation note: this materialises the per-query ``(N,)`` similarity
    row, applies the masks, argsorts, and walks the ranking — the same
    per-row reference logic as :func:`compute_perpair_map`. MTC-ANN corpora
    are ~700 rows so the O(N²) row pass is sub-second; it deliberately does
    NOT use the chunked/streaming aggregator (those don't support per-query
    gallery masks).
    """
    n = sim.size(0)
    wids = work_ids if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
    groups = None
    if gallery_groups is not None:
        groups = (
            gallery_groups
            if isinstance(gallery_groups, torch.Tensor)
            else torch.tensor(gallery_groups)
        )
    subset = None
    if query_subset is not None:
        subset = (
            query_subset if isinstance(query_subset, torch.Tensor) else torch.tensor(query_subset)
        )

    aps: list[float] = []
    for i in range(n):
        if subset is not None and not bool(subset[i]):
            continue
        # Allowed gallery: same group as query (if a group mask is given),
        # excluding self. ``-inf`` (not a finite sentinel) so genuine items
        # never sort below a masked "wall" — same class of bug as audit-2 #6.
        if groups is not None:
            allowed = groups == groups[i]
        else:
            allowed = torch.ones(n, dtype=torch.bool)
        allowed = allowed.clone()
        allowed[i] = False
        if not bool(allowed.any()):
            continue
        sims_i = sim[i].clone()
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
        aps.append(ap / n_rel)
    return float(sum(aps) / len(aps)) if aps else float("nan")


# ──────────────────────────────────────────────────────────────────────
# Within-group multi-label MAP (within-piece phrase-window same-motif)
# ──────────────────────────────────────────────────────────────────────


def compute_within_group_multilabel_map(
    embs: torch.Tensor,
    groups: list[int] | torch.Tensor,
    letters: list[set[str]],
    occ_ids: list[set[str]],
) -> float:
    """Mean within-group, multi-label, same-occurrence-excluded retrieval MAP.

    The generalisation of the leitmotifs-prototype ``within_movement_map``
    (``scripts/eval/bps_within_piece_metric.py``, shuffle-control-validated) to a
    per-group sub-matrix evaluation. Used by the BPS-Motif within-piece
    phrase-window task: each query window is a phrase-window slice carrying the
    union of its bars' motif letters; relevance is "same group (movement) AND
    shares >=1 motif letter", with the query's own overlapping windows removed so
    we measure genuine recurrence rather than self-similarity.

    For every query ``q`` with ``letters[q] != set()``:

      * gallery = items ``w`` with ``groups[w] == groups[q]``, ``w != q``, and
        ``occ_ids[w] & occ_ids[q] == set()`` (same-occurrence exclusion — drop
        the query's own overlapping windows);
      * relevant = gallery items with ``letters[w] & letters[q] != set()``;
      * rank the gallery by descending cosine similarity to ``q`` on ``embs``;
      * standard AP (precision@k averaged over the ranks of the relevant items).

    Returns the mean AP over queries that have >=1 relevant gallery item, or
    ``float('nan')`` if no such query exists (nothing genuine to retrieve).

    Mirrors :func:`compute_masked_map` discipline: per-query row materialisation,
    self-/cross-group masking via the gallery-index restriction (not a finite
    similarity sentinel), and a NaN return on the degenerate empty case.

    Args:
        embs: ``(N, H)`` per-window embeddings. Cosine is taken after an internal
            L2-normalise, so the caller may pass raw OR centered embeddings (the
            within-piece task passes both for ``map`` and ``map_centered``).
        groups: ``(N,)`` group label per window (the movement id — gallery is
            restricted to the query's own group).
        letters: length-``N`` list of motif-letter sets (``set()`` = no motif →
            never a query, but still a valid non-relevant gallery distractor).
        occ_ids: length-``N`` list of occurrence-id sets (same-occurrence
            exclusion key).

    Returns:
        Mean within-group multi-label AP, or ``float('nan')`` if unscorable.
    """
    n = embs.shape[0]
    if n == 0 or not (len(letters) == len(occ_ids) == n):
        return float("nan")

    grp = groups.tolist() if isinstance(groups, torch.Tensor) else list(groups)
    if len(grp) != n:
        return float("nan")

    # Cosine over L2-normalised rows (fp64 for a stable argsort tie-break).
    normed = torch.nn.functional.normalize(embs.to(torch.float64), dim=-1)
    sims = normed @ normed.T  # (N, N)

    aps: list[float] = []
    for q in range(n):
        q_letters = letters[q]
        if not q_letters:
            continue  # no motif → not a query
        q_occ = occ_ids[q]
        q_grp = grp[q]
        gallery_idx = [
            w for w in range(n) if w != q and grp[w] == q_grp and not (occ_ids[w] & q_occ)
        ]
        if not gallery_idx:
            continue
        relevant = torch.tensor(
            [1.0 if (letters[w] & q_letters) else 0.0 for w in gallery_idx],
            dtype=torch.float64,
        )
        if relevant.sum().item() == 0:
            continue  # no genuine same-letter recurrence to retrieve
        g_sims = sims[q, gallery_idx]
        order = torch.argsort(g_sims, descending=True)
        rel_ranked = relevant[order]
        cum_rel = torch.cumsum(rel_ranked, dim=0)
        ranks = torch.arange(1, rel_ranked.shape[0] + 1, dtype=torch.float64)
        precision_at_k = cum_rel / ranks
        ap = float((precision_at_k * rel_ranked).sum().item() / rel_ranked.sum().item())
        aps.append(ap)

    if not aps:
        return float("nan")
    return float(sum(aps) / len(aps))


def compute_within_group_multilabel_map_with_null(
    embs: torch.Tensor,
    groups: list[int] | torch.Tensor,
    letters: list[set[str]],
    occ_ids: list[set[str]],
    *,
    n_perms: int = 100,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """:func:`compute_within_group_multilabel_map` + a label-PERMUTATION NULL.

    The prevalence control for the within-piece window-size sweep, and the direct
    analog of the audio pipeline's permutation null
    (``floor_analysis.null_precision_bands`` / ``pairwise_links.permutation_null_f1``,
    EVALUATION.md "Method A"). The within-movement same-motif MAP rises
    monotonically with window size simply because wider windows carry more motif
    letters, so "shares >=1 letter" is trivially satisfied (prevalence inflation).
    This holds the embeddings, the per-group occurrence-excluded gallery, and the
    cosine ranking FIXED, then permutes *which window carries which letter-set* and
    recomputes the MAP ``n_perms`` times.

    CRITICAL — the permutation is **within-group** (within movement). The gallery
    is restricted to the query's own group, so the correct exchangeability unit is
    the group: letter-sets are shuffled only among windows of the SAME movement,
    never across movements (a global shuffle would corrupt each movement's
    prevalence and is the wrong null). This matches the leitmotifs prototype, which
    scored each movement separately (per-movement permutation) before averaging.

    The honest window signal is the LIFT = real - null_mean (and the empirical
    ``p`` = fraction of null perms >= real, +1-smoothed). Vectorised: a letter-
    overlap matrix is precomputed once, so each permutation is pure integer
    indexing (no per-perm O(n^2) Python set ops).

    Returns ``(real, null_mean, null_std, p_value)``. ``real`` is byte-identical to
    :func:`compute_within_group_multilabel_map`. If ``real`` is NaN (nothing
    scorable), returns ``(nan, nan, nan, nan)``; if no permutation is scorable,
    ``(real, nan, nan, nan)``.
    """
    import random
    import statistics
    from collections import defaultdict

    n = embs.shape[0]
    nan4 = (float("nan"), float("nan"), float("nan"), float("nan"))
    if n == 0 or not (len(letters) == len(occ_ids) == n):
        return nan4
    grp = groups.tolist() if isinstance(groups, torch.Tensor) else list(groups)
    if len(grp) != n:
        return nan4

    normed = torch.nn.functional.normalize(embs.to(torch.float64), dim=-1)
    sims = normed @ normed.T

    # Letter-overlap matrix, precomputed ONCE; label permutation is then pure
    # integer indexing. ``shares[i,j]`` = windows i and j share >=1 motif letter.
    vocab = sorted({ltr for s in letters for ltr in s})
    lut = {ltr: k for k, ltr in enumerate(vocab)}
    if vocab:
        membership = torch.zeros((n, len(vocab)), dtype=torch.float64)
        for i, s in enumerate(letters):
            for ltr in s:
                membership[i, lut[ltr]] = 1.0
        shares = (membership @ membership.T) > 0
    else:
        shares = torch.zeros((n, n), dtype=torch.bool)
    has_letter = [bool(s) for s in letters]

    # Per-query gallery (same-group, occ-excluded — label-INDEPENDENT) + ranking.
    galleries: list[torch.Tensor] = []
    orders: list[torch.Tensor] = []
    for q in range(n):
        g = [w for w in range(n) if w != q and grp[w] == grp[q] and not (occ_ids[w] & occ_ids[q])]
        g_t = torch.tensor(g, dtype=torch.long)
        galleries.append(g_t)
        orders.append(
            torch.argsort(sims[q, g_t], descending=True) if g else torch.empty(0, dtype=torch.long)
        )

    # Group -> member positions, for WITHIN-group label permutation.
    group_members: dict[int, list[int]] = defaultdict(list)
    for i, gg in enumerate(grp):
        group_members[gg].append(i)

    def _map_for(perm: torch.Tensor) -> float:
        """Mean AP when window position ``q`` is assigned source row ``perm[q]``."""
        aps: list[float] = []
        for q in range(n):
            pq = int(perm[q])
            if not has_letter[pq]:
                continue
            g = galleries[q]
            if g.numel() == 0:
                continue
            rel = shares[pq, perm[g]].to(torch.float64)
            if rel.sum().item() == 0:
                continue
            rr = rel[orders[q]]
            cum_rel = torch.cumsum(rr, dim=0)
            ranks = torch.arange(1, rr.shape[0] + 1, dtype=torch.float64)
            aps.append(float((cum_rel / ranks * rr).sum().item() / rr.sum().item()))
        return float(sum(aps) / len(aps)) if aps else float("nan")

    real = _map_for(torch.arange(n, dtype=torch.long))
    if real != real:  # NaN
        return nan4

    rng = random.Random(seed)
    nulls: list[float] = []
    for _ in range(max(1, n_perms)):
        perm = list(range(n))
        for members in group_members.values():
            shuffled = members[:]
            rng.shuffle(shuffled)
            for pos, src in zip(members, shuffled, strict=True):
                perm[pos] = src
        nm = _map_for(torch.tensor(perm, dtype=torch.long))
        if nm == nm:  # not NaN
            nulls.append(nm)
    if not nulls:
        return (real, float("nan"), float("nan"), float("nan"))
    null_mean = statistics.fmean(nulls)
    null_std = statistics.pstdev(nulls) if len(nulls) > 1 else 0.0
    p_value = (sum(1 for x in nulls if x >= real) + 1) / (len(nulls) + 1)
    return (real, null_mean, null_std, p_value)


# ──────────────────────────────────────────────────────────────────────
# Per-condition (cross-instrument / cross-soundfont) MAP
# ──────────────────────────────────────────────────────────────────────


def compute_perpair_map(
    sim: torch.Tensor,
    work_ids: list[int] | torch.Tensor,
    conditions: list[int] | torch.Tensor,
    query_condition: int | None,
    target_condition: int | None,
    variation_ids: list[int] | torch.Tensor | None = None,
    require_different_variation: bool = False,
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
    # Variation-controlled relevance: when enabled, mask out same-(work_id, variation)
    # candidates (the query's own composition re-rendered in another condition — an
    # audio near-duplicate) so cross- vs within-condition MAP is apples-to-apples
    # ("retrieve a *different* variation"). No-op if variation_ids is None.
    vars_t = None
    if require_different_variation and variation_ids is not None:
        vars_t = (
            variation_ids
            if isinstance(variation_ids, torch.Tensor)
            else torch.tensor(variation_ids)
        )

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
        if vars_t is not None:
            # Drop the same-(work, variation) twin(s) from BOTH gallery and relevance.
            allowed &= ~((wids == wids[i]) & (vars_t == vars_t[i]))
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


def _resolve_variation_tensor(
    variation_ids: list[int] | torch.Tensor | None,
    require_different_variation: bool,
) -> torch.Tensor | None:
    """Return a variation-id tensor iff variation control is active, else ``None``.

    Mirrors the gate in :func:`compute_perpair_map`: control only kicks in when the
    caller both opts in (``require_different_variation``) AND supplies ids. Either
    missing → ``None`` → the ``_all`` aggregators run their normal (uncontrolled)
    path, so the feature is a strict no-op by default.
    """
    if not require_different_variation or variation_ids is None:
        return None
    return variation_ids if isinstance(variation_ids, torch.Tensor) else torch.tensor(variation_ids)


def compute_perpair_map_all(
    sim: torch.Tensor,
    work_ids: list[int] | torch.Tensor,
    conditions: list[int] | torch.Tensor,
    *,
    query_conds: list[int] | None = None,
    target_conds: list[int] | None = None,
    batch: int = 2048,
    variation_ids: list[int] | torch.Tensor | None = None,
    require_different_variation: bool = False,
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

    vars_t = _resolve_variation_tensor(variation_ids, require_different_variation)
    return _perpair_map_all_from_chunk_iter(
        _iter_row_order_chunks(sim, batch=batch),
        wids=wids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=aps_per_cell,
        vars=vars_t,
    )


class PerPairGridChunkAggregator:
    """Stateful, device-generic per-(query_cond, target_cond) AP aggregator.

    Consumes order-chunks of shape ``(B, N-1)`` (CPU **or** CUDA) via
    :meth:`update` and computes the per-cell APs in batched tensor ops.
    The mathematical operation is identical to
    :func:`_perpair_map_all_from_iter` (the per-row variant): for each
    query and each target condition we walk the items at that target
    condition in rank order, accumulating ``hits_so_far / sub_rank``
    whenever the item is relevant.

    Vectorised via two cumulative sums per cell:
        * ``sub_rank[j] = cumsum(mask_at_t)[j]`` — 1-indexed rank of
          position ``j`` within the target-conditioned subset (positions
          outside the subset get an irrelevant value because the
          contribution is gated by ``rel_mask``).
        * ``hits_in_filtered[j] = cumsum(rel & mask_at_t)[j]`` — number
          of relevants in the subset up to and including position ``j``.
        * Per-position AP contribution: ``rel_mask * hits_in_filtered /
          sub_rank``. Sum over j and divide by per-query n_rel.

    All per-chunk math runs on the chunk's device; only the ``(B,)``
    per-query AP scalars cross to host per chunk.
    ``vars`` enables variation control (VGMIDITVar twin confound): drop
    same-(work, variation) candidates from BOTH the gallery and
    relevance — exactly the ``allowed &= ~twin`` of the per-cell path.
    """

    def __init__(
        self,
        *,
        wids: torch.Tensor,
        conds: torch.Tensor,
        query_set: set[int],
        target_list: list[int],
        aps_per_cell: dict[tuple[int, int], list[float]],
        vars: torch.Tensor | None = None,
    ):
        self._wids = wids
        self._conds = conds
        self._vars = vars
        self.query_conds_sorted = sorted(query_set)
        self.target_list = list(target_list)
        self.aps_per_cell = aps_per_cell
        self._dev: torch.device | None = None

    def _to_device(self, dev: torch.device) -> None:
        if self._dev == dev:
            return
        self._dev = dev
        self._wids = self._wids.to(dev)
        self._conds = self._conds.to(dev)
        if self._vars is not None:
            self._vars = self._vars.to(dev)

    def update(self, start: int, order_chunk: torch.Tensor) -> None:
        if not self.target_list:
            return
        self._to_device(order_chunk.device)
        wids, conds, vars = self._wids, self._conds, self._vars

        B, _ = order_chunk.shape
        end = start + B
        query_ids = torch.arange(start, end, device=order_chunk.device)
        q_conds_chunk = conds[query_ids]  # (B,)

        # Filter mask: only queries whose q_cond is in query_set
        # contribute to any cell. Skip the rest of the work for the
        # batch if no query qualifies.
        any_in_set = False
        for q_cond in self.query_conds_sorted:
            if (q_conds_chunk == q_cond).any():
                any_in_set = True
                break
        if not any_in_set:
            return

        # Per-rank conditions and is_rel: (B, N-1) each. Reused across
        # all target conditions.
        cond_in_order = conds[order_chunk]  # (B, N-1)
        wids_at_order = wids[order_chunk]  # (B, N-1)
        wids_at_query = wids[query_ids].unsqueeze(-1)  # (B, 1)
        is_rel_chunk = wids_at_order == wids_at_query  # (B, N-1) bool

        # Variation control (VGMIDITVar twin confound): drop same-(work, variation)
        # candidates from BOTH the gallery and relevance. Masking into ``mask_qt``
        # removes them from the rank denominator (cumsum) and from ``rel_mask`` in
        # one shot — exactly the ``allowed &= ~twin`` of the per-cell path.
        keep_chunk = None
        if vars is not None:
            vars_at_order = vars[order_chunk]  # (B, N-1)
            vars_at_query = vars[query_ids].unsqueeze(-1)  # (B, 1)
            keep_chunk = ~(is_rel_chunk & (vars_at_order == vars_at_query))  # (B, N-1) bool

        for t_cond in self.target_list:
            mask_qt = cond_in_order == t_cond  # (B, N-1) bool
            if keep_chunk is not None:
                mask_qt = mask_qt & keep_chunk
            rel_mask = is_rel_chunk & mask_qt  # (B, N-1) bool
            # sub_ranks: 1-indexed rank within the t_cond subset at
            # each position. clamp(min=1) avoids div-by-zero at
            # positions outside the subset (their contribution is
            # zero'd by ``rel_mask`` anyway, but we still need a safe
            # divisor for the broadcast).
            sub_ranks = mask_qt.long().cumsum(dim=-1).clamp(min=1)
            hits_in_filtered = rel_mask.long().cumsum(dim=-1)
            n_rel_per_query = rel_mask.sum(dim=-1)  # (B,) long
            per_pos = rel_mask.float() * (hits_in_filtered.float() / sub_ranks.float())
            ap_full = per_pos.sum(dim=-1) / n_rel_per_query.clamp(min=1).float()  # (B,)
            valid = n_rel_per_query > 0  # (B,)
            for q_cond in self.query_conds_sorted:
                cell_mask = valid & (q_conds_chunk == q_cond)
                if cell_mask.any():
                    self.aps_per_cell[(q_cond, t_cond)].extend(ap_full[cell_mask].tolist())

    def result(self) -> dict[tuple[int, int], tuple[float, int]]:
        return {
            (q, t): (float(sum(aps) / len(aps)) if aps else 0.0, len(aps))
            for (q, t), aps in self.aps_per_cell.items()
        }


def _perpair_map_all_from_chunk_iter(
    chunk_iter: Iterator[tuple[int, torch.Tensor]],
    *,
    wids: torch.Tensor,
    conds: torch.Tensor,
    query_set: set[int],
    target_list: list[int],
    aps_per_cell: dict[tuple[int, int], list[float]],
    vars: torch.Tensor | None = None,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Functional wrapper over :class:`PerPairGridChunkAggregator`.

    Same float32 arithmetic as the metric aggregator above. Tested for
    equivalence with :func:`_perpair_map_all_from_iter` at fp rounding
    tolerance. Args mirror :func:`_perpair_map_all_from_iter` exactly so
    callers can swap them by routing different iterators in.
    """
    agg = PerPairGridChunkAggregator(
        wids=wids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=aps_per_cell,
        vars=vars,
    )
    for start, order_chunk in chunk_iter:
        agg.update(start, order_chunk)
    return agg.result()


def _perpair_map_all_from_iter(
    order_iter: Iterator[tuple[int, torch.Tensor]],
    *,
    wids: torch.Tensor,
    conds: torch.Tensor,
    query_set: set[int],
    target_list: list[int],
    aps_per_cell: dict[tuple[int, int], list[float]],
    vars: torch.Tensor | None = None,
) -> dict[tuple[int, int], tuple[float, int]]:
    """Per-row reference body for the per-cell perpair MAP grid.

    Kept as the equivalence baseline for
    :func:`_perpair_map_all_from_chunk_iter` — the chunked path that
    production code routes through. Tests in
    ``tests/test_retrieval_metrics_streaming.py`` pin the equivalence
    at abs=1e-6 on the AP values."""
    conds_list = conds.tolist()
    for i, order_i in order_iter:
        q_cond = conds_list[i]
        if q_cond not in query_set:
            continue
        cond_in_order = conds[order_i]  # (N-1,)
        rel_in_order = wids[order_i] == wids[i]  # (N-1,)
        # Variation control: drop same-(work, variation) twins from gallery + relevance.
        keep_i = None
        if vars is not None:
            keep_i = ~(rel_in_order & (vars[order_i] == vars[i]))  # (N-1,)
        for t_cond in target_list:
            mask = cond_in_order == t_cond
            if keep_i is not None:
                mask = mask & keep_i
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
    variation_ids: list[int] | torch.Tensor | None = None,
    require_different_variation: bool = False,
) -> dict[tuple[int, int], tuple[float, int]]:
    """GPU-chunked variant of :func:`compute_perpair_map_all`.

    Drop-in replacement that takes per-file embeddings (``(N, H)``)
    instead of a pre-materialised similarity matrix. Computes sim
    row-chunks on demand on ``device`` via
    :func:`_iter_row_order_chunks_streaming`, then aggregates via the
    batched :func:`_perpair_map_all_from_chunk_iter`. Numerical output
    is the same as :func:`compute_perpair_map_all` up to argsort
    tie-breaking on the GPU path.

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

    vars_t = _resolve_variation_tensor(variation_ids, require_different_variation)
    return _perpair_map_all_from_chunk_iter(
        _iter_row_order_chunks_streaming(embs, batch=batch, device=device),
        wids=wids,
        conds=conds,
        query_set=query_set,
        target_list=target_list,
        aps_per_cell=aps_per_cell,
        vars=vars_t,
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

    Returns four numbers measuring **complementary** aspects of the
    embedding distribution. They are NOT redundant:

      - ``mean_vec_norm`` (float in [0, 1]): norm of the corpus mean
        AFTER L2-normalising each row. Measures **cone collapse** — a
        single dominant direction shared by all embeddings.
          * Isotropic: ``≈ 1 / √N`` (verified in tests).
          * Pure cone collapse (all rows along one axis): ``≈ 1``.
          * CLaMP3 audio embeddings on VGMIDITVar-timbre: ~0.85 across
            all layers — heavy cone collapse but stable across depth.

      - ``avg_pair_cos`` (float in [-1, 1]): mean cosine similarity over
        ``n_pairs`` random off-diagonal pairs of L2-normalised rows.
        **Theoretically related to ``mean_vec_norm``** via
        ``E[cos(a, b)] ≈ ||μ||² = mean_vec_norm²`` for unit vectors.
        Used as an **independent cross-check** that ``mean_vec_norm`` is
        a real cone effect and not a numerical artefact. If the two
        numbers diverge meaningfully, something is wrong with the
        implementation (see test_anisotropy_pair_cos_matches_mvn_squared).

      - ``top1_sv_share`` (float in [0, 1]): leading singular value²
        as a fraction of total variance, computed on **centered** (not
        L2-normalised) embeddings. Measures **rank collapse on the
        top-1 direction post-centering** — i.e. is there a second
        dominant direction after removing the corpus mean?
          * Isotropic random: ``~1/H`` (Marchenko-Pastur edge).
          * Cone collapse alone: small (the dominant direction was
            killed by centering — the metric does NOT measure cone
            collapse).
          * Two-cluster structure: large (the cluster-separation axis
            is now the dominant direction).

      - ``effective_rank`` (float in [1, min(N, H)]): ``exp(entropy of
        normalised SV² spectrum)`` on **centered** embeddings.
        Measures **structural diversity** — how many directions carry
        meaningful variance after removing the cone. It does NOT
        directly measure cone collapse (a pure cone has effective_rank
        near H-1 after centering because the noise is uniform across
        the remaining axes).
          * Isotropic Gaussian: ``≈ min(N, H)`` (with Marchenko-Pastur
            reduction; e.g. ~700 at H=768).
          * Single dominant cluster: high (cone gets removed → flat
            residual).
          * Multiple meaningful directions: moderate.
          * Single rank-1 subspace (after cone removal): low.
          For retrieval tasks, **a peak in effective_rank often
          coincides with the best-MAP layer** — see the CLaMP3 sweep
          where layer 4 (peak effective_rank=62) matches the best
          cross-condition MAP.

    Quick interpretation guide
    --------------------------
    The four metrics decompose anisotropy into:
      ``mean_vec_norm`` → "is there a shared common direction?"
      ``top1_sv_share`` → "is there a second dominant direction after
                          removing the first?"
      ``effective_rank`` → "how spread out is the residual variance?"
      ``avg_pair_cos`` → sanity check on ``mean_vec_norm``.

    Notes
    -----
    - SVD is capped at 4096 samples for speed (matches the offline script);
      larger corpora subsample randomly with ``seed`` for reproducibility.
    - ``mean_vec_norm`` and ``avg_pair_cos`` are scale-invariant
      (internal L2 normalisation). ``top1_sv_share`` and
      ``effective_rank`` are NOT scale-invariant — they operate on the
      raw input scale. When called from ``CoverRetrievalTask`` the input
      is already L2-normalised, so this distinction is moot in practice.
    - Returns NaN-filled dict on degenerate input (N < 2).
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


def zca_whiten(
    embs: torch.Tensor,
    alpha: float = 1.0,
    eps_rel: float = 0.0,
    eps_floor: float = 1e-12,
) -> torch.Tensor:
    """ZCA-whiten a ``(N, H)`` embedding matrix for cosine retrieval.

    Computes ``e_w = U (Λ + ε·λ_max·I)^(−α/2) Uᵀ (e − μ)`` where
    ``(μ, Λ, U)`` are the corpus mean and the eigendecomposition of the
    centered covariance. ``α=1.0`` is full whitening; ``α=0.0`` collapses
    to plain centering. The result is **not** L2-normalised — the caller
    renormalises (whitening here is a cosine-retrieval preprocessing step,
    not an identity-covariance guarantee on the sphere).

    ``eps_rel`` adds a **relative Tikhonov ridge** ``ε·λ_max`` to every
    eigenvalue before the negative power. ``eps_rel=0`` (default) is pure
    whitening. A positive ε is the fix for the small-corpus regime below:
    it damps the amplification of near-zero (null-space) eigenvalues.

    The eigendecomposition runs in **fp64** for stability: cone-collapsed
    encoders have ``λ_min/λ_max < 1e-6`` and ``Λ^(−1/2)`` blows up tiny
    eigenvalues in fp32. ``eps_floor`` clamps eigenvalues before the
    negative power so the ~zero tail doesn't divide by zero. Output is
    returned in the input dtype.

    This is a **transductive** transform (μ, Σ fit on ``embs`` itself),
    matching the centered-MAP protocol. It is a known technique, not a
    novel one — see ``docs/whitening_ablation.md`` for prior art
    (BERT-whitening, All-But-The-Top, Spectral Tempering).

    **Small-corpus regime.** Σ is ``(H, H)`` estimated from ``N`` rows.
    When ``N < H`` it is rank-deficient: ~``H−(N−1)`` directions have
    near-zero variance, and **pure** whitening (``eps_rel=0``) rescales
    them to unit variance — amplifying pure estimation noise and
    *collapsing* retrieval (verified on Covers80: α=1.0 MAP 0.04 vs raw
    0.17). A relative ridge (``eps_rel≈1e-2``) rescues it (MAP back to
    ~0.25, > raw) by leaving the null directions small. The probe uses
    ``eps_rel=1e-2`` when ``N < 2*H`` and pure whitening otherwise.
    """
    if embs.dim() != 2:
        raise ValueError(f"Expected 2D (N, H) embeddings; got {tuple(embs.shape)}")
    if not 0.0 <= alpha <= 2.0:
        raise ValueError(f"zca_whiten: alpha must be in [0, 2]; got {alpha}")
    if eps_rel < 0.0:
        raise ValueError(f"zca_whiten: eps_rel must be >= 0; got {eps_rel}")
    n = embs.shape[0]
    embs64 = embs.detach().to(torch.float64)
    centered = embs64 - embs64.mean(dim=0, keepdim=True)
    sigma = (centered.T @ centered) / float(n)
    # eigh: ascending eigenvalues; PSD by construction (clamp fp roundoff).
    eigvals, eigvecs = torch.linalg.eigh(sigma)
    eigvals = eigvals.clamp(min=0.0)
    if eps_rel > 0.0:
        eigvals = eigvals + eps_rel * float(eigvals.max())
    scale = eigvals.clamp(min=eps_floor).pow(-alpha / 2.0)
    whitened = ((centered @ eigvecs) * scale.unsqueeze(0)) @ eigvecs.T
    return whitened.to(embs.dtype)
