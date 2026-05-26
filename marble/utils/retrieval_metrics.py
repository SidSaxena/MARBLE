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

    .. note::
       Avoids cloning the full ``(N, N)`` matrix (~42 GB for VGMIDITVar-
       timbre with N=102 960). Instead, modifies the diagonal in-place
       to ``-inf``, runs ``argsort`` (which copies internally), then
       restores the original diagonal values.
    """
    diag = sim.diagonal()  # view into the diagonal — modifies sim in-place
    diag_vals = diag.clone()  # (N,) — tiny vs full (N,N) clone
    diag.fill_(float("-inf"))
    order = sim.argsort(descending=True, dim=-1)
    diag.copy_(diag_vals)  # restore original diagonal
    return order[:, :-1]  # drop the last column (always self)


# ──────────────────────────────────────────────────────────────────────
# Public metrics
# ──────────────────────────────────────────────────────────────────────


def recall_at_k(
    sim: torch.Tensor,
    work_ids: torch.Tensor,
    k: int,
    *,
    order: torch.Tensor | None = None,
) -> float:
    """Recall@K — average over queries of (# relevant in top-K) / (# relevant total).

    For queries with zero other-relevant items, contribute nothing
    (skipped — they're degenerate). If ALL queries are degenerate
    returns NaN.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.
        k: rank cutoff (1 ≤ k ≤ N − 1). Caller should skip metric
           entirely when k ≥ N.
        order: Optional precomputed ranking order from :func:`_ranking_order`.
               Pass to avoid recomputing when calling multiple metrics on
               the same similarity matrix.

    Returns:
        Average recall as a Python float in [0, 1].
    """
    N = work_ids.size(0)
    if k <= 0 or k >= N:
        raise ValueError(f"k={k} must be in [1, N-1] where N={N}")
    if order is None:
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


def hit_rate_at_k(
    sim: torch.Tensor,
    work_ids: torch.Tensor,
    k: int,
    *,
    order: torch.Tensor | None = None,
) -> float:
    """Hit Rate@K — fraction of queries with ≥1 relevant in the top-K.

    Binary per query; lower variance than Recall@K. "Did the system
    surface at least one true positive in the review budget?"

    Same args / return shape as :func:`recall_at_k`.
    """
    N = work_ids.size(0)
    if k <= 0 or k >= N:
        raise ValueError(f"k={k} must be in [1, N-1] where N={N}")
    if order is None:
        order = _ranking_order(sim)
    hits: list[float] = []
    for i in range(N):
        is_rel = work_ids[order[i]] == work_ids[i]
        if int(is_rel.sum().item()) == 0:
            continue
        hits.append(1.0 if bool(is_rel[:k].any().item()) else 0.0)
    return float(sum(hits) / len(hits)) if hits else float("nan")


def median_rank_first_hit(
    sim: torch.Tensor,
    work_ids: torch.Tensor,
    *,
    order: torch.Tensor | None = None,
) -> float:
    """Median rank (1-indexed) of the first relevant item per query.

    Diagnostic — "how deep into the ranking before I see *anything*?"
    Queries with zero relevant items are skipped.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.
        order: Optional precomputed ranking order from :func:`_ranking_order`.

    Returns:
        Median 1-indexed rank as float (use ``int(round(.))`` if a
        rank integer is desired downstream). NaN if no query has a hit.
    """
    N = work_ids.size(0)
    if order is None:
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


def r_precision(
    sim: torch.Tensor,
    work_ids: torch.Tensor,
    *,
    order: torch.Tensor | None = None,
) -> float:
    """R-Precision — precision at K = (number of relevant items for this query).

    Self-calibrating K per query: each query's natural cutoff is the
    size of its relevant set. Standard in IR literature; sidesteps the
    "which K?" question entirely.

    Args:
        sim: ``(N, N)`` similarity matrix.
        work_ids: ``(N,)`` group labels.
        order: Optional precomputed ranking order from :func:`_ranking_order`.

    Queries with zero relevant items are skipped.
    """
    N = work_ids.size(0)
    if order is None:
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
        sims_i[~allowed] = -2.0
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
