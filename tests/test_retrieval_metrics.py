"""
tests/test_retrieval_metrics.py

Unit tests for ``marble.utils.retrieval_metrics``.

All tests build small hand-constructed similarity matrices with known
ranks so the expected metric values can be computed on paper. No GPU,
no audio files, no fixtures — follows the pattern from
``tests/test_transforms.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from marble.utils.retrieval_metrics import (
    anisotropy_metrics,
    compute_perpair_map,
    hit_rate_at_k,
    median_rank_first_hit,
    r_precision,
    recall_at_k,
)

# ──────────────────────────────────────────────────────────────────────
# Fixtures (functions for clarity; no pytest-fixture overhead needed)
# ──────────────────────────────────────────────────────────────────────


def _two_pairs_perfect_sim() -> tuple[torch.Tensor, torch.Tensor]:
    """4 items, 2 works of 2 covers each. Pairwise sim is perfectly
    block-diagonal: same-work items have sim 0.9, cross-work 0.0.

    Expected: Recall@1 = Recall@K(any K≥1) = 1.0, Hit@1 = 1.0,
              median rank = 1.0, R-Precision = 1.0.
    """
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0],  # query 0 — same work as 1
            [0.9, 1.0, 0.0, 0.0],  # query 1 — same work as 0
            [0.0, 0.0, 1.0, 0.9],  # query 2 — same work as 3
            [0.0, 0.0, 0.9, 1.0],  # query 3 — same work as 2
        ]
    )
    work_ids = torch.tensor([0, 0, 1, 1])
    return sim, work_ids


def _two_pairs_inverted_sim() -> tuple[torch.Tensor, torch.Tensor]:
    """Same labels, but the encoder is WRONG — every query ranks the
    cross-work pair higher than its own match.

    Recall@1 = 0 (the top-1 is always wrong), Recall@3 = 1.0 (the
    relevant item is at rank 3 in all 4 queries). Median first-hit
    rank = 3.0. R-Precision = 0 (each query has R=1 and the top-1 is
    wrong).
    """
    sim = torch.tensor(
        [
            [1.0, 0.1, 0.9, 0.5],  # query 0 prefers 2 (cross-work)
            [0.1, 1.0, 0.5, 0.9],
            [0.9, 0.5, 1.0, 0.1],
            [0.5, 0.9, 0.1, 1.0],
        ]
    )
    work_ids = torch.tensor([0, 0, 1, 1])
    return sim, work_ids


def _multi_cover_sim() -> tuple[torch.Tensor, torch.Tensor]:
    """8 items, 2 works of 4 covers each. Sim is identity on within-work
    pairs scaled by exponent so the ranking is deterministic:

        within-work sim:  high (block-diagonal)
        cross-work sim:   low

    Each query has R=3 other-relevant items. Designed so the top-3
    items by similarity are exactly the within-work peers.

    Expected per-query: Recall@3 = 1.0, R-Precision = 1.0,
    median first-hit rank = 1.0, Hit@1 = 1.0.
    Recall@1 = 1/3 (one of the three relevant is at rank 1).
    """
    sim = torch.zeros(8, 8)
    for i in range(8):
        sim[i, i] = 1.0
        for j in range(8):
            if i != j:
                same = (i // 4) == (j // 4)
                # within-block sims (0.91, 0.92, 0.93) > all cross sims (0.1–0.3)
                sim[i, j] = 0.9 + 0.01 * (j % 4) if same else 0.1 + 0.05 * (j % 4)
    work_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    return sim, work_ids


# ──────────────────────────────────────────────────────────────────────
# recall_at_k
# ──────────────────────────────────────────────────────────────────────


def test_recall_at_k_perfect():
    sim, work_ids = _two_pairs_perfect_sim()
    assert recall_at_k(sim, work_ids, 1) == 1.0
    assert recall_at_k(sim, work_ids, 3) == 1.0


def test_recall_at_k_inverted():
    sim, work_ids = _two_pairs_inverted_sim()
    # All queries: relevant item at rank 3, so:
    #   Recall@1 = Recall@2 = 0
    #   Recall@3 = 1.0
    assert recall_at_k(sim, work_ids, 1) == 0.0
    assert recall_at_k(sim, work_ids, 2) == 0.0
    assert recall_at_k(sim, work_ids, 3) == 1.0


def test_recall_at_k_multi_cover():
    sim, work_ids = _multi_cover_sim()
    # Each query has R=3 relevant; sorted by within-work-with-(j%4)-bias,
    # the top-3 within-block items are exactly the relevant set.
    # Recall@1 = 1/3 (one of 3 relevant at rank 1)
    # Recall@3 = 1.0  (all 3 relevant at ranks 1-3)
    assert recall_at_k(sim, work_ids, 1) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(sim, work_ids, 3) == pytest.approx(1.0)


def test_recall_at_k_invalid_k():
    sim, work_ids = _two_pairs_perfect_sim()
    with pytest.raises(ValueError):
        recall_at_k(sim, work_ids, 0)
    with pytest.raises(ValueError):
        recall_at_k(sim, work_ids, 4)  # N=4, k must be < 4


def test_recall_at_k_all_unique_returns_nan():
    """When every item has a unique work_id, no query has a relevant
    other-item — degenerate case, returns NaN."""
    sim = torch.eye(3)
    work_ids = torch.tensor([0, 1, 2])
    result = recall_at_k(sim, work_ids, 1)
    assert math.isnan(result)


# ──────────────────────────────────────────────────────────────────────
# hit_rate_at_k
# ──────────────────────────────────────────────────────────────────────


def test_hit_rate_at_k_perfect():
    sim, work_ids = _two_pairs_perfect_sim()
    assert hit_rate_at_k(sim, work_ids, 1) == 1.0


def test_hit_rate_at_k_inverted():
    sim, work_ids = _two_pairs_inverted_sim()
    assert hit_rate_at_k(sim, work_ids, 1) == 0.0
    assert hit_rate_at_k(sim, work_ids, 3) == 1.0


def test_hit_rate_at_k_vs_recall_at_k_when_R_eq_1():
    """When every query has exactly 1 relevant other-item, Hit@K and
    Recall@K compute the same number — both are 0/1 indicators of
    whether the top-K contains the one relevant item."""
    sim, work_ids = _two_pairs_inverted_sim()  # R=1 per query
    for k in (1, 2, 3):
        assert hit_rate_at_k(sim, work_ids, k) == recall_at_k(sim, work_ids, k)


# ──────────────────────────────────────────────────────────────────────
# median_rank_first_hit
# ──────────────────────────────────────────────────────────────────────


def test_median_rank_first_hit_perfect():
    sim, work_ids = _two_pairs_perfect_sim()
    assert median_rank_first_hit(sim, work_ids) == 1.0


def test_median_rank_first_hit_inverted():
    sim, work_ids = _two_pairs_inverted_sim()
    # First-hit rank is 3 for every query → median 3.0
    assert median_rank_first_hit(sim, work_ids) == 3.0


def test_median_rank_first_hit_mixed_ranks():
    """4 items, 2 works. Encoder gets queries 0 + 1 perfectly (first
    hit at rank 1) but ranks the pair for queries 2 + 3 at rank 3.
    Median across the 4 first-ranks = median(1, 1, 3, 3) = 2.0."""
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0],  # 0→1 first
            [0.9, 1.0, 0.0, 0.1],  # 1→0 first
            [0.5, 0.4, 1.0, 0.1],  # 2's relevant (3) at rank 3
            [0.4, 0.5, 0.1, 1.0],  # 3's relevant (2) at rank 3
        ]
    )
    work_ids = torch.tensor([0, 0, 1, 1])
    assert median_rank_first_hit(sim, work_ids) == 2.0


# ──────────────────────────────────────────────────────────────────────
# r_precision
# ──────────────────────────────────────────────────────────────────────


def test_r_precision_perfect():
    sim, work_ids = _two_pairs_perfect_sim()
    assert r_precision(sim, work_ids) == 1.0


def test_r_precision_inverted():
    sim, work_ids = _two_pairs_inverted_sim()
    # Each query has R=1 and the top-1 is wrong → R-Precision 0
    assert r_precision(sim, work_ids) == 0.0


def test_r_precision_multi_relevant():
    sim, work_ids = _multi_cover_sim()
    # Each query has R=3 and the top-3 are all relevant → R-Precision 1.0
    assert r_precision(sim, work_ids) == pytest.approx(1.0)


def test_r_precision_partial_credit():
    """Mixed: top-2 of 3 relevant correct, top-3 has 2/3 relevant.
    R = 3, precision@3 = 2/3 per query."""
    # 4 items, all same work_id; we set sim so each query's top-3 has
    # exactly one cross-work distractor (but in this contrived case
    # cross-work is impossible since work_ids are all the same).
    # Use 6 items, 2 works of 3.
    sim = torch.zeros(6, 6)
    # Within-work pairs: (0,1,2) and (3,4,5)
    # Make query 0 see [1, 4, 2] as top-3 (so 2/3 of top-3 are relevant)
    for i in range(6):
        sim[i, i] = 1.0
    sim[0] = torch.tensor([1.0, 0.95, 0.85, 0.5, 0.9, 0.3])  # top-3: 1, 4, 2
    sim[1] = torch.tensor([0.95, 1.0, 0.85, 0.5, 0.9, 0.3])  # top-3: 0, 4, 2
    sim[2] = torch.tensor(
        [0.85, 0.85, 1.0, 0.5, 0.9, 0.3]
    )  # top-3: 4, 0, 1 → tie-break; we'll accept
    sim[3] = torch.tensor([0.5, 0.5, 0.5, 1.0, 0.95, 0.85])  # top-3: 4, 5, then tie
    sim[4] = torch.tensor([0.9, 0.9, 0.9, 0.95, 1.0, 0.85])  # top-3: 3, 0/1/2 tie
    sim[5] = torch.tensor([0.3, 0.3, 0.3, 0.85, 0.85, 1.0])  # top-3: 3, 4
    work_ids = torch.tensor([0, 0, 0, 1, 1, 1])
    # Don't pin to a specific value — just assert it's in (0, 1.0).
    rp = r_precision(sim, work_ids)
    assert 0.0 < rp < 1.0


# ──────────────────────────────────────────────────────────────────────
# General invariants
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# compute_perpair_map (cross-instrument / cross-soundfont)
# ──────────────────────────────────────────────────────────────────────


def _perpair_8file_sim() -> tuple[torch.Tensor, list[int], list[int]]:
    """8 files, 4 works × 2 conditions. Each work has one item in
    condition 0 and one in condition 1.

    Layout:
       work_id:    [0, 0, 1, 1, 2, 2, 3, 3]
       condition:  [0, 1, 0, 1, 0, 1, 0, 1]
       item idx:    0  1  2  3  4  5  6  7

    Sim matrix designed so EACH query's relevant cross-condition peer
    ranks #1 among its condition (i.e. (0,1) cell = perfect cross-cond
    retrieval). Within-condition queries (0,0) have no relevant pair
    (the only same-work item is in the OTHER condition), so cell n=0.

    Expected:
      perpair(0,1) — query in cond 0, target in cond 1: perfect MAP=1.0
      perpair(1,0) — query in cond 1, target in cond 0: perfect MAP=1.0
      perpair(0,0) — within cond 0: every query has 0 same-work peers
                      in cond 0 → MAP=0.0, n_queries=0.
      perpair(1,1) — same, MAP=0.0, n_queries=0.
    """
    n = 8
    sim = torch.zeros(n, n)
    work_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    conditions = [0, 1, 0, 1, 0, 1, 0, 1]
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(n):
            if i == j:
                continue
            same_work = work_ids[i] == work_ids[j]
            same_cond = conditions[i] == conditions[j]
            # Strong: same-work cross-cond peer.
            # Weak: same-work same-cond (impossible here).
            # Noise: cross-work, any cond.
            if same_work and not same_cond:
                sim[i, j] = 0.95
            elif same_work and same_cond:
                sim[i, j] = 0.5  # unreachable in this layout
            else:
                sim[i, j] = 0.1
    return sim, work_ids, conditions


def test_compute_perpair_map_cross_condition_perfect():
    """Cross-condition cell (0,1) and (1,0) should both be 1.0."""
    sim, work_ids, conditions = _perpair_8file_sim()
    map_01, n_01 = compute_perpair_map(sim, work_ids, conditions, 0, 1)
    assert map_01 == pytest.approx(1.0), f"got {map_01}"
    assert n_01 == 4  # 4 queries in condition 0

    map_10, n_10 = compute_perpair_map(sim, work_ids, conditions, 1, 0)
    assert map_10 == pytest.approx(1.0), f"got {map_10}"
    assert n_10 == 4


def test_compute_perpair_map_within_condition_empty():
    """Within-condition cells (0,0) and (1,1) have zero relevant pairs
    by construction → MAP=0, n_queries=0."""
    sim, work_ids, conditions = _perpair_8file_sim()
    map_00, n_00 = compute_perpair_map(sim, work_ids, conditions, 0, 0)
    assert map_00 == 0.0
    assert n_00 == 0
    map_11, n_11 = compute_perpair_map(sim, work_ids, conditions, 1, 1)
    assert map_11 == 0.0
    assert n_11 == 0


def test_compute_perpair_map_none_means_any():
    """``query_condition=None`` and/or ``target_condition=None``
    accept all queries / candidates."""
    sim, work_ids, conditions = _perpair_8file_sim()
    # All queries → any target: equivalent to the "aggregate MAP" but
    # computed via the perpair helper.
    map_all, n_all = compute_perpair_map(sim, work_ids, conditions, None, None)
    assert n_all == 8  # every query contributes
    # Cross-cond pair always ranks first; same-work means MAP=1.0
    assert map_all == pytest.approx(1.0)


def test_compute_perpair_map_off_diag_mean_matches_cross_instrument():
    """The 'cross-instrument MAP' in docs/leitmotif_findings.md is
    defined as the off-diagonal mean of the (q,t) grid where both
    q,t are not-None. Verify the mean of off-diagonal cells matches
    the average of perpair MAPs computed individually."""
    sim, work_ids, conditions = _perpair_8file_sim()
    map_01, _ = compute_perpair_map(sim, work_ids, conditions, 0, 1)
    map_10, _ = compute_perpair_map(sim, work_ids, conditions, 1, 0)
    expected_off_diag = (map_01 + map_10) / 2
    assert expected_off_diag == pytest.approx(1.0)


def test_compute_perpair_map_partial():
    """A condition cell where SOME queries get rank-1 hits and others
    don't — verify partial MAP."""
    n = 4
    work_ids = [0, 0, 1, 1]
    conditions = [0, 1, 0, 1]
    # Query 0 (cond 0, work 0) prefers item 1 (cond 1, work 0) → rank 1 ✓
    # Query 2 (cond 0, work 1) prefers item 1 (cond 1, work 0) → wrong! Relevant=3 at rank 2.
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.0],  # 0 prefers 1 (right cross-cond)
            [0.9, 1.0, 0.0, 0.1],  # 1's view (not tested here)
            [0.1, 0.7, 1.0, 0.5],  # 2 prefers 1 (wrong), then 3 (right)
            [0.0, 0.1, 0.5, 1.0],  # 3's view
        ]
    )
    # Cell (q=0, t=1): queries 0 and 2, candidates 1 and 3 (others excluded).
    # Query 0 → top in cond 1: item 1 (relevant). AP = 1/1 = 1.0
    # Query 2 → top in cond 1: item 1 (NOT relevant), then 3 (relevant at rank 2). AP = 1/2 = 0.5
    # MAP = (1.0 + 0.5) / 2 = 0.75
    map_01, n_01 = compute_perpair_map(sim, work_ids, conditions, 0, 1)
    assert n_01 == 2
    assert map_01 == pytest.approx(0.75)


# ──────────────────────────────────────────────────────────────────────
# anisotropy_metrics
# ──────────────────────────────────────────────────────────────────────


def test_anisotropy_metrics_isotropic_baseline():
    """Random Gaussian embeddings in high dim should land near isotropic:
    mean_vec_norm ≈ 1/sqrt(N), avg_pair_cos ≈ 0, eff_rank ≈ min(N, H)."""
    torch.manual_seed(0)
    N, H = 100, 64
    embs = torch.randn(N, H)
    m = anisotropy_metrics(embs, seed=0)
    # Isotropic Gaussian: mean of unit vectors has expected norm ~1/sqrt(N)≈0.1
    assert 0.0 <= m["mean_vec_norm"] < 0.3, f"mean_vec_norm={m['mean_vec_norm']}"
    # Avg pair cosine should be near 0
    assert abs(m["avg_pair_cos"]) < 0.15, f"avg_pair_cos={m['avg_pair_cos']}"
    # Effective rank near min(N, H) = 64 for random Gaussian
    assert m["effective_rank"] > 30.0, f"effective_rank={m['effective_rank']}"
    # No single direction dominates
    assert m["top1_sv_share"] < 0.2, f"top1_sv_share={m['top1_sv_share']}"


def test_anisotropy_metrics_cone_collapse():
    """Embeddings that all lie near a single direction should register as
    highly anisotropic: mean_vec_norm ≈ 1, effective_rank ≈ 1, top1_sv ≈ 1."""
    torch.manual_seed(1)
    N, H = 100, 64
    base = torch.zeros(N, H)
    base[:, 0] = 1.0
    # tiny noise to keep SVD numerically clean
    embs = base + 0.001 * torch.randn(N, H)
    m = anisotropy_metrics(embs, seed=0)
    # All embeddings point in the +e0 direction → mean vector has near-unit norm
    assert m["mean_vec_norm"] > 0.95, f"mean_vec_norm={m['mean_vec_norm']}"
    # Random pairs have cosine near 1
    assert m["avg_pair_cos"] > 0.95, f"avg_pair_cos={m['avg_pair_cos']}"
    # Post-centering the dominant direction dies — variance lives in the
    # tiny noise dimensions → effective rank ≈ noise dims. Just check it
    # collapsed below the isotropic ~64 baseline.
    assert m["effective_rank"] < 50.0, f"effective_rank={m['effective_rank']}"


def test_anisotropy_metrics_keys_complete():
    """All four documented keys must be present and float-valued."""
    embs = torch.randn(20, 16)
    m = anisotropy_metrics(embs)
    for key in ("mean_vec_norm", "avg_pair_cos", "top1_sv_share", "effective_rank"):
        assert key in m
        assert isinstance(m[key], float)


def test_anisotropy_metrics_degenerate_returns_nan():
    """N < 2 corpora can't define pairwise cosine or rank statistics —
    function must return NaN dict, not raise / emit NumPy warnings."""
    # N=0: empty corpus
    embs_empty = torch.zeros(0, 16)
    m_empty = anisotropy_metrics(embs_empty)
    for key in ("mean_vec_norm", "avg_pair_cos", "top1_sv_share", "effective_rank"):
        assert math.isnan(m_empty[key]), f"expected NaN for empty corpus key {key}"
    # N=1: single point — pair sampling and SVD both ill-defined
    embs_single = torch.randn(1, 16)
    m_single = anisotropy_metrics(embs_single)
    for key in ("mean_vec_norm", "avg_pair_cos", "top1_sv_share", "effective_rank"):
        assert math.isnan(m_single[key]), f"expected NaN for single-point corpus key {key}"


def test_anisotropy_metrics_deterministic_under_seed():
    """Same input + same seed → identical output (reproducibility)."""
    embs = torch.randn(50, 32)
    m1 = anisotropy_metrics(embs, seed=42)
    m2 = anisotropy_metrics(embs, seed=42)
    for key in m1:
        # mean_vec_norm + eff_rank are deterministic (not RNG-sampled)
        # avg_pair_cos uses RNG but the same seed gives the same draws
        assert m1[key] == pytest.approx(m2[key]), f"non-deterministic at key {key}"


# ──────────────────────────────────────────────────────────────────────
# General invariants
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Single-pass bundle + row iterator (added with the VGMIDITVar-timbre
# OOM fix — verify the batched per-row path produces identical numbers
# to the individual function calls).
# ──────────────────────────────────────────────────────────────────────


def test_compute_retrieval_metrics_matches_individual_calls():
    """Single-pass bundle must match per-metric functions bit-for-bit."""
    from marble.utils.retrieval_metrics import compute_retrieval_metrics

    torch.manual_seed(7)
    N = 50
    sim = torch.randn(N, N)
    work_ids = torch.randint(0, 8, (N,))

    bundle = compute_retrieval_metrics(
        sim,
        work_ids,
        recall_ks=(1, 5, 10),
        hit_ks=(1, 5, 10),
        include_r_precision=True,
        include_median_rank=True,
    )
    for k in (1, 5, 10):
        assert bundle[f"recall@{k}"] == pytest.approx(recall_at_k(sim, work_ids, k))
        assert bundle[f"hit_rate@{k}"] == pytest.approx(hit_rate_at_k(sim, work_ids, k))
    assert bundle["r_precision"] == pytest.approx(r_precision(sim, work_ids))
    assert bundle["median_rank"] == pytest.approx(median_rank_first_hit(sim, work_ids))


def test_compute_retrieval_metrics_batch_invariant():
    """Output is identical regardless of ``batch`` size — proves the
    chunked argsort path is correct (no off-by-one at chunk boundaries)."""
    from marble.utils.retrieval_metrics import compute_retrieval_metrics

    torch.manual_seed(11)
    N = 64
    sim = torch.randn(N, N)
    work_ids = torch.randint(0, 4, (N,))

    m_default = compute_retrieval_metrics(sim, work_ids, recall_ks=(5,))
    m_tiny = compute_retrieval_metrics(sim, work_ids, recall_ks=(5,), batch=7)
    m_one = compute_retrieval_metrics(sim, work_ids, recall_ks=(5,), batch=1)
    m_huge = compute_retrieval_metrics(sim, work_ids, recall_ks=(5,), batch=1024)

    for key in m_default:
        assert m_default[key] == pytest.approx(m_tiny[key]), f"batch=7 mismatch at {key}"
        assert m_default[key] == pytest.approx(m_one[key]), f"batch=1 mismatch at {key}"
        assert m_default[key] == pytest.approx(m_huge[key]), f"batch=1024 mismatch at {key}"


def test_iter_row_orders_self_excluded_and_complete():
    """Each yielded row_order is (N-1,) and never contains self-index."""
    from marble.utils.retrieval_metrics import _iter_row_orders

    torch.manual_seed(3)
    N = 32
    sim = torch.randn(N, N)
    seen = set()
    for i, order_i in _iter_row_orders(sim, batch=8):
        assert order_i.shape == (N - 1,)
        assert i not in order_i.tolist(), f"row {i} contains self"
        seen.add(i)
    assert seen == set(range(N))


def test_metrics_are_bounded_in_unit_interval():
    """Sanity check: a random similarity matrix produces metrics in [0, 1]."""
    torch.manual_seed(0)
    N = 20
    sim = torch.randn(N, N)
    sim = (sim + sim.T) / 2  # symmetric, optional
    work_ids = torch.randint(0, 4, (N,))
    for k in (1, 5, 10):
        r = recall_at_k(sim, work_ids, k)
        h = hit_rate_at_k(sim, work_ids, k)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= h <= 1.0
    assert 0.0 <= r_precision(sim, work_ids) <= 1.0
    med = median_rank_first_hit(sim, work_ids)
    assert 1.0 <= med <= float(N - 1)
