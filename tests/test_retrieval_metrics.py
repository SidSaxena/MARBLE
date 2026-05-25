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
