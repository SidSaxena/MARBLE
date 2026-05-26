"""Unit tests for the self-exclusion fix in CoverRetrievalTask._compute_map
(audit issue #6).

The previous implementation used ``sims_i[i] = -2.0`` to exclude self
from the ranking. Because -2.0 is finite and within the float range,
``argsort(descending=True)`` placed self at the LAST position rather
than removing it. ``is_rel`` then matched self (``work_ids[i] ==
work_ids[i]`` is always True), inflating ``n_relevant`` by 1 and adding
a spurious hit at rank N. The fix uses ``-inf`` + last-column drop,
matching ``marble.utils.retrieval_metrics._ranking_order``.

These tests construct tiny corpora with known-correct MAP values
(computable by hand) so any regression of the fix is caught immediately.
"""

from __future__ import annotations

import torch

from marble.tasks.Covers80.probe import CoverRetrievalTask


def _ap_by_hand(hit_ranks: list[int], n_relevant: int) -> float:
    """Reference: AP = (1/n_relevant) * sum over true positives of
    (hits_so_far / rank_at_TP). ``hit_ranks`` is 1-indexed."""
    hits = 0
    ap = 0.0
    for r in sorted(hit_ranks):
        hits += 1
        ap += hits / r
    return ap / n_relevant


def test_two_works_two_versions_each_perfect_retrieval():
    """4 items, 2 works (A, B), each with 2 versions. Cosine ordered
    so the matching cover is always rank 1 (other-work items follow).
    MAP should be exactly 1.0."""
    work_ids = torch.tensor([0, 0, 1, 1])
    # Custom similarities: item 0's pair is item 1 (highest); 1's pair
    # is 0; 2's pair is 3; 3's pair is 2.
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.2],  # query 0 → 1 (cover, rank 1), 3, 2
            [0.9, 1.0, 0.2, 0.1],  # query 1 → 0 (cover, rank 1), 2, 3
            [0.1, 0.2, 1.0, 0.9],  # query 2 → 3 (cover, rank 1), 1, 0
            [0.2, 0.1, 0.9, 1.0],  # query 3 → 2 (cover, rank 1), 0, 1
        ]
    )
    map_value = CoverRetrievalTask._compute_map(sim, work_ids)
    assert abs(map_value - 1.0) < 1e-6, f"Expected 1.0, got {map_value}"


def test_two_works_one_misranks_to_position_2():
    """Same 4-item / 2-works setup, but query 0's true cover is at
    rank 2 (after a distractor). AP for query 0 = 1/2 = 0.5. Other
    queries still 1.0. MAP = (0.5 + 1.0 + 1.0 + 1.0) / 4 = 0.875."""
    work_ids = torch.tensor([0, 0, 1, 1])
    sim = torch.tensor(
        [
            [1.0, 0.5, 0.9, 0.2],  # query 0 → 2 (distractor), 1 (cover, rank 2), 3
            [0.5, 1.0, 0.2, 0.1],  # query 1 → 0 (cover, rank 1), 2, 3
            [0.1, 0.2, 1.0, 0.9],  # query 2 → 3 (cover, rank 1), 1, 0
            [0.2, 0.1, 0.9, 1.0],  # query 3 → 2 (cover, rank 1), 0, 1
        ]
    )
    map_value = CoverRetrievalTask._compute_map(sim, work_ids)
    expected = (0.5 + 1.0 + 1.0 + 1.0) / 4
    assert abs(map_value - expected) < 1e-6, f"Expected {expected}, got {map_value}"


def test_three_relevants_per_query_all_at_top():
    """4 items, 1 work — all 3 others are relevant. With perfect
    ranking (all 3 at top), AP = 1.0 per query → MAP = 1.0.

    Note: this test is by design NOT diagnostic of the bug on its own
    (with OLD -2.0 code, AP also rounds to 1.0 because every position
    in the buggy is_rel is True). It guards the perfect-case baseline.
    The bug-discriminating tests are
    ``test_two_works_one_misranks_to_position_2``,
    ``test_self_not_in_is_rel_via_inf``, and
    ``test_mrr_skips_queries_with_no_relevant``."""
    work_ids = torch.tensor([0, 0, 0, 0])
    # Identity matrix scaled so self is 1.0, others are decreasing
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.8, 0.7],
            [0.9, 1.0, 0.8, 0.7],
            [0.8, 0.9, 1.0, 0.7],
            [0.7, 0.8, 0.9, 1.0],
        ]
    )
    map_value = CoverRetrievalTask._compute_map(sim, work_ids)
    assert abs(map_value - 1.0) < 1e-6, f"Expected 1.0, got {map_value}"


def test_no_relevants_skipped():
    """Each item has a unique work_id → no query has a true relevant
    peer. MAP should be 0.0 (every query is skipped via the
    ``n_relevant == 0`` guard)."""
    work_ids = torch.tensor([0, 1, 2, 3])
    sim = torch.eye(4) + 0.5  # self=1.5, others=0.5
    map_value = CoverRetrievalTask._compute_map(sim, work_ids)
    assert map_value == 0.0, f"Expected 0.0, got {map_value}"


def test_self_not_in_is_rel_via_inf():
    """Direct check: with -inf + last-column drop, self never appears
    in the ranking sequence. Single query, 5 items, all same work_id."""
    work_ids = torch.tensor([7, 7, 7, 7, 7])
    sim = torch.full((5, 5), 0.1)
    sim.fill_diagonal_(1.0)  # self-sim=1.0, others=0.1
    # Query 0: with the fix, sims_i[0]=-inf, order over remaining
    # positions = [1,2,3,4] (all relevant) → AP = 1.0.
    map_value = CoverRetrievalTask._compute_map(sim, work_ids)
    assert abs(map_value - 1.0) < 1e-6, f"Expected 1.0, got {map_value}"


def test_map_at_k_self_exclusion_at_k_equals_one():
    """_map_at_k with k=1 and 2 items same work + 2 others. Top-1 for
    each query should be the cover (rank 1), giving MAP@1 = 1.0."""
    work_ids = torch.tensor([0, 0, 1, 1])
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.9],
            [0.2, 0.1, 0.9, 1.0],
        ]
    )
    map_at_1 = CoverRetrievalTask._map_at_k(sim, work_ids, k=1)
    # n_total = 1 (only 1 relevant per query, after self excluded).
    # AP = 1/1 / 1 = 1.0 for each query.
    assert abs(map_at_1 - 1.0) < 1e-6, f"Expected 1.0, got {map_at_1}"


def test_mrr_skips_queries_with_no_relevant():
    """MRR should ignore queries with no true relevant. Without the
    last-column drop, the buggy code would treat ``is_rel[N-1] = True``
    (self) as a hit, giving every query MRR = 1/N spuriously."""
    work_ids = torch.tensor([0, 1, 2, 3])  # all unique → no true relevants
    sim = torch.eye(4) + 0.1
    mrr_value = CoverRetrievalTask._mrr(sim, work_ids)
    assert mrr_value == 0.0, f"Expected 0.0, got {mrr_value}"


def test_mrr_simple_perfect():
    """MRR with each query's cover at rank 1 → 1.0."""
    work_ids = torch.tensor([0, 0, 1, 1])
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.9],
            [0.2, 0.1, 0.9, 1.0],
        ]
    )
    mrr_value = CoverRetrievalTask._mrr(sim, work_ids)
    assert abs(mrr_value - 1.0) < 1e-6, f"Expected 1.0, got {mrr_value}"
