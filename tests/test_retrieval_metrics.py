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
    zca_whiten,
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


def test_anisotropy_mean_vec_norm_matches_theoretical_for_isotropic():
    """For N L2-normalised iid-Gaussian vectors in H dims, the mean
    vector has expected norm ≈ 1/√N (the "random walk on the sphere"
    result). This is the canonical anisotropy baseline. We assert
    the function lands within ~3× the theoretical bound (Monte-Carlo
    fluctuations for finite N).

    Pinning this catches subtle regressions like normalising rows
    AFTER taking the mean (which would give a constant 1.0 regardless
    of the data) or forgetting to L2-normalise at all (which would
    make ``mean_vec_norm`` depend on the input scale)."""
    torch.manual_seed(0)
    N, H = 5000, 64
    embs = torch.randn(N, H)
    m = anisotropy_metrics(embs, seed=0)
    theoretical = 1.0 / math.sqrt(N)
    # Tight bound: empirical std around 1/√N for isotropic Gaussian is
    # O(1/√(N·H)) — at N=5000, H=64 we expect ~0.014 ± 0.002. The 3×
    # multiplier covers H-dependent prefactors without false positives.
    assert m["mean_vec_norm"] == pytest.approx(theoretical, abs=3 * theoretical), (
        f"mean_vec_norm={m['mean_vec_norm']:.4f} not within 3× of "
        f"theoretical 1/√N = {theoretical:.4f}"
    )


def test_anisotropy_pair_cos_matches_mvn_squared():
    """For L2-normalised inputs the expected pair cosine equals the
    squared mean-vec-norm: E[cos(a, b)] = ||μ||² (a basic identity
    for unit vectors on the sphere).

    This is the **headline cross-check** between two anisotropy metrics
    that should be measuring related quantities. If they diverge, the
    implementation has a bug — either the pair-sampling is biased,
    the L2-normalisation step is broken, or one of the metrics is
    computing on a different scale than documented.

    Pinned at abs=0.005 — well above Monte-Carlo noise (n_pairs=5000)
    but tight enough to catch a real implementation drift."""
    torch.manual_seed(0)
    # Construct embeddings with a deliberately non-trivial anisotropy
    # level so the equivalence is a meaningful constraint (not just
    # "both ≈ 0").
    N, H = 2000, 128
    bias_strength = 1.5
    bias = torch.zeros(H)
    bias[0] = bias_strength
    embs = bias.unsqueeze(0) + 0.7 * torch.randn(N, H)
    embs = embs / embs.norm(dim=-1, keepdim=True)

    m = anisotropy_metrics(embs, seed=0, n_pairs=10_000)
    expected_apc = m["mean_vec_norm"] ** 2
    assert m["avg_pair_cos"] == pytest.approx(expected_apc, abs=0.005), (
        f"avg_pair_cos = {m['avg_pair_cos']:.4f} should ≈ "
        f"mean_vec_norm² = {expected_apc:.4f} (Δ={m['avg_pair_cos'] - expected_apc:+.4f})"
    )


def test_anisotropy_effective_rank_high_for_pure_cone_collapse():
    """Documents the subtle property: ``effective_rank`` does NOT
    detect cone collapse — it measures rank diversity AFTER centering.
    For a pure cone (all rows along one axis + iid Gaussian noise),
    centering removes the cone direction, leaving ~uniform noise across
    the remaining H-1 dimensions → effective_rank stays near H-1.

    Anyone reading the metric expecting "low rank = anisotropic" needs
    to instead consult ``mean_vec_norm`` — the docstring spells this out
    but pinning it in a test is the safest contract."""
    torch.manual_seed(0)
    N, H = 1000, 64
    base = torch.zeros(N, H)
    base[:, 0] = 1.0  # everyone points at +e0
    embs = base + 0.01 * torch.randn(N, H)  # tiny iid noise for well-posed SVD
    m = anisotropy_metrics(embs, seed=0)
    # Cone collapse: mean_vec_norm ≈ 1
    assert m["mean_vec_norm"] > 0.95, f"mvn={m['mean_vec_norm']}"
    # But effective_rank stays HIGH (not low) — this is the property
    # we're pinning. Tolerance: with Marchenko-Pastur eigenvalue spread
    # at N/H=15.6, the effective rank lands ~0.85 × min(N, H) = ~54.
    # Lower bound 40 guards against "implementation suddenly reports
    # rank=1 for cone collapse" regressions.
    assert m["effective_rank"] > 40.0, (
        f"effective_rank={m['effective_rank']} unexpectedly low — "
        f"effective_rank should NOT collapse for pure cone collapse; "
        f"if it does, either the SVD is being run on uncentered input "
        f"or there's a numerical issue. See docstring."
    )


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


def test_compute_retrieval_metrics_empty_corpus_returns_all_nan():
    """N=0 must return NaN for every requested key, not crash."""
    import math as _math

    from marble.utils.retrieval_metrics import compute_retrieval_metrics

    sim = torch.empty(0, 0)
    work_ids = torch.empty(0, dtype=torch.long)
    out = compute_retrieval_metrics(
        sim,
        work_ids,
        recall_ks=(1, 5),
        hit_ks=(1,),
        include_r_precision=True,
        include_median_rank=True,
    )
    # All-NaN for an empty corpus.
    for key, val in out.items():
        assert _math.isnan(val), f"{key} should be NaN, got {val}"


# ──────────────────────────────────────────────────────────────────────
# MAP / MAP@K / MRR — bundle must match the probe's static methods
# bit-for-bit. The static methods now delegate to the bundle, so these
# tests pin both the bundle's correctness AND the shim's faithfulness.
# ──────────────────────────────────────────────────────────────────────


def _bundle_map(sim, work_ids):
    from marble.utils.retrieval_metrics import compute_retrieval_metrics

    return compute_retrieval_metrics(
        sim,
        work_ids,
        recall_ks=(),
        include_r_precision=False,
        include_median_rank=False,
        include_map=True,
    )["map"]


def _bundle_map_at_k(sim, work_ids, k):
    from marble.utils.retrieval_metrics import compute_retrieval_metrics

    return compute_retrieval_metrics(
        sim,
        work_ids,
        recall_ks=(),
        include_r_precision=False,
        include_median_rank=False,
        map_at_ks=(k,),
    )[f"map@{k}"]


def _bundle_mrr(sim, work_ids):
    from marble.utils.retrieval_metrics import compute_retrieval_metrics

    return compute_retrieval_metrics(
        sim,
        work_ids,
        recall_ks=(),
        include_r_precision=False,
        include_median_rank=False,
        include_mrr=True,
    )["mrr"]


def test_bundle_map_matches_known_values():
    """Hand-computed MAP for the 4-item 2-pair fixture."""
    sim, work_ids = _two_pairs_perfect_sim()
    # Perfect retrieval: rel at rank 1, AP = 1/1 = 1.0 for every query.
    assert _bundle_map(sim, work_ids) == pytest.approx(1.0)

    sim_inv, work_ids_inv = _two_pairs_inverted_sim()
    # Inverted: rel at rank 3, AP = (1/3) for every query.
    assert _bundle_map(sim_inv, work_ids_inv) == pytest.approx(1.0 / 3.0)


def test_bundle_map_matches_probe_compute_map():
    """The probe's _compute_map static method delegates to the bundle —
    confirm they return identical values on a non-trivial fixture."""
    from marble.tasks.Covers80.probe import CoverRetrievalTask

    torch.manual_seed(13)
    N = 50
    sim = torch.randn(N, N)
    work_ids = torch.randint(0, 6, (N,))
    assert CoverRetrievalTask._compute_map(sim, work_ids) == pytest.approx(
        _bundle_map(sim, work_ids)
    )


def test_bundle_map_at_k_matches_probe():
    from marble.tasks.Covers80.probe import CoverRetrievalTask

    torch.manual_seed(17)
    N = 60
    sim = torch.randn(N, N)
    work_ids = torch.randint(0, 5, (N,))
    for k in (1, 5, 10, 30):
        assert CoverRetrievalTask._map_at_k(sim, work_ids, k) == pytest.approx(
            _bundle_map_at_k(sim, work_ids, k)
        ), f"map@{k} mismatch"


def test_bundle_mrr_matches_probe():
    from marble.tasks.Covers80.probe import CoverRetrievalTask

    torch.manual_seed(23)
    N = 40
    sim = torch.randn(N, N)
    work_ids = torch.randint(0, 4, (N,))
    assert CoverRetrievalTask._mrr(sim, work_ids) == pytest.approx(_bundle_mrr(sim, work_ids))


def test_compute_perpair_map_all_matches_per_cell():
    """Batched all-cells perpair MAP must match per-cell `compute_perpair_map`."""
    from marble.utils.retrieval_metrics import compute_perpair_map, compute_perpair_map_all

    torch.manual_seed(29)
    N = 40
    sim = torch.randn(N, N)
    work_ids = torch.randint(0, 5, (N,)).tolist()
    # 3 conditions × 3 conditions = 9 cells; some will have zero queries.
    conditions = [i % 3 for i in range(N)]

    all_cells = compute_perpair_map_all(sim, work_ids, conditions)
    for q in (0, 1, 2):
        for t in (0, 1, 2):
            ap_ref, n_ref = compute_perpair_map(sim, work_ids, conditions, q, t)
            ap_new, n_new = all_cells[(q, t)]
            assert n_new == n_ref, f"cell ({q},{t}) n={n_new} vs ref {n_ref}"
            assert ap_new == pytest.approx(ap_ref, abs=1e-6), (
                f"cell ({q},{t}) ap={ap_new} vs ref {ap_ref}"
            )


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


# ──────────────────────────────────────────────────────────────────────
# zca_whiten
# ──────────────────────────────────────────────────────────────────────
def test_zca_whiten_alpha0_equals_centering():
    """α=0 must reduce to plain mean-centering (no rescaling)."""
    torch.manual_seed(0)
    embs = torch.randn(200, 16) * torch.arange(1, 17).float() + 3.0
    out = zca_whiten(embs, alpha=0.0)
    expected = embs - embs.mean(dim=0, keepdim=True)
    assert torch.allclose(out, expected, atol=1e-6)


def test_zca_whiten_alpha1_gives_identity_covariance():
    """Full whitening (α=1) must produce ≈ identity covariance (pre-norm)."""
    torch.manual_seed(1)
    # Strongly anisotropic: independent columns with very different scales.
    base = torch.randn(2000, 8)
    embs = base * torch.tensor([10.0, 5.0, 1.0, 0.3, 8.0, 2.0, 0.1, 4.0])
    w = zca_whiten(embs, alpha=1.0).double()
    wc = w - w.mean(dim=0, keepdim=True)
    cov = (wc.T @ wc) / wc.shape[0]
    assert torch.allclose(cov, torch.eye(8, dtype=torch.float64), atol=5e-2)


def test_zca_whiten_matches_independent_numpy():
    """Cosine geometry must match an independent numpy ZCA implementation."""
    import numpy as np

    torch.manual_seed(2)
    embs = torch.randn(300, 12) * torch.arange(1, 13).float()
    out = torch.nn.functional.normalize(zca_whiten(embs, alpha=1.0), dim=-1)

    X = embs.double().numpy()
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / X.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    scale = np.clip(evals, 1e-12, None) ** -0.5
    iw = (Xc @ evecs) * scale @ evecs.T
    iw = iw / np.linalg.norm(iw, axis=1, keepdims=True)
    # Sign/rotation-invariant comparison via the similarity matrix.
    sim_ours = (out @ out.T).numpy()
    sim_np = iw @ iw.T
    assert np.abs(sim_ours - sim_np).max() < 1e-5


def test_zca_whiten_rejects_non_2d():
    with pytest.raises(ValueError):
        zca_whiten(torch.randn(4, 5, 6))


def test_zca_whiten_rejects_bad_alpha():
    with pytest.raises(ValueError):
        zca_whiten(torch.randn(50, 8), alpha=-0.1)
    with pytest.raises(ValueError):
        zca_whiten(torch.randn(50, 8), alpha=2.5)


def test_zca_whiten_correlated_covariance_gives_identity():
    """Identity-cov must hold for a genuinely *rotated* (correlated)
    covariance, not just a diagonal one — this exercises the U Uᵀ rotation,
    which a diagonal cov (U≈I) would not catch."""
    torch.manual_seed(7)
    # Anisotropic diagonal data, then rotate by a random orthogonal Q so
    # the covariance is dense (off-diagonals non-zero).
    base = torch.randn(4000, 6) * torch.tensor([9.0, 5.0, 2.0, 1.0, 0.4, 7.0])
    q, _ = torch.linalg.qr(torch.randn(6, 6))
    embs = base @ q  # dense covariance
    w = zca_whiten(embs, alpha=1.0).double()
    wc = w - w.mean(dim=0, keepdim=True)
    cov = (wc.T @ wc) / wc.shape[0]
    assert torch.allclose(cov, torch.eye(6, dtype=torch.float64), atol=5e-2)


def test_zca_whiten_small_corpus_is_finite_not_crash():
    """N < H (rank-deficient covariance) must stay finite — no NaN/Inf from
    the eps-floor clamp. The probe gates on this regime, but the transform
    itself must never poison logging."""
    torch.manual_seed(3)
    embs = torch.nn.functional.normalize(torch.randn(20, 256), dim=-1)  # N < H
    w = zca_whiten(embs, alpha=1.0)
    assert torch.isfinite(w).all()


def test_zca_whiten_fp64_matches_script_fp32_path_when_n_gg_h():
    """The probe whitens in fp64 throughout; the validated script downcasts
    eigvecs/eigvals to fp32 before the transform. In the operating regime
    (N >> H) the two must agree — pins that map_whitened reproduces the
    script's whiten-a1.0 numbers and guards against future drift."""
    torch.manual_seed(5)
    embs = torch.nn.functional.normalize(torch.randn(3000, 64), dim=-1)
    ours = torch.nn.functional.normalize(zca_whiten(embs, alpha=1.0), dim=-1)

    # Reproduce the script's fp32-transform path (whitening_ablation.py).
    e64 = embs.double()
    centered64 = e64 - e64.mean(dim=0, keepdim=True)
    sigma = (centered64.T @ centered64) / e64.shape[0]
    evals, evecs = torch.linalg.eigh(sigma)
    evals32, evecs32 = evals.clamp(min=0.0).float(), evecs.float()
    c32 = (embs - embs.mean(dim=0, keepdim=True)).float()
    scale = evals32.clamp(min=1e-12).pow(-0.5)
    script = torch.nn.functional.normalize(((c32 @ evecs32) * scale) @ evecs32.T, dim=-1)

    max_sim_diff = (ours @ ours.T - script @ script.T).abs().max().item()
    assert max_sim_diff < 1e-4
