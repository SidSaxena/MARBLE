"""Unit tests for :func:`compute_within_group_multilabel_map`.

Within-group, per-group-sub-matrix, multi-label motif retrieval MAP (the metric
behind the BPS-Motif within-piece phrase-window task). Ported + generalised from
the shuffle-control-validated leitmotifs prototype
(``scripts/eval/bps_within_piece_metric.py::within_movement_map``).

Contract under test (per query window ``q`` with ``letters[q] != {}``):
  * gallery = items in the SAME group, ``!= q``, occ-disjoint from ``q``
    (``occ_ids[w] & occ_ids[q] == {}`` — same-occurrence exclusion);
  * relevant = gallery items sharing >=1 letter with ``q``;
  * rank the gallery by descending cosine on (already-normalised) ``embs``;
  * standard AP per scorable query; mean over scorable queries;
  * NaN if no query is scorable (no group has a genuine same-letter peer).

Tiny corpora with hand-computable expectations, mirroring
``tests/test_compute_map_self_exclusion.py`` discipline.
"""

from __future__ import annotations

import math

import torch

from marble.utils.retrieval_metrics import (
    compute_within_group_multilabel_map,
    compute_within_group_multilabel_map_with_null,
)


def _orthogonal_rows(n: int, dim: int | None = None) -> torch.Tensor:
    """``n`` mutually-orthogonal unit rows (one-hot in distinct axes)."""
    dim = dim or n
    e = torch.zeros(n, dim)
    for i in range(n):
        e[i, i] = 1.0
    return e


def test_same_group_shared_letter_is_relevant():
    """Two same-group windows sharing letter 'a' (occ-disjoint), each the
    other's only same-group peer, cosine-closest to each other → MAP = 1.0."""
    # 4 items: items 0,1 in group 0 share 'a'; items 2,3 in group 1 share 'b'.
    embs = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0],  # 0
            [0.9, 1.0, 0.0, 0.0],  # 1  (closest to 0)
            [0.0, 0.0, 1.0, 0.9],  # 2
            [0.0, 0.0, 0.9, 1.0],  # 3  (closest to 2)
        ]
    )
    groups = [0, 0, 1, 1]
    letters = [{"a"}, {"a"}, {"b"}, {"b"}]
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}]
    m = compute_within_group_multilabel_map(embs, groups, letters, occ)
    assert abs(m - 1.0) < 1e-6, m


def test_cross_group_never_relevant():
    """A query whose only same-letter peer is in a DIFFERENT group has no
    relevant gallery item → query is skipped. Here every group is a singleton
    sharing 'a' across groups, so NO query is scorable → NaN."""
    embs = _orthogonal_rows(3, 4)
    groups = [0, 1, 2]
    letters = [{"a"}, {"a"}, {"a"}]  # same letter, but different groups
    occ = [{"o0"}, {"o1"}, {"o2"}]
    m = compute_within_group_multilabel_map(embs, groups, letters, occ)
    assert math.isnan(m), m


def test_same_occurrence_excluded_from_gallery():
    """A relevant same-letter same-group peer that shares an occurrence id with
    the query is EXCLUDED from the gallery. Here the only same-group same-letter
    peer shares the query's occurrence → it's removed → no relevant left → the
    query is unscorable → NaN."""
    embs = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0],  # 0  query
            [0.9, 1.0, 0.0, 0.0],  # 1  shares occ 'shared' with 0
        ]
    )
    groups = [0, 0]
    letters = [{"a"}, {"a"}]
    occ = [{"shared"}, {"shared"}]  # same occurrence id → excluded
    m = compute_within_group_multilabel_map(embs, groups, letters, occ)
    assert math.isnan(m), m


def test_two_letter_window_relevant_to_either_letter():
    """A 2-letter query {'a','b'} is relevant to a window carrying EITHER 'a'
    or 'b' (>=1-letter overlap). Build group 0 = {q(ab), wa(a), wb(b)} all
    occ-disjoint; q's gallery = {wa, wb}, both relevant. With wa & wb ranked
    above nothing else, AP(q) = 1.0. The other two queries (wa, wb) also each
    retrieve q (shares their letter) at rank 1 → MAP = 1.0."""
    embs = torch.tensor(
        [
            [1.0, 0.8, 0.8],  # q  (ab) — closest to both wa, wb
            [0.8, 1.0, 0.1],  # wa (a)
            [0.8, 0.1, 1.0],  # wb (b)
        ]
    )
    groups = [0, 0, 0]
    letters = [{"a", "b"}, {"a"}, {"b"}]
    occ = [{"oq"}, {"owa"}, {"owb"}]
    m = compute_within_group_multilabel_map(embs, groups, letters, occ)
    assert abs(m - 1.0) < 1e-6, m


def test_all_singleton_letters_nan():
    """Every window carries a unique letter (no two windows share a letter) →
    no query has a relevant peer → NaN."""
    embs = _orthogonal_rows(4)
    groups = [0, 0, 0, 0]  # all same group
    letters = [{"a"}, {"b"}, {"c"}, {"d"}]  # all distinct
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}]
    m = compute_within_group_multilabel_map(embs, groups, letters, occ)
    assert math.isnan(m), m


def test_empty_letters_query_skipped_but_serves_as_distractor():
    """A window with NO letters is never a query (empty letters), but it CAN sit
    in the gallery as a non-relevant distractor. Group 0: q(a), peer(a),
    distractor(no letters). q's gallery = {peer, distractor}; only peer is
    relevant. If the distractor outranks the peer, AP(q) = 1/2."""
    embs = torch.tensor(
        [
            [1.0, 0.5, 0.9, 0.0],  # 0  q(a): distractor(2) ranks above peer(1)
            [0.5, 1.0, 0.4, 0.0],  # 1  peer(a)
            [0.9, 0.4, 1.0, 0.0],  # 2  distractor (no letters)
            [0.0, 0.0, 0.0, 1.0],  # 3  filler, different group
        ]
    )
    groups = [0, 0, 0, 1]
    letters = [{"a"}, {"a"}, set(), {"z"}]
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}]
    m = compute_within_group_multilabel_map(embs, groups, letters, occ)
    # Only query 0 is scorable (1 is also a query: its gallery = {0(a relevant),
    # 2(distractor)}; for q=1, 0 is at cos .5, 2 at .4 → relevant 0 ranks first
    # → AP=1.0). q=0: peer at .5, distractor at .9 → distractor rank1, peer
    # rank2 → AP = 1/2. MAP = (0.5 + 1.0)/2 = 0.75.
    assert abs(m - 0.75) < 1e-6, m


# ── permutation-null variant (Method A, within-group) ───────────────────────


def test_null_real_matches_plain_metric():
    """AUDIT GATE: the quad's `real` must equal the audited plain metric exactly."""
    embs = torch.tensor(
        [
            [1.0, 0.8, 0.8],
            [0.8, 1.0, 0.1],
            [0.8, 0.1, 1.0],
        ]
    )
    groups = [0, 0, 0]
    letters = [{"a", "b"}, {"a"}, {"b"}]
    occ = [{"oq"}, {"owa"}, {"owb"}]
    plain = compute_within_group_multilabel_map(embs, groups, letters, occ)
    real, _nm, _ns, _p = compute_within_group_multilabel_map_with_null(
        embs, groups, letters, occ, n_perms=50
    )
    assert abs(real - plain) < 1e-12, (real, plain)


def test_null_separable_lifts_above_chance():
    """One same-letter pair among unique-letter distractors → real 1.0, strong
    positive lift, real at the top of the null distribution.

    A single ``a``-pair sits on the only two mutually-near windows; four
    distractors carry unique letters (never relevant) and are mutually orthogonal.
    Random within-group reassignment rarely lands the ``a``-pair on the near
    windows, so null_mean is far below the real MAP and real is in the upper tail.
    (We assert the mechanism — positive lift, upper-tail real — not a strict
    p<0.05, which a 6-window toy cannot reach; significance is a property of the
    real 32-movement corpus.)
    """
    embs = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.0, 0.0, 0.0],  # 0  {a}  (near 1)
            [0.9, 1.0, 0.0, 0.0, 0.0, 0.0],  # 1  {a}  (near 0)
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # 2  {c}  distractor
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # 3  {d}  distractor
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # 4  {e}  distractor
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # 5  {f}  distractor
        ]
    )
    groups = [0, 0, 0, 0, 0, 0]
    letters = [{"a"}, {"a"}, {"c"}, {"d"}, {"e"}, {"f"}]
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}, {"o4"}, {"o5"}]
    real, null_mean, _ns, p = compute_within_group_multilabel_map_with_null(
        embs, groups, letters, occ, n_perms=300, seed=0
    )
    assert abs(real - 1.0) < 1e-9, real
    assert null_mean < real - 0.2, (null_mean, real)  # strong positive lift
    assert p < 0.2, p  # real in the upper tail


def test_null_permutation_is_within_group_not_global():
    """Distinguishing test: FAILS if the null permuted labels globally.

    (Verified empirically — see below — that within-group and global nulls give
    DIFFERENT null_means on this fixture, so the assertions actually discriminate.
    An earlier version used [{x},{x},{x},{y}] which a global null also scored 1.0,
    giving false confidence; this fixture fixes that.)

    Layout: group 0 = positions {1,2}, both ``{c}`` and mutually nearest →
    scorable, MAP 1.0. group 1 = positions {0,3,4} = ``set(), set(), {a,b}`` →
    its only lettered window ({a,b}) has no same-group same-letter peer → group 1
    is NEVER scorable.

      * WITHIN-group null: group 0's two identical ``{c}`` sets are
        permutation-invariant (and group 1 is unscorable under any permutation),
        so every permutation reproduces MAP 1.0 → null_mean == 1.0, std 0, p 1.0.
      * GLOBAL null (the WRONG one): a ``{c}`` can migrate into group 1 while
        group 0 receives ``set()``/``{a,b}``, breaking group 0's scorability →
        null_mean ≈ 0.81 (empirically), which FAILS the ``null_mean == 1.0``
        assertion below. That is exactly the within-group property we lock here.
    """
    embs = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],  # 0  group1 set()
            [1.0, 0.9, 0.0, 0.0, 0.0],  # 1  group0 {c}
            [0.9, 1.0, 0.0, 0.0, 0.0],  # 2  group0 {c} (nearest to 1)
            [0.0, 0.0, 0.0, 1.0, 0.0],  # 3  group1 set()
            [0.0, 0.0, 0.0, 0.0, 1.0],  # 4  group1 {a,b} (no same-group peer)
        ]
    )
    groups = [1, 0, 0, 1, 1]
    letters = [set(), {"c"}, {"c"}, set(), {"a", "b"}]
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}, {"o4"}]
    real, null_mean, null_std, p = compute_within_group_multilabel_map_with_null(
        embs, groups, letters, occ, n_perms=300, seed=0
    )
    assert abs(real - 1.0) < 1e-9, real
    assert abs(null_mean - 1.0) < 1e-9, (null_mean, real)  # within-group → invariant
    assert null_std < 1e-9, null_std
    assert abs(p - 1.0) < 1e-9, p


def test_null_seed_deterministic():
    embs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.99, 0.0, 0.0],
        ]
    )
    groups = [0, 0, 0, 0]
    letters = [{"a"}, {"a"}, {"b"}, {"b"}]
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}]
    a = compute_within_group_multilabel_map_with_null(
        embs, groups, letters, occ, n_perms=100, seed=7
    )
    b = compute_within_group_multilabel_map_with_null(
        embs, groups, letters, occ, n_perms=100, seed=7
    )
    assert a == b


def test_null_nan_when_nothing_scorable():
    """Every letter unique → plain metric NaN → quad all-NaN."""
    embs = _orthogonal_rows(4)
    groups = [0, 0, 0, 0]
    letters = [{"a"}, {"b"}, {"c"}, {"d"}]
    occ = [{"o0"}, {"o1"}, {"o2"}, {"o3"}]
    real, null_mean, null_std, p = compute_within_group_multilabel_map_with_null(
        embs, groups, letters, occ
    )
    assert all(x != x for x in (real, null_mean, null_std, p))  # all NaN
