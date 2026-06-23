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

from marble.utils.retrieval_metrics import compute_within_group_multilabel_map


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
