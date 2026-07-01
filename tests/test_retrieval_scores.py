"""Tests for the retrieval score-distribution accumulator.

CoverRetrievalTask scores retrieval by cosine RANKING (MAP) and discards the raw
similarities. This module accumulates, per (query_condition, target_condition)
cell, the distribution of candidate cosine scores split into RELEVANT (same
work_id) vs DISTRACTOR (different work_id), self-excluded — so we can inspect the
score distribution and relevant-vs-distractor separation, not just the aggregate
MAP. Streaming-capable (per-query-row) so it works at N=100k without an N×N
materialisation.

Hand-computed 4×4 fixture:
  work_ids = [10, 10, 20, 20], conditions = [0, 1, 0, 1]
  sim (symmetric, diag=1.0):
        j0    j1    j2    j3
   i0 [1.0,  0.9,  0.2,  0.1]
   i1 [0.9,  1.0,  0.3,  0.4]
   i2 [0.2,  0.3,  1.0,  0.8]
   i3 [0.1,  0.4,  0.8,  1.0]
  Self excluded. Overall relevant pool (same work) = [0.9,0.9,0.8,0.8] → mean 0.85.
  Overall distractor pool = [0.2,0.1, 0.3,0.4, 0.2,0.3, 0.1,0.4] → mean 0.25.
"""

from __future__ import annotations

import torch

from marble.utils.retrieval_scores import RetrievalScoreAccumulator, score_distributions

SIM = torch.tensor(
    [
        [1.0, 0.9, 0.2, 0.1],
        [0.9, 1.0, 0.3, 0.4],
        [0.2, 0.3, 1.0, 0.8],
        [0.1, 0.4, 0.8, 1.0],
    ]
)
WORK = [10, 10, 20, 20]
COND = [0, 1, 0, 1]


def test_overall_relevant_and_distractor_means():
    d = score_distributions(SIM, WORK)
    o = d["overall"]
    assert o["relevant"]["n"] == 4
    assert abs(o["relevant"]["mean"] - 0.85) < 1e-6
    assert o["distractor"]["n"] == 8
    assert abs(o["distractor"]["mean"] - 0.25) < 1e-6
    assert abs(o["separation"] - 0.60) < 1e-6


def test_self_excluded_no_diagonal_ones():
    # If self (sim=1.0) leaked in, distractor/relevant maxima would be 1.0.
    d = score_distributions(SIM, WORK)
    assert d["overall"]["relevant"]["max"] <= 0.9 + 1e-5
    assert d["overall"]["distractor"]["max"] <= 0.4 + 1e-5


def test_conditions_split_into_cells():
    d = score_distributions(SIM, WORK, conditions=COND)
    cells = d["cells"]
    # cell (q=0, t=1): relevant [0.9, 0.8], distractor [0.1, 0.3]
    c01 = cells[(0, 1)]
    assert c01["relevant"]["n"] == 2
    assert abs(c01["relevant"]["mean"] - 0.85) < 1e-6
    assert c01["distractor"]["n"] == 2
    assert abs(c01["distractor"]["mean"] - 0.2) < 1e-6
    # cell (q=0, t=0): only different-work candidates → 0 relevant, 2 distractor [0.2,0.2]
    c00 = cells[(0, 0)]
    assert c00["relevant"]["n"] == 0
    assert c00["distractor"]["n"] == 2
    assert abs(c00["distractor"]["mean"] - 0.2) < 1e-6


def test_histograms_sum_to_counts_and_edges_span_range():
    d = score_distributions(SIM, WORK, n_bins=10, lo=-1.0, hi=1.0)
    o = d["overall"]
    assert sum(o["relevant"]["hist"]) == o["relevant"]["n"]
    assert sum(o["distractor"]["hist"]) == o["distractor"]["n"]
    assert len(o["relevant"]["edges"]) == 11  # n_bins + 1
    assert o["relevant"]["edges"][0] == -1.0 and o["relevant"]["edges"][-1] == 1.0


def test_empty_relevant_cell_has_null_separation():
    # Cell (0,0): both candidates are different-work → 0 relevant. Its separation
    # must be None (not 0.0 - distractor_mean), so callers can't average garbage in.
    d = score_distributions(SIM, WORK, conditions=COND)
    c00 = d["cells"][(0, 0)]
    assert c00["relevant"]["n"] == 0
    assert c00["separation"] is None
    # A populated cell keeps a real separation.
    assert d["cells"][(0, 1)]["separation"] is not None


def test_sentinel_condition_excluded_from_cells():
    # Condition -1 (unparsed) must not spawn cells, matching the MAP grid's c!=-1 filter.
    work = [10, 10, 20, 20]
    cond = [-1, 1, -1, 1]
    d = score_distributions(SIM, work, conditions=cond)
    assert all(-1 not in k for k in d["cells"]), "sentinel -1 leaked into cells"
    # The one real cell (1,1) still forms.
    assert (1, 1) in d["cells"]


def test_variation_control_splits_twin_out_of_relevant():
    # 4 items, one work=10 with variations [0,0] rendered at conditions [0,1]:
    #   query i0 (work10,var0,cond0). Its cond-1 relevant is i1 (work10,var0) — a TWIN.
    # Confounded 'relevant' includes the twin; 'relevant_diffvar' excludes it.
    work = [10, 10, 20, 20]
    cond = [0, 1, 0, 1]
    var = [0, 0, 0, 0]  # every same-work pair is a same-variation twin
    d = score_distributions(SIM, work, conditions=cond, variations=var)
    c01 = d["cells"][(0, 1)]
    # Cell (0,1) aggregates both cond-0 queries: i0→i1 (work10 twin) and i2→i3
    # (work20 twin). Confounded relevant pool = {i1, i3}, both same-variation twins.
    assert c01["relevant"]["n"] == 2
    # variation-controlled relevant pool excludes both twins → empty → varctl sep None
    assert c01["relevant_diffvar"]["n"] == 0
    assert c01["separation_varctl"] is None


def test_variation_control_keeps_different_variation_relevant():
    # work=10 has TWO variations; the different-variation peer survives control.
    #   idx: work var cond ; sim rows below
    sim = torch.tensor(
        [
            [1.0, 0.9, 0.5, 0.2],  # i0 work10 var0 cond0 ; i1 is twin, i? diff-var
            [0.9, 1.0, 0.3, 0.4],  # i1 work10 var0 cond1 (twin of i0)
            [0.5, 0.3, 1.0, 0.8],  # i2 work10 var1 cond1 (different variation, cond1)
            [0.2, 0.4, 0.8, 1.0],
        ]
    )
    work = [10, 10, 10, 20]
    cond = [0, 1, 1, 1]
    var = [0, 0, 1, 0]
    d = score_distributions(sim, work, conditions=cond, variations=var)
    c01 = d["cells"][(0, 1)]
    # cond-1 same-work candidates for q0: i1 (twin, sim .9) + i2 (diff-var, sim .5)
    assert c01["relevant"]["n"] == 2  # confounded: both
    assert c01["relevant_diffvar"]["n"] == 1  # controlled: only i2
    assert abs(c01["relevant_diffvar"]["mean"] - 0.5) < 1e-6
    assert c01["separation_varctl"] is not None


def test_histogram_captures_out_of_range_tail():
    # Scores above hi must land in the top bin, not vanish (sum(hist) == n always).
    sim = torch.tensor(
        [
            [1.0, 1.5, 0.2, 0.1],  # 1.5 > hi=1.0
            [1.5, 1.0, 0.3, 0.4],
            [0.2, 0.3, 1.0, 0.8],
            [0.1, 0.4, 0.8, 1.0],
        ]
    )
    d = score_distributions(sim, WORK, n_bins=10, lo=-1.0, hi=1.0)
    o = d["overall"]
    assert sum(o["relevant"]["hist"]) == o["relevant"]["n"]
    assert sum(o["distractor"]["hist"]) == o["distractor"]["n"]


def test_streaming_accumulator_matches_full_matrix():
    # Feeding rows one batch at a time must equal the whole-matrix convenience.
    acc = RetrievalScoreAccumulator(WORK, COND, n_bins=10, lo=-1.0, hi=1.0)
    acc.update([0, 1], SIM[[0, 1]])  # first 2 query rows
    acc.update([2, 3], SIM[[2, 3]])  # next 2
    streamed = acc.result()
    full = score_distributions(SIM, WORK, conditions=COND, n_bins=10)
    assert streamed["overall"]["relevant"]["n"] == full["overall"]["relevant"]["n"]
    assert abs(streamed["overall"]["separation"] - full["overall"]["separation"]) < 1e-6
    assert (
        streamed["cells"][(0, 1)]["relevant"]["hist"] == full["cells"][(0, 1)]["relevant"]["hist"]
    )
