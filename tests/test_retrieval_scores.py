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
