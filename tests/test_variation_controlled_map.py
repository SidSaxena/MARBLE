"""Variation-controlled relevance for compute_perpair_map (the VGMIDITVar confound fix).

Normal relevance = same ``work_id``. The confound: the off-diagonal (cross-instrument)
relevant set contains the query's own variation re-rendered in the target timbre (an audio
near-duplicate). With ``require_different_variation=True`` those same-(work_id, variation)
twins are masked out of BOTH relevance and the gallery, so cross vs within becomes
apples-to-apples (retrieve a *different* variation).

Hand-computed 5-item fixture (query_condition=0, target_condition=1):
  idx: work  cond  var
   0:   W1    0     0   (query)   sim[0] = [1.0, 0.9, 0.4, 0.7, 0.5]
   1:   W1    1     0   (twin of 0, in gallery)
   2:   W1    1     1   (different-variation relevant for q0)
   3:   W2    1     0   (relevant for q4)          sim[4] = [0.5, 0.1, 0.2, 0.95, 1.0]
   4:   W2    0     0   (query)

Gallery (cond 1) = {1, 2, 3}.
  q0 relevant (same work W1) = {1 (twin), 2 (diff-var)}.
    no control: rank {1:0.9, 3:0.7, 2:0.4} → rel 1@1, 2@3 → AP = (1 + 2/3)/2 = 0.8333.
    control:    mask twin 1 → rank {3:0.7, 2:0.4} → rel 2@2 → AP = 0.5.
  q4 relevant (same work W2) = {3}. var(3)=0 == var(q4)=0 → a twin.
    no control: rank {3:0.95, 2:0.2, 1:0.1} → rel 3@1 → AP = 1.0.
    control:    twin masked → no relevant left → query SKIPPED.
  ⇒ no control: MAP = mean(0.8333, 1.0) = 0.91667, n=2.
     control:   MAP = 0.5, n=1.
"""

from __future__ import annotations

import torch

from marble.utils.retrieval_metrics import compute_perpair_map

SIM = torch.tensor(
    [
        [1.00, 0.90, 0.40, 0.70, 0.50],
        [0.90, 1.00, 0.30, 0.30, 0.30],
        [0.40, 0.30, 1.00, 0.30, 0.30],
        [0.70, 0.30, 0.30, 1.00, 0.95],
        [0.50, 0.10, 0.20, 0.95, 1.00],
    ]
)
WORK = [1, 1, 1, 2, 2]
COND = [0, 1, 1, 1, 0]
VAR = [0, 0, 1, 0, 0]


def test_no_control_matches_hand_computation():
    m, n = compute_perpair_map(SIM, WORK, COND, query_condition=0, target_condition=1)
    assert n == 2
    assert abs(m - 0.916667) < 1e-5


def test_variation_controlled_excludes_twins():
    m, n = compute_perpair_map(
        SIM,
        WORK,
        COND,
        query_condition=0,
        target_condition=1,
        variation_ids=VAR,
        require_different_variation=True,
    )
    assert n == 1  # q4 skipped (its only relevant is a same-variation twin)
    assert abs(m - 0.5) < 1e-5


def test_control_is_noop_without_variation_ids():
    # require_different_variation with no variation_ids provided must not change anything.
    base, _ = compute_perpair_map(SIM, WORK, COND, query_condition=0, target_condition=1)
    m, _ = compute_perpair_map(
        SIM,
        WORK,
        COND,
        query_condition=0,
        target_condition=1,
        variation_ids=None,
        require_different_variation=True,
    )
    assert abs(m - base) < 1e-9
