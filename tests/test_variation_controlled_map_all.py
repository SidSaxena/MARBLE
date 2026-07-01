"""Variation control on the shared-argsort grid (compute_perpair_map_all + streaming).

The per-cell ``compute_perpair_map`` already implements + tests the VGMIDITVar
same-composition-twin confound fix (see test_variation_controlled_map.py). The
full-scale run does NOT use that per-cell path — it uses the batched
``compute_perpair_map_all`` (CPU sim) / ``compute_perpair_map_all_streaming``
(GPU, N=102 960). These tests pin those batched paths to the per-cell reference
*with variation control on*, cell-by-cell, so the confound fix is identical at
scale (not silently skipped on the streaming path).
"""

from __future__ import annotations

import torch

from marble.utils.retrieval_metrics import (
    compute_perpair_map,
    compute_perpair_map_all,
    compute_perpair_map_all_streaming,
)

# Same hand-computed fixture as test_variation_controlled_map.py.
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


def test_all_matches_percell_with_varctl():
    conds = sorted(set(COND))
    got = compute_perpair_map_all(
        SIM,
        WORK,
        COND,
        query_conds=conds,
        target_conds=conds,
        variation_ids=VAR,
        require_different_variation=True,
    )
    for q in conds:
        for t in conds:
            m, n = compute_perpair_map(
                SIM, WORK, COND, q, t, variation_ids=VAR, require_different_variation=True
            )
            gm, gn = got[(q, t)]
            assert gn == n, f"cell ({q},{t}) n mismatch: {gn} != {n}"
            assert abs(gm - m) < 1e-5, f"cell ({q},{t}) MAP mismatch: {gm} != {m}"


def test_all_varctl_differs_from_uncontrolled_cross_cell():
    conds = sorted(set(COND))
    base = compute_perpair_map_all(SIM, WORK, COND, query_conds=conds, target_conds=conds)
    ctl = compute_perpair_map_all(
        SIM,
        WORK,
        COND,
        query_conds=conds,
        target_conds=conds,
        variation_ids=VAR,
        require_different_variation=True,
    )
    # Cross cell (0,1): uncontrolled MAP=0.91667 n=2, controlled MAP=0.5 n=1.
    assert base[(0, 1)] != ctl[(0, 1)]
    assert ctl[(0, 1)][1] == 1
    assert abs(ctl[(0, 1)][0] - 0.5) < 1e-5


def test_all_varctl_is_noop_without_variation_ids():
    conds = sorted(set(COND))
    base = compute_perpair_map_all(SIM, WORK, COND, query_conds=conds, target_conds=conds)
    same = compute_perpair_map_all(
        SIM,
        WORK,
        COND,
        query_conds=conds,
        target_conds=conds,
        variation_ids=None,
        require_different_variation=True,
    )
    assert base == same


def test_streaming_matches_percell_with_varctl():
    torch.manual_seed(0)
    n, h = 12, 8
    embs = torch.randn(n, h)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work = [i // 3 for i in range(n)]  # 4 works of 3 items
    cond = [i % 2 for i in range(n)]
    var = [(i // 3) % 2 for i in range(n)]  # variation label within work
    sim = embs @ embs.T
    conds = sorted(set(cond))
    stream = compute_perpair_map_all_streaming(
        embs,
        work,
        cond,
        query_conds=conds,
        target_conds=conds,
        device="cpu",
        variation_ids=var,
        require_different_variation=True,
    )
    for q in conds:
        for t in conds:
            m, nq = compute_perpair_map(
                sim, work, cond, q, t, variation_ids=var, require_different_variation=True
            )
            sm, sn = stream[(q, t)]
            assert sn == nq, f"cell ({q},{t}) n mismatch: {sn} != {nq}"
            assert abs(sm - m) < 1e-5, f"cell ({q},{t}) MAP mismatch: {sm} != {m}"
