"""Equivalence tests for the fused GPU-resident retrieval pass.

The fused pass (marble/utils/retrieval_fused.py) must be numerically
IDENTICAL to the individual streaming passes it replaces — these are the
tests that make the ~10-20x faster path trustworthy for the confound
numbers. Everything runs on CPU here (the code is device-generic); the
full-scale CUDA validation is the MuQ-L11 VGMIDITVar-timbre oracle run.
"""

from __future__ import annotations

import numpy as np
import torch

from marble.utils.retrieval_fused import fused_retrieval_pass
from marble.utils.retrieval_metrics import (
    compute_perpair_map_all_streaming,
    compute_retrieval_metrics,
    compute_retrieval_metrics_streaming,
)
from marble.utils.retrieval_scores import RetrievalScoreAccumulator, score_distributions


def _fixture(n=64, h=16, n_works=12, n_conds=3, n_vars=2, seed=0):
    torch.manual_seed(seed)
    embs = torch.randn(n, h)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    work = torch.tensor([i % n_works for i in range(n)])
    cond = torch.tensor([(i // n_works) % n_conds for i in range(n)])
    var = torch.tensor([(i // (n_works * n_conds)) % n_vars for i in range(n)])
    return embs, work, cond, var


BASE_KWARGS = dict(
    recall_ks=(1, 5, 10),
    hit_ks=(1, 5, 10),
    include_r_precision=True,
    include_median_rank=True,
    include_map=True,
    map_at_ks=(1,),
    include_mrr=True,
)


def test_fused_base_matches_streaming():
    embs, work, _, _ = _fixture()
    fused = fused_retrieval_pass(
        embs,
        work,
        base_kwargs=BASE_KWARGS,
        with_grid=False,
        with_varctl=False,
        with_scores=False,
        device="cpu",
        batch=17,  # deliberately ragged chunks
    )
    ref = compute_retrieval_metrics_streaming(embs, work, device="cpu", batch=17, **BASE_KWARGS)
    assert set(fused["base"].keys()) == set(ref.keys())
    for k, v in ref.items():
        assert abs(fused["base"][k] - v) < 1e-6, f"{k}: {fused['base'][k]} != {v}"


def test_fused_base_matches_materialised():
    # Cross-check against the non-streaming implementation too.
    embs, work, _, _ = _fixture(seed=3)
    fused = fused_retrieval_pass(
        embs,
        work,
        base_kwargs=BASE_KWARGS,
        with_grid=False,
        with_varctl=False,
        with_scores=False,
        device="cpu",
        batch=16,
    )
    ref = compute_retrieval_metrics(embs @ embs.T, work, **BASE_KWARGS)
    for k, v in ref.items():
        assert abs(fused["base"][k] - v) < 1e-5, f"{k}: {fused['base'][k]} != {v}"


def test_fused_grid_and_varctl_match_streaming():
    embs, work, cond, var = _fixture(seed=1)
    fused = fused_retrieval_pass(
        embs,
        work,
        conditions=cond,
        variations=var,
        base_kwargs=None,
        with_scores=False,
        device="cpu",
        batch=13,
    )
    conds = sorted({int(c) for c in cond.tolist()})
    ref = compute_perpair_map_all_streaming(
        embs, work, cond, query_conds=conds, target_conds=conds, device="cpu", batch=13
    )
    ref_v = compute_perpair_map_all_streaming(
        embs,
        work,
        cond,
        query_conds=conds,
        target_conds=conds,
        device="cpu",
        batch=13,
        variation_ids=var,
        require_different_variation=True,
    )
    for cell, (m, n) in ref.items():
        fm, fn = fused["grid"][cell]
        assert fn == n and abs(fm - m) < 1e-6, f"grid {cell}"
    for cell, (m, n) in ref_v.items():
        fm, fn = fused["grid_varctl"][cell]
        assert fn == n and abs(fm - m) < 1e-6, f"varctl {cell}"


def test_fused_scores_match_score_distributions():
    embs, work, cond, var = _fixture(seed=2)
    fused = fused_retrieval_pass(
        embs,
        work,
        conditions=cond,
        variations=var,
        base_kwargs=None,
        with_grid=False,
        with_varctl=False,
        with_scores=True,
        score_n_bins=20,
        device="cpu",
        batch=11,
    )
    sim = embs @ embs.T  # raw sim — accumulator self-excludes via masks
    ref = score_distributions(sim, work, conditions=cond, variations=var, n_bins=20)
    f, r = fused["scores"], ref
    for scope_key in ["overall"]:
        for pool in ("relevant", "relevant_diffvar", "distractor"):
            assert f[scope_key][pool]["n"] == r[scope_key][pool]["n"]
            # chunked-vs-oneshot matmul differ in the last fp32 ulp of sim itself
            assert abs(f[scope_key][pool]["mean"] - r[scope_key][pool]["mean"]) < 1e-7
            assert f[scope_key][pool]["hist"] == r[scope_key][pool]["hist"]
    assert set(f["cells"].keys()) == set(r["cells"].keys())
    for cell in r["cells"]:
        for pool in ("relevant", "relevant_diffvar", "distractor"):
            assert f["cells"][cell][pool]["n"] == r["cells"][cell][pool]["n"], f"{cell}/{pool}"
            assert f["cells"][cell][pool]["hist"] == r["cells"][cell][pool]["hist"]
            assert abs(f["cells"][cell][pool]["mean"] - r["cells"][cell][pool]["mean"]) < 1e-7


# ── bulk score accumulator vs a reference re-implementation of the previous
#    per-cell masking algorithm (the behavior the 10 unit tests were written
#    against), on random data with a sentinel condition present ─────────────


def _reference_score_distributions(sim, work, cond, var, n_bins=25, lo=-1.0, hi=1.0):
    """Reference: the previous (pre-bulk) accumulator algorithm, verbatim
    semantics — per-cell boolean masking + np.histogram on clipped values."""
    edges = np.linspace(lo, hi, n_bins + 1)
    work = torch.as_tensor(work)
    cond_t = None if cond is None else torch.as_tensor(cond)
    var_t = None if var is None else torch.as_tensor(var)
    uc = (
        [c for c in sorted({int(x) for x in cond_t.tolist()}) if c != -1]
        if cond_t is not None
        else []
    )

    def new_pool():
        return {
            "n": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "max": float("-inf"),
            "min": float("inf"),
            "hist": np.zeros(n_bins, dtype=np.int64),
        }

    def add(pool, scores):
        if scores.numel() == 0:
            return
        s = scores.to(torch.float64).numpy()
        pool["n"] += int(s.size)
        pool["sum"] += float(s.sum())
        pool["sumsq"] += float((s * s).sum())
        pool["max"] = max(pool["max"], float(s.max()))
        pool["min"] = min(pool["min"], float(s.min()))
        pool["hist"] += np.histogram(np.clip(s, lo, hi), bins=edges)[0]

    n = sim.shape[0]
    cells = {}

    def cell(key):
        if key not in cells:
            cells[key] = {"rel_diff": new_pool(), "rel_twin": new_pool(), "distr": new_pool()}
        return cells[key]

    for qi in range(n):
        row = sim[qi]
        not_self = torch.ones(n, dtype=torch.bool)
        not_self[qi] = False
        same_work = (work == work[qi]) & not_self
        twin = (
            same_work & (var_t == var_t[qi]) if var_t is not None else torch.zeros_like(same_work)
        )
        rel_diff = same_work & ~twin
        distr = (~(work == work[qi])) & not_self
        ov = cell("overall")
        add(ov["rel_diff"], row[rel_diff])
        add(ov["rel_twin"], row[twin])
        add(ov["distr"], row[distr])
        if cond_t is not None and int(cond_t[qi]) in uc:
            for tc in uc:
                tm = cond_t == tc
                c = cell((int(cond_t[qi]), tc))
                add(c["rel_diff"], row[rel_diff & tm])
                add(c["rel_twin"], row[twin & tm])
                add(c["distr"], row[distr & tm])
    return cells


def test_bulk_accumulator_matches_reference_with_sentinel_cond():
    torch.manual_seed(7)
    n = 48
    embs = torch.randn(n, 8)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    sim = embs @ embs.T
    work = torch.tensor([i % 9 for i in range(n)])
    cond = torch.tensor([[-1, 0, 1, 2][i % 4] for i in range(n)])  # includes sentinel
    var = torch.tensor([i % 2 for i in range(n)])

    acc = RetrievalScoreAccumulator(work, cond, var, n_bins=25)
    # feed in two ragged batches to exercise streaming
    acc.update(list(range(0, 20)), sim[:20])
    acc.update(list(range(20, n)), sim[20:])
    got = acc.result()

    ref_cells = _reference_score_distributions(sim, work, cond, var, n_bins=25)
    # overall
    for pool, refkey in (
        ("relevant_diffvar", "rel_diff"),
        ("distractor", "distr"),
    ):
        rp = ref_cells["overall"][refkey]
        gp = got["overall"][pool]
        assert gp["n"] == rp["n"]
        assert gp["hist"] == rp["hist"].tolist()
        if rp["n"]:
            assert abs(gp["mean"] - rp["sum"] / rp["n"]) < 1e-9
            assert abs(gp["max"] - rp["max"]) < 1e-12
            assert abs(gp["min"] - rp["min"]) < 1e-12
    # merged relevant pool (diff + twin)
    r_all_n = ref_cells["overall"]["rel_diff"]["n"] + ref_cells["overall"]["rel_twin"]["n"]
    assert got["overall"]["relevant"]["n"] == r_all_n
    # cells: same keys (sentinel -1 must not appear), same counts/hists
    ref_cell_keys = {k for k in ref_cells if k != "overall"}
    assert set(got["cells"].keys()) == ref_cell_keys
    for key in ref_cell_keys:
        for pool, refkey in (
            ("relevant_diffvar", "rel_diff"),
            ("distractor", "distr"),
        ):
            rp = ref_cells[key][refkey]
            gp = got["cells"][key][pool]
            assert gp["n"] == rp["n"], f"{key}/{pool}"
            assert gp["hist"] == rp["hist"].tolist(), f"{key}/{pool}"


def test_bulk_accumulator_inf_poisoned_self_is_harmless():
    # The fused iterator feeds sim chunks whose self positions are -inf.
    # Stats must be identical to feeding the raw sim.
    embs, work, cond, var = _fixture(seed=5, n=32)
    sim = embs @ embs.T
    sim_poisoned = sim.clone()
    sim_poisoned[torch.arange(32), torch.arange(32)] = float("-inf")
    a = RetrievalScoreAccumulator(work, cond, var, n_bins=15)
    a.update(list(range(32)), sim)
    b = RetrievalScoreAccumulator(work, cond, var, n_bins=15)
    b.update(list(range(32)), sim_poisoned)
    ra, rb = a.result(), b.result()
    assert ra["overall"] == rb["overall"]
    assert ra["cells"] == rb["cells"]
