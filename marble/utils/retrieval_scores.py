"""Per-cell cosine-score distributions for retrieval tasks.

``CoverRetrievalTask`` scores retrieval by cosine *ranking* (MAP) and discards
the raw similarities. This module accumulates, per (query_condition,
target_condition) cell, the distribution of candidate cosine scores split into
RELEVANT (same ``work_id``) vs DISTRACTOR (different ``work_id``), self-excluded —
so one can inspect the score distribution and the relevant-vs-distractor
separation, not only the aggregate MAP.

Variation control (VGMIDITVar twin confound): when ``variations`` is supplied,
the RELEVANT pool is split into

  * ``relevant``          — all same-work candidates (the *confounded* pool, which
                            in a cross-condition cell contains the query's own
                            composition re-rendered in the target timbre — an
                            audio near-duplicate "twin"), and
  * ``relevant_diffvar``  — same-work candidates with a *different* variation id,
                            i.e. the twins removed.

so ``separation`` (confounded) and ``separation_varctl`` (twins removed) can be
compared directly — mirroring the ``condition_gap`` vs ``condition_gap_varctl``
split of the MAP grid. Without ``variations`` the two coincide.

Empty pools: a cell whose RELEVANT pool is empty (e.g. a work with no same-work
peer at that condition) reports ``separation = None`` rather than a bogus
``0.0 - distractor_mean``, so callers cannot silently average garbage in.

Sentinel conditions: condition ``-1`` (the "unparsed" sentinel used across the
probe) is excluded from cell enumeration, matching the MAP grid's ``c != -1``
filter, so the two diagnostics score the same cell universe.

Streaming + device residency: :meth:`RetrievalScoreAccumulator.update` takes a
batch of query rows and accumulates with vectorised scatter/bincount ops **on
the device the rows live on** — feed it CUDA sim chunks and the whole
accumulation runs on GPU with only fixed-size count/moment tensors held; feed
it CPU rows and it runs on CPU. Works at N≈100k without materialising the full
N×N similarity matrix. Self positions may carry ``-inf`` (the shared streaming
iterator poisons the diagonal before argsort) — they are excluded by the
self-mask before any pool statistics, so the poisoning is harmless.

Histogram semantics match ``np.histogram`` on values clipped into ``[lo, hi]``
(so ``sum(hist) == n`` always): bin = ``searchsorted(edges, clamp(x), right) - 1``
clamped to the last bin, computed in float64 exactly like the previous
numpy-based implementation.
"""

from __future__ import annotations

import numpy as np
import torch


def _new_pool(n_bins: int) -> dict:
    return {
        "n": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "max": float("-inf"),
        "min": float("inf"),
        "hist": np.zeros(n_bins, dtype=np.int64),
    }


def _merge_pools(a: dict, b: dict) -> dict:
    """Additive combine of two moment/histogram pools (RELEVANT = diffvar ∪ twin)."""
    return {
        "n": a["n"] + b["n"],
        "sum": a["sum"] + b["sum"],
        "sumsq": a["sumsq"] + b["sumsq"],
        "max": max(a["max"], b["max"]),
        "min": min(a["min"], b["min"]),
        "hist": a["hist"] + b["hist"],
    }


# Pool codes used by the bulk accumulation. SKIP covers self positions and
# (for the cell scope) sentinel-condition elements.
_POOL_REL_DIFF = 0
_POOL_TWIN = 1
_POOL_DISTR = 2
_POOL_SKIP = 3


class _ScopeAccumulators:
    """Fixed-size tensor accumulators for one scope (overall, or all cells).

    ``groups`` = 3 pools for the overall scope, ``n_cells * 3`` for the cell
    scope. One extra trailing slot absorbs SKIP elements so the hot loop never
    needs boolean compaction — scatter everything, slice the junk slot off.
    """

    def __init__(self, groups: int, n_bins: int, device: torch.device):
        self.groups = groups
        self.n_bins = n_bins
        self.hist = torch.zeros(groups * n_bins + 1, dtype=torch.long, device=device)
        self.n = torch.zeros(groups + 1, dtype=torch.long, device=device)
        self.sum = torch.zeros(groups + 1, dtype=torch.float64, device=device)
        self.sumsq = torch.zeros(groups + 1, dtype=torch.float64, device=device)
        self.min = torch.full((groups + 1,), float("inf"), dtype=torch.float64, device=device)
        self.max = torch.full((groups + 1,), float("-inf"), dtype=torch.float64, device=device)

    def add(self, group_idx: torch.Tensor, bins: torch.Tensor, values: torch.Tensor) -> None:
        """Scatter one flattened batch. ``group_idx`` == ``self.groups`` → junk slot."""
        g = group_idx.reshape(-1)
        v = values.reshape(-1)
        flat_hist = g * self.n_bins + bins.reshape(-1)
        # Junk-slot elements land past groups*n_bins; clamp into the single
        # trailing slot (their bin offset would overrun it otherwise).
        flat_hist = flat_hist.clamp(max=self.groups * self.n_bins)
        self.hist += torch.bincount(flat_hist, minlength=self.groups * self.n_bins + 1)
        self.n += torch.bincount(g, minlength=self.groups + 1)
        self.sum += torch.bincount(g, weights=v, minlength=self.groups + 1)
        self.sumsq += torch.bincount(g, weights=v * v, minlength=self.groups + 1)
        self.min.scatter_reduce_(0, g, v, reduce="amin")
        self.max.scatter_reduce_(0, g, v, reduce="amax")

    def pool_dict(self, group: int) -> dict:
        """Materialise one group's accumulators as a host-side pool dict."""
        n = int(self.n[group].item())
        return {
            "n": n,
            "sum": float(self.sum[group].item()),
            "sumsq": float(self.sumsq[group].item()),
            "max": float(self.max[group].item()),
            "min": float(self.min[group].item()),
            "hist": self.hist[group * self.n_bins : (group + 1) * self.n_bins]
            .cpu()
            .numpy()
            .astype(np.int64),
        }


class RetrievalScoreAccumulator:
    """Accumulate relevant/distractor cosine-score histograms + moments per cell.

    The RELEVANT side is tracked as two sub-pools — ``rel_diff`` (different
    variation) and ``rel_twin`` (same-(work, variation) near-duplicate) — so both
    the confounded and the variation-controlled separation are recoverable in a
    single pass. ``rel_twin`` is only ever populated when ``variations`` is given.
    """

    def __init__(
        self,
        work_ids,
        conditions=None,
        variations=None,
        *,
        n_bins: int = 50,
        lo: float = -1.0,
        hi: float = 1.0,
        exclude_condition: int | None = -1,
    ):
        self.work = torch.as_tensor(work_ids)
        self.cond = None if conditions is None else torch.as_tensor(conditions)
        self.var = None if variations is None else torch.as_tensor(variations)
        self.n_bins = int(n_bins)
        self.lo = float(lo)
        self.hi = float(hi)
        self.edges = np.linspace(lo, hi, self.n_bins + 1)
        # Cell enumeration excludes the ``exclude_condition`` sentinel (default -1)
        # so cells match the MAP grid, which filters ``c != -1``.
        if self.cond is not None:
            uc = [int(c) for c in torch.unique(self.cond).tolist()]
            self._unique_conds = [
                c for c in uc if exclude_condition is None or c != exclude_condition
            ]
        else:
            self._unique_conds = []
        self.n_c = len(self._unique_conds)
        # Lazy device state (built on first update, on the rows' device).
        self._dev: torch.device | None = None
        self._overall: _ScopeAccumulators | None = None
        self._cells_acc: _ScopeAccumulators | None = None

    # ── device state ────────────────────────────────────────────────────
    def _ensure_state(self, dev: torch.device) -> None:
        if self._dev == dev:
            return
        if self._dev is not None and self._overall is not None:
            # Mid-stream device switch: carry accumulated state over.
            for acc in (self._overall, self._cells_acc):
                if acc is None:
                    continue
                for name in ("hist", "n", "sum", "sumsq", "min", "max"):
                    setattr(acc, name, getattr(acc, name).to(dev))
            self._work_d = self._work_d.to(dev)
            self._cond_pos = None if self._cond_pos is None else self._cond_pos.to(dev)
            self._var_d = None if self._var_d is None else self._var_d.to(dev)
            self._edges_d = self._edges_d.to(dev)
            self._dev = dev
            return
        self._dev = dev
        self._work_d = self.work.to(dev)
        self._var_d = None if self.var is None else self.var.to(dev)
        self._edges_d = torch.tensor(self.edges, dtype=torch.float64, device=dev)
        # Per-item cell position of each candidate's condition (-1 = not a cell
        # condition, e.g. the sentinel). Precomputed once for the target axis.
        if self.cond is not None and self.n_c:
            uc_t = torch.tensor(self._unique_conds, dtype=self.cond.dtype, device=dev)
            cond_d = self.cond.to(dev)
            pos = torch.searchsorted(uc_t, cond_d.clamp(min=int(uc_t.min()), max=int(uc_t.max())))
            pos = pos.clamp(max=self.n_c - 1)
            match = uc_t[pos] == cond_d
            self._cond_pos = torch.where(match, pos, torch.full_like(pos, -1))
        else:
            self._cond_pos = None
        self._overall = _ScopeAccumulators(3, self.n_bins, dev)
        self._cells_acc = (
            _ScopeAccumulators(self.n_c * self.n_c * 3, self.n_bins, dev)
            if self._cond_pos is not None
            else None
        )

    # ── accumulation ────────────────────────────────────────────────────
    def update(self, query_idx, sim_rows) -> None:
        """Process a batch of query rows with vectorised scatter ops on the
        rows' device.

        ``query_idx`` = the global row indices (length B); ``sim_rows`` =
        ``(B, N)`` similarities for those queries. Self positions may be
        ``-inf``-poisoned — they are excluded before any statistic.
        """
        sim = torch.as_tensor(sim_rows)
        if sim.ndim == 1:
            sim = sim.unsqueeze(0)
        dev = sim.device
        self._ensure_state(dev)
        qidx = torch.as_tensor(
            [int(x) for x in query_idx] if not isinstance(query_idx, torch.Tensor) else query_idx,
            dtype=torch.long,
            device=dev,
        )
        b, n = sim.shape
        rows = torch.arange(b, device=dev)

        wq = self._work_d[qidx].unsqueeze(1)  # (B, 1)
        same_work = self._work_d.unsqueeze(0) == wq  # (B, N)
        not_self = torch.ones(b, n, dtype=torch.bool, device=dev)
        not_self[rows, qidx] = False
        if self._var_d is not None:
            vq = self._var_d[qidx].unsqueeze(1)
            twin = same_work & (self._var_d.unsqueeze(0) == vq)
        else:
            twin = torch.zeros(b, n, dtype=torch.bool, device=dev)

        # Pool code per element: rel_diff / twin / distr / skip(self).
        pool = torch.full((b, n), _POOL_SKIP, dtype=torch.long, device=dev)
        pool = torch.where((~same_work) & not_self, torch.tensor(_POOL_DISTR, device=dev), pool)
        pool = torch.where(
            same_work & not_self & ~twin, torch.tensor(_POOL_REL_DIFF, device=dev), pool
        )
        pool = torch.where(twin & not_self, torch.tensor(_POOL_TWIN, device=dev), pool)

        # float64 values + np.histogram-equivalent binning on clipped values.
        s64 = sim.to(torch.float64)
        # Neutralise skip elements' values so -inf can't leak into moments via
        # the junk slot arithmetic (they only ever land in the junk slot, but
        # -inf * 0 style traps are avoided entirely by overwriting).
        s64 = torch.where(pool == _POOL_SKIP, torch.zeros_like(s64), s64)
        bins = (
            torch.searchsorted(self._edges_d, s64.clamp(self.lo, self.hi).contiguous(), right=True)
            - 1
        )
        bins = bins.clamp(min=0, max=self.n_bins - 1)

        # Overall scope: group = pool (skip → junk slot 3).
        self._overall.add(pool, bins, s64)

        # Cell scope: group = (qc_pos * n_c + tc_pos) * 3 + pool; anything with
        # an invalid condition or skip pool → junk slot.
        if self._cells_acc is not None:
            qpos = self._cond_pos[qidx]  # (B,)
            tpos = self._cond_pos.unsqueeze(0)  # (1, N)
            cell = qpos.unsqueeze(1) * self.n_c + tpos  # (B, N)
            valid = (qpos.unsqueeze(1) >= 0) & (tpos >= 0) & (pool < _POOL_SKIP)
            group = torch.where(
                valid,
                cell * 3 + pool,
                torch.full_like(cell, self._cells_acc.groups),
            )
            self._cells_acc.add(group, bins, s64)

    # ── finalisation ────────────────────────────────────────────────────
    def _finalize(self, pool: dict) -> dict:
        n = pool["n"]
        mean = pool["sum"] / n if n else 0.0
        var = (pool["sumsq"] / n - mean * mean) if n else 0.0
        return {
            "n": n,
            "mean": float(mean),
            "std": float(np.sqrt(max(var, 0.0))),
            "max": float(pool["max"]) if n else 0.0,
            "min": float(pool["min"]) if n else 0.0,
            "hist": pool["hist"].tolist(),
            "edges": self.edges.tolist(),
        }

    def _finalize_cell(self, rel_diff: dict, rel_twin: dict, distr: dict) -> dict:
        rel_all = self._finalize(_merge_pools(rel_diff, rel_twin))
        rel_diff_f = self._finalize(rel_diff)
        distr_f = self._finalize(distr)
        # separation is None for an empty RELEVANT (or DISTRACTOR) pool so callers
        # can't average a bogus ``0.0 - distractor_mean`` in.
        sep = rel_all["mean"] - distr_f["mean"] if rel_all["n"] > 0 and distr_f["n"] > 0 else None
        sep_ctl = (
            rel_diff_f["mean"] - distr_f["mean"]
            if rel_diff_f["n"] > 0 and distr_f["n"] > 0
            else None
        )
        return {
            "relevant": rel_all,
            "distractor": distr_f,
            "separation": sep,
            "relevant_diffvar": rel_diff_f,
            "separation_varctl": sep_ctl,
        }

    def result(self) -> dict:
        empty_cell = self._finalize_cell(
            _new_pool(self.n_bins), _new_pool(self.n_bins), _new_pool(self.n_bins)
        )
        if self._overall is None:  # no update() ever ran
            return {"overall": empty_cell, "cells": {}}
        overall = self._finalize_cell(
            self._overall.pool_dict(_POOL_REL_DIFF),
            self._overall.pool_dict(_POOL_TWIN),
            self._overall.pool_dict(_POOL_DISTR),
        )
        cells: dict = {}
        if self._cells_acc is not None:
            for qi, qc in enumerate(self._unique_conds):
                for ti, tc in enumerate(self._unique_conds):
                    base = (qi * self.n_c + ti) * 3
                    cells[(qc, tc)] = self._finalize_cell(
                        self._cells_acc.pool_dict(base + _POOL_REL_DIFF),
                        self._cells_acc.pool_dict(base + _POOL_TWIN),
                        self._cells_acc.pool_dict(base + _POOL_DISTR),
                    )
        return {"overall": overall, "cells": cells}


def score_distributions(
    sim, work_ids, conditions=None, variations=None, *, n_bins: int = 50, lo=-1.0, hi=1.0
) -> dict:
    """Whole-matrix convenience wrapper (feeds every row through the accumulator).

    For large N use :class:`RetrievalScoreAccumulator` with batched rows instead.
    """
    acc = RetrievalScoreAccumulator(work_ids, conditions, variations, n_bins=n_bins, lo=lo, hi=hi)
    sim = torch.as_tensor(sim)
    acc.update(list(range(sim.shape[0])), sim)
    return acc.result()
