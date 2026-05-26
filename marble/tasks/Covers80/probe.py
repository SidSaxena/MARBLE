# marble/tasks/Covers80/probe.py
"""
Covers80 cover-song retrieval probe.

This task does NOT train a probe head.  The pre-trained encoder is evaluated
directly (zero-shot retrieval), making the evaluation entirely unsupervised.

Evaluation procedure (test stage):
  1. Encode every track → mean-pool over time → L2-normalise → embedding.
  2. Mean-pool all clip embeddings that belong to the same audio file.
  3. Build a (160 × 160) cosine-similarity matrix.
  4. For each query track, rank the remaining 159 tracks by similarity.
  5. AP = 1 / rank_of_the_single_matching_cover  (Covers80 has exactly
     2 versions per work, so there is always exactly 1 relevant result).
  6. MAP = mean(AP) over all 160 queries.

Note:  The 'fit' command runs with max_epochs=0 (a no-op) so that the
       normal run_sweep_local.py pipeline (fit → test) works without
       modification.

Usage
-----
python cli.py fit  -c configs/probe.OMARRQ-multifeature25hz.Covers80.yaml
python cli.py test -c configs/probe.OMARRQ-multifeature25hz.Covers80.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from lightning.pytorch import LightningModule

from marble.core.utils import instantiate_from_config
from marble.utils.emb_cache import EmbeddingCacheMixin


class CoverRetrievalTask(LightningModule, EmbeddingCacheMixin):
    """
    Zero-shot cover-song retrieval probe.

    Parameters
    ----------
    sample_rate      : int  target audio sample rate (for documentation only).
    encoder          : dict config for the pre-trained audio encoder.
    emb_transforms   : list[dict] LayerSelector + TimeAvgPool configs.
    cache_embeddings : bool, default False. If True, the post-time-pool
                       per-layer embedding ``(L, H)`` for every clip is
                       written to ``output/.emb_cache/<encoder>/<task>__<hash>/``
                       on first encounter and re-used on subsequent calls.
                       Skips the encoder + ``LayerSelector`` + ``TimeAvgPool``
                       forward for cache hits — saves the ~33 min/layer
                       encoder pass on every layer-sweep job after the first.
                       Cache is opt-in per config (see plan: 14 cache-safe
                       tasks). Set False to fall back to the un-cached path.
    """

    # Disable Lightning's automatic gradient-step machinery — we have
    # nothing to optimise (max_epochs=0).
    automatic_optimization = False

    def __init__(
        self,
        sample_rate: int,
        encoder: dict,
        emb_transforms: list[dict],
        cache_embeddings: bool = False,
        cache_pool_time: bool = True,
        log_extended_retrieval_metrics: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = instantiate_from_config(encoder)
        self.emb_transforms = nn.ModuleList([instantiate_from_config(c) for c in emb_transforms])
        # Tiny dummy parameter so Lightning's model-summary and optimizer
        # don't complain about zero-parameter models.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Cache plumbing inherited from EmbeddingCacheMixin — slot init
        # + lazy build on first forward + audio-I/O bypass injection.
        # cache_pool_time=True (default) stores (L, H); False stores
        # (L, T, H) for frame-level probes. See emb_cache.py.
        self.cache_embeddings = bool(cache_embeddings)
        self.cache_pool_time = bool(cache_pool_time)
        self._init_cache_state()

        # Default trim set logs ~7 retrieval metrics (map, map_centered,
        # recall@10, r_precision, median_rank, anisotropy/{mean_vec_norm,
        # effective_rank}) plus the per-condition triplet when present.
        # Set this True to also log map@1, mrr, the full recall@K range
        # (1/5/10/50/100), hit_rate@K, the _centered duplicates of
        # secondary metrics, and the two extra anisotropy lines —
        # ~26 additional keys. Useful for leitmotif/no-ground-truth runs
        # where the K-sweep is the actual scientific question.
        self.log_extended_retrieval_metrics = bool(log_extended_retrieval_metrics)

    def setup(self, stage: str | None = None) -> None:
        """Hook into Lightning's per-stage setup to wire the cache check
        into the datasets so they can skip audio I/O on cache hits."""
        super().setup(stage)
        self._ensure_cache()
        self._inject_cache_check_into_datasets()

    # ── forward: encoder → transforms → flatten → L2-normalise ──────────────

    def forward(
        self,
        x: torch.Tensor,
        clip_ids: list[str] | None = None,
    ) -> torch.Tensor:
        """
        Returns L2-normalised embeddings of shape (B, H).

        If ``cache_embeddings=True`` and ``clip_ids`` is provided, the
        per-clip post-time-pool ``(L, H)`` tensor is fetched from /
        written to ``output/.emb_cache/...`` and the encoder forward is
        skipped on cache hits.
        """
        h = self._cached_forward_layer_tuple(x, clip_ids)
        for t in self.emb_transforms:
            h = t(h)
        # h: (B, L_sel, T, H) → collapse to (B, H). For cache-hit / cache-warm
        # paths T==1 so the reduce is a no-op on the time axis.
        h = reduce(h, "b l t h -> b h", "mean")
        return F.normalize(h, dim=-1)

    # ── fit stage is a no-op ─────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        # Called 0 times with max_epochs=0 — here for completeness.
        return None

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # No optimisation steps will be taken (max_epochs=0).
        return []

    # ── test stage: accumulate embeddings, then compute MAP ──────────────────

    def on_test_start(self) -> None:
        self._test_embeddings: list[torch.Tensor] = []
        self._test_work_ids: list[torch.Tensor] = []
        self._test_paths: list[str] = []
        # Per-condition metadata for cross-condition MAP (cross-instrument
        # for VGMIDITVar-leitmotif, cross-soundfont for VGMIDITVar-multisf).
        # Stays empty for 4-tuple datamodules (Covers80, SHS100K) — the
        # per-condition log block is gated on the list being non-empty.
        self._test_conditions: list[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        # batch is one of:
        #   3-tuple (waveform, work_ids, paths)             — legacy
        #   4-tuple (..., clip_ids)                         — Covers80, SHS100K
        #   5-tuple (..., clip_ids, conditions)             — VGMIDITVar variants
        # ``conditions`` carries gm_program (leitmotif) OR soundfont_id
        # (multisf) OR -1 (base VGMIDITVar). See VGMIDITVar/datamodule.py.
        conditions = None
        if len(batch) == 5:
            x, work_ids, paths, clip_ids, conditions = batch
        elif len(batch) == 4:
            x, work_ids, paths, clip_ids = batch
        else:
            x, work_ids, paths = batch
            clip_ids = None
        embeddings = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        self._test_embeddings.append(embeddings.detach().cpu())
        self._test_work_ids.append(
            work_ids.cpu() if isinstance(work_ids, torch.Tensor) else torch.tensor(work_ids)
        )
        self._test_paths.extend(paths)
        if conditions is not None:
            self._test_conditions.append(
                conditions.cpu()
                if isinstance(conditions, torch.Tensor)
                else torch.tensor(conditions)
            )

    def on_test_epoch_end(self) -> None:
        all_embs = torch.cat(self._test_embeddings)  # (N_clips, H)
        all_work_ids = torch.cat(self._test_work_ids)  # (N_clips,)
        all_paths = self._test_paths

        # Conditions are per-clip; aggregate to per-file by first-seen
        # (every clip from a single file shares the same condition).
        has_conditions = bool(self._test_conditions)
        all_conditions = torch.cat(self._test_conditions).tolist() if has_conditions else None

        # ── per-file mean-pool (aggregates clips from the same track) ────────
        path2work: dict[str, int] = {}
        path2embs: dict[str, list] = {}
        path2cond: dict[str, int] = {}
        for idx, (emb, wid, path) in enumerate(
            zip(all_embs, all_work_ids.tolist(), all_paths, strict=True)
        ):
            path2work.setdefault(path, wid)
            path2embs.setdefault(path, []).append(emb)
            if has_conditions:
                path2cond.setdefault(path, all_conditions[idx])

        file_embs: list[torch.Tensor] = []
        file_work_ids: list[int] = []
        file_conditions: list[int] = []
        for path, embs_list in path2embs.items():
            stacked = torch.stack(embs_list)  # (n_clips, H)
            mean_emb = stacked.mean(0)
            mean_emb = F.normalize(mean_emb, dim=-1)  # re-normalise
            file_embs.append(mean_emb)
            file_work_ids.append(path2work[path])
            if has_conditions:
                file_conditions.append(path2cond[path])

        embs = torch.stack(file_embs)  # (N, H)
        work_ids = torch.tensor(file_work_ids)  # (N,)
        N = len(work_ids)

        # Prefix is generic ("[CoverRetrieval]") because this class is
        # reused by VGMIDITVar / SHS100K / etc. — the prior "[Covers80]"
        # was misleading when the same code runs against any work-id-keyed
        # retrieval dataset. The originating task is also visible in the
        # WandB run's group/tags.
        print(
            f"\n[CoverRetrieval] Evaluating retrieval MAP over {N} tracks "
            f"({len(set(file_work_ids))} works)."
        )

        # ── cosine similarity (embeddings already L2-normalised) ─────────────
        sim = embs @ embs.T  # (N, N)

        # ── Centering variant: remove cone-effect anisotropy ─────────────────
        # The anisotropy diagnostic (scripts/diagnostics/anisotropy_diag.py)
        # found OMARRQ embeddings live in a cone (mean_vec_norm ≈ 0.5). For
        # cosine retrieval that shared direction inflates every pairwise
        # similarity and squashes discrimination. Subtracting the corpus
        # mean ("all-but-the-top", Mu 2018) removes it. We log BOTH the raw
        # and centered MAP so the comparison is automatic for every encoder.
        embs_c = embs - embs.mean(dim=0, keepdim=True)
        embs_c = F.normalize(embs_c, dim=-1)
        sim_c = embs_c @ embs_c.T

        # ── compute MAP for both variants ─────────────────────────────────────
        map_raw = self._compute_map(sim, work_ids)
        map_centered = self._compute_map(sim_c, work_ids)
        print(f"[CoverRetrieval] MAP (raw)      = {map_raw:.4f}")
        print(f"[CoverRetrieval] MAP (centered) = {map_centered:.4f}")

        self.log("test/map", map_raw, prog_bar=True, rank_zero_only=True)
        self.log("test/map_centered", map_centered, prog_bar=False, rank_zero_only=True)
        if self.log_extended_retrieval_metrics:
            # map@1 ≈ recall@1 (identical when one relevant per query); mrr
            # ≈ inverse of median_rank. Kept available for K-sweep analysis.
            self.log("test/map@1", self._map_at_k(sim, work_ids, k=1), rank_zero_only=True)
            self.log(
                "test/map@1_centered", self._map_at_k(sim_c, work_ids, k=1), rank_zero_only=True
            )
            self.log("test/mrr", self._mrr(sim, work_ids), rank_zero_only=True)
            self.log("test/mrr_centered", self._mrr(sim_c, work_ids), rank_zero_only=True)

        # ── Recall / Hit Rate / median rank / R-Precision ────────────────────
        # Headline metrics for review-budget-K-aware retrieval evaluation
        # (leitmotif workflow). See marble/utils/retrieval_metrics.py and
        # docs/benchmarking_methodology.md for rationale. Both raw and
        # centered variants logged. K values that exceed corpus size are
        # silently skipped — small smoke runs don't pollute the log with
        # NaN keys.
        from marble.utils.retrieval_metrics import (
            hit_rate_at_k,
            median_rank_first_hit,
            r_precision,
            recall_at_k,
        )

        # Default trim set: recall@10, r_precision, median_rank — all raw.
        # Centered variants of the secondary metrics are rarely flipped
        # by anisotropy and are gated behind the extended flag.
        if N > 10:
            self.log("test/recall@10", recall_at_k(sim, work_ids, 10), rank_zero_only=True)
        self.log("test/r_precision", r_precision(sim, work_ids), rank_zero_only=True)
        self.log("test/median_rank", median_rank_first_hit(sim, work_ids), rank_zero_only=True)

        if self.log_extended_retrieval_metrics:
            # Full K-sweep + _centered duplicates + hit_rate. ~22 keys.
            # Compute is cheap (sim matrix already in memory); cost is
            # dashboard clutter, hence opt-in.
            K_RECALL_EXTRA = [k for k in (1, 5, 50, 100) if k < N]
            K_HIT = [k for k in (1, 5, 10) if k < N]
            for k in K_RECALL_EXTRA:
                self.log(f"test/recall@{k}", recall_at_k(sim, work_ids, k), rank_zero_only=True)
            for suffix, S in (("", sim), ("_centered", sim_c)):
                for k in K_HIT:
                    self.log(
                        f"test/hit_rate@{k}{suffix}",
                        hit_rate_at_k(S, work_ids, k),
                        rank_zero_only=True,
                    )
            # centered duplicates of the headline trim set
            if N > 10:
                self.log(
                    "test/recall@10_centered", recall_at_k(sim_c, work_ids, 10), rank_zero_only=True
                )
            K_RECALL_ALL_C = [k for k in (1, 5, 10, 50, 100) if k < N]
            for k in K_RECALL_ALL_C:
                if k == 10:
                    continue
                self.log(
                    f"test/recall@{k}_centered",
                    recall_at_k(sim_c, work_ids, k),
                    rank_zero_only=True,
                )
            self.log("test/r_precision_centered", r_precision(sim_c, work_ids), rank_zero_only=True)
            self.log(
                "test/median_rank_centered",
                median_rank_first_hit(sim_c, work_ids),
                rank_zero_only=True,
            )

        # ── Per-condition MAP (cross-instrument / cross-soundfont) ───────────
        # Only meaningful when the dataset carries a per-item condition
        # field (VGMIDITVar-leitmotif: gm_program; VGMIDITVar-multisf:
        # soundfont_id). Covers80 / SHS100K skip silently because their
        # datamodules emit 4-tuples → has_conditions=False.
        #
        # Cross-condition MAP (off-diagonal mean of the (q,t) grid) is THE
        # leitmotif-relevant metric per docs/leitmotif_findings.md: it
        # measures retrieval performance specifically when the relevant
        # peer is in a different timbre/instrument than the query, which
        # is the actual operational scenario for cross-orchestration
        # leitmotif retrieval.
        if has_conditions and any(c != -1 for c in file_conditions):
            from marble.utils.retrieval_metrics import compute_perpair_map

            unique_conds = sorted({c for c in file_conditions if c != -1})
            same_aps: list[float] = []
            cross_aps: list[float] = []
            # Use the centered similarity matrix for cross-condition MAP —
            # consistent with offline analysis in
            # scripts/analysis/vgmiditvar_leitmotif_breakdown.py, which
            # subtracts the corpus mean before per-pair MAP to remove
            # cone-effect anisotropy (relevant for OMARRQ esp.).
            for q in unique_conds:
                for t in unique_conds:
                    ap, n = compute_perpair_map(sim_c, file_work_ids, file_conditions, q, t)
                    if n == 0:
                        continue
                    (same_aps if q == t else cross_aps).append(ap)
            if same_aps:
                same_mean = float(sum(same_aps) / len(same_aps))
                self.log("test/map_same_condition", same_mean, rank_zero_only=True)
                print(f"[CoverRetrieval] MAP same-condition  = {same_mean:.4f}")
            if cross_aps:
                cross_mean = float(sum(cross_aps) / len(cross_aps))
                self.log("test/map_cross_condition", cross_mean, rank_zero_only=True)
                print(f"[CoverRetrieval] MAP cross-condition = {cross_mean:.4f}")
            if same_aps and cross_aps:
                gap = float(sum(same_aps) / len(same_aps) - sum(cross_aps) / len(cross_aps))
                self.log("test/condition_gap", gap, rank_zero_only=True)

        # ── Anisotropy diagnostics ───────────────────────────────────────────
        # Cone-effect / rank-collapse measurements on the per-file embedding
        # matrix (pre-centering, since centering itself is what we're
        # diagnosing). High mean_vec_norm → ``map_centered`` is the trustworthy
        # number for this encoder. See marble/utils/retrieval_metrics.py.
        from marble.utils.retrieval_metrics import anisotropy_metrics

        ani = anisotropy_metrics(embs)
        # Default trim: mean_vec_norm (cone-effect headline) + effective_rank
        # (rank collapse). avg_pair_cos and top1_sv_share are correlated
        # with these and gated behind the extended flag.
        self.log("test/anisotropy/mean_vec_norm", float(ani["mean_vec_norm"]), rank_zero_only=True)
        self.log(
            "test/anisotropy/effective_rank",
            float(ani["effective_rank"]),
            rank_zero_only=True,
        )
        if self.log_extended_retrieval_metrics:
            self.log(
                "test/anisotropy/avg_pair_cos",
                float(ani["avg_pair_cos"]),
                rank_zero_only=True,
            )
            self.log(
                "test/anisotropy/top1_sv_share",
                float(ani["top1_sv_share"]),
                rank_zero_only=True,
            )
        print(
            f"[CoverRetrieval] Anisotropy: mean_vec_norm={ani['mean_vec_norm']:.3f}  "
            f"avg_pair_cos={ani['avg_pair_cos']:.3f}  "
            f"top1_sv={ani['top1_sv_share']:.3f}  "
            f"eff_rank={ani['effective_rank']:.1f}"
        )

    @staticmethod
    def _compute_map(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
        """Standard MAP from a similarity matrix and work_id labels."""
        N = len(work_ids)
        aps: list[float] = []
        for i in range(N):
            sims_i = sim[i].clone()
            sims_i[i] = -2.0  # exclude self
            order = sims_i.argsort(descending=True)
            is_rel = work_ids[order] == work_ids[i]
            n_relevant = int(is_rel.sum().item())
            if n_relevant == 0:
                continue
            hits = 0
            ap = 0.0
            for rank, rel in enumerate(is_rel.tolist(), start=1):
                if rel:
                    hits += 1
                    ap += hits / rank
            ap /= n_relevant
            aps.append(ap)
        return float(torch.tensor(aps).mean().item()) if aps else 0.0

    # ── helper metrics ────────────────────────────────────────────────────────

    @staticmethod
    def _map_at_k(sim: torch.Tensor, work_ids: torch.Tensor, k: int) -> float:
        N = len(work_ids)
        aps = []
        for i in range(N):
            s = sim[i].clone()
            s[i] = -2.0
            top_k = s.argsort(descending=True)[:k]
            is_rel = (work_ids[top_k] == work_ids[i]).float()
            n_total = int((work_ids == work_ids[i]).sum().item()) - 1
            if n_total == 0:
                continue
            hits = 0
            ap = 0.0
            for rank, rel in enumerate(is_rel.tolist(), start=1):
                if rel:
                    hits += 1
                    ap += hits / rank
            ap /= n_total
            aps.append(ap)
        return float(torch.tensor(aps).mean().item()) if aps else 0.0

    @staticmethod
    def _mrr(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
        N = len(work_ids)
        recip = []
        for i in range(N):
            s = sim[i].clone()
            s[i] = -2.0
            order = s.argsort(descending=True)
            is_rel = work_ids[order] == work_ids[i]
            nz = is_rel.nonzero(as_tuple=True)[0]
            if len(nz) == 0:
                continue
            recip.append(1.0 / (nz[0].item() + 1))
        return float(torch.tensor(recip).mean().item()) if recip else 0.0
