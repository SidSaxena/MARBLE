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
python cli.py fit  -c configs/probe.OMARRQ-multifeature-25hz.Covers80.yaml
python cli.py test -c configs/probe.OMARRQ-multifeature-25hz.Covers80.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from lightning.pytorch import LightningModule

from marble.core.utils import instantiate_from_config
from marble.utils.emb_cache import EmbeddingCacheMixin


def auto_whiten_params(n_works: int, n_files: int, hidden: int) -> tuple[float, float]:
    """Works/size-aware (alpha, eps_rel) for transductive map_whitened.

    Grounded in three labeled datapoints (docs/whitening_ablation.md §10-11):
      - n_files < 2*hidden  -> rank-deficient covariance: relative-Tikhonov
        ridge eps_rel=1e-2 (else pure whitening amplifies null-space noise;
        Covers80 N=160<H=768 collapsed without it).
      - n_works < hidden    -> covariance dominated by few per-work centroids,
        so transductive alpha=1.0 over-flattens/self-defeats; fractional
        alpha=0.6 is robust (SHS100K 111 works: +15-76% at a~0.6, a=1.0 hurt).
      - n_works >= hidden with n_files>>hidden -> pure alpha=1.0 best
        (VGMIDITVar 5040 works, the +100-425% regime).
    """
    eps_rel = 1e-2 if n_files < 2 * hidden else 0.0
    alpha = 1.0 if n_works >= hidden else 0.6
    return alpha, eps_rel


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
        metric_device: str = "auto",
        whiten_alpha: float | None = None,
        whiten_eps_rel: float | None = None,
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

        # Default trim set logs ~8 retrieval metrics (map, map_centered,
        # map_whitened, recall@10, r_precision, median_rank,
        # anisotropy/{mean_vec_norm, effective_rank}) plus the per-condition
        # triplet when present. Set this True to also log map@1, mrr, the
        # full recall@K range (1/5/10/50/100), hit_rate@K, the _centered and
        # _whitened duplicates of secondary metrics, and the two extra
        # anisotropy lines. Useful for leitmotif/no-ground-truth runs
        # where the K-sweep is the actual scientific question.
        self.log_extended_retrieval_metrics = bool(log_extended_retrieval_metrics)

        # Whitening (test/map_whitened) strength. None = works/size-aware
        # auto (see on_test_epoch_end): alpha=1.0 if n_works>=H else 0.6;
        # eps_rel=1e-2 if N<2H else 0. Set a float to force a fixed value.
        self.whiten_alpha = whiten_alpha
        self.whiten_eps_rel = whiten_eps_rel

        # Where to run the heavy metric-block work (sim + argsort + per-cell
        # grid). For VGMIDITVar-timbre (N=102 960, sim = 42 GB on CPU
        # which overflows into pagefile on a 32 GB-RAM box), the GPU
        # streaming path drops the metric block from ~50 min to ~30-60 s
        # by computing sim row-chunks on demand on the GPU and never
        # materialising the full (N, N) matrix. See
        # ``marble.utils.retrieval_metrics.compute_retrieval_metrics_streaming``.
        #
        # ``"auto"`` picks ``"cuda"`` when ``torch.cuda.is_available()``
        # else ``"cpu"``. Force ``"cpu"`` to reproduce pre-streaming
        # numbers bit-for-bit (no argsort tie-break delta).
        self.metric_device = str(metric_device)

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
        # for VGMIDITVar-timbre, cross-soundfont for VGMIDITVar-multisf).
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

        # ── Single batched pass: map, secondary metrics, all in one sweep ────
        # ``compute_retrieval_metrics`` runs the per-row argsort once per
        # similarity matrix and aggregates every requested metric inside
        # that single pass. This is the only sane shape for large N:
        # the earlier "precompute (N, N-1) order once and share it"
        # pattern OOM'd for VGMIDITVar-timbre (N=102 960 → 84 GB int64
        # order tensor); the previous "one call per metric" pattern still
        # paid the row-sort cost for each call (MAP family was up to
        # 6 redundant passes over ``sim``).
        from marble.utils.retrieval_metrics import (
            compute_perpair_map_all,
            compute_perpair_map_all_streaming,
            compute_retrieval_metrics,
            compute_retrieval_metrics_streaming,
            zca_whiten,
        )

        # Resolve metric device. ``"auto"`` → cuda if available, else cpu.
        # GPU path: streams sim row-chunks on demand on the GPU, never
        # materialises the full (N, N) sim. Drops the metric block from
        # ~50 min to ~30-60 s for VGMIDITVar-timbre at N=102 960.
        # See docs/layer_sweeps_plan.md + tests/test_retrieval_metrics_streaming.py.
        # ``getattr`` fallback supports test fixtures that build the
        # task via ``__new__`` without running ``__init__``.
        _md = getattr(self, "metric_device", "auto")
        if _md == "auto":
            _md = "cuda" if torch.cuda.is_available() else "cpu"
        use_streaming = _md == "cuda"
        if use_streaming:
            print(f"[CoverRetrieval] metric_device = {_md} (streaming GPU path)")
        else:
            print(f"[CoverRetrieval] metric_device = {_md} (materialised CPU sim)")

        # Default trim set: recall@10, r_precision, median_rank — all raw.
        # Centered variants of the secondary metrics are rarely flipped
        # by anisotropy and are gated behind the extended flag.
        recall_ks_raw: list[int] = [10] if N > 10 else []
        if self.log_extended_retrieval_metrics:
            recall_ks_raw = sorted(set(recall_ks_raw + [k for k in (1, 5, 50, 100) if k < N]))
        hit_ks_raw: list[int] = (
            [k for k in (1, 5, 10) if k < N] if self.log_extended_retrieval_metrics else []
        )
        map_at_ks_raw: list[int] = [1] if self.log_extended_retrieval_metrics and N > 1 else []

        raw_kwargs = dict(
            recall_ks=recall_ks_raw,
            hit_ks=hit_ks_raw,
            include_r_precision=True,
            include_median_rank=True,
            include_map=True,
            map_at_ks=map_at_ks_raw,
            include_mrr=self.log_extended_retrieval_metrics,
        )

        if use_streaming:
            # No (N, N) sim allocation at all — chunks on GPU only.
            metrics_raw = compute_retrieval_metrics_streaming(
                embs, work_ids, device=_md, **raw_kwargs
            )
        else:
            # ── cosine similarity (embeddings already L2-normalised) ──
            sim = embs @ embs.T  # (N, N) — ~42 GB at N=102 960
            metrics_raw = compute_retrieval_metrics(sim, work_ids, **raw_kwargs)
            # Free the raw similarity matrix before allocating ``sim_c`` — on
            # a 32 GB RAM + 81 GB pagefile machine, holding both
            # simultaneously (84 GB) overruns the commit limit. Halves
            # peak memory.
            del sim

        map_raw = metrics_raw["map"]
        print(f"[CoverRetrieval] MAP (raw)      = {map_raw:.4f}")
        self.log("test/map", map_raw, prog_bar=True, rank_zero_only=True)
        if "recall@10" in metrics_raw:
            self.log("test/recall@10", metrics_raw["recall@10"], rank_zero_only=True)
        self.log("test/r_precision", metrics_raw["r_precision"], rank_zero_only=True)
        self.log("test/median_rank", metrics_raw["median_rank"], rank_zero_only=True)
        if self.log_extended_retrieval_metrics:
            for k in (1, 5, 50, 100):
                key = f"recall@{k}"
                if key in metrics_raw:
                    self.log(f"test/{key}", metrics_raw[key], rank_zero_only=True)
            for k in (1, 5, 10):
                key = f"hit_rate@{k}"
                if key in metrics_raw:
                    self.log(f"test/{key}", metrics_raw[key], rank_zero_only=True)
            if "map@1" in metrics_raw:
                self.log("test/map@1", metrics_raw["map@1"], rank_zero_only=True)
            self.log("test/mrr", metrics_raw["mrr"], rank_zero_only=True)

        # ── Centering variant: remove cone-effect anisotropy ─────────────────
        # The anisotropy diagnostic (scripts/diagnostics/anisotropy_diag.py)
        # found OMARRQ embeddings live in a cone (mean_vec_norm ≈ 0.5). For
        # cosine retrieval that shared direction inflates every pairwise
        # similarity and squashes discrimination. Subtracting the corpus
        # mean ("all-but-the-top", Mu 2018) removes it. We log BOTH the raw
        # and centered MAP so the comparison is automatic for every encoder.
        embs_c = embs - embs.mean(dim=0, keepdim=True)
        embs_c = F.normalize(embs_c, dim=-1)

        recall_ks_c = (
            [k for k in (1, 5, 10, 50, 100) if k < N] if self.log_extended_retrieval_metrics else []
        )
        hit_ks_c = [k for k in (1, 5, 10) if k < N] if self.log_extended_retrieval_metrics else []
        map_at_ks_c = [1] if self.log_extended_retrieval_metrics and N > 1 else []
        centered_kwargs = dict(
            recall_ks=recall_ks_c,
            hit_ks=hit_ks_c,
            include_r_precision=self.log_extended_retrieval_metrics,
            include_median_rank=self.log_extended_retrieval_metrics,
            include_map=True,
            map_at_ks=map_at_ks_c,
            include_mrr=self.log_extended_retrieval_metrics,
        )
        if use_streaming:
            metrics_c = compute_retrieval_metrics_streaming(
                embs_c, work_ids, device=_md, **centered_kwargs
            )
            sim_c = None  # streaming path doesn't materialise sim_c
        else:
            sim_c = embs_c @ embs_c.T
            metrics_c = compute_retrieval_metrics(sim_c, work_ids, **centered_kwargs)
        map_centered = metrics_c["map"]
        print(f"[CoverRetrieval] MAP (centered) = {map_centered:.4f}")
        self.log("test/map_centered", map_centered, prog_bar=False, rank_zero_only=True)
        if self.log_extended_retrieval_metrics:
            for k in (1, 5, 10, 50, 100):
                key = f"recall@{k}"
                if key in metrics_c:
                    self.log(f"test/{key}_centered", metrics_c[key], rank_zero_only=True)
            for k in (1, 5, 10):
                key = f"hit_rate@{k}"
                if key in metrics_c:
                    self.log(f"test/{key}_centered", metrics_c[key], rank_zero_only=True)
            self.log("test/r_precision_centered", metrics_c["r_precision"], rank_zero_only=True)
            self.log("test/median_rank_centered", metrics_c["median_rank"], rank_zero_only=True)
            if "map@1" in metrics_c:
                self.log("test/map@1_centered", metrics_c["map@1"], rank_zero_only=True)
            self.log("test/mrr_centered", metrics_c["mrr"], rank_zero_only=True)

        # ── Whitening variant: ZCA-whiten then cosine ───────────────────────
        # Full whitening (α=1.0) rescales every principal direction to unit
        # variance before the L2-norm — up-weighting the low-variance
        # directions that carry work-identity and downweighting the
        # high-variance nuisance (e.g. timbre) directions. On the
        # cone-collapsed music encoders this lifts cross-condition retrieval
        # MAP substantially (docs/whitening_ablation.md). Transductive fit on
        # the test corpus — same protocol as map_centered. Known technique
        # (BERT-whitening, Su 2021); logged as a first-class metric.
        #
        # WORKS/SIZE-AWARE auto-(alpha, eps_rel). The optimal whitening
        # strength is corpus-dependent (docs/whitening_ablation.md §10-11):
        #   - N < 2*H  -> rank-deficient covariance: a relative-Tikhonov
        #     ridge (eps_rel) is required, else pure whitening amplifies the
        #     ~H-(N-1) null-space directions and collapses retrieval
        #     (Covers80, N=160<H=768: pure alpha=1.0 MAP 0.04 vs raw 0.17).
        #   - n_works < H -> the corpus covariance is dominated by the few
        #     per-work centroids, so transductive alpha=1.0 over-flattens /
        #     self-defeats; fractional alpha=0.6 is robust (SHS100K: +15-76%
        #     vs centering at a~0.6, while a=1.0 hurt CLaMP3/MuQ).
        #   - n_works >= H with N>>H -> pure alpha=1.0 is best (VGMIDITVar,
        #     5040 works, the +100-425% regime).
        # Override either via the whiten_alpha / whiten_eps_rel task args
        # (None = auto). NOTE: still transductive (fit on the test corpus);
        # an inductive fit can be stronger on few-work corpora (§11) but
        # needs a reference set the probe doesn't have.
        H_dim = embs.shape[1]
        n_works = int(torch.unique(work_ids).numel())
        auto_alpha, auto_eps_rel = auto_whiten_params(n_works, N, H_dim)
        _wa = getattr(self, "whiten_alpha", None)
        _we = getattr(self, "whiten_eps_rel", None)
        w_alpha = auto_alpha if _wa is None else float(_wa)
        w_eps_rel = auto_eps_rel if _we is None else float(_we)
        print(
            f"[CoverRetrieval] map_whitened: n_works={n_works} N={N} H={H_dim} "
            f"-> alpha={w_alpha} eps_rel={w_eps_rel}"
        )
        self.log("test/map_whitened_alpha", w_alpha, rank_zero_only=True)
        self.log("test/map_whitened_eps_rel", w_eps_rel, rank_zero_only=True)
        embs_w = F.normalize(zca_whiten(embs, alpha=w_alpha, eps_rel=w_eps_rel), dim=-1)
        recall_ks_w = (
            [k for k in (1, 5, 10, 50, 100) if k < N] if self.log_extended_retrieval_metrics else []
        )
        hit_ks_w = [k for k in (1, 5, 10) if k < N] if self.log_extended_retrieval_metrics else []
        map_at_ks_w = [1] if self.log_extended_retrieval_metrics and N > 1 else []
        whiten_kwargs = dict(
            recall_ks=recall_ks_w,
            hit_ks=hit_ks_w,
            include_r_precision=self.log_extended_retrieval_metrics,
            include_median_rank=self.log_extended_retrieval_metrics,
            include_map=True,
            map_at_ks=map_at_ks_w,
            include_mrr=self.log_extended_retrieval_metrics,
        )
        if use_streaming:
            metrics_w = compute_retrieval_metrics_streaming(
                embs_w, work_ids, device=_md, **whiten_kwargs
            )
        else:
            metrics_w = compute_retrieval_metrics(embs_w @ embs_w.T, work_ids, **whiten_kwargs)
        map_whitened = metrics_w["map"]
        print(f"[CoverRetrieval] MAP (whitened) = {map_whitened:.4f}")
        self.log("test/map_whitened", map_whitened, prog_bar=False, rank_zero_only=True)
        if self.log_extended_retrieval_metrics:
            for k in (1, 5, 10, 50, 100):
                key = f"recall@{k}"
                if key in metrics_w:
                    self.log(f"test/{key}_whitened", metrics_w[key], rank_zero_only=True)
            for k in (1, 5, 10):
                key = f"hit_rate@{k}"
                if key in metrics_w:
                    self.log(f"test/{key}_whitened", metrics_w[key], rank_zero_only=True)
            self.log("test/r_precision_whitened", metrics_w["r_precision"], rank_zero_only=True)
            self.log("test/median_rank_whitened", metrics_w["median_rank"], rank_zero_only=True)
            if "map@1" in metrics_w:
                self.log("test/map@1_whitened", metrics_w["map@1"], rank_zero_only=True)
            self.log("test/mrr_whitened", metrics_w["mrr"], rank_zero_only=True)

        # ── Per-condition MAP (cross-instrument / cross-soundfont) ───────────
        # Only meaningful when the dataset carries a per-item condition
        # field (VGMIDITVar-timbre: gm_program; VGMIDITVar-multisf:
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
            unique_conds = sorted({c for c in file_conditions if c != -1})
            # Use the centered similarity matrix for cross-condition MAP —
            # consistent with offline analysis in
            # scripts/analysis/vgmiditvar_leitmotif_breakdown.py, which
            # subtracts the corpus mean before per-pair MAP to remove
            # cone-effect anisotropy (relevant for OMARRQ esp.).
            #
            # ``compute_perpair_map_all`` does a single batched argsort
            # pass and slices per (q, t) cell — for VGMIDITVar-timbre
            # (8 GM programs → 64 cells, N=102 960) this drops the grid
            # from ~4 h of unbatched per-cell sorts to ~3 min.
            if use_streaming:
                # Streaming path: chunk sim_c on GPU on-the-fly via embs_c.
                cell_results = compute_perpair_map_all_streaming(
                    embs_c,
                    file_work_ids,
                    file_conditions,
                    query_conds=unique_conds,
                    target_conds=unique_conds,
                    device=_md,
                )
            else:
                cell_results = compute_perpair_map_all(
                    sim_c,
                    file_work_ids,
                    file_conditions,
                    query_conds=unique_conds,
                    target_conds=unique_conds,
                )
            same_aps: list[float] = []
            cross_aps: list[float] = []
            for q in unique_conds:
                for t in unique_conds:
                    ap, n = cell_results.get((q, t), (0.0, 0))
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

            # Per-cell logging of the full (query_cond × target_cond) grid.
            # For VGMIDITVar-timbre with 8 GM programs that's 64 keys —
            # tractable in wandb. Use ``map_grid/q_to_t`` keys (slashes
            # group them under one dashboard section). Also dump CSV +
            # PNG heatmap next to the wandb run for paper figures.
            for q in unique_conds:
                for t in unique_conds:
                    ap, n = cell_results.get((q, t), (0.0, 0))
                    if n == 0:
                        continue
                    self.log(f"test/map_grid/{q}_to_{t}", ap, rank_zero_only=True)
                    self.log(f"test/map_grid/{q}_to_{t}_n", float(n), rank_zero_only=True)
            self._dump_condition_grid_artifacts(unique_conds, cell_results)

        # ── Anisotropy diagnostics ───────────────────────────────────────────
        # Cone-effect / rank-collapse measurements on the per-file embedding
        # matrix (pre-centering, since centering itself is what we're
        # diagnosing). High mean_vec_norm → ``map_centered`` is the trustworthy
        # number for this encoder. See marble/utils/retrieval_metrics.py.
        from marble.utils.retrieval_metrics import anisotropy_metrics

        ani = anisotropy_metrics(embs)
        # Log all four anisotropy metrics by default. Each is <1 ms to
        # compute and serves a distinct diagnostic role:
        #   - mean_vec_norm  : cone-collapse headline (single shared direction)
        #   - effective_rank : structural rank diversity (how many independent
        #                      directions carry signal after centering)
        #   - avg_pair_cos   : cross-check on mean_vec_norm (should ≈ mvn²
        #                      for L2-normalised input). Useful as an
        #                      independent confirmation that the cone-collapse
        #                      number is real and not a numerical artefact.
        #   - top1_sv_share  : fraction of post-centering variance explained
        #                      by the leading singular direction. Distinct
        #                      from mean_vec_norm because it's measured AFTER
        #                      removing the corpus mean.
        # Audit (2026-05-28) found avg_pair_cos and top1_sv_share were gated
        # behind log_extended_retrieval_metrics — that silently dropped them
        # from default sweeps and made anisotropy plots incomplete. Moved out.
        self.log("test/anisotropy/mean_vec_norm", float(ani["mean_vec_norm"]), rank_zero_only=True)
        self.log(
            "test/anisotropy/effective_rank",
            float(ani["effective_rank"]),
            rank_zero_only=True,
        )
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

    # ── per-condition grid CSV + heatmap PNG ──────────────────────────────────

    def _dump_condition_grid_artifacts(
        self,
        unique_conds: list[int],
        cell_results: dict[tuple[int, int], tuple[float, int]],
    ) -> None:
        """Write ``condition_grid.{csv,png}`` next to the wandb run dir so
        the 8x8 (query_cond x target_cond) MAP table is available offline
        for paper figures. Failures are logged but never raise -- this is a
        side-channel artefact, not a load-bearing metric.

        Matplotlib is imported lazily so headless envs without it still
        get the CSV. The output dir is derived from the trainer's logger;
        if no logger is attached (smoke tests) we skip silently.

        ALL print statements use ASCII-only characters because on Windows
        Python's default stdout encoding is cp1252 and the wandb console-
        capture wrapper re-encodes through it -- a UnicodeEncodeError on
        e.g. a U+2192 (right-arrow) here would propagate up through
        Lightning's on_test_epoch_end hook and crash the entire run AFTER
        all metrics are already logged. The whole method is wrapped in a
        broad try/except for the same defensive reason.
        """
        import contextlib

        try:
            self._dump_condition_grid_artifacts_inner(unique_conds, cell_results)
        except Exception as e:  # noqa: BLE001 — defensive; never break the run
            # Print at most a short ASCII summary so even this fallback
            # can't itself trigger a UnicodeEncodeError. Suppress any
            # further exception from the fallback print itself.
            with contextlib.suppress(Exception):
                print(f"[CoverRetrieval] WARN: condition grid dump failed: {type(e).__name__}")

    def _dump_condition_grid_artifacts_inner(
        self,
        unique_conds: list[int],
        cell_results: dict[tuple[int, int], tuple[float, int]],
    ) -> None:
        # Derive output dir from the wandb logger; fall back gracefully.
        out_dir = None
        try:
            logger = self.trainer.logger  # type: ignore[attr-defined]
            save_dir = getattr(logger, "save_dir", None) or getattr(logger, "_save_dir", None)
            run_dir = (
                getattr(logger.experiment, "dir", None) if hasattr(logger, "experiment") else None
            )
            out_dir = run_dir or save_dir
        except Exception:
            out_dir = None
        if out_dir is None:
            return
        import csv
        from pathlib import Path

        out_path = Path(out_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            return

        csv_path = out_path / "condition_grid.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["query_program", "target_program", "map", "n_queries"])
                for q in unique_conds:
                    for t in unique_conds:
                        ap, n = cell_results.get((q, t), (0.0, 0))
                        w.writerow([q, t, f"{ap:.6f}", n])
            print(f"[CoverRetrieval] wrote condition grid CSV -> {csv_path}")
        except OSError as e:
            print(f"[CoverRetrieval] WARN: could not write {csv_path}: {e}")
            return

        # PNG heatmap (matplotlib optional -- skip if missing or fails).
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import numpy as np  # noqa: PLC0415
            from matplotlib import pyplot as plt  # noqa: PLC0415

            n_c = len(unique_conds)
            grid = np.full((n_c, n_c), np.nan, dtype=float)
            for i, q in enumerate(unique_conds):
                for j, t in enumerate(unique_conds):
                    ap, n = cell_results.get((q, t), (0.0, 0))
                    if n > 0:
                        grid[i, j] = ap
            fig, ax = plt.subplots(figsize=(max(4, n_c * 0.7), max(4, n_c * 0.7)))
            im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0.0)
            ax.set_xticks(range(n_c))
            ax.set_yticks(range(n_c))
            ax.set_xticklabels([str(c) for c in unique_conds], rotation=45, ha="right")
            ax.set_yticklabels([str(c) for c in unique_conds])
            ax.set_xlabel("target condition (gm_program)")
            ax.set_ylabel("query condition (gm_program)")
            ax.set_title("Per-condition MAP grid (centered sim)")
            for i in range(n_c):
                for j in range(n_c):
                    v = grid[i, j]
                    if not np.isnan(v):
                        ax.text(
                            j,
                            i,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            color="white" if v < 0.5 else "black",
                            fontsize=8,
                        )
            fig.colorbar(im, ax=ax, label="MAP")
            fig.tight_layout()
            png_path = out_path / "condition_grid.png"
            fig.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"[CoverRetrieval] wrote condition grid PNG -> {png_path}")
        except ImportError:
            pass  # matplotlib not installed -- CSV only.
        except Exception as e:
            print(f"[CoverRetrieval] WARN: matplotlib heatmap failed: {e}")

    # ── back-compat shims (delegate to compute_retrieval_metrics) ─────────────
    # These three static methods used to contain their own per-row argsort
    # loops. Production paths (`on_test_epoch_end`) now run through
    # ``marble.utils.retrieval_metrics.compute_retrieval_metrics`` directly,
    # which folds map / map@K / mrr into the same batched pass as the
    # secondary metrics. The shims below are kept because
    # ``tests/test_compute_map_self_exclusion.py`` invokes them directly to
    # pin the self-exclusion convention. They delegate so we have ONE source
    # of truth.

    @staticmethod
    def _compute_map(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
        """Standard MAP — delegates to
        :func:`marble.utils.retrieval_metrics.compute_retrieval_metrics`.
        Self-exclusion convention: ``-inf`` sentinel + last-column drop
        (audit-2 #6, commit ac121f0). For large N use the bundle directly.
        """
        from marble.utils.retrieval_metrics import compute_retrieval_metrics

        return compute_retrieval_metrics(
            sim,
            work_ids,
            recall_ks=(),
            include_r_precision=False,
            include_median_rank=False,
            include_map=True,
        )["map"]

    @staticmethod
    def _map_at_k(sim: torch.Tensor, work_ids: torch.Tensor, k: int) -> float:
        """MAP@K — delegates to
        :func:`marble.utils.retrieval_metrics.compute_retrieval_metrics`.
        Uses the non-standard normalisation that divides AP@K by total
        ``n_relevant`` (not ``min(K, n_relevant)``); back-compat with the
        original ``test/map@1`` wandb key.
        """
        from marble.utils.retrieval_metrics import compute_retrieval_metrics

        return compute_retrieval_metrics(
            sim,
            work_ids,
            recall_ks=(),
            include_r_precision=False,
            include_median_rank=False,
            map_at_ks=(k,),
        )[f"map@{k}"]

    @staticmethod
    def _mrr(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
        """Mean reciprocal rank — delegates to
        :func:`marble.utils.retrieval_metrics.compute_retrieval_metrics`.
        """
        from marble.utils.retrieval_metrics import compute_retrieval_metrics

        return compute_retrieval_metrics(
            sim,
            work_ids,
            recall_ks=(),
            include_r_precision=False,
            include_median_rank=False,
            include_mrr=True,
        )["mrr"]
