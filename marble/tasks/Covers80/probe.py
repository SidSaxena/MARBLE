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
from marble.utils.emb_cache import (
    EmbeddingCache,
    compute_config_hash,
    encoder_tuple_to_pooled,
    stacked_to_layer_tuple,
)


class CoverRetrievalTask(LightningModule):
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
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.encoder = instantiate_from_config(encoder)
        self.emb_transforms = nn.ModuleList([instantiate_from_config(c) for c in emb_transforms])
        # Tiny dummy parameter so Lightning's model-summary and optimizer
        # don't complain about zero-parameter models.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Embedding cache (lazy-built on first batch — we need access to
        # the trainer's datamodule to compute the config hash from the
        # actual audio pipeline. See ``_ensure_cache``.)
        self.cache_embeddings = bool(cache_embeddings)
        self._cache: EmbeddingCache | None = None
        self._cache_init_attempted = False

    # ── cache plumbing ───────────────────────────────────────────────────────

    def _ensure_cache(self) -> None:
        """Lazily construct the EmbeddingCache once the trainer + datamodule
        are wired up. Safe to call repeatedly; only the first call does work."""
        if self._cache is not None or self._cache_init_attempted:
            return
        self._cache_init_attempted = True
        if not self.cache_embeddings:
            return
        # Derive a task tag from the WandB group name when available
        # ("OMARRQ-multifeature-25hz / SHS100K" → ("OMARRQ-multifeature-25hz",
        # "SHS100K")). Fall back to the encoder class name + a generic
        # "task" if WandB metadata isn't present.
        encoder_slug, task_name = self._derive_cache_slugs()
        model_id = getattr(self.encoder, "HUGGINGFACE_MODEL_NAME", encoder_slug)
        sr = int(getattr(self.encoder, "sampling_rate", self.sample_rate))
        # Pull clip_seconds from the (test) datamodule when available.
        clip_seconds = self._derive_clip_seconds()
        # Pipeline signature: class names of the test-stage transforms,
        # so changing the audio preprocessor invalidates the cache.
        pipeline_sig = self._derive_pipeline_signature()
        config_hash = compute_config_hash(
            encoder_model_id=model_id,
            sample_rate=sr,
            clip_seconds=clip_seconds,
            pipeline_signature=pipeline_sig,
        )
        self._cache = EmbeddingCache(
            encoder_slug=encoder_slug,
            task_name=task_name,
            config_hash=config_hash,
            metadata={
                "encoder_model_id": str(model_id),
                "sample_rate": sr,
                "clip_seconds": float(clip_seconds),
                "pipeline_signature": pipeline_sig,
            },
        )

    def _derive_cache_slugs(self) -> tuple[str, str]:
        """Return ``(encoder_slug, task_name)`` for the cache directory."""
        group = None
        try:
            logger = self.trainer.logger
            init_args = getattr(logger, "_wandb_init", None) or {}
            group = init_args.get("group")
        except Exception:
            group = None
        if isinstance(group, str) and " / " in group:
            enc, task = group.split(" / ", 1)
            return enc.strip(), task.strip()
        # Fallbacks
        enc = type(self.encoder).__name__
        task = type(self).__name__
        return enc, task

    def _derive_clip_seconds(self) -> float:
        try:
            test_cfg = self.trainer.datamodule.test_config  # type: ignore[attr-defined]
            init_args = test_cfg.get("init_args", {})
            cs = init_args.get("clip_seconds")
            if cs is not None:
                return float(cs)
        except Exception:
            pass
        return 0.0

    def _derive_pipeline_signature(self) -> str:
        try:
            transforms = self.trainer.datamodule.audio_transforms.get("test", [])  # type: ignore[attr-defined]
            # `transforms` here is a list of config dicts, not instances —
            # use the class_path string as the signature.
            sig_parts = [t.get("class_path", repr(t)) for t in transforms]
            return "|".join(sig_parts)
        except Exception:
            return ""

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
        self._ensure_cache()
        use_cache = self._cache is not None and clip_ids is not None

        if use_cache and self._cache.has_all(clip_ids):
            # Cache hit — skip encoder + time-pool entirely.
            cached = self._cache.get_batch(clip_ids).to(x.device)  # (B, L, H)
            self._cache.maybe_log(hit=True, n_clips=len(clip_ids))
            layer_tuple = stacked_to_layer_tuple(cached)  # tuple of (B, 1, H)
        else:
            # Miss path — run the encoder, time-pool, persist to cache.
            layer_outputs = self.encoder(x)  # tuple of (B, T, H)
            if use_cache:
                pooled = encoder_tuple_to_pooled(layer_outputs)  # (B, L, H)
                self._cache.put_batch(clip_ids, pooled)
                self._cache.maybe_log(hit=False, n_clips=len(clip_ids))
                layer_tuple = stacked_to_layer_tuple(pooled)
            else:
                layer_tuple = layer_outputs

        h = layer_tuple
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

    def test_step(self, batch, batch_idx):
        # batch is either the legacy 3-tuple (waveform, work_ids, paths)
        # or the new 4-tuple (..., clip_ids). The retrieval datamodules
        # in this repo now always emit the 4-tuple; older external
        # datamodules still work via the unpack fallback.
        if len(batch) == 4:
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

    def on_test_epoch_end(self) -> None:
        all_embs = torch.cat(self._test_embeddings)  # (N_clips, H)
        all_work_ids = torch.cat(self._test_work_ids)  # (N_clips,)
        all_paths = self._test_paths

        # ── per-file mean-pool (aggregates clips from the same track) ────────
        path2work: dict[str, int] = {}
        path2embs: dict[str, list] = {}
        for emb, wid, path in zip(all_embs, all_work_ids.tolist(), all_paths, strict=True):
            path2work.setdefault(path, wid)
            path2embs.setdefault(path, []).append(emb)

        file_embs: list[torch.Tensor] = []
        file_work_ids: list[int] = []
        for path, embs_list in path2embs.items():
            stacked = torch.stack(embs_list)  # (n_clips, H)
            mean_emb = stacked.mean(0)
            mean_emb = F.normalize(mean_emb, dim=-1)  # re-normalise
            file_embs.append(mean_emb)
            file_work_ids.append(path2work[path])

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
        self.log("test/map@1", self._map_at_k(sim, work_ids, k=1), rank_zero_only=True)
        self.log("test/map@1_centered", self._map_at_k(sim_c, work_ids, k=1), rank_zero_only=True)
        self.log("test/mrr", self._mrr(sim, work_ids), rank_zero_only=True)
        self.log("test/mrr_centered", self._mrr(sim_c, work_ids), rank_zero_only=True)

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
