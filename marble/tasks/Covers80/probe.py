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


class CoverRetrievalTask(LightningModule):
    """
    Zero-shot cover-song retrieval probe.

    Parameters
    ----------
    sample_rate    : int  target audio sample rate (for documentation only).
    encoder        : dict config for the pre-trained audio encoder.
    emb_transforms : list[dict] LayerSelector + TimeAvgPool configs.
    """

    # Disable Lightning's automatic gradient-step machinery — we have
    # nothing to optimise (max_epochs=0).
    automatic_optimization = False

    def __init__(
        self,
        sample_rate: int,
        encoder: dict,
        emb_transforms: list[dict],
    ):
        super().__init__()
        self.sample_rate   = sample_rate
        self.encoder       = instantiate_from_config(encoder)
        self.emb_transforms = nn.ModuleList(
            [instantiate_from_config(c) for c in emb_transforms]
        )
        # Tiny dummy parameter so Lightning's model-summary and optimizer
        # don't complain about zero-parameter models.
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    # ── forward: encoder → transforms → flatten → L2-normalise ──────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalised embeddings of shape (B, H).
        """
        h = self.encoder(x)
        for t in self.emb_transforms:
            h = t(h)
        # h: (B, L, T, H) after LayerSelector + TimeAvgPool → (B, L=1, T=1, H)
        # Collapse L and T: (B, H)
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
        self._test_work_ids:   list[torch.Tensor] = []
        self._test_paths:      list[str]           = []

    def test_step(self, batch, batch_idx):
        x, work_ids, paths = batch          # work_ids: (B,) int tensor
        embeddings = self(x)                # (B, H) — L2 normalised
        self._test_embeddings.append(embeddings.detach().cpu())
        self._test_work_ids.append(
            work_ids.cpu() if isinstance(work_ids, torch.Tensor)
            else torch.tensor(work_ids)
        )
        self._test_paths.extend(paths)

    def on_test_epoch_end(self) -> None:
        all_embs     = torch.cat(self._test_embeddings)  # (N_clips, H)
        all_work_ids = torch.cat(self._test_work_ids)    # (N_clips,)
        all_paths    = self._test_paths

        # ── per-file mean-pool (aggregates clips from the same track) ────────
        path2work: dict[str, int] = {}
        path2embs: dict[str, list] = {}
        for emb, wid, path in zip(all_embs, all_work_ids.tolist(), all_paths):
            path2work.setdefault(path, wid)
            path2embs.setdefault(path, []).append(emb)

        file_embs:     list[torch.Tensor] = []
        file_work_ids: list[int]          = []
        for path, embs_list in path2embs.items():
            stacked  = torch.stack(embs_list)              # (n_clips, H)
            mean_emb = stacked.mean(0)
            mean_emb = F.normalize(mean_emb, dim=-1)       # re-normalise
            file_embs.append(mean_emb)
            file_work_ids.append(path2work[path])

        embs     = torch.stack(file_embs)                  # (N, H)
        work_ids = torch.tensor(file_work_ids)             # (N,)
        N        = len(work_ids)

        # Prefix is generic ("[CoverRetrieval]") because this class is
        # reused by VGMIDITVar / SHS100K / etc. — the prior "[Covers80]"
        # was misleading when the same code runs against any work-id-keyed
        # retrieval dataset. The originating task is also visible in the
        # WandB run's group/tags.
        print(f"\n[CoverRetrieval] Evaluating retrieval MAP over {N} tracks "
              f"({len(set(file_work_ids))} works).")

        # ── cosine similarity (embeddings already L2-normalised) ─────────────
        sim = embs @ embs.T   # (N, N)

        # ── Centering variant: remove cone-effect anisotropy ─────────────────
        # The anisotropy diagnostic (scripts/diagnostics/anisotropy_diag.py)
        # found OMARRQ embeddings live in a cone (mean_vec_norm ≈ 0.5). For
        # cosine retrieval that shared direction inflates every pairwise
        # similarity and squashes discrimination. Subtracting the corpus
        # mean ("all-but-the-top", Mu 2018) removes it. We log BOTH the raw
        # and centered MAP so the comparison is automatic for every encoder.
        embs_c = embs - embs.mean(dim=0, keepdim=True)
        embs_c = F.normalize(embs_c, dim=-1)
        sim_c  = embs_c @ embs_c.T

        # ── compute MAP for both variants ─────────────────────────────────────
        map_raw      = self._compute_map(sim,   work_ids)
        map_centered = self._compute_map(sim_c, work_ids)
        print(f"[CoverRetrieval] MAP (raw)      = {map_raw:.4f}")
        print(f"[CoverRetrieval] MAP (centered) = {map_centered:.4f}")

        self.log("test/map",          map_raw,      prog_bar=True,  rank_zero_only=True)
        self.log("test/map_centered", map_centered, prog_bar=False, rank_zero_only=True)
        self.log("test/map@1",          self._map_at_k(sim,   work_ids, k=1), rank_zero_only=True)
        self.log("test/map@1_centered", self._map_at_k(sim_c, work_ids, k=1), rank_zero_only=True)
        self.log("test/mrr",          self._mrr(sim,   work_ids), rank_zero_only=True)
        self.log("test/mrr_centered", self._mrr(sim_c, work_ids), rank_zero_only=True)

    @staticmethod
    def _compute_map(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
        """Standard MAP from a similarity matrix and work_id labels."""
        N = len(work_ids)
        aps: list[float] = []
        for i in range(N):
            sims_i = sim[i].clone()
            sims_i[i] = -2.0                               # exclude self
            order      = sims_i.argsort(descending=True)
            is_rel     = (work_ids[order] == work_ids[i])
            n_relevant = int(is_rel.sum().item())
            if n_relevant == 0:
                continue
            hits = 0
            ap   = 0.0
            for rank, rel in enumerate(is_rel.tolist(), start=1):
                if rel:
                    hits += 1
                    ap   += hits / rank
            ap /= n_relevant
            aps.append(ap)
        return float(torch.tensor(aps).mean().item()) if aps else 0.0

    # ── helper metrics ────────────────────────────────────────────────────────

    @staticmethod
    def _map_at_k(sim: torch.Tensor, work_ids: torch.Tensor, k: int) -> float:
        N   = len(work_ids)
        aps = []
        for i in range(N):
            s = sim[i].clone()
            s[i] = -2.0
            top_k   = s.argsort(descending=True)[:k]
            is_rel  = (work_ids[top_k] == work_ids[i]).float()
            n_total = int((work_ids == work_ids[i]).sum().item()) - 1
            if n_total == 0:
                continue
            hits = 0
            ap   = 0.0
            for rank, rel in enumerate(is_rel.tolist(), start=1):
                if rel:
                    hits += 1
                    ap   += hits / rank
            ap /= n_total
            aps.append(ap)
        return float(torch.tensor(aps).mean().item()) if aps else 0.0

    @staticmethod
    def _mrr(sim: torch.Tensor, work_ids: torch.Tensor) -> float:
        N     = len(work_ids)
        recip = []
        for i in range(N):
            s = sim[i].clone()
            s[i] = -2.0
            order = s.argsort(descending=True)
            is_rel = (work_ids[order] == work_ids[i])
            nz = is_rel.nonzero(as_tuple=True)[0]
            if len(nz) == 0:
                continue
            recip.append(1.0 / (nz[0].item() + 1))
        return float(torch.tensor(recip).mean().item()) if recip else 0.0
