# marble/tasks/BPSMotif/probe.py
"""
BPS-Motif probe tasks.

Two parallel tasks share this module:

* :class:`BPSMotifMNIDTask` — clip-level binary classification on
  motif-window vs sampled non-motif-window MIDI slices. Inherits
  fit/test/metric/cache plumbing from :class:`BaseTask`; the
  per-file aggregation in BaseTask is a no-op here since each window
  is already a separate file.

* :class:`BPSMotifRetrievalTask` — within-piece within-letter motif
  retrieval. Trivial subclass of :class:`CoverRetrievalTask`: the only
  difference is that the datamodule encodes ``(piece_id, motif_letter)``
  jointly into ``work_id`` so the standard MAP scoring counts
  "same piece + same letter" occurrences as relevant — exactly the
  within-movement motif identity we want, because motif letters in this
  dataset are movement-local. No method overrides needed; this class
  exists only to give the WandB run a distinct ``task`` tag and to give
  the LightningCLI a stable import path.
"""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection

from marble.core.base_task import BaseTask
from marble.core.utils import instantiate_from_config
from marble.tasks.Covers80.probe import CoverRetrievalTask


class BPSMotifMNIDTask(BaseTask):
    """Binary motif-window classification probe on BPS-Motif.

    See :class:`marble.core.base_task.BaseTask` for the underlying
    fit/test/metric machinery. This subclass exists to:

    * Build encoder / transforms / decoders / losses / metrics from the
      YAML config (the standard MARBLE wiring pattern).
    * Hold a stable import path so the LightningCLI ``class_path`` is
      ``marble.tasks.BPSMotif.probe.BPSMotifMNIDTask``.
    """

    def __init__(
        self,
        sample_rate: int,
        use_ema: bool,
        encoder: dict,
        emb_transforms: list[dict],
        decoders: list[dict],
        losses: list[dict],
        metrics: dict[str, dict[str, dict]],
        cache_embeddings: bool = False,
    ):
        enc = instantiate_from_config(encoder)
        tfs = [instantiate_from_config(cfg) for cfg in emb_transforms]
        decs = [instantiate_from_config(cfg) for cfg in decoders]
        loss_fns = [instantiate_from_config(cfg) for cfg in losses]
        metric_maps = {
            split: {name: instantiate_from_config(cfg) for name, cfg in metrics[split].items()}
            for split in ("train", "val", "test")
        }
        super().__init__(
            encoder=enc,
            emb_transforms=tfs,
            decoders=decs,
            losses=loss_fns,
            metrics=metric_maps,
            sample_rate=sample_rate,
            use_ema=use_ema,
            cache_embeddings=cache_embeddings,
        )

    # ── per-window test aggregation ────────────────────────────────────────
    #
    # BPS-Motif windows are 1:1 with files (each motif occurrence / sampled
    # negative is its own per-window MIDI), so there is no per-file
    # aggregation step like HookTheoryKey's per-slice→per-song majority
    # vote. We rely on BaseTask's default test_step + on_test_epoch_end,
    # which compute the standard torchmetrics (F1, accuracy, precision,
    # recall via the metric collection defined in the YAML).

    def on_test_start(self) -> None:
        self._test_outputs: list[dict] = []

    def test_step(self, batch, batch_idx):
        # 4-tuple (patches, label, midi_path, clip_id).
        if isinstance(batch, (tuple, list)) and len(batch) >= 4:
            x, labels, _paths, clip_ids = batch[0], batch[1], batch[2], batch[3]
        else:
            x, labels, _paths = batch
            clip_ids = None
        logits = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        for logit, lb in zip(logits, labels, strict=False):
            self._test_outputs.append({"logit": logit, "label": lb})

    def on_test_epoch_end(self) -> None:
        if not self._test_outputs:
            return
        batched_logits = torch.stack([e["logit"] for e in self._test_outputs])
        batched_labels = torch.stack([e["label"] for e in self._test_outputs])
        mc: MetricCollection | None = getattr(self, "test_metrics", None)
        if mc is not None:
            metrics_out = mc(batched_logits, batched_labels)
            self.log_dict(
                metrics_out,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )


class BPSMotifRetrievalTask(CoverRetrievalTask):
    """Within-piece within-letter motif retrieval on BPS-Motif.

    Subclass only for the import path. All MAP / centering / aggregation
    logic is inherited from :class:`CoverRetrievalTask`. The datamodule
    encodes (piece_id, motif_letter) jointly into ``work_id`` so the
    standard ``_compute_map`` scores same-piece-same-letter occurrences
    as relevant — which IS within-piece within-letter retrieval, because
    BPS-Motif's letters are movement-local.
    """

    pass


class BPSMotifWithinPieceTask(CoverRetrievalTask):
    """Within-piece phrase-window same-motif retrieval on BPS-Motif.

    Slides N-bar phrase windows (stride 1) over each WHOLE movement (dataset
    built by ``scripts/data/build_bps_motif_within_piece.py``) and measures, PER
    MOVEMENT, whether two windows that share a motif letter retrieve each other —
    genuine within-movement recurrence. The semantics are the
    shuffle-control-validated leitmotifs prototype
    (``scripts/eval/bps_within_piece_metric.py::within_movement_map``).

    The inherited :class:`CoverRetrievalTask` MAP is single-label, full-gallery,
    self-only-excluded — it CANNOT express this task, which is multi-label
    (relevant = shares >=1 motif letter), per-movement-gallery, and
    same-occurrence-excluded. So this subclass reuses the encoder / forward /
    cache plumbing but overrides the three test hooks to accumulate the
    6-tuple labels and call
    :func:`marble.utils.retrieval_metrics.compute_within_group_multilabel_map`.

    Logs:

    * ``test/map`` — raw within-group multi-label MAP (the headline the sweep
      parser reads).
    * ``test/map_centered`` — the SAME metric on PER-MOVEMENT-centered
      embeddings (subtract each movement's mean, then re-L2-normalise — NOT a
      global corpus mean, because the gallery is per-movement so the cone to
      remove is per-movement too).
    """

    def on_test_start(self) -> None:
        self._wp_embeddings: list[torch.Tensor] = []
        self._wp_groups: list[torch.Tensor] = []
        self._wp_occ: list[str] = []
        self._wp_letters: list[str] = []
        self._wp_keys: list[str] = []

    def test_step(self, batch, batch_idx):
        # 6-tuple: (patches, movement_id_int, occ_ids_str, letters_str,
        #           clip_key, clip_id).
        x, movement_ids, occ_ids_str, letters_str, clip_keys, clip_ids = batch
        embeddings = self(x, clip_ids=list(clip_ids) if clip_ids is not None else None)
        self._wp_embeddings.append(embeddings.detach().cpu())
        self._wp_groups.append(
            movement_ids.cpu()
            if isinstance(movement_ids, torch.Tensor)
            else torch.tensor(movement_ids)
        )
        # occ_ids_str / letters_str / clip_keys ride collation as list-of-str.
        self._wp_occ.extend(list(occ_ids_str))
        self._wp_letters.extend(list(letters_str))
        self._wp_keys.extend(list(clip_keys))

    def on_test_epoch_end(self) -> None:
        import torch.nn.functional as F

        from marble.utils.retrieval_metrics import (
            anisotropy_metrics,
            compute_within_group_multilabel_map,
        )

        if not self._wp_embeddings:
            return

        all_embs = torch.cat(self._wp_embeddings)  # (N, H), already L2-normed
        all_groups = torch.cat(self._wp_groups).tolist()  # (N,)

        # ── per-file mean-pool (window == file here, so this is 1:1; we keep
        #    the aggregation for parity with CoverRetrievalTask and to be safe
        #    if a window ever splits into >1 cached clip) ──────────────────────
        key2idx: dict[str, int] = {}
        file_embs: list[torch.Tensor] = []
        file_groups: list[int] = []
        file_occ: list[set[str]] = []
        file_letters: list[set[str]] = []
        file_clip_buf: dict[str, list[torch.Tensor]] = {}
        order: list[str] = []
        for emb, grp, occ_s, let_s, key in zip(
            all_embs, all_groups, self._wp_occ, self._wp_letters, self._wp_keys, strict=True
        ):
            if key not in key2idx:
                key2idx[key] = len(order)
                order.append(key)
                file_groups.append(grp)
                file_occ.append(set(occ_s.split("|")) - {""})
                file_letters.append(set(let_s.split("|")) - {""})
                file_clip_buf[key] = []
            file_clip_buf[key].append(emb)
        for key in order:
            stacked = torch.stack(file_clip_buf[key])
            mean_emb = F.normalize(stacked.mean(0), dim=-1)
            file_embs.append(mean_emb)

        embs = torch.stack(file_embs)  # (N, H)
        N = embs.shape[0]
        n_movements = len(set(file_groups))
        print(
            f"\n[BPSMotifWithinPiece] Evaluating within-movement same-motif MAP "
            f"over {N} windows ({n_movements} movements)."
        )

        # ── raw within-group multi-label MAP (the headline) ──────────────────
        map_raw = compute_within_group_multilabel_map(
            embs, file_groups, file_letters, file_occ
        )
        print(f"[BPSMotifWithinPiece] MAP (raw)      = {map_raw:.4f}")
        self.log("test/map", map_raw, prog_bar=True, rank_zero_only=True)

        # ── per-movement-centered MAP ────────────────────────────────────────
        # Subtract each movement's own mean (the gallery is per-movement, so the
        # anisotropy cone to remove is per-movement too — a GLOBAL mean would
        # leave each movement's local cone intact), then re-L2-normalise.
        groups_t = torch.tensor(file_groups)
        embs_c = embs.clone()
        for g in torch.unique(groups_t):
            mask = groups_t == g
            embs_c[mask] = embs[mask] - embs[mask].mean(dim=0, keepdim=True)
        embs_c = F.normalize(embs_c, dim=-1)
        map_centered = compute_within_group_multilabel_map(
            embs_c, file_groups, file_letters, file_occ
        )
        print(f"[BPSMotifWithinPiece] MAP (centered) = {map_centered:.4f}")
        self.log("test/map_centered", map_centered, prog_bar=False, rank_zero_only=True)

        # ── anisotropy diagnostics (same as CoverRetrievalTask) ──────────────
        ani = anisotropy_metrics(embs)
        self.log("test/anisotropy/mean_vec_norm", float(ani["mean_vec_norm"]), rank_zero_only=True)
        self.log(
            "test/anisotropy/effective_rank", float(ani["effective_rank"]), rank_zero_only=True
        )
        print(
            f"[BPSMotifWithinPiece] Anisotropy: mean_vec_norm={ani['mean_vec_norm']:.3f}  "
            f"eff_rank={ani['effective_rank']:.1f}"
        )
