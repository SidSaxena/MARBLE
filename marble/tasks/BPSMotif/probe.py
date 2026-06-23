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
        map_raw = compute_within_group_multilabel_map(embs, file_groups, file_letters, file_occ)
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


# ──────────────────────────────────────────────────────────────────────────────
# Whole-piece-context within-piece task
# ──────────────────────────────────────────────────────────────────────────────

import re  # noqa: E402 — local to the whole-piece block

_VOICE_TAG_RE = re.compile(r"\[V:([^\]]+)\]")


def _bar_of_patch_from_texts(patch_texts: list[str]) -> list[int]:
    """Physical bar number for each patch, from its decoded ABC text.

    Bit-identical port of the leitmotifs canonical kernel
    (``leitmotifs-symbolic/scripts/diagnostics/annotate_clusters_with_bars.py::
    _bar_of_patch_from_texts``) so the whole-piece probe's patch→bar mapping
    matches the dataset assembler's physical-bar axis exactly.

    Bar 0 = header patches (no ``[V:n]`` tag, or before the first one). Bars 1+
    count from the first ``[V:first_voice]`` occurrence onward; the bar number
    increments every time another ``[V:first_voice]`` patch is seen. "First voice"
    is the first voice that appears in patch order (defensive against pieces where
    V:1 has no content). ``search`` (not ``match``) because the tag often appears
    mid-patch in the patchilised stream.
    """
    first_voice: str | None = None
    voice_per_patch: list[str | None] = []
    for t in patch_texts:
        m = _VOICE_TAG_RE.search(t)
        v = m.group(1).strip() if m else None
        voice_per_patch.append(v)
        if first_voice is None and v is not None:
            first_voice = v

    bars: list[int] = []
    bar = 0
    for voice in voice_per_patch:
        if voice == first_voice and voice is not None:
            bar += 1
        bars.append(bar if voice is not None else 0)
    return bars


class BPSMotifWithinPieceWholeTask(CoverRetrievalTask):
    """WHOLE-PIECE-CONTEXT within-movement same-motif retrieval on BPS-Motif.

    The whole-piece-context counterpart of :class:`BPSMotifWithinPieceTask`. Both
    score the SAME metric (``compute_within_group_multilabel_map``) on the SAME
    windows (same physical-bar spans, labels, occurrence ids). The ONLY difference
    is the ENCODING:

    * :class:`BPSMotifWithinPieceTask` (clip-isolated): the datamodule tokenises
      each 4-bar window's OWN ABC slice and the encoder time-avg-pools it — the
      encoder never sees the rest of the movement. Peaked shallow-mid (≈L4).

    * THIS task (whole-piece): the datamodule yields the WHOLE movement's patches
      (un-truncated) + the window specs. Per movement we encode the whole ABC ONCE
      at PER-PATCH resolution (segmenting >512, stitching — the encoder's
      datamodule cap of 512 would otherwise drop most of a movement), map
      patches→physical bars, then pool each window's bar-patches → window vectors
      IN MOVEMENT CONTEXT. This reproduces the leitmotifs whole-piece prototype,
      which peaked DEEP (≈L8: raw L8≈0.598, L2≈0.498).

    The selected hidden layer (or meanall) is read from the ``LayerSelector`` in
    ``emb_transforms`` — the standard ``forward`` pipeline is bypassed because it
    pools patches away; we need the pre-pool per-patch states. ``cache_embeddings``
    is ignored here (the per-patch whole-movement encode is not the cacheable
    post-pool ``(L, H)`` tensor).

    Logs ``test/map`` (raw) and ``test/map_centered`` (per-movement centered),
    identical to the clip-isolated task.
    """

    # ── layer selection (read off the LayerSelector transform) ──────────────
    def _selected_layers(self) -> tuple[list[int], str]:
        """Return ``(layer_indices, mode)`` from the configured LayerSelector.

        ``mode`` is ``"select"`` (single layer in ``layer_indices``) or one of
        ``"mean"``/``"sum"`` (meanall over ``layer_indices``). Defaults to layer 6
        if no LayerSelector is present (matches the config default).
        """
        from marble.modules.transforms import LayerSelector

        for t in self.emb_transforms:
            if isinstance(t, LayerSelector):
                return list(t.layers), t.mode
        return [6], "select"

    # ── whole-movement per-patch encode (segmenting >512) ───────────────────
    @torch.no_grad()
    def _encode_movement_all_hidden(self, patches_full: torch.Tensor) -> torch.Tensor:
        """One whole movement's patches ``(P, PATCH_SIZE)`` → ``(13, P, H)``.

        Reaches into the symbolic encoder's inner ``BertModel`` to request
        per-patch ``output_hidden_states=True``, segmenting inputs >``PATCH_LENGTH``
        (final segment overlaps back) and stitching — mirroring the leitmotifs
        adapter's ``encode_symbolic_per_patch_all_hidden`` exactly. SEGMENTATION,
        not truncation: every patch of a 144–1250-patch movement is kept.
        """
        import torch.nn.functional as F

        enc = self.encoder
        cfg = enc.config
        device = next(enc.model.parameters()).device
        sym = enc.model.symbolic_model  # M3PatchEncoder
        bert = sym.base  # HF BertModel
        patch_embed = sym.patch_embedding  # Linear(PATCH_SIZE*128 → H)
        pad_id = enc.patchilizer.pad_token_id
        max_len = cfg.PATCH_LENGTH
        n_layers = cfg.PATCH_NUM_LAYERS + 1  # 13
        H = cfg.M3_HIDDEN_SIZE

        total = patches_full.size(0)
        if total == 0:
            return torch.zeros((n_layers, 0, H))

        segments = [patches_full[i : i + max_len] for i in range(0, total, max_len)]
        if len(segments) > 1 and segments[-1].size(0) < max_len:
            segments[-1] = patches_full[-max_len:]

        per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
        for segment in segments:
            seg_len = segment.size(0)
            pad_len = max_len - seg_len
            mask = F.pad(torch.ones(seg_len, device=device), (0, pad_len), value=0.0)
            pad_token = torch.full(
                (pad_len, cfg.PATCH_SIZE), pad_id, dtype=segment.dtype, device=device
            )
            seg_padded = torch.cat((segment.to(device), pad_token), dim=0).long()

            oh = F.one_hot(seg_padded, num_classes=128).float()  # (P, PS, 128)
            flat = oh.reshape(seg_padded.size(0), -1)  # (P, PS*128)
            emb = patch_embed(flat).unsqueeze(0)  # (1, P, H)

            out = bert(
                inputs_embeds=emb,
                attention_mask=mask.unsqueeze(0),
                output_hidden_states=True,
            )
            valid = int(mask.sum().item())
            for li in range(n_layers):
                per_layer[li].append(out.hidden_states[li][0, :valid, :].cpu())

        remainder = total % max_len
        layers: list[torch.Tensor] = []
        for li in range(n_layers):
            outs = per_layer[li]
            if len(outs) > 1 and remainder != 0:
                outs[-1] = outs[-1][-remainder:]
            layers.append(torch.cat(outs, dim=0))  # (P, H)
        return torch.stack(layers, dim=0)  # (13, P, H)

    def _select_layer_feats(self, all_layers: torch.Tensor) -> torch.Tensor:
        """``(13, P, H)`` → ``(P, H)`` for the configured layer / meanall."""
        layer_idx, mode = self._selected_layers()
        if mode in ("mean", "sum") and len(layer_idx) > 1:
            sel = all_layers[layer_idx]  # (L, P, H)
            return sel.mean(dim=0) if mode == "mean" else sel.sum(dim=0)
        return all_layers[layer_idx[0]]  # (P, H)

    # ── test hooks ───────────────────────────────────────────────────────────
    def on_test_start(self) -> None:
        self._wp_embeddings: list[torch.Tensor] = []
        self._wp_groups: list[int] = []
        self._wp_occ: list[set] = []
        self._wp_letters: list[set] = []

    def test_step(self, batch, batch_idx):
        import json

        import torch.nn.functional as F

        from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer

        patchilizer = M3Patchilizer()

        # batch is a python list (identity collate); typically one movement.
        for item in batch:
            patches_full, movement_id_int, windows_json, _movement_id = item
            group = int(movement_id_int)
            windows = json.loads(windows_json)

            all_layers = self._encode_movement_all_hidden(patches_full)  # (13, P, H)
            feats = self._select_layer_feats(all_layers)  # (P, H)

            # Decode each patch to its ABC text → physical bar number per patch.
            patch_texts = [patchilizer.patch2bar(p.tolist()) for p in patches_full]
            bar_of_patch = _bar_of_patch_from_texts(patch_texts)
            if len(bar_of_patch) != feats.shape[0]:
                raise RuntimeError(
                    f"{_movement_id}: bar map len {len(bar_of_patch)} != "
                    f"patch feats {feats.shape[0]} (rows misaligned)"
                )

            # Group patch indices by physical bar (drop header bar 0).
            bar_to_idx: dict[int, list[int]] = {}
            for pi, b in enumerate(bar_of_patch):
                if b > 0:
                    bar_to_idx.setdefault(b, []).append(pi)

            # Pool each window's bar-patches → one vector per window.
            for spec in windows:
                idxs: list[int] = []
                for b in range(spec["bar_start"], spec["bar_end"] + 1):
                    idxs.extend(bar_to_idx.get(b, []))
                if not idxs:
                    # Window covers only empty/header bars — fall back to the
                    # whole-movement mean so the window still yields a vector
                    # (vanishingly rare; keeps the window count aligned).
                    win_vec = feats.mean(dim=0)
                else:
                    win_vec = feats[idxs].mean(dim=0)
                self._wp_embeddings.append(F.normalize(win_vec, dim=-1))
                self._wp_groups.append(group)
                self._wp_occ.append(set(spec["occurrence_ids"]))
                self._wp_letters.append(set(spec["letters"]))

    def on_test_epoch_end(self) -> None:
        import torch.nn.functional as F

        from marble.utils.retrieval_metrics import (
            anisotropy_metrics,
            compute_within_group_multilabel_map,
        )

        if not self._wp_embeddings:
            return

        embs = torch.stack(self._wp_embeddings)  # (N, H), L2-normed
        file_groups = list(self._wp_groups)
        file_letters = list(self._wp_letters)
        file_occ = list(self._wp_occ)

        N = embs.shape[0]
        n_movements = len(set(file_groups))
        layer_idx, mode = self._selected_layers()
        print(
            f"\n[BPSMotifWithinPieceWhole] Evaluating WHOLE-PIECE-CONTEXT "
            f"within-movement same-motif MAP over {N} windows "
            f"({n_movements} movements); layers={layer_idx} mode={mode}."
        )

        # ── raw within-group multi-label MAP (the headline) ──────────────────
        map_raw = compute_within_group_multilabel_map(embs, file_groups, file_letters, file_occ)
        print(f"[BPSMotifWithinPieceWhole] MAP (raw)      = {map_raw:.4f}")
        self.log("test/map", map_raw, prog_bar=True, rank_zero_only=True)

        # ── per-movement-centered MAP ────────────────────────────────────────
        groups_t = torch.tensor(file_groups)
        embs_c = embs.clone()
        for g in torch.unique(groups_t):
            mask = groups_t == g
            embs_c[mask] = embs[mask] - embs[mask].mean(dim=0, keepdim=True)
        embs_c = F.normalize(embs_c, dim=-1)
        map_centered = compute_within_group_multilabel_map(
            embs_c, file_groups, file_letters, file_occ
        )
        print(f"[BPSMotifWithinPieceWhole] MAP (centered) = {map_centered:.4f}")
        self.log("test/map_centered", map_centered, prog_bar=False, rank_zero_only=True)

        # ── anisotropy diagnostics ───────────────────────────────────────────
        ani = anisotropy_metrics(embs)
        self.log("test/anisotropy/mean_vec_norm", float(ani["mean_vec_norm"]), rank_zero_only=True)
        self.log(
            "test/anisotropy/effective_rank", float(ani["effective_rank"]), rank_zero_only=True
        )
        print(
            f"[BPSMotifWithinPieceWhole] Anisotropy: mean_vec_norm={ani['mean_vec_norm']:.3f}  "
            f"eff_rank={ani['effective_rank']:.1f}"
        )
