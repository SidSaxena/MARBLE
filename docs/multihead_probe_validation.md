# Multi-head parallel layer probing — GPU validation plan

**Status: pending GPU validation.** The implementation is CPU-unit-tested
(`tests/test_multihead_probe.py`, incl. a bitwise update-equivalence proof),
but the claim that ONE multi-head run reproduces the committed per-layer
single-head results has to be validated on the PC before any sweep is
replaced by it.

## What this is

`configs/probe.MuQ-multihead.MedleyDBMelody.yaml` trains a per-layer
`MLPDecoderKeepTime` head for **all 13 MuQ layers + a meanall head in one
run** (task `ProbeAudioTaskMultiHead`, decoder `PerLayerHeads`, callback
`PerHeadBestCheckpoint`), replacing the 13 per-layer runs + 1 meanall run of
the fold-0 `probe.MuQ-layers` / `probe.MuQ-meanall` sweep. Heads are
parameter-disjoint and the loss is the per-head sum, so with per-parameter
Adam each head receives *exactly* the updates its independent run would at
the same LR (proven bitwise in
`tests/test_multihead_probe.py::test_update_equivalence`). Everything else —
probe architecture, data, folds, epochs, batch/accumulation, cache — is
identical to the single-head protocol.

## (a) Run fold-0 multi-head MuQ on MedleyDBMelody

On the PC (`ssh my-pc` → WSL `~/developer/marble`):

```bash
cd ~/developer/marble

# fit (~40 epochs; reuses the frame-level (L,T,H) cache the layer sweep
# built — watch for "[emb_cache] HIT" on the first batch; a MISS means the
# cache key drifted, stop and check the wandb group / audio pipeline)
.venv/bin/python cli.py fit -c configs/probe.MuQ-multihead.MedleyDBMelody.yaml \
    --trainer.logger.init_args.name "multihead-fold0-fit" \
    --trainer.logger.init_args.job_type "fit"

# test (LoadLatestCheckpointCallback loads best.ckpt, then
# PerHeadBestCheckpoint restores EACH head to its own best-val-epoch weights
# — the console must print "[PerHeadBestCheckpoint] restored 14 head(s): ...")
.venv/bin/python cli.py test -c configs/probe.MuQ-multihead.MedleyDBMelody.yaml \
    --trainer.logger.init_args.name "multihead-fold0-test" \
    --trainer.logger.init_args.job_type "test"
```

Notes:
* Name/job_type are set explicitly per stage (same convention as the
  meanall-fold runs in `scripts/sweeps/run_medleydb_melody_folds.sh`); the
  config's `name: MuQ-multihead` is only a fallback.
* Do **not** rename the wandb `group` ("MuQ / MedleyDBMelody") — it is the
  embedding-cache directory key; renaming re-extracts the whole corpus.
* Other folds: `sed -e "s/fold_idx: 0/fold_idx: ${F}/g" -e
  "s#MuQ-multihead/#MuQ-multihead.fold${F}/#g"` on the config, exactly like
  the fold script does (checkpoint dirs must not collide across folds).

## (b) Compare per-layer test RPA against the single-head anchors

The test run logs `test/acc_rpa_l{0..12}`, `test/acc_rpa_meanall`,
`test/acc_rca_*`, and `test/acc_rpa_best` — read them from the Lightning
test table printed at the end, or from the wandb run summary
(`wandb.Api().run(...).summary`). Summary scripts should read these keys
from the ONE run instead of per-run summaries.

Committed single-head anchors (MuQ, MedleyDBMelody):

| quantity | single-head result |
|---|---|
| fold-0 layer curve peak | layer 1, RPA ≈ 0.638 |
| layer 11, 5-fold | RPA 0.557 ± 0.040 |
| meanall, 5-fold | RPA 0.635 ± 0.044 |

Acceptance: **same-fold** per-layer deltas (`test/acc_rpa_l{k}` vs the
fold-0 single-head run of layer k) within seed/run noise ≈ **±0.01**, and
the *shape* of the layer curve preserved (argmax layer identical or an
adjacent layer within noise). The 5-fold numbers are cross-fold aggregates —
only compare against them after running all 5 folds.

Expected benign discrepancy sources (all sub-±0.01):
* **Init/dropout RNG** — head k's init draws and dropout masks differ from a
  solo run's at the same seed (the K heads share the global RNG stream).
  Seed-level noise, not bias.
* **Test-metric aggregation** — the multi-head test path logs the metric
  *objects* (exact corpus-level Σcorrect/Σtotal); the single-head test path
  logs per-batch values that Lightning batch-size-weight-averages. ~1e-3
  difference for RPA/RCA.
* **LR schedule** — see below; usually small because heads plateau together.

## If the deltas do NOT sit within ±0.01

Check, in this order:

1. **LR coupling** (most likely). The multi-head `ReduceLROnPlateau` is
   keyed to `val/acc_rpa_best`, so LR halvings follow the *best* head; a
   single-head run's LR followed its *own* metric. Compare the logged
   `lr-Adam` trajectory against the affected layer's single-head run. To
   isolate: rerun both the multi-head config and one affected single-head
   config with the `lr_scheduler` block deleted (constant 1e-3) — under
   constant LR the update-equivalence invariant is exact, so remaining gaps
   are NOT scheduler-related.
2. **Normalization / state leakage across heads.** The invariant requires
   heads to be parameter-disjoint and the shared trunk stateless. Verify no
   emb_transform with trainable/statistics-carrying state (LayerWeightedSum,
   BatchNorm-anything) is in the config, and that
   `sum(p.numel() for p in task.encoder.parameters() if p.requires_grad) == 0`.
   Also verify global gradient clipping is OFF (`trainer.gradient_clip_val`
   unset) — a global norm couples heads through the clip factor.
3. **Label / layer alignment.** Head k must see hidden-state k:
   `LayerSelector(layers=["0..12"])` preserves encoder order, and
   `PerLayerHeads` slices `emb[:, k]`. A tell-tale of an off-by-one is the
   fold-0 layer curve shifted by one position vs the single-head curve. Also
   confirm `label_freq: 25` and the same `time_dim_mismatch_tol: 5`.
4. **Cache integrity.** Confirm the fit log printed `[emb_cache] HIT` (the
   run consumed the same fp16 frame cache the single-head sweep used). A
   re-extracted cache is fine too — but a MISS + different pipeline
   signature means the inputs differ from the anchors'.
5. **Per-head restore at test.** The test log must show
   `[PerHeadBestCheckpoint] restored 14 head(s)` with per-head epochs that
   differ across heads (all-identical epochs suggests the callback monitored
   a missing key and warned during fit — grep the fit log for
   `PerHeadBestCheckpoint] WARNING`).

## Documented deviations from the single-head protocol (deliberate)

* **Fixed `max_epochs: 40`, no per-layer early stopping.** Single-head runs
  early-stop (patience 7) per layer; one run cannot stop per head.
  Mitigation: `PerHeadBestCheckpoint` restores each head to its own best val
  epoch, which is what early stopping's checkpointing achieved. Cost: the
  run always trains the full 40 epochs.
* **Shared LR schedule.** One optimizer, one `ReduceLROnPlateau` keyed to
  `val/acc_rpa_best`. Under a *constant* LR the per-head updates are exactly
  those of independent runs (proven); the plateau scheduler re-introduces a
  mild coupling (checklist item 1).
* **One wandb run for all layers.** `sweep/layer` is not stamped per head;
  summary tooling reads `test/acc_rpa_l{k}` keys from the single run.
* **Exact-global test aggregation** (see discrepancy sources above).

## Cost expectation

One fit ≈ the wall-clock of ~one single-head fit (encoder forward is
cache-served either way; 14 heads ≈ 4.1 M params total vs 0.3 M — head
FLOPs are negligible next to the data path), replacing 14 fits: ~14×
end-to-end reduction for the fold-0 sweep, minus the early-stopping epochs
single-head runs saved.

## Next adopters

`ProbeAudioTaskMultiHead` lives in `marble/tasks/HookTheoryMelody/probe.py`
(MedleyDBMelody re-exports it), so HookTheoryMelody needs only a config:
copy its `-layers` config, switch the task class, set `LayerSelector
layers: ["0..L-1"]`, swap the decoder for `PerLayerHeads(num_layers=L,
include_meanall=true)`, add `PerHeadBestCheckpoint`, monitor
`val/acc_rpa_best`, drop EarlyStopping.

## Claims wording for the thesis (adversarial-review outcome, 2026-07-06)

Review verdict: SHIP-WITH-FIXES; no result-corrupting code bug found. Binding
rules on how multi-head numbers may be claimed:

1. **Never write "identical/bitwise-equivalent to independent single-layer
   runs."** The bitwise proof covers constant LR + dropout 0. The shipped
   config's shared `ReduceLROnPlateau` (on `val/acc_rpa_best`) gives every
   head ONE common LR schedule — a *systematic* deviation from per-run
   adaptive schedules, active precisely because heads plateau at different
   epochs. Correct phrasing: *"multi-head approximation, empirically
   validated against independent single-layer runs within run noise
   (±0.01 RPA, argmax layer preserved)."*
2. Bitwise equivalence **to the committed single-head anchors is impossible
   by construction** (they used per-run ReduceLROnPlateau + per-run
   EarlyStopping, which a joint run cannot replicate per head).
3. The fold-0 anchor comparison is a **sanity check, not an equivalence
   proof**: the acceptance band also absorbs a disclosed aggregation-method
   difference (single-head test logs batch-weighted per-batch RPA; multi-head
   object-logs the exact global ratio; ~1e-3, larger for unequal
   voiced-frame counts).
4. No per-layer number is published before the fold-0 GPU validation in this
   doc has actually been run and passed.

**Smoke-test footgun:** running the multihead config with the WandbLogger
disabled changes the cache-slug fallback (`type(self).__name__` →
`ProbeAudioTaskMultiHead`), silently splitting the cache dir and
re-extracting 26 GB. Always keep the wandb group `"MuQ / MedleyDBMelody"`
(mode=offline is fine; logger *removal* is not).

## Layer-aggregation protocol (July 2026 research synthesis)

Web-research verdict: per-layer curves + softmax weighted head + normalized
gate logging is a defensible 2026 protocol, PROVIDED the reporting hierarchy
below is respected. Key sources: Feng et al., TASLP 2024 (arXiv:2404.09385,
the SUPERB follow-up: norm confound + weights-vs-accuracy Spearman rho only
0.37-0.49); Zaiem et al., Interspeech 2023 / CS&L 2024 (probe capacity
reorders rankings); Zhou et al., Interspeech 2025 (arXiv:2505.16306 — on
MusicFM/MuQ, single best layer often BEATS the weighted sum in music);
Tuned Lens, arXiv:2303.08112 (the one-pass-many-probes precedent); MARBLE
paper's own protocol (its Appendix A hyperparameter grid is
{every single layer, weighted sum} — our multi-head+weighted run is the
efficient realization of exactly that grid).

Historical note: MARBLE v1 (benchmark/models/probers.py, layer="all") HAD
the softmax featurizer with per-epoch weight logging; the June-2025 v2
refactor dropped it (only the unused Conv1d LayerWeightedSum survived) and
pinned per-task fixed layers. Our implementation restores the v1 capability
with the 2024+ fixes.

Reporting hierarchy (binding for the thesis):
1. Per-layer single-layer probe curves = the ground-truth layer-contribution
   measure (the multi-head run produces them directly).
2. Weighted-sum head = a headline aggregate reported ALONGSIDE the best
   single layer — never as an importance measure. Expect single-layer to
   possibly beat it (Zhou 2025); report both honestly.
3. Softmax gates = secondary panel only, and only because the weighted head
   LayerNorms each layer before mixing (Feng et al.); report the
   gates-vs-per-layer-accuracy Spearman rho as an explicit check.

Roadmap items adopted from the research (not yet implemented):
- Cross-task weight transfer to zero-shot retrieval: melody-learned gates
  applied frozen to VGMIDITVar-timbre embeddings — defensible (no retrieval
  labels touched) but flag: weights are strongly task-dependent and may
  pick the wrong layer band; treat as one baseline vs meanall/best-layer.
- Probe-capacity sensitivity check (linear vs 512-MLP) on one task
  (answers Zaiem); attentive-pooling-over-time head as a second design
  axis (arXiv:2605.10494); both cheap from the frame cache.
- OMAR-RQ has NO published layer map (paper probes last layer only) — our
  24-layer sweep is a novel contribution, not a reproduction.
