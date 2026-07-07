# Post-defense master TODO list (canonical, written 2026-07-07)

*You will have forgotten everything by the time you read this. This doc is
self-contained: every item carries its context, project, priority, effort,
and pointers. Detailed method plans live in the linked docs; this is the
index you start from after the defense.*

**Projects:** `marble` = the benchmark/probing repo (this repo).
`leitmotifs` = the deployed Zelda leitmotif-discovery pipeline.
`both` = result feeds both (and any future paper).

---

## State of the world (as of 2026-07-07, for cold restart)

- **Deployed pipeline** uses **MuQ L11** (mean-centered cosine ranking),
  chosen on CSI-style / VGMIDITVar-timbre retrieval. The melody campaigns
  showed melody peaks EARLY (MuQ L1–L2) — this is a **validated dissociation,
  not an oversight**: L2 (best melody layer) loses 34 % cross-instrument MAP
  vs L11. Numbers + hypothesis: [leitmotif_layer_transfer_todos.md](leitmotif_layer_transfer_todos.md).
- **Completed campaigns** (all data in committed CSVs under `docs/figures/`):
  VGMIDITVar-timbre 4-encoder layer sweeps (MuQ 0.226@L8 > OMAR 0.188@L15 >
  CLaMP3 0.130@L4 > MERT 0.113@L11, varctl); MedleyDB melody 3-encoder 5-fold
  multi-head (MuQ 0.655@L1 > OMAR 0.621@L5 > MERT 0.606@L9); HookTheory
  melody multi-head+weighted (MuQ best L2=0.473, weighted 0.482; MERT done;
  OMAR was finishing on defense-eve).
- **Melody peak sits below the invariance peak in every encoder** (MuQ 1<8,
  OMAR 5<15, MERT 9<11) — the thesis's cross-encoder law.
- **Learned gates** (SUPERB weighted head): MuQ concentrates early (HTM:
  L2=0.45), MERT late (L8-11), OMAR diffuse; gates≠importance exhibit
  (Spearman 0.15–0.97). Gate CSVs committed under
  `docs/figures/{medleydb,hooktheory}_melody_multihead/`.
- **Infrastructure you built and will forget**: multi-head probe
  (`ProbeAudioTaskMultiHead`, validated ≤0.0091 vs single-layer anchors);
  `LayerSoftmaxSum(learnable=False, init_weights=…)` frozen-gate transfer
  (41cd456); top-k config generator (`scripts/sweeps/gen_topk_configs.py`);
  fp16 embedding caches (MedleyDB frame caches exist for all 3 encoders —
  top-k experiments there are config-only, ~15 min/run; HookTheory has NO
  cache, TB-scale, runs live ~5-6 h).
- **Bug you fixed and must not reintroduce**: torchmetrics
  `MetricCollection(compute_groups=True)` silently aliased RPA↔RCA in
  multi-head TEST metrics (4e24707 fixed; memory file
  `project_metriccollection_compute_groups_bug.md`).
- Runs live on the **PC** (`ssh my-pc` → WSL `~/developer/marble`), wandb
  project `marble`, groups `<encoder> / <Task>`; clean HTM test runs are
  named `multihead-weighted-test-fix`.

---

## P0 — do these first (highest value per hour)

| # | Item | Project | Effort | Context & pointers |
|---|---|---|---|---|
| P0.1 | **Benchmark layer-fusion search** (Phase 1 of [layer_combination_search_plan.md](layer_combination_search_plan.md)): build the per-layer score cache (~21 GB, ~30 min GPU), then fixed candidates (single layers, uniform, RRF{L2,L8,L11}, **frozen HTM melody gates** — prediction: loses to L11), greedy subset, Dirichlet weights. Work-disjoint holdout, report shrinkage. | both | ½ day human + <1 h GPU | Subsumes TODOs 1–3 of the transfer doc. Decides "does adding melody layers help retrieval" with labels. |
| P0.2 | **Zelda validation assets** (Phase 2): curate 40–80 verified theme-restatement pairs from your motif annotations + concert-album↔OST cross-version pairs (Symphony of the Goddesses etc.). | leitmotifs | ½ day human, no compute | Turns the unlabeled corpus into a recall@K benchmark; unblocks every deployment claim. Highest-leverage curation in the plan. |
| P0.3 | **Top-k matrix conclusions**: read the 10-arm fold-0 results (wandb tag `topk`, names `topk-<arm>-test`); if an arm beats/ties best-single or full-13, extend that arm to 5 folds (~75 min/arm, config-only on cache). First result on defense-eve: MuQ htm-top3 {2,1,12} = 0.641 > best-single 0.630 > full-13 weighted 0.616. | marble | 1 h analysis + ~2 h GPU if extending | Generator: `scripts/sweeps/gen_topk_configs.py`. Arms: htm-top3/top5, own-top3, bottom3 (neg control), htm-top3-frozen (pure transfer). |

## P1 — the deployment decision + hypothesis discriminators

| # | Item | Project | Effort | Context & pointers |
|---|---|---|---|---|
| P1.1 | **Deployment A/B with pooled labeling** (Phase 3): run 3–4 configs (L11 baseline, L8, P0.1 winner, union-of-nominations) on the Zelda corpus; pool top-K nominations, label the pool blind, score precision/recall-in-pool. Union is judged on *recall at reviewable precision*, not MAP. | leitmotifs | 1 day (mostly listening) + pipeline runs | Needs P0.1 + P0.2. This picks what ships. Also settles L8-vs-L11 (benchmark argmax is L8; L11 wins separability 0.424 & condition gap 0.073). |
| P1.2 | **NSynth monophonic pitch probe** (per encoder): if MERT's pure-pitch peak is also late → its objective buries the spectral surface; if early → MERT's L9 melody peak is about *predominance* (voice selection). | marble | ~1 day (new task wiring + 3 short runs) | The discriminator for the "peak position tracks the pre-training target" hypothesis (MedleyDB appendix + defense Q&A). |
| P1.3 | **Linear-vs-MLP probe at MuQ L1/L2** (Zaiem check): is pitch surface-linear early? | marble | ~2 h (config-only on MedleyDB cache) | Cheap; strengthens the Mel-RVQ-surface story. |

## P2 — worth doing, not urgent

| # | Item | Project | Effort | Context & pointers |
|---|---|---|---|---|
| P2.1 | **Unsupervised layer-selection criteria** (Phase 4): rank-correlate eff_rank / hubness / top-K stability with holdout MAP across P0.1 candidates. Either outcome is a contribution ("label-free recipe" or its documented absence). | both | ½ day once P0.1 cache exists | eff_rank per layer already in `docs/figures/vgmiditvar_timbre_muq_varctl/summary_table.csv`. |
| P2.2 | **eff_rank-for-supervised analogue**: does effective rank predict the best *probing* layer too? | marble | ~2 h offline on cached embeddings | Ties the supervised and unsupervised stories. |
| P2.3 | **Defensive MedleyDB re-verify post compute_groups fix**: re-run one multihead test per encoder, confirm numbers match committed CSVs (they were empirically clean — 0/185 aliased — but verify before any paper). | marble | ~30 min GPU | Bug memory: `project_metriccollection_compute_groups_bug.md`. |
| P2.4 | **HTM frozen-mix caching infra** (only if HTM-side top-k ever matters): post-transform cache flag whose key includes transform config (~40 lines + validation). Enables HTM top-k at ~3 h instead of ~11 h. | marble | ½ day careful infra | Current cache is pre-transform; key excludes `emb_transforms` (verified in `compute_config_hash`). |
| P2.5 | **HXMSA build step**: Harmonix BigVGAN (912 tracks, rendered audio + MSA jsonl) downloaded to D: + Modal on 2026-06-18; the MARBLE task build was never done. | marble | ~1 day | Memory: `project_hxmsa_harmonix_bigvgan.md`. Only if structure tasks return to scope. |
| P2.6 | **Perf leftovers from the optimization roadmap** (items 4–8: CachedStubEncoder, consolidated memmap cache, int8 A/B, MERT SDPA, CLaMP3 compile): only matter if new large sweeps are planned. | marble | varies, ½–1 day each | [optimization_roadmap_2026-07.md](optimization_roadmap_2026-07.md) §Ranked plan. |

## In flight on defense-eve (finish/verify FIRST, may already be done)

- **OMAR-RQ HookTheory**: fit was running (ETA ~16:00 on 2026-07-07); its
  retest (`retest` tmux, `-fix` name) fires automatically after
  `HTM_ALL_DONE`. Then: re-run `scripts/analysis/hooktheory_melody_export.py`
  on the PC (auto-includes OMAR), pull CSVs, commit, re-run
  `scripts/analysis/hooktheory_melody_report.py` → 3-encoder figures →
  HTM appendix section in the thesis (mirror the MedleyDB one,
  `app:medleydb` in `chapters/A_appendix.tex`).
- **Top-k matrix** (tmux `topk`): all 10 arms; see P0.3.
- Check watchers/tmux are gone; PC repo on `main` (the old
  `feat/bps-within-piece` was merged & deleted; PC was force-synced 2026-07-07).

## Priority rationale (one paragraph, so future-you trusts the ordering)

P0 items are cheap and decide the *scientific* question (do combinations
beat L11, and on what evidence); P1 turns that into the *deployment*
decision and closes the two open hypothesis threads that a paper reviewer
would poke at; P2 is infrastructure and garnish that only pays off if the
project continues past the paper. Within each tier, items are ordered by
(value ÷ effort). The single most valuable non-obvious asset: **P0.2 — the
labeled Zelda pairs.** Everything deployment-facing routes through it.

---

# Complete two-repo TODO inventory (exhaustive sweep, 2026-07-07)

*Both repos were swept file-by-file (code comments + every markdown doc). The
P0–P2 tables above cover the layer-fusion/deployment thread; this section is
everything else that exists, so nothing lives outside this doc. Detail stays
in the source docs — this is the index.*

## ⚠️ PRE-DEFENSE items found in the sweep (do BEFORE the defense, not after)

From `~/leitmotifs/docs/thesis/2026-07-03-final-review-findings-and-plan.md`
(flagged there as defense-critical, ~2 days):
- **T1.2 chance-level null baseline** (~½ day; new ~100-line null generator)
- **T1.3 cover-pair stratification**
- **T1.4 family-bootstrap CIs** (`family_bootstrap.py`)
- plus T1.1/T1.5–T1.8 (circularity paragraph, chroma/LAMA baselines decision,
  leitmotif-definition paragraph, Table 4.1 re-render, cluster metrics)
- One **live QbE Find smoke test** before the demo (model warm-up proof) —
  `docs/results_inventory_2026-07-06.md:108`
- Two storage confirmations in that same inventory doc: Mac-viewer
  full-corpus audio source; WD-Black holds canonical raw embeddings before
  deleting the Mac copy.

## Project: leitmotifs (`~/leitmotifs`) — canonical doc: `docs/ROADMAP.md`

**Repo state**: code is TODO-clean (all pending work tracked in markdown).
**Cleanup flag**: `docs/TODO.md` there is a **stale duplicate** of ROADMAP.md
(pre-2026-06-12, "261-track") but `DOC_INDEX.md` still points to it as "★
outstanding work" — deprecate it. `docs/doc_factcheck_audit_2026-06-27.md`
lists the stale line-refs inside ROADMAP itself.

Open work by cluster (status per the sweep):
- **The one unbuilt executable plan**: `docs/superpowers/plans/`
  `2026-06-22-label-driven-themes-pipeline.md` — its target scripts
  (`build_label_results.py`, `check_label_taxonomy.py`) exist in NO branch;
  prereqs (golden set, taxonomy) all exist. Unblocked, unimplemented,
  highest-value unfinished plan in that repo.
- **12.5 Hz final-run evaluation** — run completed (6f44bd0c); eval blocked
  on new labels + `corpus_exclusions.txt`; like-for-like 25 Hz re-eval queued
  (`docs/plans/2026-07-02-run-12p5hz-final.md` §3–4; thesis W1–W8 backlog in
  `docs/thesis/2026-07-02-writing-plan-ch04-08.md:521`).
- **ROADMAP §B assumption tests** (floor-transfer, dual-mode-at-power,
  Leiden purity) — flagged highest-leverage there.
- **ROADMAP §A5 QbE Phases 3–4** (server RAM LRU; CLaMP3 panel) and **§A6
  CLaMP3 pseudo-labeling** (blocked: symbolic data uncommitted, submodule
  uninit); `extract_modal.py` CLaMP3 stub is open-by-design until A5 Phase 4.
- **ROADMAP §D operational** — 263-track raw+whitened full runs, cluster
  validations (Dark Beast Ganon/Mipha/Sidon), MERT blind A/B, raw∪whitened
  union mode, int8/lazy viewer storage, `development`→`main` merge.
- **ROADMAP Improvement roadmap Tiers 0–2** (#3 window sweep, #4 mask
  analysis, #6 symbolic fusion, #7 audio→MIDI, #8 source-sep, #9 learned
  projection, #10–13).
- **Waveform-sync defects** on the currently-checked-out branch
  (`hotfix/waveform-cursor-sync`): D1/D2/D4 likely fixed by recent commits;
  **D3 (isPlaying crop-mode trap) and D5 (double-seek) plausibly open**;
  §9 open questions in `docs/waveform_sync_audit_2026-06.md`.
- **Review-findings residue**: 24 low/cosmetic open + 4 deferred-by-choice
  (`docs/review_findings.md`); `min_edge_weight` default decision.
- Matrix-profile result cache (8M pairwise scalars, avoids 4–5 h recompute) —
  tracked in *marble's* `docs/TODO.md:194` but belongs to this repo.

## Project: marble — threads OUTSIDE the layer-fusion story (from `docs/TODO.md` legacy log + plan docs)

- **Symbolic-motif thread** (entire, untracked by the P0–P2 tables):
  `symbolic_motif_benchmark_roadmap.md` (MTC-ANN build ~1 day, full-BPS
  extension, Essen fallback, **LoRA/InfoNCE contrastive re-head on CLaMP3
  L6-L7** to lift the 0.84/0.49 ceiling); `clamp3_contrastive_training_plan.md`
  (unchecked execution checklist + **5 open decisions for the user**);
  `mtc_ann_scoping.md`; `bps_motif_*` (audio variant deferred, MNID
  integer-boundary-negative rebuild, `_layer_done` fold-awareness bug);
  `kern_sourcing_bps_jkupdd.md`; `symbolic_encoder_landscape.md` (Aria /
  MidiBERT / Moonbeam integration; re-check ISMIR 2026 accepts Jul–Aug).
- **Robustness experiments** (`docs/TODO.md`): background-leitmotif level
  mismatch (−6/−12/−18/−24 dB renders, cross-level MAP grid); ecological
  per-instrument reverb variant (`VGMIDITVar-timbre-ecological`).
- **VGMIDITVar analysis leftovers**: confound-free control 8×8 grid NOT yet
  run (needs audio staged back from D: + re-extraction); MERT/CLaMP3 re-runs
  under control; whitening layer studies for CLaMP3/MERT/OMARRQ (only MuQ
  done — `docs/whitening_ablation.md:157`).
- **MedleyDB probe plans** (planning-status docs): instrument-activation
  multi-label probe; remix-invariance probe (CUT/GATED by audit — 5 open
  decisions); leitmotif-eval T-suite sequencing (T2→T5→T3→T4→T8).
- **SuperMario / structure thread**: boundary-detection variant (branch
  recoverable from reflog `b0748c3`); secondary heads (8-label sections,
  IsAdaptive/IsStinger, compound-similarity regression); Bricasti re-render;
  audio-encoder sweep pending; structure-dataset queue (SongFormBench, JSD,
  TAVERN, BPSD, SALAMI…).
- **Layer-analysis follow-up flagged highest-priority there**: re-run
  MuQ × HookTheoryStructure and MuQ × HookTheoryKey with corrected FE configs
  (invalidated by audit fix #1) — `docs/layer_analysis.md:78`.
- **Infra/robustness deferrals**: LeitmotifDetection task has NO embedding
  cache wiring (~10× waste); UTF-8 `open()` audit; wandb-core Windows spawn
  failures; GPU offload of the 102960² metric block; Modal
  `setup_vgmiditvar`; torch.compile for MuQ/OMARRQ/Qwen2Audio + sparse Modal
  entrypoints (beyond roadmap item 8); OMAR-RQ beat-tracking gap hypotheses;
  attentive-pooling-over-time head (multihead doc §roadmap).
- **Code TODOs** (only real ones in first-party code): theory utils —
  `marble/utils/theory/lead_sheet.py:100` (downbeat alignment),
  `theorytab.py:349` (chord-inversion parsing `NotImplementedError`).
- **2-minute verification**: `docs/superpowers/plans/2026-06-19-vgmloop-`
  `audit-fixes.md` — all boxes unchecked but target branch deleted; confirm
  the 3 fixes are on `main`, then mark the plan done.

## Sweep provenance

Marble sweep: code + all ~30 docs; the four consolidated docs
(master/transfer/combination-plan/optimization-roadmap) cover ONLY the
layer-fusion + perf threads — the symbolic, structure, robustness, and
MedleyDB-probe threads above were outside them until this section.
Leitmotifs sweep: code clean; `ROADMAP.md` is canonical; eval-harness
worktree no longer exists (harness shipped: `scripts/eval/*`,
`tests/golden/leitmotif_instances.json`).
