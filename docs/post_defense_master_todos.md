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
