# Analysis-script inventory — marble · leitmotifs · smc-msc-thesis

Compiled 2026-07-06. The cross-repo map of every analysis/pipeline/utility script:
what it does and where it lives. One line per script; CLI details live in each
script's own docstring (leitmotifs additionally has a full CLI reference —
see §2). Classes: **CORE** (pipeline stage), **EVAL** (metrics/benchmarks),
**FIG** (plots/reports/wandb logging), **DATA** (dataset build/convert),
**RUN** (launch wrapper), **INFRA** (maintenance/serving), **STALE**
(superseded/one-time-done — safe-to-delete candidates).

Machines: marble runs on the PC (`ssh my-pc` → WSL `~/developer/marble`,
git-synced with the Mac clone this doc lives in); leitmotifs is Windows-native
on the PC (`C:/Users/Sid/developer/python/leitmotifs`); the thesis is Mac-only
(`~/Developer/UPF/SMC/Thesis/smc-msc-thesis`).

---

## 1. marble (`scripts/`) — encoder probing & retrieval benchmarks

### scripts/analysis/ — result analysis & reports
| script | purpose | class |
|---|---|---|
| `vgm_timbre_sweep_from_cache.py` | Offline VGMIDITVar-timbre layer sweep from the embedding cache (fused GPU metrics, varctl + score dump, wandb per layer). THE sweep engine for the timbre task | CORE |
| `vgm_timbre_report.py` | **The figure/CSV generator**: `report` (per-encoder fig1-6+figS1-3+CSVs), `compare` (N-encoder panels), `thesis` (all thesis-styled vgm_* figures + defense PNGs straight into the thesis repo). Data-driven annotations; zero edits per new encoder | FIG |
| `compare_encoders_vgmiditvar_timbre.py` | Older cross-encoder timbre comparison (pre-`vgm_timbre_report`) | STALE |
| `whitening_ablation.py` | Whitening treatments ablation (raw/centered/ZCA×alpha) on retrieval tasks | EVAL |
| `best_layer.py` | Cross-task/encoder best-layer consistency views from wandb | EVAL |
| `fix_wandb_runs.py` | First-class W&B run ops: list/rename/retag/archive (scalar-only moveRuns) | INFRA |
| `reconstruct_condition_grid_from_cache.py` | Rebuild an 8×8 instrument grid from cached embeddings (spot re-derivation) | EVAL |
| `regen_instrument_grids.py` | Restyle/regenerate the `vgm_instrument_grids_<enc>` thesis+defense figures from the **committed** best-layer slices (`docs/figures/vgmiditvar_timbre_<enc>_varctl/best_layer_condition_grid{,_varctl}.csv`) — no sweep re-run. Seam-free heatmaps (`imshow(interpolation='nearest')`+`set_rasterized(True)`+`grid(False)`). `--extra "Disp:layer:work_csv:varctl_csv"` renders non-best deployed layers (e.g. MuQ L11, MERT L7) → `_l<layer>` suffix. Keep all encoders on ONE script version so styling stays consistent | FIG |
| `sms_clamp3_symbolic_report.py` (+`_extras.py`) | SuperMarioStructure CLaMP3-symbolic ABC-vs-MIDI benchmark report | EVAL |
| `ceiling_mnid_bar_granularity.py` | BPS-Motif bar-granularity labeling F1 ceiling | EVAL |
| `inspect_leitmotif_labels.py` | Print labels/counts in a LeitmotifDetection JSONL | EVAL |

### scripts/embeddings/ — the cache
| script | purpose | class |
|---|---|---|
| `extract.py` | Pre-warm the per-clip embedding cache for any probe config (`--precision {fp32,tf32,bf16}`, bf16 default = TF32+autocast+inference_mode, ~2-4× on tensor cores) | CORE |
| `manage.py` | Cache CLI: list/info/clear | INFRA |
| `audit_cache_integration.py` | Static audit that every task wires the cache correctly | EVAL |

### scripts/sweeps/ — launchers & summaries
| script | purpose | class |
|---|---|---|
| `run_sweep_local.py` / `run_all_sweeps.py` | THE canonical local launchers (skip-completed via wandb test/ keys, auto-resume, `--concurrency`, per-layer logs) — never launch `cli.py` bare | CORE |
| `gen_sweep_configs.py` / `gen_within_piece_n_configs.py` | Generate per-layer sweep configs from a template | CORE |
| `autotune_concurrency.py` | Recommend concurrency + cache-RAM mode from live RAM/GPU measurements | INFRA |
| `run_medleydb_melody_folds.sh` | MedleyDB melody: 5 folds × layer sweep launcher (workers default 6) | RUN |
| `run_bps_*.sh` ×7 | BPS-Motif sweep launchers (MNID/retrieval/within-piece × MTF/ABC × folds/windows) | RUN |
| `bps_*_summary.py` ×4, `jkupdd_*_summary.py` ×2, `mtc_ann_abc_vs_mtf_summary.py` | Aggregate the respective sweeps into tables | FIG |
| `plot_bps_*.py` ×2, `plot_jkupdd_*.py` ×2, `plot_mtc_ann_abc_vs_mtf.py` | Per-layer figures for those sweeps | FIG |
| `modal/modal_sweep.py` / `modal/modal_run_all_sweeps.py` | Parallel per-layer sweeps on Modal GPUs | RUN |
| `backfill_retrieval_metrics.sh`, `smoke_one_layer.sh` | Metric backfill; single-layer smoke | RUN |

### scripts/data/ — dataset builders (all DATA)
`build_vgmiditvar_dataset.py` + `postprocess_vgmiditvar.py` + `rewrite_vgmidi_programs.py` +
`audit_soundfont_assignment.py` (the VGMIDITVar/timbre render chain);
`build_bps_motif_{dataset,abc,within_piece}.py`; `build_jkupdd_{abc,retrieval}.py`;
`build_mtc_ann_dataset.py`; `build_medleydb_melody_jsonl.py`; `build_hooktheory_melody_jsonl.py`;
`build_supermario_dataset.py`; `build_hxmsa_dataset.py`; `convert_vgm_corpus_to_jsonl.py`;
`convert_{audio_format,mxl_to_abc,mxl_to_midi,shs100k_to_flac}.py`; `kern_to_mxl_abc.py`
(+ vendored `_vendor/xml2abc.py`); `download_{covers80,hooktheory,ninsheetmusic,nsynth,shs100k}.py`;
`cache_audio_info_in_jsonl.py` (kills per-record torchaudio.info at init);
`rewrite_jsonl_audio_paths.py`; `build_smoke_jsonl.py`; `verify_retrieval_jsonl.py`.

### scripts/{diagnostics,verify,maintenance}/
`diagnostics/`: `anisotropy_diag.py`, `probe_omarrq_shapes.py`, `smoke_test_compile.py`
(compile_mode A/B), `smoke_test_wav.py`, `count_hooktheorymelody_slices.py`,
`hooktheorymelody_missing_audio.py`, `test_clamp3_crossmodal_semantic.py`, `test_mps_compat.py` — all EVAL/diagnostic.
`verify/`: `verify_hooktheory.py`, `verify_shs100k.py` — dataset integrity (DATA).
`maintenance/`: `wandb_checkpoint_audit.py` — delete local ckpts whose runs are complete in W&B (INFRA).

### repo root
`cli.py` (LightningCLI entry), `modal_marble.py` (Modal probe runner), `download.py`, `setup.py`.

---

## 2. leitmotifs (`C:/Users/Sid/developer/python/leitmotifs`) — BotW discovery pipeline

**Already self-documented**: `docs/scripts_reference.md` is the full CLI reference
(defer to `docs/pipeline_parameters.md` on conflicts); also `docs/PIPELINE.md`,
`docs/EVALUATION.md`, `docs/pipeline_db.md`, `docs/viewer.md`, `docs/research_log.md`.
This section is the map; that reference is the manual.

### scripts/pipeline/ — the audio discovery pipeline (CORE)
`run_pipeline.sh` (orchestrator: extract → MP → DTW → canon → Leiden → viewer) →
`matrix_profile_pipeline.py` (GPU windowed-cosine MP nomination) → `dtw_enrich.py`
(canonical DTW pass; path_mean_cos feeds edge weights) → `canonicalize_occurrences.py` →
`segment_match.py` (Leiden themes/variants/occurrences) → `build_browsable.py` /
`build_scope.py` (viewer data; scoped rebuilds from pipeline.db) → `write_manifest.py`.
Plus `add_tracks.py` (incremental AB-join). INFRA: `tag_corpus.py`, `backfill_track_durations.py`.
STALE: `migrate_results_to_db.py` (done), `overnight_*.sh` ×3 (dated runners).

### scripts/extract/ — embeddings (CORE)
`extract_local.py`, `extract_modal.py` (all layers, parallel L4s),
`extract_clamp3_symbolic.py`; INFRA: `fetch_symbolic.py`, `download_embeddings.py`,
`verify_embeddings{,_local}.py`. STALE: `repack_embeddings.py`/`repack_modal.py` (root cause fixed).

### scripts/eval/ — evaluation vs golden labels (EVAL)
Harness: `../evaluate.py` (match-level CLI), `build_golden.py`. Metric families:
`ceiling_decomposition.py` + `null_ceiling.py` (recall ceiling + chance control),
`floor_analysis.py` (Method A), `sibling_recall.py` (Method B), `pairwise_links.py`
(Method C), `discrimination.py` + `pair_score_hist.py` (AUC/P@K),
`family_bootstrap.py` (grouped CIs), `pair_stratification.py`, `encoder_overlap.py`
(fusion motivation), `threshold_floor.py`, `topk_cross_sweep.py`, `mask_coverage.py` +
`mask_sweep.sh`, `probe_rerank.py`, `themes_vs_families.py`, `occurrence_durations.py`,
`low_end_matches.py`, `inspect_motif.py`, `report.py`. Libs (imported, no CLI):
`db_source/dedup/iou/labels_io/matching/metrics/taxonomy.py`.
FIG: `plot_{eval_figs,pr_curves,score_dist,score_distribution,durations,nms_sweep,xmodel_nms15}.py`;
wandb loggers `wandb_log{,_nms_sweep,_xmodel_nms15}.py`, `wandb_report{,_synthetic}.py`
(pin `wandb==0.19.11`). STALE: `wandb_log_nms15.py` (superseded by xmodel version).

### scripts/diagnostics/ — synthetic probes & investigations
EVAL campaigns: `floor_clearance.py` (invariance gate), `clearance_survival.py`,
`make_{augmented,mixture}_audio.py`, `build_e2e_corpus.py` + `e2e_nomination.py` (C1),
`c4_separation.py` (C4 SMR), `dtw_warp_recovery.py` (C3), `downsample_sweep.py`,
`eval_whitening_{synthetic,windows}.py` + `whiten_eval_common.py`, `qbe_roundtrip.py`,
`blind_match_test.py`, `compare_summaries.py`.
RUN drivers: `run_all_invariance.sh`, `run_invariance_gate.sh`, `run_{all_,}breaking_points.sh`,
`run_all_downsample.sh`, `run_c1_e2e.sh`, `run_c4_superposition.sh`, `run_whiten_{ab,evals}.sh`.
FIG: `plot_breaking_points.py`, `plot_c4_smr.py`, `summarize_whiten_campaign.py`,
`wandb_log_{campaign,c1,c4,breaking_points,cross_encoder,downsample,dtw_warp}.py`,
`diagnose_anisotropy_v2.py` (v1 STALE), `loop_{detector,audio,map}.py`, `corpus_redundancy.py`,
`inspect_matches.py`, `tail_match_report.py`, `warp_query.py`, `mask_diagnostics.py`,
`visualize_masks.py`, `stereo_correlation.py`, `compute_dtw_distributions.py`, `qbe.py`,
`clamp3_text_query.py`.
**⚠ The SYMBOLIC discovery pipeline is misfiled here** — it's a second, parallel
CORE pipeline: `symbolic_motif_discovery.py` → `cluster_motif_pairs.py` →
`annotate_clusters_with_bars.py` → `visualize_motifs.py` (colored .mxl), shapes in
`utils/motif_schemas.py`.

### scripts/{sweeps,dev,utils}/ + root
`sweeps/sweep_{dtw,leiden}.py` (param sweeps, EVAL); `dev/serve_viewer.py` +
`dev/dev_serve.sh` (viewer serving, INFRA), `dev/add_track.sh` (RUN).
`utils/` = shared library (TrackEmbedding/masks/whitening, numba DTW, pipeline.db I/O,
QbE core, favourites/annotations stores, vendored CLaMP3 bridge + xml2abc).
Root: `pc_offload.sh`+`offload.bat` (C:→D: junction offload, INFRA — note: junctions since
dissolved, D: copy merged back 2026-07-05), `run_eval_*.sh`/`run_nms15_mert_omar.sh`/
`run_gate.bat`/`breakpoints.bat`/`viewer_rebuild_job.sh` (RUN),
`pc_viewer_fix.py` (one-time INFRA), `_xcheck.py`+`mert75.{sh,bat}` (STALE).

### marble_benchmark/scripts/ (benchmark sub-tree inside leitmotifs)
`vgmiditvar_leitmotif_sweep.py` (all encoders × all layers, no selection bias),
`vgmiditvar_leitmotif_breakdown.py` (per-instrument-pair MAP), `inspect_leitmotif_labels.py` — EVAL.

---

## 3. smc-msc-thesis (`~/Developer/UPF/SMC/Thesis/smc-msc-thesis`)

| script | purpose | class |
|---|---|---|
| `compile.sh` | Build the thesis PDF via latexmk (output to `build/`; `.latexmkrc` handles Overleaf) | INFRA |
| `figures/src/vgm_*.py` ×3 (+ data JSONs) | **SUPERSEDED** by marble `vgm_timbre_report.py thesis` mode (see `figures/src/README.md`) — kept for history | STALE |
| `defense/clips/_export_fav_clips.py` | Export favourite audio clips for the defense demo | INFRA |

Thesis figures are regenerated from marble:
`uv run python scripts/analysis/vgm_timbre_report.py thesis --results <dir>:<Name> ... --thesis-dir <thesis>`
(first spec = lead encoder). Deck PDF: export `defense/defense-master.pptx` from
PowerPoint (AppleScript: `save active presentation in (POSIX file ...) as save as PDF`).

---

## 4. WSL home `/home/sid/*.sh` on the PC (~83 scratch runners, no repo)

~90% are marble sweep/extraction launchers and log-poll waiters written per-campaign;
one is leitmotifs (`serve_viewer_defense.sh` — serves the viewer for the defense demo, KEEP).
Groups: BPS window-sweep launchers+monitors; CLaMP3/OMAR-RQ/MusicFM/MuQ/VGMLoop/HookTheory
sweep launchers; the **fp16-cache-migration audit chain** (`ab_prepare_run.sh`,
`run_ab_compare.sh`, `verify_fp16.sh`, `verify_muq.sh`, `muq_tf32_audit.sh`,
`convert_retrieval.sh` — one-time but documents the load-bearing fp16/TF32 verification);
VGM tooling builds (`build_libvgm.sh` etc.); profiling probes; and dead iteration chains
(`observe{,2,3}.sh`, `verify{1,2}.sh`, `throughput_*` ×3, `wait_omar*` ×3 — STALE).
These are disposable by design; anything worth keeping should be promoted into
`marble/scripts/` and committed.

---

## 5. Known supersessions & cleanup candidates

| superseded | by |
|---|---|
| thesis `figures/src/vgm_*.py` | marble `vgm_timbre_report.py thesis` |
| marble `compare_encoders_vgmiditvar_timbre.py` | `vgm_timbre_report.py compare` |
| leitmotifs `diagnose_anisotropy.py` | `diagnose_anisotropy_v2.py` |
| leitmotifs `wandb_log_nms15.py` | `wandb_log_xmodel_nms15.py` |
| leitmotifs `repack_embeddings.py`/`repack_modal.py`, `migrate_results_to_db.py`, `pc_viewer_fix.py`, `_xcheck.py`, `mert75.*`, `overnight_*.sh` | one-time, done |
| WSL-home dead chains (observe/verify/throughput/wait_omar) | done |

## 6. Figure provenance (thesis/defense)

- **VGMIDITVar-timbre figures** (chapters + slides): marble `vgm_timbre_report.py`
  (`thesis` mode) → `thesis/figures/chapters/vgm_*.pdf` + `thesis/defense/vgm_*.png`.
  Per-encoder marble versions in `marble/docs/figures/vgmiditvar_timbre_*_varctl/`.
- **BotW discovery eval figures**: leitmotifs `scripts/eval/plot_*.py` +
  `scripts/diagnostics/plot_*.py` → `leitmotifs/docs/assets/` and
  `thesis/figures/chapters/` (breaking_points_summary, eval_panel_nms3,
  nms_sweep, null_sweep, pr_curves, score_distribution, occ_duration_coverage).
- **W&B assets**: catalogued in `leitmotifs/docs/thesis/wandb_asset_inventory.md`.

### Per-instrument MAP grids (`vgm_instrument_grids_<enc>`) — data source & two paths
The 8×8 grid figures have **two** data paths; know which you're using:
- **Committed portable slices (source of truth)**: `docs/figures/vgmiditvar_timbre_<enc>_varctl/best_layer_condition_grid.csv`
  (work-level) + `best_layer_condition_grid_varctl.csv` (variation-controlled / twin-masked).
  Long format `query_program,target_program,map,n_queries`, one 8×8 per file, best-layer slice.
  These are self-contained and **guaranteed to match the thesis**. Best layers: CLaMP3 L4,
  MERT-v1-95M L11, MuQ L8, OMAR-RQ L15. **Regenerate from these** via `regen_instrument_grids.py`.
- **Any layer (incl. non-best) → WANDB uploaded files**: the 2026-07-03 from-cache sweep logged one
  run per layer (groups `MuQ / VGMIDITVar-timbre`, `MERT-v1-95M / VGMIDITVar-timbre`), each with
  `condition_grid.csv` + `condition_grid_varctl.csv` **uploaded as run files** — pull via
  `wandb.Api().run(id).file("condition_grid_varctl.csv").download()`. NOT in the run summary (only
  work grid cells + varctl *aggregate* scalars are) and NOT on PC disk (out-dir cleaned in a reorg).
  Verified byte-identical to the committed best-layer slice. Deployed-layer grids
  `vgm_instrument_grids_muq_l11` / `_mert_l7` were built this way (run IDs `2f280cr0`, `2cs4z8va`).
- **Raw sweep tree (what `vgm_timbre_report.py thesis` reads)**: `<results>/layer{BEST}/condition_grid.csv`
  (+`condition_grid_varctl.csv`, no `best_layer_` prefix). Point `--results-dir` at the encoder's
  sweep-output tree, not `docs/figures/`. Prefer the committed slices or wandb files over this.
- **The white-seam bug (fixed 2026-07-06)**: the old grids drew `imshow` unrasterized, so PDF viewers
  anti-aliased the page-white (`#fcfcfb`) between cells → thin white gridlines across every row/column.
  Fix (in both `vgm_timbre_report.py` fig5/thesis grids AND `regen_instrument_grids.py`):
  `imshow(interpolation='nearest')` + `im.set_rasterized(True)` + explicit `ax.grid(False)`.
  Always regenerate **all** encoders from the same script version (a prior agent restyled divergently).
