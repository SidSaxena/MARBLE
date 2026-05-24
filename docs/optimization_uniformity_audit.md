# Optimization uniformity audit (2026-05-24) — SNAPSHOT, superseded

> **NOTE**: This doc is a point-in-time audit from the middle of the
> optimisation rollout. The current canonical reference (with all
> post-audit changes — OMARRQ compile bug fix, OMARRQ + MuQ rollout to
> all configs, NSynth Modal infra, etc.) is
> **[docs/performance_optimizations.md](performance_optimizations.md)**.
>
> Kept for traceability.

---


Captures the state of all optimisations applied across MARBLE
encoders × tasks during the recent HookTheoryMelody + NSynth rollout, and
identifies real vs apparent gaps. Source of truth for what's deployed
where; check this before assuming a config has a given fix.

## Task × optimisation matrix

`✓` = applied, `✗` = absent, `N/A` = doesn't apply to this task.

| Task | prefetch_factor=4 | num_workers | WAV jsonls | audio_ext param | precompute_labels | Modal warmup | Sweep entrypoints |
|---|---|---|---|---|---|---|---|
| HookTheoryMelody | ✓ | 8 | ✓ | ✓ | ✓ | ✓ | 5 encoders |
| HookTheoryKey | ✓ | 8 | ✓ | N/A (path from JSONL) | N/A (trivial labels) | partially | 0 |
| HookTheoryStructure | ✓ | 8 | ✓ | N/A | N/A | partially | 0 |
| NSynth | ✓ | 8 | N/A (already WAV) | N/A | N/A | ✓ | 4 encoders |
| GTZANGenre | ✓ | 8 | N/A (WAV) | N/A | N/A | ✓ | 2 encoders |
| GTZANBeatTracking | ✓ | 8 | N/A (WAV) | N/A | could benefit | ✗ | 0 |
| Chords1217 | ✓ | 16 | N/A (FLAC) | N/A | could benefit | ✗ | 1 |
| MTT | ✓ | 32 | N/A | N/A | N/A | ✗ | 0 |
| MTG{Genre,Inst,Mood,Top50} | ✓ | 32 | N/A | N/A | N/A | ✗ | 0 |
| EMO | ✓ | 8 | N/A (WAV) | N/A | N/A | ✓ | 2 |
| GS | ✓ | 8 | N/A (WAV) | N/A | N/A | ✓ | 2 |
| Covers80 | ✓ | 4 | N/A | N/A | N/A | ✗ | 0 |
| SHS100K | ✓ | 4 | N/A | N/A | N/A | ✗ | 0 |
| BPSMotif, VGMIDITVar, SuperMario, HXMSA | ✓ | 4-8 | N/A | N/A | N/A | ✗ | 0 |

## Encoder × `compile_mode` matrix

| Encoder | `compile_mode` init param | Configs that set it |
|---|---|---|
| MERT-v1-95M | ✓ supported | 2 (HTM layers + meanall) |
| MERT-v1-330M | ✓ supported (inherits) | 0 |
| MuQ | ✗ not implemented | — |
| MusicFM | ✗ not implemented | — |
| OMARRQ-multifeature-25hz | ✗ not implemented | — |
| CLaMP3 | ✗ not implemented | — |
| Qwen2AudioInstructEncoder | ✗ not implemented | — |

## Real gaps vs apparent gaps

### Real (could close cheaply)

1. **`compile_mode` deployment coverage for MERT.** Only 2 HookTheoryMelody configs set it; the other 66 MERT configs (NSynth, Key, Structure, GS, EMO, Chords1217, …) inherit the same encoder class but don't opt in. This was deliberately added as "an A/B test point" for HTM. Now that HTM has run with it, can be extended to other tasks selectively as wins are validated. **Recommended fix**: add to MERT configs for the actively-run tasks (HTM done, NSynth + Key + Structure next).

2. **`num_workers=16` outlier in `configs/probe.*.Chords1217.yaml`.** Single task that's larger than 8 without a clear comment justifying it. Either bring down to 8 for consistency OR add a comment explaining why 16 is appropriate (and matching MTG/MTT at 32 if that turns out to be the same rationale).

### Real (substantial work — defer)

3. **`torch.compile` support absent from MuQ, MusicFM, OMARRQ, CLaMP3, Qwen2Audio.** Adding it requires per-encoder changes (init param + conditional `.compile()` call + capability gate). Each encoder has different internals, so this is encoder-level work, not a uniformity sweep. **Recommendation**: file as 5 small follow-ups, one per encoder. Lowest-hanging is MuQ (simplest forward path); CLaMP3 is hardest (symbolic + audio dual-mode).

### Apparent — not real gaps

4. **HookTheoryKey/Structure datamodules lack `audio_ext` + `precompute_labels`.** The earlier audit flagged this but it's deliberate:
   - `audio_ext` is N/A — the datamodules read `audio_path` directly from JSONL records, not derived from id+ext. The WAV switch was done via JSONL rewrites instead (cleaner; no datamodule change).
   - `precompute_labels` is N/A — labels are a single dict lookup (`LABEL2IDX[info["label"]]`), <1 µs/call. No precompute benefit.

5. **`num_workers=32` on MTG/MTT, =4 on Covers80/SHS100K/BPSMotif.** These look like outliers but are likely task-tuned:
   - 32 on MTG/MTT makes sense: ~467 k file corpora, each .mp3/.flac decode is expensive, dataloader can use many cores.
   - 4 on Covers80/SHS100K (zero-shot retrieval) and symbolic tasks (BPSMotif, VGMIDITVar): small corpora or fast MIDI parsing, more workers buys nothing.
   - Leave alone unless a per-task profile shows otherwise.

6. **Modal sweep convenience entrypoints sparse (only 9 tasks have them).** Adding `sweep_<encoder>_<task>` for the other 11 tasks would be boilerplate without clear demand. The generic `run_sweep` works for any config; convenience entrypoints just save typing. Opt-in expansion as tasks become actively-swept.

## What got applied in this rollout (commit-by-commit)

| Commit | Scope | Notes |
|---|---|---|
| `5aea2af` (perf) | HTM datamodule + tests + dev pytest | interp1d cache, vectorised labels, precompute_labels, audio_ext, prefetch_factor=4 |
| `d8ff5f1` (feat) | MP3→WAV pipeline + smoke validator + Modal infra | convert_hooktheory_to_wav, warmup helpers, smoke_test_wav |
| `095d4b0` (feat) | 10 HTM configs → WAV | jsonl→.wav.jsonl, audio_dir→audio_wav, audio_ext: .wav |
| `b03cadd` (chore) | num_workers split | configs=8, Modal sweeps override to 16 via cli_overrides |
| `40d3f0f` (feat) | Key/Structure WAV infra | rewrite_jsonl_audio_paths.py, convert_hooktheory_clips_to_wav |
| `3ee8657` (feat) | 24 Key/Structure configs → WAV | + 3 num_workers normalisations |
| `acf2e9c` (docs) | NSynth optimisation plan | plan-only |
| (this commit) | NSynth Modal infra | setup_nsynth, warmup_nsynth_audio, 4 sweep_*_nsynth |

## Recommendations

- **Apply now**: add `compile_mode: default` to MERT configs for the 3 actively-running tasks (NSynth, HookTheoryKey, HookTheoryStructure). Low risk, mirrors HTM. Run a smoke first to confirm the val metric is unchanged.
- **Defer**: torch.compile on other encoders. Track as 5 separate follow-ups.
- **Skip**: num_workers "normalisation". Current values are task-tuned.
- **Skip**: Modal sweep entrypoint expansion. Add only when those tasks are actively swept.
