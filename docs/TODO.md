# MARBLE — open follow-ups

Living list of things we've decided to defer rather than not do. Add an
entry per item with (a) the motivation, (b) the rough design, (c) the
cost estimate, and (d) what would trigger us to actually do it.

---

## 1. Per-layer embedding cache for zero-shot + clip-level sweeps

**Motivation.** Every per-layer subprocess in `run_sweep_local.py`
re-runs the encoder forward pass over the full dataset. For OMARRQ × a
24-layer sweep, that's the same audio embedded 24 times. The encoder
already returns all-layers in a single forward (`extract_embeddings(x,
layers=set(range(24)))` in `marble/encoders/OMAR_RQ/model.py`), so this
is pure waste — ~10–20× speedup is on the table for zero-shot and
clip-level supervised tasks.

**Design.** Cache the **post-`TimeAvgPool`** embedding (shape `(L, H)`
per track) rather than the pre-pool `(L, T, H)` tensor. That keeps the
disk footprint trivially small:

| Task | Tracks | (L, H) fp32 size |
|---|---|---|
| Covers80 | 160 | ~15 MB |
| VGMIDITVar | 12,870 | ~1.2 GB |
| SHS100K (~5K tracks) | 5,000 | ~470 MB |

Cache key = `hash(encoder_id + sample_rate + clip_seconds + audio_path
+ mtime)` so changing the encoder or audio preprocessing invalidates
correctly. Cache location: `output/.emb_cache/<encoder>/<task>/...`.

The probe task pipeline (`CoverRetrievalTask` for retrieval, the
clip-level supervised base for GS/HookTheoryKey/Structure) needs a
branch: if cache hit, load `(L, H)`; otherwise run the encoder forward
and write the cache. The cache is populated once (during the meanall
baseline that runs first now) and then reused for every per-layer
job inside the same sweep.

**Frame-level tasks (BeatTracking, Chords1217, HookTheoryMelody)** need
`(T, H)` per layer per track, which blows up disk by ~ T-frames-per-
clip. Defer those — accept the cost until we actually run those
sweeps.

**Cost.** ~3–4 hours of refactor. Touches:
- `marble/tasks/<Task>/probe.py` (the task-specific Lightning module)
- Probably a small `marble/utils/emb_cache.py` helper
- Smoke-test on Covers80 (small + fast to verify)

**Trigger.** Once the Covers80 + SHS100K sweeps complete on the
current pipeline and we're confident the new taxonomy is settled. Then
this becomes the main speedup for any future encoder-variant or
encoder-comparison work.

---
