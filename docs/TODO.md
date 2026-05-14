# MARBLE — open follow-ups

Living list of things we've decided to defer rather than not do. Add an
entry per item with (a) the motivation, (b) the rough design, (c) the
cost estimate, and (d) what would trigger us to actually do it.

---

## Done

### ✅ Per-layer embedding cache (retrieval tasks)

**Shipped 2026-05-14** in commit `58871bc` + `bdaef8b`. See full
documentation at [`docs/embedding_cache.md`](embedding_cache.md).

- Cache utility (`marble/utils/emb_cache.py`) with atomic writes,
  deterministic key derivation, batch helpers.
- Cache integration in `CoverRetrievalTask` (used by Covers80, SHS100K,
  VGMIDITVar via re-export). `cache_embeddings: true` opt-in on all 44
  retrieval configs.
- Datamodule batches return `clip_id` as a 4th tuple element.
- Pre-warm + inspection CLIs: `scripts/embeddings/extract.py`,
  `scripts/embeddings/manage.py`.

Validation: disk math exact (96 KB/clip predicted, 100 KB observed on
OMARRQ-25hz with `torch.save` header); meanall + per-layer queries served
from same cache by the commutativity property.

### ✅ Cache extension to clip-level supervised tasks (11 tasks)

**Shipped 2026-05-14** as the follow-up to the retrieval cache.

- Cache plumbing factored into `EmbeddingCacheMixin`
  (`marble/utils/emb_cache.py`). Both `BaseTask` and `CoverRetrievalTask`
  inherit; supervised tasks (`GS`, `EMO`, `GTZANGenre`, `NSynth`,
  `HookTheoryKey`, `HookTheoryStructure`, `MTGGenre/Instrument/Mood/Top50/MTT`)
  get caching transparently through `BaseTask`.
- `BaseTask.forward(x, clip_ids=...)` routes through the mixin's
  hit/miss paths; `_shared_step`/`test_step` unpack 4-tuple batches.
- All 11 supervised datamodules now emit `clip_id` as the 4th tuple
  element (using `make_clip_id`).
- ~97 supervised configs updated with `cache_embeddings: true`.

### ✅ Audio-I/O bypass on cache hits

**Shipped 2026-05-14** with the supervised extension. Pushes warm-cache
wall-clock from ~10 min per layer (audio decode dominated) to
estimated <1 min per layer.

- `BaseAudioDataset.cache_check_fn` (optional `Callable[[str], bool]`)
  injected by the task at `setup()` time via
  `EmbeddingCacheMixin._inject_cache_check_into_datasets`.
- On hit, dataset returns a zero-placeholder waveform and skips
  `torchaudio.load + resample + pad` entirely. The task's `forward()`
  ignores `x` on cache hits and uses the cached tensor.
- Same pattern added to the 7 custom-dataset classes (Covers80,
  SHS100K, VGMIDITVar, GS, EMO, GTZANGenre, NSynth).

---

## Open

### 1. Leitmotifs matrix-profile result cache (separate repo)

In `/Users/sid/leitmotifs/`. The matrix-profile cosine-similarity step
takes 4–5 hours of GPU compute over 8M pair-wise comparisons (5-second
windows × 259 tracks). DTW after that is another 2 hours. Per-window
embeddings are already cached in `embeddings/<model>/L<N>/*.pt`, so
MARBLE-style embedding caching won't help. What WILL help is caching
the **matrix-profile results** themselves:

- Per-pair `(pair_id, peak_score, peak_pos)` → ~100 MB for 8M pairs
- Per-pair DTW scalar → ~32 MB

Cache key: `sha256(encoder_id + layer + window_seconds + step_seconds + sample_rate)[:8]`.
Re-iterations on downstream filter thresholds or clustering
parameters reuse the cached matrix-profile output instead of paying
the 4–5h GPU cost again.

**Cost.** ~50 LOC wrapping the existing matrix-profile script.
**Trigger.** Next time you iterate on the leitmotifs pipeline.

### 2. Frame-level task caching (deferred indefinitely)

`HookTheoryMelody`, `GTZANBeatTracking`, `Chords1217` need the full
`(T, H)` time-axis tensor for per-frame predictions. A cache that
preserves time would be ~73 MB/clip for OMARRQ-25hz on 30-sec clips,
or 500 GB+ for SHS100K. Tabled until disk budget or a more compact
encoding makes it worth revisiting.

### 3. Tensor compression (fp16 / int8 quantization)

The current cache is fp32. fp16 would halve disk usage (96 KB → 48 KB
per clip) with negligible precision impact on cosine similarity. int8
would quarter it but needs careful per-clip scale storage. Not urgent
at current disk usage — defer until aggregate cache passes ~10 GB.

### 4. Encoder fine-tuning

The cache assumes a frozen encoder; if `train_mode != "freeze"`, the
cache key would need to include encoder weight hashes and invalidate
every epoch. Not in scope for the current probe-only experiments.
