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

---

## Open

### 1. Cache integration for clip-level supervised tasks (11 tasks)

The cache utility + retrieval integration already shipped. Extending to
clip-level supervised tasks (`GS`, `EMO`, `GTZANGenre`, `NSynth`,
`HookTheoryKey`, `HookTheoryStructure`, `MTGGenre/Instrument/Mood/Top50/MTT`)
is mechanical:

- Each task's datamodule needs to return `clip_id` (same `make_clip_id(path, slice_idx)`
  pattern as retrieval).
- Each task's `forward()` needs to accept an optional `clip_ids` kwarg
  and route through the cache miss/hit branches — same shape as the
  retrieval task's integration. The `BaseTask._shared_step` may want a
  small adapter so `training_step` / `validation_step` / `test_step`
  all pass `clip_ids` through.

**Cost.** ~2–3 hours; mostly find-and-replace across 11 datamodule
files plus 11 probe.py files. Most of those probes follow a shared
pattern (encoder → emb_transforms → MLPDecoder head) so a single
`_forward_with_cache` helper on `BaseTask` could cover them all and
the per-task changes drop to 1 line each.

**Benefit.** Even bigger speedup than retrieval. Supervised tasks
train for 40–60 epochs; without cache, each epoch re-runs the
encoder over all clips. With cache, only epoch 1 pays the encoder
cost (and only on uncached clips); epochs 2–40 are pure head
training. Estimated 10–20× speedup on a 13-layer × 60-epoch GS sweep.

**Trigger.** When you start the supervised sweeps (after the SHS100K
+ Covers80 retrieval sweeps land).

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
