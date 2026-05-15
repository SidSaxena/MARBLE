# Embedding cache — correctness, caveats, and what to watch out for

This is the companion doc to `docs/embedding_cache.md`. The reference doc
explains *how* the cache works and *how* to use it. This doc explains
**what the cache changes about training semantics**, where those changes
are safe, and where they would silently produce wrong results.

Read this once before adding the cache to a new config, especially if
that config uses a non-default `emb_transforms` pipeline or random data
augmentation. The cache trades a real architectural restriction for ~10×
end-to-end speedup; you need to know what you traded.

---

## 1. The fundamental restriction

The cache stores `(L, H)` per clip, where `L` = number of encoder
layers and `H` = encoder hidden dim. **The time dimension has been
collapsed by `mean` before storage.**

Concretely, the cache write path is:

```
encoder(audio)         →   (L, T, H)        # raw encoder output
encoder_tuple_to_pooled  →  (L, H)          # mean over T
cache.put(clip_id, ...)
```

The cache read path round-trips through a singleton-time dim:

```
cache.get(clip_id)        →   (B, L, H)
stacked_to_layer_tuple    →   tuple of L × (B, 1, H)
                              # T = 1, value = the stored mean
```

That `(B, 1, H)` tuple is what the `emb_transforms` see.

**Consequence:** any transform that expects to operate on the time
dimension is now operating on a length-1 sequence whose single value is
already a mean. This is mathematically harmless for some transforms and
silently wrong for others. See section 2.

---

## 2. Pooling compatibility table

`marble/modules/poolings.py` defines several time-pooling strategies.
Here's how each one interacts with the cache:

| Pooling class | Time op | Cache-safe? | Why |
|---|---|---|---|
| `TimeAvgPool` / `GlobalAvgPool1D` | `mean(x, dim=time)` | ✅ Identical | Mean is its own collapse: `mean([mean])` = mean. The cache stores the mean; mean-pooling it again is a no-op. |
| `TimeAdaptivePool` (mean mode) | adaptive avg over T → fixed K | ✅ Equivalent if K=1 | Otherwise depends on K vs cached T (we store T=1). |
| `GlobalMaxPool1D` | `max(x, dim=time)` | ❌ Wrong | Max over a length-1 sequence returns that single value (= the mean). You wanted the temporal max; you get the temporal mean. |
| `MaxAvgPool1D` | `α·max + (1-α)·mean` | ❌ Wrong | The max half degenerates to the mean (see above), so the output is `α·mean + (1-α)·mean = mean`. The whole pooling collapses. |
| `AttentionPooling1D` | `softmax(W x) · x` over T | ❌ Wrong | Softmax over a length-1 sequence is always `[1.0]`. The pooled output is exactly the cached mean — the learned attention scoring is bypassed entirely. The `nn.Linear(H, 1)` weights would train but contribute nothing to the loss. |
| `GatedAttentionPooling1D` | gated attention over T | ❌ Wrong | Same failure mode as above; gating + softmax over T=1 is identity. |
| `AutoPool1D` | `logsumexp(α·x) / α` over T | ❌ Wrong | `logsumexp` over length-1 is `α·x` (single element), so the output reduces to `x` (the cached mean) regardless of `α`. |

**Bottom line:** the cache is **only correct for mean time-pooling**.
Every MARBLE config in this repo currently uses `TimeAvgPool`, so we're
safe. But if you ever switch a probe to learned/attention/max pooling,
you must set `cache_embeddings: false` for that config, or extend the
cache to store `(L, T, H)` (T-preserving). The latter is straightforward
but multiplies disk usage by T (~125 for 5s × 25Hz, ~750 for 30s × 25Hz).

---

## 3. Random augmentation interaction

Dataset `__getitem__` can be non-deterministic. Two sources to track:

### `channel_mode: random`

Used in train splits for HookTheoryKey, HookTheoryStructure, Chords1217,
and a few others. Each `__getitem__` picks a random channel (or mixes).
Under no cache:

```
epoch 0: random_channel_choice = ch_2  → encoder → loss
epoch 1: random_channel_choice = ch_0  → encoder → loss      # fresh randomness
epoch 2: random_channel_choice = ch_1  → encoder → loss
```

Under cache + audio-I/O bypass:

```
epoch 0: random_channel_choice = ch_2  → encoder → cache_put → loss
epoch 1: cache_check_fn(clip_id) → hit → audio NEVER LOADED → cache_get → loss
epoch 2: same as epoch 1
```

The first epoch's random channel choice is frozen for the rest of
training. Effective conversion: `channel_mode: random` →
`channel_mode: random-fixed-at-epoch-0`.

**Impact:** small. Channel selection is a weak augmentation —
typically worth <1% on validation. For linear probes (decoder is
~1k–10k params) the regularization barely matters; there's not much
capacity to overfit. For deeper probes or small datasets, it might
matter more.

**Detection:** if your train loss drops faster with cache enabled and
your val score stays roughly the same, you're fitting a (very slightly)
smaller dataset. That's the augmentation freezing.

### Random clip sampling (not currently used)

Some MARBLE-style datasets randomly choose the *window* within a longer
audio file per epoch. None of our current datamodules do this — they
all build a deterministic `index_map` of (file_idx, slice_idx). If you
add a random-window dataset, the cache will freeze the first epoch's
window per clip_id, which is a much larger behavior change. Plan
accordingly (disable cache, or extend `make_clip_id` to include window
offset).

### Encoder-internal randomness

MuQ and MusicFM include SpecAugment inside the encoder for
pre-training. For probe runs the encoder is in `eval()` mode, which
disables SpecAugment (and dropout). The cache is safe with respect to
encoder-internal stochasticity, but only because eval mode kills it.
If you ever run a probe with `encoder.train()`, the cache will freeze
the first epoch's random SpecAugment masks.

---

## 4. What the cache does NOT change

To balance the scary list above — here's what's untouched:

- **Decoder training.** Same shuffling, same optimizer state, same
  random init, same loss.
- **Val / test reproducibility.** Val/test typically use deterministic
  `channel_mode` and deterministic clip slicing. Val/test metrics under
  cache should match no-cache to floating-point rounding.
- **Layer selection.** All L layers are stored; the LayerSelector picks
  from the cache exactly as it would from a fresh encoder.
- **Learning-rate schedulers, early stopping, checkpointing.** Cache is
  upstream of all of this.
- **Loss functions, metrics.** Computed on decoder output, downstream
  of cache.

---

## 5. Versioning & invalidation

Cache directories are keyed by an 8-char hash of:

```
encoder_id || clip_seconds || sample_rate || time_pool_mode || layer_count
```

Change any of those and you get a fresh directory. Hash collision risk
is negligible (~10⁻⁹).

**What is NOT in the hash** (and therefore re-uses the cache):

- Decoder architecture, hidden size, dropout
- Loss function, learning rate, batch size, optimizer
- Random seed (this is intentional — same encoder output regardless of
  seed)
- `channel_mode`, `min_clip_ratio`, dataset path

The last one is worth flagging: if you change the training JSONL but
keep the encoder/clip settings the same, you'll reuse the cache for
overlapping `clip_id`s. That's usually what you want (no need to
recompute the same audio under the same encoder), but if you've
changed the *contents* of a JSONL entry (e.g. different `audio_path`
for the same `ori_uid`), be aware that `make_clip_id(audio_path,
slice_idx)` hashes the path, so the cache will correctly diverge.

---

## 6. Pickle / state mismatch (resolved 2025-11)

Caught by an actual production run on Windows:

```
AttributeError: 'HookTheoryStructureAudioTest' object has no
attribute 'cache_check_fn'
```

**Root cause:** Windows DataLoader workers use spawn mode, which
pickles the parent's dataset `__dict__` and unpickles it in each worker.
Unpickling does **not** re-run `__init__`. If the parent created the
dataset under code that didn't have `self.cache_check_fn = None`
(pre-fix), but a worker imports post-fix code (e.g. after a mid-run
`git pull`), the worker reads `self.cache_check_fn` → AttributeError.

**Fix shipped (commit `23f8e36`):** every audio-I/O bypass check uses
`getattr(self, "cache_check_fn", None)` instead of bare attribute
access. A missing attribute now degrades to "bypass doesn't fire"
rather than crashing the worker.

**Operational rule:** if you `git pull` during a sweep, kill and
restart the run. Don't trust that an in-flight worker is on the latest
code.

---

## 7. Verifying cache equality (run this once per pooling shape)

The fastest, cheapest, most decisive test:

```bash
# 1. Make sure no EMO cache exists.
uv run python scripts/embeddings/manage.py clear MuQ/EMO__... 2>/dev/null
# (or just: rm -rf data/embeddings_cache/MuQ/EMO__*)

# 2. Run twice with the same seed — once cache OFF, once cache ON.
uv run python cli.py fit \
    -c configs/probe.MuQ-meanall.EMO.yaml \
    --model.init_args.cache_embeddings=false \
    --trainer.max_epochs=10 \
    --seed_everything=1234

uv run python cli.py fit \
    -c configs/probe.MuQ-meanall.EMO.yaml \
    --model.init_args.cache_embeddings=true \
    --trainer.max_epochs=10 \
    --seed_everything=1234

# 3. Compare final val/r2 in each WandB run. Should match to within 1e-3.
```

Pick a task with deterministic train augmentation (EMO uses
`channel_mode: first` — perfect). If you run this on a task with
`channel_mode: random`, the two runs will diverge slightly even
*without* the cache, because each no-cache epoch resamples channels.

Re-run this whenever you touch:

- `marble/utils/emb_cache.py` (round-trip math)
- `marble/core/base_task.py:forward` (cache routing)
- `encoder_tuple_to_pooled` or `stacked_to_layer_tuple`

---

## 8. The static audit script (catches plumbing bugs)

`scripts/embeddings/audit_cache_integration.py` walks every
cache-enabled config, resolves the task + dataset classes, and verifies:

- Task `__init__` accepts `cache_embeddings`
- Task inherits `EmbeddingCacheMixin`
- Task `forward` (if overridden) calls `_cached_forward_layer_tuple`
- Task step methods (`*_step`, `_shared_step`) propagate `clip_ids` to
  `forward`
- Dataset `__init__` initializes `cache_check_fn`
- Dataset `__getitem__` calls `make_clip_id` and emits the 4-tuple
- Dataset `__getitem__` has the audio-I/O bypass branch

Run before opening a PR that touches any cache code:

```bash
uv run python scripts/embeddings/audit_cache_integration.py
```

The audit caught the MTGMood `validation_step` bug (was dropping
`clip_ids` from the 4-tuple unpack) that would have silently disabled
the cache for 5 MTG* tasks.

---

## 9. When NOT to enable the cache

| Scenario | Reason |
|---|---|
| `freeze_encoder: false` (encoder is fine-tuned) | Encoder weights change every step; cache becomes stale immediately. |
| Non-mean time pooling (any AttnPool / MaxPool / AutoPool variant) | See section 2 — silently wrong. |
| Random window sampling per epoch | Cache freezes the first window per clip_id. |
| Train-time encoder dropout (encoder in `.train()` mode) | Cache freezes the first epoch's dropout mask. |
| Tiny dataset where one full encoder forward fits in DataLoader cache anyway | Cache doesn't help, adds disk footprint. |
| Debugging a new encoder | Disable while iterating on encoder code; re-enable when stable. |

---

## 10. Future extensions (not implemented)

- **Time-preserving cache.** Store `(L, T, H)` instead of `(L, H)`.
  Unblocks attention pooling but multiplies disk usage by T. Reasonable
  for short clips (5s × 25Hz = 125 frames).
- **GPU cache.** Pre-load the entire cache into VRAM at training start.
  Removes disk I/O from the warm-epoch path. Only feasible for small
  datasets.
- **Augmentation-aware cache.** Include augmentation parameters in
  `clip_id` so each augmented variant is its own cache entry. Disk cost
  scales with number of augmentation variants per epoch.
- **Distributed cache invalidation.** Right now if two processes write
  the same clip_id simultaneously, one wins via atomic rename. For
  multi-rank training this is fine. For more exotic setups, add a
  proper lock.

---

## TL;DR for future-you

1. **Cache is correct iff:** frozen encoder + mean time pooling +
   deterministic train augmentation.
2. **Your current sweeps satisfy all three.** Headline numbers are
   honest.
3. **Don't enable cache on a new config without checking** that the
   `emb_transforms` only do mean pooling and the train `channel_mode`
   is `first` or `mix`. Or if it's `random`, accept the mild
   regularization loss.
4. **Run the audit script** before merging cache-touching PRs.
5. **Run the equality test** (section 7) when you change any cache
   round-trip code.
