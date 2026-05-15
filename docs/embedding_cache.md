# MARBLE — per-clip embedding cache

A disk-backed cache for the **per-clip, all-layers, post-time-pool**
embedding tensor that frozen encoders produce. Lets every per-layer (and
meanall) job in a layer sweep skip the encoder forward — the slowest
part of zero-shot retrieval and a major chunk of clip-level supervised
training — and read tensors instead.

For the OMARRQ-multifeature-25hz × SHS100K sweep this cuts wall-clock
from **~13 hours to ~1 hour** (one cold meanall pass, then 24 layer
tests that each take <1 minute).

---

## How it works (the (L, H) insight)

The encoder forward returns a `(L, T, H)` tensor per clip:

| Axis | Meaning | OMARRQ-25hz × 30-sec clip |
|---|---|---|
| **L** | Number of transformer/conformer layers | 24 |
| **T** | Number of tokens / time-steps | 750 (30 sec × 25 Hz) |
| **H** | Hidden dimension per layer | 1024 |

The probe's `TimeAvgPool` collapses **T** (mean over time), giving
`(L, H)`. The probe head only ever sees this pooled vector — `LayerSelector`
just picks one slice of the L axis (or means over it for meanall). So
caching `(L, H)` per clip is sufficient to answer **every** per-layer
*and* meanall query from the same data.

What we *don't* cache is **T**. Caching `(L, T, H)` would blow up:

| Cache boundary | Per-clip size (OMARRQ-25hz, 30-sec) | SHS100K (5K tracks ≈ 6.9K clips) |
|---|---|---|
| `(L, T, H)` pre-pool — *rejected* | ~73 MB | ~500 GB |
| `(L, H)` post-pool — **what we built** | 96 KB | ~660 MB |

The factor-of-T = 750× reduction is what makes the cache feasible.

### Critical commutativity (why meanall is also free)

Mean over layers commutes with mean over time (both linear). So the
same `(L, H)` cache serves both kinds of queries:

```
per-layer-N:  result = mean_T(encoder(x)[N])           # pick layer N from L, then pool T
            = encoder_pooled[N]                          # cached as embedding[N]

meanall:      result = mean_T(mean_L(encoder(x)))      # mean over both, in any order
            = mean_L(mean_T(encoder(x)))               # by Fubini's theorem
            = mean_L(encoder_pooled)                    # ← cached form, mean over L
```

Verified end-to-end by `marble.utils.emb_cache.encoder_tuple_to_pooled`
+ `stacked_to_layer_tuple` round-trip in the smoke test.

### One file per clip vs one file per (clip, layer)?

The cache stores **one `.pt` file per clip** containing the full
`(L=24, H=1024)` tensor — not one file per (clip, layer). All layers
live in the same file, indexed by the L axis. Trade-offs we considered:

| Design | File layout | Pros | Cons |
|---|---|---|---|
| **A — what we built** | `<clip_id>.pt` shape `(L, H)` | matches encoder output; 1 read per cache hit; atomic writes trivial; ~7K files for SHS100K | reads 96 KB even when only 4 KB are needed |
| B (alternative) | `<clip_id>__layer_N.pt` shape `(H,)` | smaller per-read | 24× more files; meanall needs 24 reads per clip; race-condition surface multiplies |

Design A wins because a 96 KB sequential read is ~1–2 ms — negligible
versus the ~30+ minutes of encoder forward it replaces.

---

## Where the cache applies

### Cache-safe tasks (14)

Encoder is frozen, head consumes `(B, H)`, no random audio augmentation
at train time. `cache_embeddings: true` is enabled in their configs.

| Type | Tasks |
|---|---|
| Retrieval (zero-shot, max_epochs=0) | `Covers80`, `SHS100K`, `VGMIDITVar` |
| Clip-level supervised | `GS`, `EMO`, `GTZANGenre`, `NSynth`, `HookTheoryKey`, `HookTheoryStructure`, `MTGGenre`, `MTGInstrument`, `MTGMood`, `MTGTop50`, `MTT` |

> **Status (2026-05-14):** all 14 cache-safe tasks are shipped.
> Retrieval landed in commit `58871bc`; clip-level supervised + the
> audio-I/O bypass landed as a follow-up. Cache plumbing is now
> factored into `EmbeddingCacheMixin` (in `marble/utils/emb_cache.py`);
> both `BaseTask` and `CoverRetrievalTask` inherit it.

### Cache-unsafe tasks (3)

Frame-level heads consume `(B, T, H)` — caching across the full time
axis is impractical (hundreds of GB per task per encoder).

| Task | Why | What the cache shape would be |
|---|---|---|
| `GTZANBeatTracking` | Per-frame beat/downbeat detection | `(L, T, H)` per clip → ~18 MB/clip × ~1K clips = 18 GB per encoder |
| `Chords1217` | Per-frame chord recognition | Same scale problem |
| `HookTheoryMelody` | Per-frame MIDI pitch (128 classes) | Same |

These tasks fall back to the normal encoder-every-batch pipeline. A
frame-level cache is plausible if disk grows (~50–500 GB per encoder)
but deferred for now.

---

## Disk math per dataset

For each (encoder, task) pair the cache materializes lazily — only
clips you actually run through populate. Worst-case full-population
sizes:

| Encoder | L | H | per-clip | Covers80 (160) | SHS100K (≈6.9K clips) | VGMIDITVar (12.87K) |
|---|---|---|---|---|---|---|
| OMARRQ-multifeature-25hz | 24 | 1024 | 96 KB | 15 MB | 660 MB | 1.2 GB |
| MERT-v1-95M | 13 | 768 | 40 KB | 6 MB | 280 MB | 510 MB |
| MERT-v1-330M | 25 | 1024 | 100 KB | 16 MB | 690 MB | 1.3 GB |
| CLaMP3 | 13 | 768 | 40 KB | 6 MB | 280 MB | 510 MB |
| MuQ / MusicFM | 13 | 1024 | 52 KB | 8 MB | 360 MB | 670 MB |

Per-clip cost is constant — independent of clip duration — because
the time axis is pooled away. Validated against the smoke test:
predicted 98,304 bytes for OMARRQ; observed 100,142 (1.8 KB
`torch.save` header).

For all 7 encoders × all retrieval tasks: ~10 GB total. Add 11
clip-level supervised tasks (when integrated) and the worst case
climbs to ~50–100 GB. `manage.py clear <encoder>` reclaims disk
whenever you're done with an experiment.

---

## Usage

### Enabling the cache on a config

Add `cache_embeddings: true` under `model.init_args`. Already done
for all 14 cache-safe configs (commit `58871bc`):

```yaml
# configs/probe.OMARRQ-multifeature-25hz.SHS100K.yaml
model:
  class_path: marble.tasks.SHS100K.probe.CoverRetrievalTask
  init_args:
    sample_rate: 24000
    cache_embeddings: true        # ← this line opts in
    encoder:
      class_path: marble.encoders.OMAR_RQ.model.OMARRQ_Multifeature25hz_Encoder
      ...
```

Set `false` (or omit) to disable for one specific run while keeping
the cache for others.

### Lazy population (the natural path)

Just run the sweep. The first job to touch each clip pays the
encoder-forward cost; every subsequent job hits cache. The
meanall-first sweep ordering means the meanall populates everything
before any per-layer job runs:

```bash
uv run python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.OMARRQ-multifeature-25hz.SHS100K.yaml \
    --num-layers 24 --model-tag OMARRQ-multifeature-25hz --task-tag SHS100K \
    --concurrency 1
```

Watch for these one-time log lines confirming the cache engaged:

```
[emb_cache] MISS on first lookup (8 clips, dir=.emb_cache/OMARRQ-multifeature-25hz/SHS100K__abc12345)
...
[emb_cache] HIT on first lookup  (8 clips, dir=.emb_cache/OMARRQ-multifeature-25hz/SHS100K__abc12345)
```

### Explicit pre-warm via `extract.py`

Useful when you want to populate the cache offline (overnight, on a
different machine, or to avoid a duplicate WandB run). Mirrors the
runtime cache-key derivation exactly, so the subsequent sweep finds
the files as cache hits.

```bash
# Pre-warm the cache without WandB, without Lightning.
uv run python scripts/embeddings/extract.py \
    --config configs/probe.OMARRQ-multifeature-25hz-meanall.SHS100K.yaml

# Different split (default: test)
uv run python scripts/embeddings/extract.py \
    --config <config> --split train

# Force re-extract even if all clips are already cached
uv run python scripts/embeddings/extract.py --config <config> --no-skip

# Override datamodule batch_size / num_workers
uv run python scripts/embeddings/extract.py \
    --config <config> --batch-size 8 --num-workers 4

# Pick a specific device
uv run python scripts/embeddings/extract.py --config <config> --device cuda:1
```

By default `extract.py` first scans the dataset once and exits early
if every clip is already cached. Pass `--no-skip` to force re-write.

### Inspecting + clearing via `manage.py`

```bash
# Table of every cache directory with clip count, on-disk size, and meta
uv run python scripts/embeddings/manage.py list

# Dump _meta.json for one cache directory
uv run python scripts/embeddings/manage.py info \
    output/.emb_cache/OMARRQ-multifeature-25hz/SHS100K__abc12345/

# Delete all caches for one encoder (dry-run by default)
uv run python scripts/embeddings/manage.py clear OMARRQ-multifeature-25hz

# Delete one specific cache directory
uv run python scripts/embeddings/manage.py clear \
    OMARRQ-multifeature-25hz/SHS100K__abc12345

# Actually delete (omit --apply for dry-run)
uv run python scripts/embeddings/manage.py clear <target> --apply
```

`manage.py list` output looks like:

```
cache dir                                                    clips         size  meta-summary
-----------------------------------------------------------------------------------
OMARRQ-multifeature-25hz/SHS100K__abc12345                   6,939     662.4 MB  mtg-upf/omar-rq-multifeature-25hz @ 30.0s
OMARRQ-multifeature-25hz/VGMIDITVar__9f8e7d6c                12,870      1.2 GB  mtg-upf/omar-rq-multifeature-25hz @ 15.0s
MERT-v1-95M/Covers80__0a1b2c3d                                  160      6.3 MB  m-a-p/MERT-v1-95M @ 30.0s
-----------------------------------------------------------------------------------
TOTAL                                                                      1.9 GB
```

---

## Cache key invariants — when does the cache get bypassed?

Each cache directory's name encodes a deterministic hash of the
**encoder input pipeline**:

```
output/.emb_cache/
  <encoder_slug>/
    <task_name>__<8-char hash>/
      _meta.json
      <clip_id>.pt
      ...
```

`<8-char hash>` = first 8 hex chars of
`sha256(model_id + sample_rate + clip_seconds + audio_pipeline_signature)`,
where `audio_pipeline_signature` is the pipe-separated list of
`audio_transforms.test` class names from the YAML.

Changing any of:

- The HuggingFace `model_id` (e.g. swapping fsq ↔ non-fsq variants)
- The target `sample_rate`
- The dataset's `clip_seconds`
- The audio preprocessing transforms

→ produces a new hash → new directory → old cache untouched, new sweep
re-populates. Useful for A/B experiments: you can keep both variants'
caches and the runtime auto-picks the right one based on which config
is loaded.

`<clip_id>` = `<audio_path_stem>__<sha1(audio_path)[:8]>__c<slice_idx>`.
The path hash defends against same-stem-different-folder collisions
(e.g. `track01.mp3` in different artist directories).

### What the cache does NOT key on

- **Layer index** — `clip_id.pt` contains all L layers; per-layer
  queries slice the L axis at read time.
- **Aggregation mode** (per-layer vs meanall) — commutativity means
  both are served from the same data.
- **The audio file's mtime** — if you swap out audio under the same
  path, the cache won't notice. Clear the cache manually via
  `manage.py` if you do this.

---

## Cache-safe vs cache-unsafe in detail

The runtime auto-disables itself in unsafe situations rather than
producing wrong results:

| Situation | Behavior |
|---|---|
| `cache_embeddings: false` in the config | Cache code path is skipped; runs as before |
| Datamodule batch has no `clip_id` field (3-tuple instead of 4-tuple) | Falls back to encoder forward, no cache writes |
| `forward(x)` called without `clip_ids` kwarg | Same fallback |
| `--no-skip` flag (sweep runner) | Cache is still consulted, but the run isn't skipped |
| `task.eval()` / no training | Cache hits read tensors directly, no encoder forward |

The runtime never writes a partially-pooled or partially-correct
tensor: writes only happen via `cache.put_batch(clip_ids, pooled)`
where `pooled` is the post-`TimeAvgPool` `(B, L, H)` tensor. If the
encoder forward succeeds but the cache write fails (disk full, etc.),
the run continues normally — cache failures are non-fatal.

---

## Troubleshooting

**"I don't see the `[emb_cache] HIT/MISS` log line on a fresh run."**
- Confirm `cache_embeddings: true` is in your config:
  `grep cache_embeddings configs/probe.*.yaml`.
- Confirm the datamodule emits a `clip_id` field. Run
  `python -c "from marble.tasks.<Task>.datamodule import <DatasetClass>; ds = <DatasetClass>(...); print(len(ds[0]))"`
  — should print `4` (waveform, label, path, clip_id).

**"I expected a cache hit but got a miss."**
- Run `manage.py list` to confirm the cache directory exists.
- Run `manage.py info <dir>` and check that
  `clip_seconds`, `model_id`, `pipeline_signature` match what your
  sweep config uses. A mismatch produces a different `<config_hash>`
  → different directory → cache miss as expected.
- If you renamed `audio_path` for any clip (e.g. `.m4a` → `.flac` via
  `convert_shs100k_to_flac.py`), the `clip_id` hash changes too —
  this is correct behavior; the cache was keyed to the old paths.

**"My cache directory is much bigger / smaller than predicted."**
- Each `.pt` is `L × H × 4` bytes + a ~1.8 KB `torch.save` header.
  Total = `n_clips × (L × H × 4 + 1800)`.
- For chunked datasets (anything with `clip_seconds < song_duration`)
  the clip count is `total_audio_seconds / clip_seconds`, not the
  track count. SHS100K's 5,000 tracks → ~6,900 clips at 30-sec.

**"I want to start over."**
```bash
uv run python scripts/embeddings/manage.py clear <encoder> --apply
```

**"I want to share the cache between machines."**
Cache files are plain `.pt` tensors keyed by a deterministic hash of
the encoder + audio pipeline. As long as both machines run the same
config, you can `rsync output/.emb_cache/ user@otherhost:.../output/.emb_cache/`
and the second machine will see all hits.

---

## Implementation reference

| Component | File |
|---|---|
| Cache class + key derivation + atomic I/O | `marble/utils/emb_cache.py:EmbeddingCache` |
| Reusable cache plumbing (mixin) | `marble/utils/emb_cache.py:EmbeddingCacheMixin` — shared by BaseTask + CoverRetrievalTask |
| Cache-aware base task (covers 11 supervised tasks) | `marble/core/base_task.py:BaseTask` |
| Cache-aware retrieval task | `marble/tasks/Covers80/probe.py` (used by SHS100K + VGMIDITVar via re-export) |
| `clip_id` + audio-I/O bypass in base datamodule (inherited by 10 supervised + symbolic tasks) | `marble/core/base_datamodule.py:BaseAudioDataset` |
| `clip_id` + audio-I/O bypass in 9 custom datasets | `marble/tasks/{Covers80,SHS100K,VGMIDITVar,GS,EMO,GTZANGenre,NSynth,HookTheoryKey,HookTheoryStructure}/datamodule.py` |
| Pass-through of extras in audio-transform wrapper | `marble/modules/transforms.py:AudioTransformDataset` |
| Pre-warm CLI | `scripts/embeddings/extract.py` |
| Inspect + cleanup CLI | `scripts/embeddings/manage.py` |
| Static audit of cache integration | `scripts/embeddings/audit_cache_integration.py` |

## Verifying cache integration is wired up (the audit script)

The cache has several silent failure modes — `cache_embeddings: true`
in a config + cache dir + `_meta.json` get created, but nothing
actually populates because some link in the chain is missing
(dataset doesn't emit `clip_id`, task overrides `forward` without
calling `_cached_forward_layer_tuple`, custom `validation_step`
drops `clip_ids`, etc.). These are nasty to detect from runtime
wall-clock alone — fit time looks "about right" because the encoder
runs cold every epoch, exactly like it did before caching shipped.

`scripts/embeddings/audit_cache_integration.py` catches all of these
statically:

```bash
# Audit every cache-enabled config in the repo
uv run python scripts/embeddings/audit_cache_integration.py

# Filter to a specific task family
uv run python scripts/embeddings/audit_cache_integration.py --filter 'HookTheory'

# Verbose mode — show every audited config, not just failures
uv run python scripts/embeddings/audit_cache_integration.py -v

# Best-effort runtime probe (instantiates one dataset, pulls one item,
# validates 4-tuple shape — skips if jsonl isn't on disk)
uv run python scripts/embeddings/audit_cache_integration.py --runtime
```

What it checks per config:
1. Task class accepts `cache_embeddings` in its `__init__` chain
2. Task class inherits `EmbeddingCacheMixin` transitively
3. If `forward()` is overridden, the override actually **calls**
   `_cached_forward_layer_tuple` (AST analysis, not substring match)
4. If any `*_step` is overridden AND its body references `batch`, the
   body must also reference `clip_ids` (no-op overrides that just
   `return None` are correctly exempted)
5. Dataset classes either inherit `BaseAudioDataset` or have all three
   explicit-integration markers in source

Exit code 0 if everything is wired up, 1 on any failure.

**Run this before:**
- Adding a new cache-enabled config
- Changing the cache plumbing (any of the mixin / BaseTask / dataset code)
- Investigating a suspicious sweep wall-clock

**Audio-I/O bypass** plumbing:
- Each cache-aware dataset has a `cache_check_fn: Callable[[str], bool]
  | None` field (default `None`). The task's `setup()` injects
  `self._cache.has` via
  `EmbeddingCacheMixin._inject_cache_check_into_datasets` after the
  trainer + datamodule are wired up.
- `__getitem__` computes `clip_id` BEFORE doing audio I/O. If
  `cache_check_fn(clip_id)` returns True, it returns a
  `torch.zeros(channels, clip_len_target)` placeholder waveform and
  skips `torchaudio.load + resample + pad` entirely. The task's
  `forward()` ignores `x` on cache hits and uses the cached
  `(L, H)` tensor instead — so the placeholder zeros are never seen
  by the encoder.

The runtime cache lookup happens in `BaseTask.forward()` (and
`CoverRetrievalTask.forward()`, which uses the same mixin helper).
Hit path uses `stacked_to_layer_tuple(cached)` to round-trip a
`(B, L, H)` tensor back into the tuple-of-layers format
`LayerSelector` expects; miss path uses
`encoder_tuple_to_pooled(layer_outputs)` to time-pool every layer in
one go and persist.
