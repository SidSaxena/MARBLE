# Leitmotif Detection with CLaMP3

End-to-end plan for using CLaMP3 audio embeddings to detect and localise
leitmotifs (recurring musical themes) in video-game / film soundtracks.

This document covers everything from data preparation through embedding
extraction, probe training, inference at scale, evaluation, and known
failure modes.  Each section ends with the exact commands you'd run.

**Companion documents:**
- [`leitmotif_swtc.md`](./leitmotif_swtc.md) — applies this methodology
  to the Star Wars Thematic Corpus, with concrete workflows and a
  proposed MARBLE task design.

**Implementation status (last reviewed below in §14):**
The cross-modal API described in §13 is **implemented and end-to-end
tested**.  The VGMIDI-TVar audio + symbolic probe tasks described in
§12 are **shipped as MARBLE tasks**.

---

## 0. Scope and assumptions

**The task.** Given a target catalogue of leitmotifs *L = {ℓ₁ … ℓ_K}*
defined by audio examples (a few seconds each), identify every occurrence
of each motif inside arbitrary unseen audio tracks.

Two operating modes that share the same encoder but differ in inference:

| Mode | Input | Output | Use case |
|---|---|---|---|
| **Clip classification** | a 5-30 s pre-segmented audio clip | one of K motif labels (or "none") | Studying a curated set of motif statements; ablation experiments. |
| **Localisation** | a full track (minutes long) | timestamped detections per motif | The real downstream task: "find every Force theme moment in the soundtrack album." |

**Why CLaMP3.** Per the layer sweep results, CLaMP3 produces the strongest
clip-level musical-content embeddings of the three encoders. It was
contrastively trained on music+text pairs, so its embedding space clusters
by *musical identity* (same melody, same harmonic content, same
orchestration character) more than by pure acoustic similarity. That's
exactly the property a leitmotif identifier needs.

**Granularity ceiling.** CLaMP3's audio encoder operates on **5-second
chunks**. This document is built around that fact:

- A leitmotif statement is typically 5–15 s — matches CLaMP3's chunk size.
- Sub-second localisation (e.g. "the theme starts at 0:43.2") is **not**
  achievable with CLaMP3 alone. We get chunk-level resolution (~5 s).
  If you need tighter timing, see §9 for hybrid approaches.

**What this plan does NOT do:**
- Train CLaMP3 itself (it stays frozen).
- Discover new motifs from scratch (the catalogue is given).
- Generalise across instrumentation that's *radically* different from
  CLaMP3's training distribution (synth-pop ≠ orchestral; results may
  degrade and that should be measured, not assumed).

---

## 1. Data preparation

### 1.1 The label catalogue

Decide your motif vocabulary up front. For BotW you might have something
like:

```python
labels = [
    "main_theme",
    "ganon",
    "zelda_lullaby",
    "korok_forest",
    "rito_village",
    # …
    "none",   # background / non-motif clips
]
```

The `"none"` class is **essential**. Without it the probe will assign
*some* motif to every clip, including ones that contain no motif at all.
Recall is meaningless without that null option.

### 1.2 Per-clip JSONL (training/evaluation)

The existing `LeitmotifDetection` datamodule reads JSONL of the form:

```json
{"audio_path": "/data/botw/clips/main_theme_001.wav",
 "label":       "main_theme",
 "sample_rate": 44100,
 "num_samples": 220500}
```

Required: `audio_path`, `label`, `sample_rate`, `num_samples`. Optional:
`channels` (defaults to 1).

For each motif:

- Collect **5–15 positive clips** of unambiguous statements (different
  arrangements, instrumentations, dynamics). Variety matters more than
  raw count — 10 well-chosen clips beat 100 near-duplicates.
- Create train/val/test splits where the same source track is **not** in
  more than one split — otherwise CLaMP3's chunk embeddings of similar
  surrounding audio leak labels across splits.

For the `"none"` class:

- Sample 2–3× as many negative clips as the largest positive class to
  avoid the classifier collapsing to "always predict motif X."
- Source negatives from the same soundtrack universe (same composer,
  same orchestra, similar production) so the classifier learns *motif
  identity*, not "is this a video game vs is this silence."

### 1.3 Splits

| Split | Purpose | Suggested fraction |
|---|---|---|
| `train` | Probe head training | 70% of clips |
| `val` | Best-layer selection, early stopping | 15% |
| `test` | Final reported numbers | 15% |

Stratify by motif label so each split has all classes. If you have a
**held-out track** (a full track that no clips from were used for
training), use it as the **localisation test set** in §5 — that's the
realistic evaluation.

### 1.4 Practical: building the JSONL

Write `scripts/build_leitmotif_jsonl.py` (sketch):

```python
import json
from pathlib import Path
import soundfile as sf

CLIP_DIR = Path("data/Leitmotif/clips")     # per-clip wavs
OUT_DIR  = Path("data/Leitmotif")
LABELS   = json.loads(Path("data/Leitmotif/labels.json").read_text())

for split in ("train", "val", "test"):
    out = OUT_DIR / f"Leitmotif.{split}.jsonl"
    with out.open("w") as f:
        for clip in (CLIP_DIR / split).rglob("*.wav"):
            info = sf.info(clip)
            label = clip.parent.name      # foldername == label
            f.write(json.dumps({
                "audio_path":  str(clip),
                "label":       label,
                "sample_rate": info.samplerate,
                "num_samples": info.frames,
                "channels":    info.channels,
            }) + "\n")
```

---

## 2. Embedding extraction strategy

### 2.1 What "extracting an embedding" actually means here

CLaMP3 produces a **(B, T_chunks, H=768)** tensor where `T_chunks =
clip_seconds / 5` (rounded). For a 5-sec clip → `T_chunks=1`; for a 15-sec
clip → `T_chunks=3`.

The MARBLE pipeline composes:

```
encoder (CLaMP3)  →  emb_transforms  →  decoder
   ↓                      ↓                ↓
 (B, L, T, H)         (B, T, H)         (B, K)
 L = 13 layers     (LayerSelector)   (logits)
```

`LayerSelector([N])` picks one of the 13 layer outputs (CNN feature
extractor + 12 BERT layers). `TimeAvgPool` collapses the `T_chunks`
dimension to one vector per clip. The decoder maps `H=768 → K`.

### 2.2 Choosing the layer

Don't pick blindly. **Run the layer sweep on your leitmotif data first.**
The sweep file lives at:

```
configs/probe.CLaMP3-layers.LeitmotifDetection.yaml   ← to be created (§4)
```

Once it runs, the layer with the highest `val/acc` (or `val/file_acc`)
is your best layer. From your existing GS/HookTheoryKey sweeps, expect
the best CLaMP3 layer to be in the **0–3 range** (early layers carry the
strongest melodic-identity signal in this encoder family).

### 2.3 The probe head

A **linear probe** (no hidden layer) is the cleanest measurement: it
tells you whether the leitmotif identity is *linearly recoverable* from
the chosen layer. For practical performance an MLP with one hidden layer
(e.g. 256 units) typically gives a few percentage points more
file-level accuracy.

For clip classification the head is:

```
Linear(768 → K)            # K = len(labels)
```

For localisation we'll add temperature scaling (§5.3) to get well-calibrated
probabilities.

---

## 3. Pre-flight: cache embeddings (optional but recommended)

If your dataset is small and you'll iterate a lot, **extract the embeddings
once and cache them** rather than re-running CLaMP3 every probe-training
epoch. Saves hours.

```python
# scripts/extract_clamp3_embeddings.py  (sketch)
import json, torch
from pathlib import Path
from tqdm import tqdm
from marble.encoders.CLaMP3.model import (
    CLaMP3_Encoder, CLaMP3_FeatureExtractor,
)

encoder = CLaMP3_Encoder().eval().cuda()
fx      = CLaMP3_FeatureExtractor(squeeze=True)

BEST_LAYER = 0       # set from your sweep results
CLIP_SECONDS = 15

with torch.no_grad():
    for line in Path("data/Leitmotif/Leitmotif.test.jsonl").read_text().splitlines():
        rec = json.loads(line)
        wav = fx(load_audio(rec["audio_path"], sr=24000))   # (1, 1, T)
        h   = encoder(wav.cuda())                            # (1, L, T, H)
        emb = h[:, BEST_LAYER].mean(dim=1)                   # (1, H) — TimeAvgPool
        torch.save(emb.cpu(), f"cache/{Path(rec['audio_path']).stem}.pt")
```

Then the probe trains on `(emb, label)` pairs in seconds. This is purely
an optimisation — the regular MARBLE pipeline works without it.

---

## 4. Training the clip classifier

### 4.1 Create the leitmotif sweep config

Mirror the GS/HookTheoryKey layer-sweep config, but pointed at the
`ProbeLeitmotifTask` (which already does file-level aggregation):

```yaml
# configs/probe.CLaMP3-layers.LeitmotifDetection.yaml  (excerpt)
model:
  class_path: marble.tasks.LeitmotifDetection.probe.ProbeLeitmotifTask
  init_args:
    sample_rate: 24000
    encoder:
      class_path: marble.encoders.CLaMP3.model.CLaMP3_Encoder
    emb_transforms:
      - class_path: marble.modules.transforms.LayerSelector
        init_args: { layers: [0] }     # sweep replaces this
      - class_path: marble.modules.transforms.TimeAvgPool
    decoders:
      - class_path: marble.modules.decoders.MLPDecoder
        init_args:
          in_dim: 768
          out_dim: 10                  # = len(labels); set per project
          hidden_layers: [256]
          dropout: 0.2
    losses:
      - class_path: torch.nn.CrossEntropyLoss
    metrics:
      train: { acc: { class_path: torchmetrics.Accuracy,
                       init_args: { task: multiclass, num_classes: 10 } } }
      val:   { … same … }
      test:  { … same … }

data:
  class_path: marble.tasks.LeitmotifDetection.datamodule.LeitmotifDetectionDataModule
  init_args:
    batch_size: 16
    num_workers: 8
    audio_transforms:
      train: [ { class_path: marble.encoders.CLaMP3.model.CLaMP3_FeatureExtractor,
                 init_args: { squeeze: true } } ]
      val:   [ … same … ]
      test:  [ … same … ]
    train:
      class_path: marble.tasks.LeitmotifDetection.datamodule.LeitmotifAudioTrain
      init_args:
        sample_rate: 24000
        channels: 1
        clip_seconds: 15.0
        channel_mode: random
        jsonl: data/Leitmotif/Leitmotif.train.jsonl
        labels: ["main_theme", "ganon", "zelda_lullaby", …, "none"]
    val: { … same paths, channel_mode: mix … }
    test: { … same paths, channel_mode: mix … }
```

### 4.2 Run the layer sweep

```bash
python scripts/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-layers.LeitmotifDetection.yaml \
    --num-layers 13 \
    --model-tag CLaMP3-layers \
    --task-tag LeitmotifDetection
```

Watch `test/acc` and `test/file_acc` per layer in WandB. Pick the layer
with the best `val/file_acc` (file-level accuracy aggregates clip
predictions, more stable than per-clip).

### 4.3 What "good" looks like

| `test/file_acc` | Interpretation |
|---|---|
| < 1/K + 5% | Random; representations don't separate motifs at all |
| 1/K + 10% to ~50% | Some signal; check label hygiene, class balance |
| 60–80% | Working; usable for downstream localisation |
| > 80% | Strong; CLaMP3 has clearly learned motif-relevant features |

Don't compare across motif catalogues — accuracy depends heavily on K and
how acoustically distinct your motifs are.

---

## 5. Localisation: finding motifs in a full track

This is the actual deployable system. Once §4 gives you a trained head
and a known best layer, here's how to apply it to a full track.

### 5.1 Sliding window

For each full track:

1. Resample to 24 kHz mono.
2. Slide a window of `W = 5 s` with hop `H = 1 s` across the track.
3. For each window, compute the CLaMP3 best-layer embedding (mean-pooled
   across the single chunk).
4. Run the trained head → softmax probabilities over K classes.
5. You now have a per-class probability curve at 1 Hz over the track.

```python
# pseudocode
sr = 24000
W = 5 * sr          # 5-second window
H = 1 * sr          # 1-second hop
probs = []          # list of length-K vectors
for start in range(0, len(audio) - W + 1, H):
    wav  = audio[start:start+W]
    feat = fx(wav)
    h    = encoder(feat)[:, BEST_LAYER]    # (1, T_chunks=1, 768)
    z    = head(h.mean(dim=1))             # (1, K)
    probs.append(softmax(z))
probs = torch.stack(probs)                 # (N_windows, K)
```

### 5.2 Peak picking

For each motif class `k`, find detections:

```python
from scipy.signal import find_peaks

curve = probs[:, k]                       # (N_windows,)
peaks, props = find_peaks(
    curve,
    height=0.5,         # threshold; tune on val track
    distance=10,        # minimum 10 s between detections of the same motif
    prominence=0.1,     # ignore tiny bumps
)
# Convert window indices back to time
detection_seconds = peaks * (H / sr)
```

Threshold selection: choose to maximise F1 on a held-out track with
ground-truth timestamps (§6). A reasonable starting point is 0.5; cleaner
data usually allows 0.6–0.7.

### 5.3 Calibration (optional but valuable)

Raw softmax probabilities tend to be over-confident on out-of-distribution
audio (silence, dialogue, foley). Two simple fixes:

1. **Temperature scaling.** Fit a scalar `T` on the val set so that
   `softmax(logits / T)` is well-calibrated. Tiny modification, large
   downstream benefit on the threshold.

2. **None-class anchoring.** During training, include explicit `"none"`
   clips (§1.1). At inference, require both `argmax == k` *and*
   `prob[k] - prob["none"] > margin` (e.g. 0.2).

### 5.4 Output format

For each track and each motif, emit:

```json
{
  "track": "track_03.flac",
  "detections": [
    { "label": "main_theme",     "start": 12.0, "end": 17.0, "score": 0.81 },
    { "label": "main_theme",     "start": 142.0, "end": 147.0, "score": 0.73 },
    { "label": "zelda_lullaby",  "start": 200.0, "end": 205.0, "score": 0.69 }
  ]
}
```

---

## 6. Evaluation

### 6.1 Clip-level metrics (already in the probe)

- `test/acc` — per-clip top-1 accuracy
- `test/file_acc` — per-file accuracy after averaging clip probs

### 6.2 Localisation metrics (write a separate evaluator)

You need a **held-out evaluation track** with ground-truth motif timing
(start, end, label). Recommended metrics:

- **Event-level F1** (mir_eval.onset.f_measure-style): a detection is a
  true positive if its centre is within `±τ` seconds of a ground-truth
  centre and the predicted label matches. Typical `τ = 2.5` s for
  5-second windows.
- **Per-class precision / recall** so you can see which motifs are
  detected reliably vs missed.
- **Confusion matrix** of predicted-label vs actual-label among true
  positives — tells you which motifs CLaMP3 confuses (often related
  motifs sharing harmonic material).

Sketch:

```python
def event_f1(pred, gt, tolerance_sec=2.5):
    """pred, gt = list of (start, label) tuples."""
    matched = set()
    tp = 0
    for ps, plbl in pred:
        for i, (gs, glbl) in enumerate(gt):
            if i in matched: continue
            if abs(ps - gs) <= tolerance_sec and plbl == glbl:
                matched.add(i); tp += 1; break
    fp = len(pred) - tp
    fn = len(gt) - tp
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)
```

### 6.3 Sanity checks before believing the numbers

| Check | Why |
|---|---|
| Random-label baseline (shuffle test labels and re-train probe) | Confirms `acc` isn't due to artefacts |
| Single-motif vs multi-motif difficulty | Easy motifs may inflate macro-avg |
| Pure-silence test clip → predicts `"none"` | Confirms `"none"` class is doing its job |
| Predict on the *training* tracks → expect very high acc | Confirms pipeline isn't broken |
| Predict on a non-game track (e.g. a pop song) → should predict `"none"` | Out-of-distribution behaviour |

---

## 7. Concrete end-to-end recipe

Assuming your motif clips are organised as
`data/Leitmotif/clips/{train,val,test}/{label}/*.wav` and you have a
held-out track at `data/Leitmotif/tracks/track_03.flac` with annotations
at `data/Leitmotif/tracks/track_03.jsonl`:

```bash
# 1. Build JSONL splits
python scripts/build_leitmotif_jsonl.py

# 2. Verify the JSONL parses and audio is loadable (reuse SHS verifier shape)
python -c "
import json
for split in 'train val test'.split():
    n = sum(1 for _ in open(f'data/Leitmotif/Leitmotif.{split}.jsonl'))
    print(f'{split}: {n} clips')
"

# 3. CLaMP3 13-layer sweep on the clip classifier
python scripts/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-layers.LeitmotifDetection.yaml \
    --num-layers 13 \
    --model-tag CLaMP3-layers \
    --task-tag LeitmotifDetection

# 4. Read the best-layer index from the sweep summary (or WandB).
#    Suppose it's layer 1.  Save the trained head:
ls output/probe.LeitmotifDetection.CLaMP3-layers.layer1/checkpoints/best.ckpt

# 5. Run localisation inference on full tracks
python scripts/leitmotif_localise.py \
    --ckpt output/probe.LeitmotifDetection.CLaMP3-layers.layer1/checkpoints/best.ckpt \
    --layer 1 \
    --tracks data/Leitmotif/tracks/*.flac \
    --window-seconds 5 --hop-seconds 1 \
    --output predictions/track_03.json

# 6. Compute event-level F1 against ground truth
python scripts/leitmotif_eval.py \
    --predictions predictions/track_03.json \
    --ground-truth data/Leitmotif/tracks/track_03.jsonl \
    --tolerance-seconds 2.5
```

Step 1 needs to be written once per project. Steps 5 and 6 are small
scripts (~80 LOC each) — sketches above.

---

## 8. Known limitations

| Limitation | Why | Workaround |
|---|---|---|
| Sub-second timing | CLaMP3 chunks are 5 s | Combine with MERT for fine timing once a CLaMP3 window flags a motif region |
| Cross-arrangement generalisation | CLaMP3 was trained on commercial music, not orchestral video-game scores | Include diverse arrangements per motif in training |
| Cross-motif confusion | Related motifs (e.g. main theme variations) share harmonic content | Train with explicit "variation" sub-classes; report confusion matrix |
| Background music vs motif | The `"none"` class needs careful curation — too easy makes a degenerate classifier | Include musically active non-motif backgrounds |
| Polyphonic motif overlaps | When two motifs play simultaneously, softmax picks one | Switch to multi-label sigmoid head (BCE loss) and threshold per class |

---

## 9. Extensions worth exploring (after a working baseline)

1. **Hybrid CLaMP3 + MERT.** CLaMP3 finds the *region*, MERT-pitched
   features lock the *exact start*. Run MERT (75 Hz) only on the
   ±5-second neighbourhood around CLaMP3 detections.

2. **Few-shot CLaMP3.** Use CLaMP3 in a nearest-centroid setup: average
   embeddings of each motif's clips into a prototype, classify by cosine
   distance to prototypes. No probe head trained at all — pure CLaMP3.
   Useful when you have ~3 clips per motif.

3. **Self-supervised pre-clustering.** Embed every 5-second window of
   your entire soundtrack universe; cluster. Manually label each cluster.
   This discovers motifs you didn't know existed.

4. **Multi-label classification.** Replace softmax-with-CE by
   sigmoid-with-BCE so a single window can carry multiple motif labels
   simultaneously. Necessary if motifs overlap in your data.

5. **Sliding-window CLaMP3 with overlap-add aggregation.** Smooth the
   per-window probability curve with a Gaussian before peak-picking
   (current §5.2 has no smoothing). Less false positives on noisy
   curves.

---

## 10. Open questions for your specific data

These are worth thinking through before the first sweep — they will
shape your label catalogue and split strategy:

1. **What counts as the same motif?** "Main theme" and "Main theme,
   tense variation" — same class or different? Decide before labelling.
2. **What's the minimum motif duration in your data?** If some
   statements are 2-3 s, CLaMP3's 5-s window may dilute the signal —
   consider padding short clips with silence vs surrounding context.
3. **Are tracks album-form or game-form?** Game music with abrupt
   transitions has different distribution than album mix-downs.
4. **Do you have ground-truth onsets** or just "this track contains the
   theme"? The latter doesn't support localisation evaluation.

---

## Appendix A: Why the `LayerSelector + TimeAvgPool` choice?

CLaMP3's `LayerSelector([N])` picks the Nth layer of CLaMP3's *internal
BERT* (12 layers + 1 CNN). It does **not** select MERT layers — CLaMP3
internally averages all MERT layers before feeding them to BERT, so
that's already done.

`TimeAvgPool` then collapses the chunk axis (3 chunks for a 15-s clip)
to a single 768-dim vector. This is the simplest pooling; alternatives
include max-pool, attention-pool, or concatenating the chunks. For
clip-classification, mean-pool consistently performs best in MARBLE
results.

## Appendix B: File layout this plan assumes

```
project_root/
├── configs/
│   └── probe.CLaMP3-layers.LeitmotifDetection.yaml
├── data/
│   └── Leitmotif/
│       ├── labels.json                    # ordered list of motif strings
│       ├── clips/{train,val,test}/{label}/*.wav
│       ├── tracks/*.flac                  # full tracks for localisation
│       ├── tracks/*.jsonl                 # ground-truth timing
│       ├── Leitmotif.train.jsonl          # built by build_leitmotif_jsonl.py
│       ├── Leitmotif.val.jsonl
│       └── Leitmotif.test.jsonl
├── scripts/
│   ├── build_leitmotif_jsonl.py
│   ├── leitmotif_localise.py
│   └── leitmotif_eval.py
└── marble/tasks/LeitmotifDetection/         # already exists
    ├── datamodule.py
    └── probe.py
```

---

## 11. Encoder-agnostic general approach

Sections 0–10 above are written around CLaMP3 because that's where the
preliminary probe results currently point.  The same methodology
generalises to any audio encoder — what changes is window size, embedding
dimension, and best-layer index; the pipeline shape stays identical.
This section is the encoder-free recipe.

### 11.1 Encoder selection decision tree

```
Q1.  Does your downstream task need frame-level timing?
       ──── yes ───►  Pick a frame-level encoder
       │                MERT-v1-95M     (75 Hz, H=768, 12+1 layers)
       │                OMARRQ          (25 Hz, H=1024, 24 layers)
       │                MuQ / MusicFM   (similar profile)
       │
       ──── no, clip-level is fine ───►  Q2

Q2.  Do your motifs come in many arrangements / instrumentations?
       ──── yes ───►  Prefer contrastive music+text models
       │                CLaMP3, MuQ-MuLan
       │                (their embeddings cluster by musical identity
       │                 rather than acoustic surface)
       │
       ──── motifs are acoustically consistent ───►  Either family works;
              prefer MERT for the cheaper compute budget
```

For leitmotifs in soundtrack data specifically, the bias is strongly
toward Q2's "many arrangements" branch — a leitmotif is recognised
across orchestration changes, tempo shifts, key modulations, and timbre
variation.  That's the situation contrastive embeddings are built for.

### 11.2 What every encoder gives you

Every audio encoder in MARBLE exposes the same shape contract:

```
encoder(waveform: [B, 1, T_samples])  →  [B, L, T_tokens, H]
                                          │   │   │          │
                                          │   │   │          embedding dim
                                          │   │   token-time axis
                                          │   layer axis
                                          batch
```

What varies:

| Encoder        | H     | L   | tokens/sec | sample rate | natural window |
|----------------|-------|-----|------------|-------------|----------------|
| MERT-v1-95M    | 768   | 13  | 75         | 24 kHz      | any            |
| OMARRQ         | 1024  | 24  | 25         | 24 kHz      | any            |
| CLaMP3         | 768   | 13  | 1 / 5 s    | 24 kHz      | 5 s            |
| MuQ            | 1024  | 13  | 25         | 24 kHz      | any            |
| MusicFM        | 768   | 13  | 25         | 24 kHz      | any            |

"Natural window" is the smallest segment that produces a meaningful
embedding.  For chunk-rate encoders (CLaMP3) you can't go below the
chunk; for token-rate encoders (everything else) any window length works
but very short windows (< 1 s) get sparse and noisy.

### 11.3 Pipeline that works for every encoder

```
   waveform                                     classifier head
      │                                                ▲
      ▼                                                │
  ┌────────┐    ┌──────────────┐    ┌────────────┐    │
  │encoder │ →  │LayerSelector │ →  │pooling     │ →──┘
  │(frozen)│    │(best layer N)│    │(see §11.4) │
  └────────┘    └──────────────┘    └────────────┘
       (B, L, T, H)      (B, T, H)         (B, H)
```

The encoder stays **frozen** for every layer-sweep probe — what we're
measuring is whether motif identity is recoverable from the
representations the encoder already produces.  Training the encoder
itself is a separate (and much more expensive) question; do that only
after probes prove no frozen layer suffices.

### 11.4 Pooling choices

How `(B, T, H) → (B, H)`?  Three options, ranked by what they cost vs
what they buy:

| Pooling | Cost | Buys | When to use |
|---|---|---|---|
| **Mean-pool over time** | none | a single robust vector | default; works for almost all clip-level tasks |
| **Max-pool over time** | none | the strongest token | when one moment in the clip is decisive (e.g. a sting) |
| **Attention-pool**  | small (1 layer) | learnable per-token weighting | when motifs occupy a small fraction of the clip |
| **Concat first+mean+last** | none | crude position-awareness | rarely worth it; mean-pool usually wins |

Mean-pool is the default everywhere in MARBLE.  If mean-pool's accuracy
plateaus and you suspect the motif occupies a sub-region of the clip,
swap in attention-pool and re-run.  Don't tune this before the layer
sweep — pick the best layer with mean-pool first, then revisit pooling.

### 11.5 Sliding-window inference, encoder-agnostic

For full-track localisation, the parameters generalise as:

| Parameter | How to pick |
|---|---|
| Window `W` | = natural window for chunk-rate encoders; = 5–10 s for token-rate (matches typical motif statement length) |
| Hop `H` | 10–20% of `W` for fine localisation; 50% for fast scanning |
| Smoothing | Gaussian over the per-class curve, σ ≈ `W/2` |
| Threshold | tuned per-encoder, per-dataset on val set |

**Multi-resolution scan:** if motif lengths vary widely in your data,
run the window sweep at multiple `W` values (e.g. 5 s, 10 s, 15 s) and
take the per-time maximum across resolutions.  Catches both short
stings and long statements without committing to one window size.

### 11.6 Multi-encoder ensembles

Once you have probes trained on each encoder individually, you can
ensemble at three levels (increasing cost):

1. **Probability-level**.  Average the softmax outputs from each
   encoder's trained head.  Zero training cost on top of the
   individual probes.  Typical gain: 2–5 absolute % in file-level acc.
2. **Embedding-level**.  Concatenate embeddings from each encoder into
   one wide vector and train one combined head.  Requires the per-clip
   embeddings to be cached for all encoders.  Typical gain: 5–10%.
3. **Layer-level fusion**.  Concatenate the *best layer* of each
   encoder and train a head on that.  Same cost as 2; can be slightly
   better than 2 if some encoders' early layers complement others'
   middle layers.

For leitmotif detection, MERT (acoustic / pitch detail) + CLaMP3
(musical identity / arrangement-invariant) is the natural pairing — the
two encoders capture different views and rarely make the same mistakes.

### 11.7 Honest comparison checklist

When evaluating a new encoder for your leitmotif task, run these in
order before adopting it:

1. **Identical evaluation harness.** Same JSONL, same splits, same
   pooling, same head architecture, same hyperparameters.  Only the
   encoder changes.
2. **Layer sweep, not single-layer evaluation.** A single layer can
   misrepresent an encoder — always pick best-of-N.
3. **Sanity-check on a related public task.** If the encoder doesn't
   beat random on GS/HookTheoryKey, expect leitmotif performance to be
   poor too.
4. **Wall-clock cost.** Per-clip embedding time, per-track localisation
   time, peak GPU memory.  A 2% accuracy gain at 10× the latency may
   not be worth it for batch processing of an album.
5. **Failure mode analysis.** Look at the confusion matrix.  A model
   with lower top-1 but cleaner confusions (only confuses related
   motifs) may be more useful than one with higher top-1 but wild
   off-class errors.

### 11.8 When the frozen-probe ceiling isn't enough

If every encoder × layer × pooling combination plateaus below what you
need:

1. **Fine-tune the last few encoder layers.** Use the same probe head
   but unfreeze layers 10–12 of the encoder.  Requires more data
   (1000s of labelled clips) and careful learning-rate scheduling
   (encoder LR 10× smaller than head LR).  Often gives 5–15% more.
2. **Train your own encoder on your domain.** Self-supervised on a
   corpus of soundtracks (BTW + similar games), then probe.  Months of
   work; only justifiable if you have lots of domain audio.
3. **Switch to symbolic.** If your data has MIDI versions
   (game-rip MIDI is often available), embedding-based motif matching
   in symbolic space is dramatically easier than from audio.

---

## 12. Candidate datasets for leitmotif / variation probes

Two external datasets came up as potential additions to the MARBLE probe
suite.  Both are plausible but have very different integration costs.

### 12.1 Variation-Transformer (POP909-TVar + VGMIDI-TVar)

**What it is.**  Theme-and-variation pairs extracted from POP909 (Western
pop) and VGMIDI (video-game piano arrangements).  Filenames encode the
relationship: `{piece_id}_{section}_0.mid` is a theme,
`{piece_id}_{section}_N.mid` for N>0 is its Nth variation.

| | |
|---|---|
| Source | https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model |
| Format | **MIDI only**, no audio |
| Splits | Pre-defined train/test in each dataset |
| POP909-TVar | ~9,440 files, pop |
| VGMIDI-TVar | smaller; video-game piano |

**Relevance to leitmotifs.**  High and direct — a leitmotif "statement
across arrangements" is conceptually the same relationship as a theme
and its variations.  The dataset gives **explicit theme–variation
groupings**, which makes it usable as a retrieval probe analogous to
SHS-100K but at the motif scale instead of the song scale.

**Integration cost.**

1. **MIDI → audio rendering** (one-time).  Pipeline: `fluidsynth` +
   a piano-leaning SoundFont (e.g. SGM-V2.01, Salamander).  Render at
   44.1 kHz, mono, 16-bit WAV.  ~1–2 s per file.  ~5 hours total for
   both datasets.
2. **Datamodule.**  Almost identical to Covers80 / SHS100K — the
   "work_id" becomes the theme-section identifier (`piece_id_section`),
   tracks within the same theme group are "covers" of each other.
   Re-use `CoverRetrievalTask` with no changes.
3. **Probe.**  Zero-shot retrieval; no training.  Same MAP-based eval as
   SHS100K.
4. **Caveat.**  Rendering MIDI to audio produces a *synthetic*
   distribution that may not match what the encoders were trained on
   (which is mostly studio-recorded music).  Expect lower absolute
   numbers than on natural-audio retrieval tasks — but for *relative*
   comparison across encoders / layers / pooling choices, that's fine.

**Recommendation: implement VGMIDI-TVar first** as a small MARBLE retrieval
task.  Why VGMIDI before POP909:

- Game music distribution is what you actually care about for the
  leitmotif downstream
- Smaller dataset → faster iteration during integration
- Once VGMIDI-TVar works, POP909-TVar is the same datamodule with
  different JSONL — trivial to add as a sanity check

The new task could be called `VGMIDITVar` (or `Variation` if generic).
Configs follow the Covers80 layer-sweep template line for line — only
the JSONL paths change.

### 12.2 Super Mario Structure Annotation

**What it is.**  554 Super Mario pieces with bar-level structural
annotations (Intro / Loop / Transition / Bridge / Outro / Stinger and
finer A/B/C section labels), plus 3,304 within-piece section pairs with
pre-computed similarity scores in three buckets (high / mid / low).

| | |
|---|---|
| Source | https://github.com/ShxLuo-Saxon/supermario-structure-annotation |
| Format | Annotations are JSON; source audio is **NOT redistributed** |
| Audio source | NinSheetMusic (manual download per `metadata/pieces.csv`) |
| Audio format on source | MUS (Finale), MIDI also available |
| Splits | Pre-defined train/val/test (70/15/15, piece-stratified) on pairs |

**Relevance to leitmotifs.**  Medium.  Two distinct sub-tasks are
embedded here, with different value:

| Sub-task | What it teaches | Leitmotif relevance |
|---|---|---|
| **Function classification** (section → In/Lp/Tr/Br/Ou/St) | Where in a track this section sits structurally | Low — section function ≠ motif identity |
| **Section-pair similarity** (compute similarity between two sections of the same piece, classify into 3 buckets) | Whether the encoder represents intra-piece variation faithfully | **High** — directly tests "do two clips with the same musical material map close in embedding space" |

The section-pair sub-task is essentially "is this encoder good at
recognising that two sections of the same Mario piece share material" —
which is the leitmotif question in miniature.

**Integration cost.**

1. **Audio acquisition** is the bottleneck.  Three paths:
   1. **Use the MIDI route** — `metadata/pieces.csv` has `url_mid`
      pointing at NinSheetMusic MIDI files.  Free, scriptable
      download.  Then MIDI → audio via fluidsynth (same as §12.1).
   2. The MXL/MUS route in the README requires **Finale** (commercial
      software) — skip.
   3. Render bar ranges to clips on demand from MIDI using bar→time
      maps computed from the MIDI tempo track.  Cleaner but more code.
2. **MIDI → audio rendering** (one-time, ~10 min for 554 pieces).
3. **Bar-to-time alignment.**  Annotations are in **bar numbers**, but
   the audio rendering produces a continuous waveform.  You need a
   reliable bar→time map for each MIDI to extract correct clips.
   `pretty_midi` exposes this via `pm.get_beats()` + tempo information.
4. **Datamodule + probes.**
   - **Function classification.**  Per-clip multi-class task, same
     pattern as HookTheoryStructure.  Simple to add.
   - **Section similarity bucketing.**  Same pattern as Covers80
     retrieval-style: embed both sections, predict bucket from cosine
     similarity (with a learned threshold or a small linear head).

**Recommendation: implement the section-similarity sub-task** as a new
MARBLE task; **skip the function-classification sub-task for now** —
it duplicates what HookTheoryStructure already measures, with the added
overhead of the MIDI pipeline.

Suggested task name: `SuperMarioPairs` or simply `MarioStructure`.  The
metric would be 3-class accuracy on similarity-bucket prediction, with
a per-clip retrieval-style alternative (MAP within a piece) as a
sanity check.

### 12.3 Side-by-side: which to add first?

| | VGMIDI-TVar | MarioStructure (pairs) | Both |
|---|---|---|---|
| Time to working datamodule | ~1 day | ~2 days | ~3 days |
| Audio quality risk | Synthetic, piano-only | Synthetic, multi-instrument | — |
| Direct leitmotif fit | High (theme ↔ variation) | High (section-pair similarity) | — |
| Dataset size | Small | Small | — |
| Adds new MARBLE capability | Variation retrieval probe | Within-piece similarity probe | — |
| Reuses existing infrastructure | Mostly Covers80 / SHS100K | Mostly HookTheoryStructure | — |

**My recommendation:**

1. **Finish current sweeps first.**  Don't expand the dataset suite until
   the MERT / OMARRQ / CLaMP3 sweeps on the existing 9 tasks have
   converged numbers — otherwise you'll be comparing apples and oranges
   when you do add these.
2. **Add VGMIDI-TVar next.**  Lowest integration cost, most direct
   leitmotif relevance, smallest dataset for fast iteration.  Reuses
   `CoverRetrievalTask` essentially unchanged.
3. **Defer MarioStructure** until VGMIDI-TVar results clarify whether
   the encoders generalise to synthesised MIDI audio at all.  If MIDI
   rendering tanks every encoder's performance, MarioStructure won't
   add new signal.  If the rendering works, MarioStructure becomes the
   richer follow-up.

### 12.4 What to skip

Both datasets share one risk worth naming explicitly: **synthetic MIDI
audio is out-of-distribution for the audio encoders.**  MERT, OMARRQ,
and CLaMP3 were all pretrained on natural audio recordings — a
fluidsynth render with a single piano SoundFont may produce embeddings
that cluster by SoundFont quirks rather than by musical content.

Mitigation: render with **varied SoundFonts** (rotate among 3–5 high-quality
SoundFonts per file) so the encoder can't latch onto a single timbre
signature.  This is a pre-processing step worth building once and
reusing for both datasets.

---

## 13. Cross-modal leitmotif discovery with CLaMP3

CLaMP3 was trained with contrastive loss across **three modality branches**:

* **Audio** — MERT features → BERT-style transformer
* **Symbolic (M3)** — MIDI patches → BERT-style transformer
* **Text** — XLM-RoBERTa

All three project into a **shared 768-dim embedding space** via per-branch
projection heads (``audio_proj`` / ``symbolic_proj`` / ``text_proj``).
A vector from any branch can be compared by cosine similarity to a vector
from any other branch.

This unlocks several leitmotif workflows that pure-audio probes can't
support.

### 13.1 The cross-modal API exposed in MARBLE

The CLaMP3 encoder wrappers expose three convenience methods that return
L2-normalised shared-space embeddings:

```python
from marble.encoders.CLaMP3.model import CLaMP3_Symbolic_Encoder

enc = CLaMP3_Symbolic_Encoder().eval().cuda()   # one class handles all 3 paths

#   each returns Tensor of shape (B, 768), L2-normalised
e_audio    = enc.embed_audio(wavs)                   # (B, 1, T_samples) → (B, 768)
e_symbolic = enc.embed_symbolic(midi_patches)        # (B, P, 64)       → (B, 768)
e_text     = enc.embed_text(["fanfare in C major"])  # str | list[str]  → (B, 768)

# Cross-modal cosine similarity matrix — comparable across all three branches
sims = e_audio @ e_symbolic.T          # (B_audio, B_symbolic)
```

Each method:

- Always uses the **final projection head** (the same one CLaMP3 was
  contrastively trained to align).  Per-layer outputs aren't available
  here — for layer-probe analysis use the standard ``forward()`` /
  ``LayerSelector`` pipeline instead.
- Pools globally over chunks/segments with length-weighted averaging
  before the projection — so long inputs (a whole track) collapse to a
  single 768-dim vector that compares fairly against short ones.
- Returns L2-normalised vectors — cosine similarity is just `A @ B.T`,
  no extra normalisation step needed.

### 13.2 Workflow A: MIDI-template motif search

The cleanest leitmotif workflow this enables:

**You have** a MIDI snippet of a leitmotif theme (one file, a few bars).
**You want** to find every occurrence of that theme in a long audio track.

```python
# 1. Tokenise the MIDI motif
from marble.encoders.CLaMP3.midi_util import midi_to_mtf
from marble.encoders.CLaMP3.clamp3_util import M3Patchilizer

motif_text   = midi_to_mtf("themes/main_theme.mid")
patches      = M3Patchilizer().encode(motif_text, add_special_patches=True)
motif_tensor = torch.tensor(patches).unsqueeze(0)   # (1, P, 64)
motif_emb    = enc.embed_symbolic(motif_tensor)     # (1, 768)

# 2. Slide a 5-sec window over the audio track
W, H_hop, sr = 5 * 24000, 1 * 24000, 24000
windows = [audio[:, :, i:i+W] for i in range(0, audio.size(-1)-W+1, H_hop)]
audio_embs = torch.stack([
    enc.embed_audio(w.unsqueeze(0)) for w in windows
], dim=0).squeeze(1)                                # (N_windows, 768)

# 3. Cosine similarity → detection curve
scores = (audio_embs @ motif_emb.T).squeeze(-1)     # (N_windows,)
detections = (scores > THRESHOLD).nonzero()          # indices into the curve
```

**Why this is valuable.**  No labelled *audio* training data of the
motif is needed — a single MIDI annotation is enough.  This works
particularly well for game soundtracks where the original MIDI scores
are often available (game-rip MIDI archives) but labelled audio
occurrences are not.

### 13.3 Workflow B: Audio-prototype motif search (no probe)

When you have a few audio examples of a motif but no MIDI:

```python
prototype_audios = [load("clip1.wav"), load("clip2.wav"), load("clip3.wav")]
motif_emb = torch.stack([
    enc.embed_audio(c.unsqueeze(0)) for c in prototype_audios
], dim=0).mean(dim=0)                # (1, 768)  — averaged prototype
motif_emb = F.normalize(motif_emb, dim=-1)
```

Then run the same sliding-window cosine-similarity pass over the target
track. This is the "few-shot CLaMP3" approach mentioned in §9 of the
audio plan, formalised: **no probe head training**, just nearest-prototype
matching in the shared space.

### 13.4 Workflow C: Hybrid (MIDI + audio prototypes)

When you have *both* the MIDI score and a couple of audio examples:

```python
e_midi   = enc.embed_symbolic(motif_patches)     # (1, 768)
e_audio  = enc.embed_audio(audio_examples)       # (N, 768)
motif_emb = F.normalize((e_midi.mean(0) + e_audio.mean(0)) / 2, dim=-1)
```

The MIDI anchor pins down the *musical identity*; the audio examples
encode *arrangement / orchestration cues*.  Averaging in the shared
space combines both signals.  In practice this is the strongest
single-pass approach we expect to see for leitmotif work.

### 13.5 Workflow D: Text-prompt motif search

Limited by CLaMP3's training corpus but worth trying for descriptive
queries:

```python
candidates = enc.embed_text([
    "heroic march in B♭ major with brass and percussion",
    "wistful piano melody in F♯ minor",
    "mysterious low-strings ostinato in D minor",
])
scores = (audio_embs @ candidates.T)              # (N_windows, len(candidates))
```

**Honest caveat.**  CLaMP3 was trained on commercial music with
captions; game-soundtrack-specific phrases ("Hyrule Castle theme")
won't match.  Generic musical-content prompts (key + instrumentation +
mood) work moderately well.  Always validate on a held-out track
before trusting it.

### 13.6 Workflow E: Cross-version alignment

You have two arrangements of the same soundtrack (orchestral vs piano,
or game OST vs concert performance).  Question: which moments
correspond?

```python
# 5-sec windows from both versions
e_v1 = embed_audio_sliding(version_1)            # (N1, 768)
e_v2 = embed_audio_sliding(version_2)            # (N2, 768)

# Pairwise similarity matrix
sim = e_v1 @ e_v2.T                              # (N1, N2)

# Dynamic time warping over the similarity matrix gives the alignment path
from librosa.sequence import dtw
_, path = dtw(C=1.0 - sim.numpy(), backtrack=True)
```

CLaMP3's arrangement-invariant embeddings make this dramatically more
robust than chroma-DTW alignment for orchestrational changes.

### 13.7 What this can NOT do

| Want | Why CLaMP3 isn't right | Alternative |
|---|---|---|
| Frame-accurate MIDI→audio alignment | 5-sec chunk ceiling | SyncToolbox, chroma-DTW |
| Note-level transcription | No per-pitch output head | Basic Pitch, OnsetsAndFrames |
| Detect specific named-but-unfamous theme by text | Captions in training don't include "Hyrule Castle" | Audio or MIDI prototype workflows above |

### 13.8 The "alignment" angle, briefly

In MIR, "alignment" usually means **frame-level correspondence between
two music representations** (e.g. score-to-audio synchronisation).
CLaMP3 is *not* a frame-level alignment tool — it sees in 5-sec
chunks.

But the workflows above all involve a coarser form of alignment:

- §13.2 / §13.3 align a motif template to **windows of a track**
  (5-sec granularity).
- §13.6 aligns full **arrangements** by computing chunk-level
  similarity then running DTW over the matrix.

For genuinely frame-accurate work (sub-second motif onset detection),
the pattern is:

1. **CLaMP3 finds the region** (which 5-sec window contains the motif)
2. **A frame-level encoder finishes the job** within that region — MERT
   (75 Hz) for tonal motifs, OMARRQ (25 Hz) for rhythmic ones — by
   running DTW or peak-picking on a finer feature

This hybrid is sketched in §9 (#1) of the original plan; with the
cross-modal API in place, the "find the region" step can now use a MIDI
template instead of requiring labelled audio.

---

## 14. Implementation status

This document was first drafted as a forward-looking plan.  Several
pieces have since been built and tested.  This section is the canonical
truth about what's actually shipped in the repository.

### 14.1 Cross-modal API (§13)

**Status: implemented and end-to-end tested with the real CLaMP3 checkpoint.**

| Method | File | Tested |
|---|---|---|
| `CLaMP3_Encoder.embed_audio(wavs)` | `marble/encoders/CLaMP3/model.py` | ✓ Returns L2-normalised `(B, 768)` |
| `CLaMP3_Encoder.embed_symbolic(patches)` | `marble/encoders/CLaMP3/model.py` | ✓ Returns L2-normalised `(B, 768)` |
| `CLaMP3_Encoder.embed_text(strs)` | `marble/encoders/CLaMP3/model.py` | ✓ Returns L2-normalised `(B, 768)` |

All three are inherited by `CLaMP3_Symbolic_Encoder`.  Cosine similarity
across modalities returns sane values in `[-1, +1]` — a confirmed
cross-modal verification was run with audio, MIDI, and text inputs.

### 14.2 Symbolic CLaMP3 encoder (§13)

**Status: implemented and end-to-end tested.**

```python
from marble.encoders.CLaMP3.model import CLaMP3_Symbolic_Encoder
enc = CLaMP3_Symbolic_Encoder().eval()
# forward(patches) returns tuple of 13 × (B, 1, H) — same contract as audio path
out = enc(patches_tensor)
```

Tested with the real CLaMP3 checkpoint on a synthetic 4-note test MIDI:
- 13 layer outputs, each `(B=1, T=1, H=768)`
- Layer norms increase as expected through the BERT stack
- Layers are differentiated (not collapsed)
- Forward time on CPU: ~0.2 s per item after weights are cached

The `CLaMP3_Encoder` (audio path) wrapper produces per-layer hidden
states for layer-probe analysis; the cross-modal helpers in §14.1 are
the separate route for shared-space embeddings.

### 14.3 MIDI ↔ MTF conversion (§13)

**Status: implemented and tested.**

```python
from marble.encoders.CLaMP3.midi_util import midi_to_mtf

# Adapted from upstream clamp3/preprocessing/midi/batch_midi2mtf.py
mtf_text = midi_to_mtf("path/to/file.mid", m3_compatible=True)
# Output: newline-separated MTF starting with "ticks_per_beat N"
```

This is the canonical text format that `M3Patchilizer.encode()`
expects.  Tested on synthetic MIDI; produces valid byte-level patches.

### 14.4 VGMIDI-TVar task (§12.1)

**Status: shipped — both audio and symbolic paths.**

| Path | Encoder | Layers | Config | Dataset class |
|---|---|---|---|---|
| Audio | `CLaMP3_Encoder` | 13 | `probe.CLaMP3-layers.VGMIDITVar.yaml` | `VGMIDITVarAudioAll` |
| Audio | `MERT_v1_95M_Encoder` | 13 | `probe.MERT-v1-95M-layers.VGMIDITVar.yaml` | `VGMIDITVarAudioAll` |
| Audio | `OMARRQ_Multifeature25hz_Encoder` | 24 | `probe.OMARRQ-multifeature25hz.VGMIDITVar.yaml` | `VGMIDITVarAudioAll` |
| Symbolic | `CLaMP3_Symbolic_Encoder` | 13 | `probe.CLaMP3-symbolic-layers.VGMIDITVar.yaml` | `VGMIDITVarSymbolicAll` |

Files:
```
marble/tasks/VGMIDITVar/
  __init__.py
  datamodule.py    # Audio + Symbolic Base / All / Dummy / Test classes
  probe.py         # Re-exports Covers80's CoverRetrievalTask
scripts/
  build_vgmiditvar_dataset.py    # MIDI zip → fluidsynth render → JSONL
```

All four sweeps are registered in `scripts/run_all_sweeps.py`.  Each
verified to parse via `cli.py --print_config` and appear in
`run_all_sweeps.py --dry-run --tasks VGMIDITVar`.

### 14.5 SuperMario Structure (§12.2)

**Status: planned but not implemented.**  Awaiting decision after
VGMIDI-TVar results confirm whether the MIDI-render pipeline is
viable for these encoders.

### 14.6 LeitmotifDetection clip classifier (§4)

**Status: skeleton shipped; full pipeline not built.**

The base scaffold lives in `marble/tasks/LeitmotifDetection/`:
- `datamodule.py` — `_LeitmotifAudioBase` (JSONL-driven clip loader)
- `probe.py` — `ProbeLeitmotifTask` (with file-level aggregation)

What's still missing for an actual run:
- `configs/probe.CLaMP3-layers.LeitmotifDetection.yaml` (sketched in §4
  but not committed)
- Actual leitmotif clip JSONL with motif labels (user's data)
- `scripts/leitmotif_localise.py` (§5; sketched only)
- `scripts/leitmotif_eval.py` (§6; sketched only)

These are built when the user has actual motif-annotated data.

### 14.7 Star Wars (SWTC) integration

**Status: documented in [`leitmotif_swtc.md`](./leitmotif_swtc.md);
nothing implemented yet.**

The companion doc walks through ten workflows, the Path A
(CLaMP3 CLI) recipe, and a proposed `SWTCLeitmotif` MARBLE task design.
Recommended order:
1. Validate Workflow 1 via the CLI before any MARBLE code
2. Build the MARBLE task only if Workflow 1 produces meaningful
   detections

---

## 15. Quick reference: which doc covers what

| Topic | Doc |
|---|---|
| Methodology (data prep, splits, evaluation, sliding window) | This doc, §1–§10 |
| Encoder choice and trade-offs | This doc, §11 |
| Candidate datasets for the broader probe suite | This doc, §12 |
| Cross-modal CLaMP3 workflows | This doc, §13 |
| What's actually implemented | This doc, §14 |
| Star Wars / SWTC specifics | [`leitmotif_swtc.md`](./leitmotif_swtc.md) |
| The 67 themes and Lehman's catalogue | [`leitmotif_swtc.md`](./leitmotif_swtc.md), §1 |
| Path A: CLaMP3 CLI workflow (no MARBLE) | [`leitmotif_swtc.md`](./leitmotif_swtc.md), §4 |
| Path B: MARBLE `SWTCLeitmotif` task design | [`leitmotif_swtc.md`](./leitmotif_swtc.md), §5 |
| 10 SWTC-specific workflows (detection, family map, etc.) | [`leitmotif_swtc.md`](./leitmotif_swtc.md), §6 |
| Scientific questions worth answering | [`leitmotif_swtc.md`](./leitmotif_swtc.md), §8 |

