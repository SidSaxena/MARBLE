# Leitmotif Detection with CLaMP3

End-to-end plan for using CLaMP3 audio embeddings to detect and localise
leitmotifs (recurring musical themes) in video-game / film soundtracks.

This document covers everything from data preparation through embedding
extraction, probe training, inference at scale, evaluation, and known
failure modes.  Each section ends with the exact commands you'd run.

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
