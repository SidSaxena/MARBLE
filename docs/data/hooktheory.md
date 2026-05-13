# HookTheory dataset setup

[m-a-p/HookTheory](https://huggingface.co/datasets/m-a-p/HookTheory)
provides crowd-annotated chord / melody / structure data for 26,175
songs (~15K unique YouTube IDs). Three MARBLE tasks consume it:

| Task | Audio source | Annotation schema | Sweep cost |
|---|---|---|---|
| **HookTheoryKey** | pre-segmented clips (~4 GB) | flat `{audio_path, label, ...}` JSONL | small |
| **HookTheoryStructure** | pre-segmented clips (same) | same flat schema | small |
| **HookTheoryMelody** | full-song audio (~104 GB) | rich annotation tree (beat-aligned notes) | bigger; full download required |

The dataset is **gated** — accept the terms at
https://huggingface.co/datasets/m-a-p/HookTheory before downloading.

---

## Key + Structure (the easy ones, ~4 GB)

Pre-segmented audio clips + per-task JSONLs come bundled in the HF repo.

### Download

```bash
uv run python scripts/data/download_hooktheory.py
```

Output:

```
data/HookTheory/
├── hooktheory_clips/<clip_id>.mp3    # ~17K MP3 clips
├── HookTheoryKey.{train,val,test}.jsonl
└── HookTheoryStructure.{train,val,test}.jsonl
```

Both JSONLs share the same flat schema:

```json
{"audio_path": "data/HookTheory/hooktheory_clips/abc.mp3",
 "ori_uid": "youtube_id_of_source",
 "label": "C major",          // Key: 24 classes (major + minor)
                              // Structure: ["intro","verse","chorus",...]
 "duration": 10.68,
 "sample_rate": 48000,
 "num_samples": 512640,
 "bit_depth": 16,
 "channels": 2}
```

`.mp3` is libsndfile-readable on every platform — no ffmpeg drama.

### Verification

```bash
uv run python scripts/verify/verify_hooktheory.py
```

---

## Melody (the hard one, ~104 GB)

Melody pitch detection needs the **full-song audio** because note
onsets/offsets are anchored to song-relative beat numbers — the
pre-segmented clips drop that frame of reference.

### Schema mismatch

The Melody datamodule expects a per-song record:

```json
{
  "youtube": {"id": "abc123", "url": "...", "duration": 225.36},
  "alignment": {"refined": {"beats": [0, 1, 2, ...],
                            "times": [51.85, 52.49, ...]}},
  "annotations": {
    "num_beats": 44,
    "melody": [{"onset": 1, "offset": 2, "octave": 0, "pitch_class": 11}, ...]
  },
  "split": "TRAIN" | "VALID" | "TEST"
}
```

This is the **`Hooktheory.json.gz`** schema from the upstream HF repo —
**not** the flat `HookTheoryKey.*.jsonl` format.

### Modal-recommended setup (one-shot)

The full audio is 104 GB — generally easier to keep on Modal than
ship to a laptop.

```bash
modal run modal_marble.py::setup_hooktheory_full
```

What it does (~30–60 min on Modal, mostly download bandwidth):

1. `snapshot_download` of m-a-p/HookTheory full (clips + 104 GB audio).
2. Extracts both `zips/hooktheory_clips/*.tar` and `zips/audio/*.tar`
   into `data/HookTheory/{hooktheory_clips,audio}/`.
3. Loads `Hooktheory.json.gz` (19 MB), groups by split, filters out
   entries with empty melody (~2,300) or missing YouTube id (~250),
   and entries whose `audio/<ytid>.mp3` doesn't exist on disk.
4. Writes
   `data/HookTheory/HookTheory.{train,val,test}.jsonl` to the volume.
5. Commits the volume.

Expected record counts: ~19,142 train / ~1,947 val / ~2,496 test
(out of 26,175 raw entries).

### Local build (if you have the disk)

If you've manually downloaded the full audio with
`download.py HookTheory` (modified to keep `zips/audio/*`), you can
run the builder directly:

```bash
uv run python scripts/data/build_hooktheory_melody_jsonl.py \
    --audio-dir data/HookTheory/audio \
    --filter-by-audio \
    --out-dir data/HookTheory
```

Auto-downloads `Hooktheory.json.gz` from HF if not present.

---

## Sweep configs

| Task | MERT | CLaMP3 | OMARRQ |
|---|:---:|:---:|:---:|
| HookTheoryKey | ✓ | ✓ | ✓ |
| HookTheoryStructure | ✓ | ✓ | ✓ |
| HookTheoryMelody | ✓ | ✗ (no config) | ✓ |

CLaMP3 doesn't have a HookTheoryMelody config because CLaMP3's
variable token rate doesn't align cleanly with frame-level melody
labels.

Per-encoder configs live at:

- `configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml`
- `configs/probe.CLaMP3-layers.HookTheoryKey.yaml`
- `configs/probe.OMARRQ-multifeature25hz.HookTheoryKey.yaml`
- (similarly for Structure and Melody)
