# SHS100K dataset setup

[SHS100K](https://github.com/NovaFrost/SHS100K2) is a YouTube-based
cover-retrieval benchmark — ~6,900 unique tracks, ~100 cover groups.
MARBLE uses the test split (zero-shot retrieval, `max_epochs=0`).

## Local acquisition

```bash
uv run python scripts/data/download_shs100k.py --browser firefox
```

YouTube cookies (extracted via yt-dlp from Firefox/Chrome/Edge) are
required because some videos are age-gated or region-locked.
`--cookies-file cookies.txt` works if you've already exported them.

Result:

```
data/SHS100K/
├── SHS100K.test.jsonl       # 6905 records: {audio_path, work_id, performance_id, ...}
└── audio/<youtube_id>.m4a   # 6905 files, ~21 GB total
```

## Why `.m4a` is a problem

torchaudio 2.7's default audio backends are:

| Platform | Backend | M4A (AAC) support |
|---|---|---|
| Linux | ffmpeg (if libs available), soundfile | ✓ via ffmpeg |
| Modal container | ffmpeg (apt-installed) | ✓ |
| macOS | soundfile only (unless ffmpeg ≤7 dylibs present) | ✗ |
| Windows | soundfile only (unless ffmpeg ≤7 DLLs present) | ✗ |

If torchaudio falls back to soundfile-only, you get
`LibsndfileError: Format not recognised` on the first batch.

See [local_sweeps.md](../local_sweeps.md#ffmpeg-on-windows) for the
ffmpeg install fix.

## Recommended: convert to FLAC

The most durable cross-platform solution — FLAC is lossless and
libsndfile-native, so it works on every platform with zero
ffmpeg-version drama.

```bash
uv run python scripts/data/convert_shs100k_to_flac.py
```

What it does:

- Reads `data/SHS100K/SHS100K.test.jsonl`.
- Converts each `<ytid>.m4a` → `<ytid>.flac` via parallel ffmpeg CLI.
- Refreshes `sample_rate / num_samples / duration` from `ffprobe`.
- Rewrites the JSONL in place to point at the FLAC files.
- Optionally deletes the originals (`--keep-originals` to keep both).

Disk: ~30 GB FLAC vs ~21 GB M4A (lossless is bigger).
Time: ~30–60 min at `--workers 8` on a modern CPU.

Skip if the source audio is moved to a different drive:

```bash
uv run python scripts/data/convert_shs100k_to_flac.py \
    --audio-dir "D:/datasets/SHS100K" \
    --out-dir "D:/datasets/SHS100K"
```

## Modal volume layout

After running the Modal flatten + setup:

```
marble-data:/SHS100K/
├── SHS100K.test.jsonl         # paths reference data/SHS100K/audio/<ytid>.m4a (Modal mount)
└── audio/<youtube_id>.m4a     # 6905 files (flatten previously corrected nested layout)
```

To rebuild the JSONL on Modal (drops missing/corrupt entries, repoints
to mount path):

```bash
# One-time: upload the local cleaned JSONL to the volume
modal volume put marble-data data/SHS100K/SHS100K.test.jsonl SHS100K/SHS100K.test.jsonl

# Verify + drop bad entries (uses scripts/verify/verify_shs100k.py --rewrite)
modal run modal_marble.py::setup_shs100k_jsonl
```

## Verification

```bash
# Local — entry/size + optional ffprobe + optional torchaudio decode
uv run python scripts/verify/verify_shs100k.py
uv run python scripts/verify/verify_shs100k.py --ffprobe --torchaudio

# Drop bad entries in-place
uv run python scripts/verify/verify_shs100k.py --rewrite
```

## Splits + schema

Only the test split is used (the upstream train split has licensing
issues for the audio). One record per audio file:

```json
{
  "audio_path": "data/SHS100K/audio/abc123.m4a",
  "work_id": 303,
  "performance_id": 303,
  "title": "All My Loving",
  "artist": "The Beatles",
  "youtube_id": "abc123",
  "sample_rate": 44100,
  "num_samples": 5638316,
  "channels": 2,
  "duration": 127.853
}
```

Retrieval task: for each `audio_path` in the test set, retrieve the
k-nearest neighbors and check whether top results share its `work_id`.
Metrics: MAP, MRR.

## Sweep configs

Per-encoder configs already exist:

- [configs/probe.MERT-v1-95M-layers.SHS100K.yaml](../../configs/probe.MERT-v1-95M-layers.SHS100K.yaml)
- [configs/probe.CLaMP3-layers.SHS100K.yaml](../../configs/probe.CLaMP3-layers.SHS100K.yaml)
- [configs/probe.OMARRQ-multifeature25hz.SHS100K.yaml](../../configs/probe.OMARRQ-multifeature25hz.SHS100K.yaml)

All zero-shot (`max_epochs=0`), test-only.
