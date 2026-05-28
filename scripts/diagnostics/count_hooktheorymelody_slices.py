"""Count training/val/test slice counts for HookTheoryMelody.

Used once to budget MuQ / MERT cache disk before running a sweep —
HookTheoryMelody is sliced into 15-second clips, so the total slice
count drives the cache size (slices × ~20 MB for MuQ, ~45 MB for
MERT at the time of writing). Re-run whenever the slice geometry
changes or the JSONL is regenerated.

Read-only inspection. Prints per-split slice counts and a total.

Run from the repo root:
    uv run python scripts/diagnostics/count_hooktheorymelody_slices.py
"""

from marble.tasks.HookTheoryMelody.datamodule import _HookTheoryMelodyDataset

total = 0
for split in ["train", "val", "test"]:
    ds = _HookTheoryMelodyDataset(
        sample_rate=24000,
        channels=1,
        clip_seconds=15.0,
        jsonl=f"data/HookTheory/HookTheory.{split}.jsonl",
        label_freq=25,
        audio_dir="data/HookTheory/audio",
        channel_mode="first",
        min_clip_ratio=0.5,
    )
    n = len(ds.index_map)
    print(f"  {split:5s}  {n:>7,} slices")
    total += n
print(f"  TOTAL  {total:>7,} slices")
# Rule of thumb (24 kHz, 15 s clips):
#   MuQ   cache size ≈ total × 20 MB
#   MERT  cache size ≈ total × 45 MB
