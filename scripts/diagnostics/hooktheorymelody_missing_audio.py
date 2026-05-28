"""List HookTheoryMelody YouTube IDs whose audio is missing locally.

Loads the upstream HookTheory metadata JSON (cached by HuggingFace
datasets) and compares against the local audio dir
``data/HookTheory/audio/``. Writes the set difference to
``data/HookTheory/missing_ytids.txt`` so the user can pass that
list to ``yt-dlp`` for a manual backfill pass.

Read-only inspection — does NOT download anything itself.

Run from the repo root:
    uv run python scripts/diagnostics/hooktheorymelody_missing_audio.py
    yt-dlp --batch-file data/HookTheory/missing_ytids.txt -o ...   # follow-up
"""

import gzip
import json
from pathlib import Path

# Locate the HuggingFace-cached HookTheory metadata snapshot.
src = Path.home() / ".cache/huggingface/hub/datasets--m-a-p--HookTheory/snapshots"
src = next(src.iterdir()) / "Hooktheory.json.gz"
data = json.loads(gzip.decompress(src.read_bytes()).decode())

# Collect every YouTube ID that has a melody annotation upstream.
all_ytids: set[str] = set()
for _hid, song in data.items():
    if not song.get("annotations", {}).get("melody"):
        continue
    yt = song.get("youtube", {}).get("id")
    if yt:
        all_ytids.add(yt)

# Compare against what's actually on disk.
have = {p.stem for p in Path("data/HookTheory/audio").glob("*.mp3")}
missing = all_ytids - have
print(f"{len(missing):,} ytids could be tried via yt-dlp")

# Write to a deterministic path the caller can feed to yt-dlp.
out = Path("data/HookTheory/missing_ytids.txt")
out.write_text("\n".join(sorted(missing)))
print(f"wrote {out}")
