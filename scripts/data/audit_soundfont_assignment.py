#!/usr/bin/env python3
"""
scripts/data/audit_soundfont_assignment.py
──────────────────────────────────────────
Reproduce the per-piece SoundFont rotation that
``build_vgmiditvar_dataset.py`` uses, so you can audit which renders came
from which SoundFont **after the fact**.

The render script's filename → SoundFont mapping is deterministic:

    idx = int(md5(stem)[:8], 16) % len(soundfonts)

Same input list, same order → same assignment.  Pass the same
``--soundfont`` list (in the same order) you gave the renderer and this
prints a histogram of which WAVs landed on which SoundFont, plus a few
example filenames you can A/B audibly.

Usage
─────
    uv run python scripts/data/audit_soundfont_assignment.py \\
        --audio-dir data/VGMIDITVar-multisf/audio \\
        --soundfont ~/sf2/FluidR3_GM.sf2 \\
        --soundfont ~/sf2/Shan_SGM_Pro_14.sf2

If the histogram is roughly 50/50 (with two SoundFonts) you know the
rotation worked.  If it's 100/0, the second ``--soundfont`` flag never
took effect when you ran the renderer.

To listen to the same MIDI rendered with each SoundFont, use the
``--side-by-side`` flag to re-render one MIDI individually.
"""

import argparse
import shutil
import subprocess
from hashlib import md5
from pathlib import Path


def assign(stem: str, soundfonts: list[str]) -> int:
    """Deterministic SoundFont index — must match build_vgmiditvar_dataset.py."""
    if len(soundfonts) == 1:
        return 0
    return int(md5(stem.encode("utf-8")).hexdigest()[:8], 16) % len(soundfonts)


def cmd_histogram(args: argparse.Namespace) -> None:
    audio = Path(args.audio_dir)
    wavs = sorted(audio.glob("*.wav"))
    if not wavs:
        print(f"No .wav files in {audio}")
        return

    buckets: dict[str, list[str]] = {sf: [] for sf in args.soundfont}
    for wav in wavs:
        idx = assign(wav.stem, args.soundfont)
        buckets[args.soundfont[idx]].append(wav.name)

    total = sum(len(v) for v in buckets.values())
    print(f"\nAssignment counts (of {total} WAVs in {audio}):\n")
    for sf, files in buckets.items():
        pct = 100 * len(files) / total if total else 0
        print(f"  {Path(sf).name:50s}  {len(files):>5}  ({pct:5.1f}%)")
        for ex in files[: args.show]:
            print(f"      {ex}")
        print()


def cmd_side_by_side(args: argparse.Namespace) -> None:
    """Render one MIDI with each SoundFont so you can A/B them audibly."""
    if shutil.which("fluidsynth") is None:
        print("fluidsynth not found on PATH")
        return
    midi = Path(args.midi)
    if not midi.exists():
        print(f"MIDI not found: {midi}")
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sf in args.soundfont:
        tag = Path(sf).stem
        out = out_dir / f"{midi.stem}__{tag}.wav"
        if out.exists() and out.stat().st_size > 1024:
            print(f"  skip (exists): {out}")
            continue
        cmd = [
            "fluidsynth",
            "-ni",
            "-g",
            "1.0",
            "-r",
            str(args.sample_rate),
            "-T",
            "wav",
            "-F",
            str(out),
            sf,
            str(midi),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            print(f"  FAILED for {tag}: {r.stderr[:200]}")
        else:
            print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd")

    h = sub.add_parser("histogram", help="(default) print SF→file assignment counts")
    h.add_argument("--audio-dir", default="data/VGMIDITVar-multisf/audio")
    h.add_argument(
        "--soundfont",
        action="append",
        required=True,
        help="Same list, same order as the renderer.",
    )
    h.add_argument("--show", type=int, default=3, help="Example filenames per SF (default: 3)")
    h.set_defaults(func=cmd_histogram)

    s = sub.add_parser(
        "side-by-side",
        help="Render one MIDI with each SoundFont individually for A/B listening.",
    )
    s.add_argument("--midi", required=True, help="Path to a single .mid file")
    s.add_argument("--soundfont", action="append", required=True)
    s.add_argument("--out-dir", default="data/VGMIDITVar-multisf/_audit_renders")
    s.add_argument("--sample-rate", type=int, default=44100)
    s.set_defaults(func=cmd_side_by_side)

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
