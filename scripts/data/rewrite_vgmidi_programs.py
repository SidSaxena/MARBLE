#!/usr/bin/env python3
"""
scripts/data/rewrite_vgmidi_programs.py
───────────────────────────────────────
Rewrite the GM program of every VGMIDI-TVar MIDI based on its variation
index, producing a cross-instrument theme/variation dataset for the
"leitmotif" benchmark.

The source dataset is piano-only (every track on GM Program 0). To
test whether an encoder is *cross-instrument* invariant — the real
leitmotif scenario, where a theme stated on piano recurs on French
horn or strings — we rewrite each MIDI's Program Change events so
that the theme stays on piano and each variation lands on a different
instrument.

Schedule (idx → GM program):
    idx 0 (theme) → 0   Acoustic Grand Piano
    idx 1         → 48  String Ensemble 1
    idx 2         → 60  French Horn
    idx 3         → 73  Flute
    idx 4         → 56  Trumpet
    idx ≥ 5       → cycle (idx % 5)

Rewrite policy (in-place, not strip-and-insert):
    1. For every track, for every existing program_change message on a
       non-drum channel, rewrite its `program` field to the schedule's
       target for this file. Mid-piece program-change events that
       existed in the source are preserved (their values are just
       updated). VGMIDI files are piano-only so this is rare in
       practice but the policy is robust.
    2. For any non-drum channel that fires a note_on BEFORE any
       program_change on that channel, insert a single
       program_change(program=target, time=0) at the head of the
       track. Guarantees the target instrument is selected before
       any note is played.
    3. Channel 9 (GM drum channel, 0-indexed) is left untouched —
       drum programs select kits, not pitched instruments.

Idempotency:
    The instrument schedule is hashed and written to
    `<dst-midi-dir>/instrument_map.json` on first run. Subsequent runs
    compare the hash; mismatch refuses to proceed without --force,
    preventing silent stale data when the schedule changes.

Usage:
    # Pilot (10 files), then verify
    uv run python scripts/data/rewrite_vgmidi_programs.py \\
        --src-midi-dir data/VGMIDITVar/midi \\
        --dst-midi-dir data/VGMIDITVar-leitmotif/midi \\
        --max-files 10
    uv run python scripts/data/rewrite_vgmidi_programs.py \\
        --dst-midi-dir data/VGMIDITVar-leitmotif/midi --verify

    # Full run
    uv run python scripts/data/rewrite_vgmidi_programs.py \\
        --src-midi-dir data/VGMIDITVar/midi \\
        --dst-midi-dir data/VGMIDITVar-leitmotif/midi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import mido

# Import filename parsing utility from the renderer so we agree on the
# canonical {piece_id}_{section}_{idx}.mid convention.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_vgmiditvar_dataset import _parse_filename  # noqa: E402

log = logging.getLogger(__name__)

# ── Instrument schedule (also written to instrument_map.json) ────────────────
# Keys are variation indices (idx in the filename); values are GM program
# numbers (0..127). idx≥len(SCHEDULE) cycles modulo len(SCHEDULE).
SCHEDULE: dict[int, int] = {
    0: 0,  # Acoustic Grand Piano (theme)
    1: 48,  # String Ensemble 1
    2: 60,  # French Horn
    3: 73,  # Flute
    4: 56,  # Trumpet
}
GM_DRUM_CHANNEL = 9  # 0-indexed; channel 10 in 1-indexed convention


def target_program_for_idx(idx: int) -> int:
    """Return the GM program number to assign for this variation index."""
    return SCHEDULE[idx % len(SCHEDULE)]


def schedule_hash() -> str:
    """Stable hash of the SCHEDULE dict — used for the idempotency guard."""
    payload = json.dumps(SCHEDULE, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ── Rewrite logic ────────────────────────────────────────────────────────────


def rewrite_midi(src: Path, dst: Path, target_program: int) -> None:
    """Read a MIDI from ``src``, rewrite program changes per policy, save to ``dst``."""
    mid = mido.MidiFile(str(src))

    for track in mid.tracks:
        # First pass: collect (channel, first_event_kind, first_event_idx)
        # to decide whether to insert a leading program_change.
        first_event_per_channel: dict[int, tuple[str, int]] = {}
        for i, msg in enumerate(track):
            ch = getattr(msg, "channel", None)
            if ch is None:
                continue
            if ch == GM_DRUM_CHANNEL:
                continue
            if ch in first_event_per_channel:
                continue
            if msg.type in ("note_on", "program_change"):
                first_event_per_channel[ch] = (msg.type, i)

        # Second pass: rewrite every existing program_change on non-drum
        # channels. Mid-piece program_change values get rewritten too —
        # the file is one homogeneous instrument throughout.
        for msg in track:
            if msg.type != "program_change":
                continue
            if getattr(msg, "channel", None) == GM_DRUM_CHANNEL:
                continue
            msg.program = target_program

        # Third pass: for each non-drum channel whose first event is a
        # note_on (not a program_change), prepend a program_change.
        # We insert in reverse channel order so list-indexed inserts at
        # position 0 don't shift each other.
        insertions: list[mido.Message] = []
        for ch, (kind, _idx) in first_event_per_channel.items():
            if kind == "note_on":
                insertions.append(
                    mido.Message(
                        "program_change",
                        program=target_program,
                        channel=ch,
                        time=0,
                    )
                )
        # Order doesn't matter semantically (all time=0) — just insert.
        for msg in insertions:
            track.insert(0, msg)

    dst.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(dst))


# ── Idempotency guard ────────────────────────────────────────────────────────


def read_or_init_instrument_map(dst_root: Path, force: bool) -> None:
    """Manage <dst_root>/instrument_map.json — refuse to proceed if it disagrees
    with the in-script SCHEDULE, unless --force."""
    map_path = dst_root / "instrument_map.json"
    current = {"hash": schedule_hash(), "schedule": SCHEDULE}
    if map_path.exists():
        try:
            prior = json.loads(map_path.read_text())
        except json.JSONDecodeError:
            prior = None
        if prior and prior.get("hash") != current["hash"]:
            if not force:
                log.error(
                    "Instrument schedule has CHANGED since the previous run.\n"
                    "  Previous: %s\n"
                    "  Current:  %s\n"
                    "Re-running would mix MIDIs from two different schedules.\n"
                    "Pass --force to proceed and overwrite. If you want both,"
                    " choose a different --dst-midi-dir.",
                    prior.get("schedule"),
                    SCHEDULE,
                )
                sys.exit(2)
    dst_root.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(current, indent=2) + "\n")


# ── Verification ─────────────────────────────────────────────────────────────


def verify_dir(dst_root: Path) -> int:
    """Re-open every rewritten MIDI, assert each non-drum track has at least
    one program_change at t=0 matching the expected target program."""
    failures = 0
    total = 0
    for split in ("train", "test"):
        d = dst_root / split
        if not d.exists():
            continue
        for midi_path in sorted(d.glob("*.mid")):
            total += 1
            parsed = _parse_filename(midi_path.stem)
            if parsed is None:
                log.warning("verify: unrecognised filename %s", midi_path.name)
                failures += 1
                continue
            expected = target_program_for_idx(parsed["idx"])
            try:
                mid = mido.MidiFile(str(midi_path))
            except Exception as e:
                log.warning("verify: cannot open %s: %s", midi_path.name, e)
                failures += 1
                continue
            # Track-level check: every non-drum channel that has a note_on
            # must have a program_change(t=0) matching expected, OR the
            # rewritten existing program_change events all match expected.
            ok = True
            for track in mid.tracks:
                channels_with_notes = {
                    msg.channel
                    for msg in track
                    if msg.type == "note_on"
                    and getattr(msg, "channel", None) is not None
                    and msg.channel != GM_DRUM_CHANNEL
                }
                if not channels_with_notes:
                    continue
                pcs = [m for m in track if m.type == "program_change"]
                pcs_match = all(
                    m.program == expected
                    for m in pcs
                    if getattr(m, "channel", None) != GM_DRUM_CHANNEL
                )
                if not pcs_match:
                    ok = False
                    break
                # Each non-drum channel with notes must have at least one
                # program_change setting it to `expected` (either an
                # original or the inserted leading one).
                for ch in channels_with_notes:
                    has_pc = any(
                        m.type == "program_change"
                        and getattr(m, "channel", None) == ch
                        and m.program == expected
                        for m in track
                    )
                    if not has_pc:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                log.warning(
                    "verify: %s (idx=%d, target=%d) is missing a matching program_change",
                    midi_path.name,
                    parsed["idx"],
                    expected,
                )
                failures += 1

    if total == 0:
        log.error("verify: no MIDIs found under %s", dst_root)
        return 1
    log.info(
        "verify: %d/%d MIDIs match the schedule (%d failures)",
        total - failures,
        total,
        failures,
    )
    return 0 if failures == 0 else 1


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--src-midi-dir",
        default="data/VGMIDITVar/midi",
        help="Source MIDI directory (with train/ and test/ subdirs). Default: %(default)s",
    )
    ap.add_argument(
        "--dst-midi-dir",
        default="data/VGMIDITVar-leitmotif/midi",
        help="Destination MIDI directory. Will be created. Default: %(default)s",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Pilot mode: process at most this many MIDIs total.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Override the instrument_map.json hash guard.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Skip rewriting; verify that each MIDI under --dst-midi-dir "
        "matches its scheduled target instrument. Exits 0 on success.",
    )
    args = ap.parse_args()

    dst_root = Path(args.dst_midi_dir)

    if args.verify:
        sys.exit(verify_dir(dst_root))

    read_or_init_instrument_map(dst_root, args.force)

    src_root = Path(args.src_midi_dir)
    if not src_root.exists():
        log.error("Source MIDI dir does not exist: %s", src_root)
        sys.exit(1)

    # Render plan preamble — make misconfiguration visible in seconds.
    log.info("")
    log.info("─" * 60)
    log.info("Rewrite plan")
    log.info("─" * 60)
    log.info("  src-midi-dir : %s", src_root)
    log.info("  dst-midi-dir : %s", dst_root)
    log.info("  schedule     : %s", SCHEDULE)
    if args.max_files:
        log.info("  max-files    : %d (pilot mode)", args.max_files)
    log.info("─" * 60)

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for split in ("train", "test"):
        d = src_root / split
        if not d.exists():
            log.warning("split missing in source: %s", d)
            continue
        midis = sorted(d.glob("*.mid"))
        for src in midis:
            if args.max_files is not None and n_done >= args.max_files:
                break
            parsed = _parse_filename(src.stem)
            if parsed is None:
                log.warning("skip unrecognised filename: %s", src.name)
                n_failed += 1
                continue
            target = target_program_for_idx(parsed["idx"])
            dst = dst_root / split / src.name
            if dst.exists():
                n_skipped += 1
                continue
            try:
                rewrite_midi(src, dst, target)
            except Exception as e:
                log.warning("rewrite failed for %s: %s", src.name, e)
                n_failed += 1
                continue
            n_done += 1
            if n_done % 500 == 0:
                log.info("  rewritten: %d", n_done)
        if args.max_files is not None and n_done >= args.max_files:
            break

    log.info("")
    log.info("=" * 60)
    log.info(" Rewrote %d MIDIs (%d skipped existing, %d failed)", n_done, n_skipped, n_failed)
    log.info(" Schedule hash: %s", schedule_hash())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
