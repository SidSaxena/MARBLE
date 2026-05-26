#!/usr/bin/env python3
"""
scripts/data/rewrite_vgmidi_programs.py
───────────────────────────────────────
Rewrite the GM program of every VGMIDI-TVar MIDI, producing a
cross-instrument theme/variation dataset.

Two output modes:

1. **Schedule mode (default, ``--mode schedule``)**: each variation idx
   maps to one fixed program — instrument identity is confounded with
   variation index. Generic schedule-driven variant builder; not in
   the active sweep set as of audit cleanup.

   Schedule (idx → GM program):
       idx 0 (theme) → 0   Acoustic Grand Piano
       idx 1         → 48  String Ensemble 1
       idx 2         → 60  French Horn
       idx 3         → 73  Flute
       idx 4         → 56  Trumpet
       idx ≥ 5       → cycle (idx % 5)

2. **Cross-product mode (``--mode cross-product --programs ...``)**:
   each source MIDI gets ONE output per program in the user-supplied
   list. Output filenames carry a ``_p<program>`` suffix so the
   variation idx and program are recorded independently.

       src/train/piece_A_0.mid → dst/train/piece_A_0_p0.mid     (piano)
                              → dst/train/piece_A_0_p48.mid    (strings)
                              → dst/train/piece_A_0_p52.mid    (choir)
                              → ...

   Drives the ``VGMIDITVar-timbre`` variant. Lets the analysis
   disentangle three orthogonal axes: cross-instrument MAP (same
   work, same variation, different program), cross-variation MAP
   (same work, same program, different variation), and the combined
   axis (legacy "cross-instrument MAP"). Disk cost: N MIDI copies per
   source file (still ~50 MB per program × 5-8 programs ≈ negligible).

Rewrite policy (in-place, not strip-and-insert):
    1. For every track, for every existing program_change message on a
       non-drum channel, rewrite its `program` field to the target.
       Mid-piece program-change events that existed in the source are
       preserved (their values are just updated). VGMIDI files are
       piano-only so this is rare in practice but the policy is robust.
    2. For any non-drum channel that fires a note_on BEFORE any
       program_change on that channel, insert a single
       program_change(program=target, time=0) at the head of the track.
    3. Channel 9 (GM drum channel, 0-indexed) is left untouched.

Idempotency:
    Schedule mode writes ``<dst-midi-dir>/instrument_map.json`` with a
    hash of the SCHEDULE dict. Cross-product mode writes
    ``<dst-midi-dir>/programs.json`` with the program list (sorted).
    Both refuse to proceed on a hash/list mismatch unless ``--force``.

Usage:
    # SCHEDULE mode (schedule-driven variant): one output per MIDI, program by idx
    uv run python scripts/data/rewrite_vgmidi_programs.py \\
        --src-midi-dir data/VGMIDITVar/midi \\
        --dst-midi-dir data/VGMIDITVar-schedule/midi

    # CROSS-PRODUCT mode (timbre variant): N outputs per MIDI, one per program
    uv run python scripts/data/rewrite_vgmidi_programs.py \\
        --src-midi-dir data/VGMIDITVar/midi \\
        --dst-midi-dir data/VGMIDITVar-timbre/midi \\
        --mode cross-product \\
        --programs 0,24,48,52,60,73,80,89

    # Verify a previously-written dir
    uv run python scripts/data/rewrite_vgmidi_programs.py \\
        --dst-midi-dir data/VGMIDITVar-timbre/midi --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import mido
from tqdm.auto import tqdm

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


def read_or_init_programs_list(dst_root: Path, programs: list[int], force: bool) -> None:
    """Cross-product analogue of read_or_init_instrument_map.

    Writes ``<dst_root>/programs.json`` with the sorted program list. On
    re-runs, refuses to proceed if the list disagrees with the prior
    one unless ``--force`` — same intent as the schedule-mode guard
    (prevent silently mixing outputs from two different program sets).
    """
    progs = sorted(set(programs))
    map_path = dst_root / "programs.json"
    current = {"programs": progs}
    if map_path.exists():
        try:
            prior = json.loads(map_path.read_text())
        except json.JSONDecodeError:
            prior = None
        if prior and sorted(prior.get("programs", [])) != progs:
            if not force:
                log.error(
                    "Program list has CHANGED since the previous run.\n"
                    "  Previous: %s\n"
                    "  Current:  %s\n"
                    "Re-running would mix MIDIs from two different program sets.\n"
                    "Pass --force to proceed and overwrite. If you want both,"
                    " choose a different --dst-midi-dir.",
                    sorted(prior.get("programs", [])),
                    progs,
                )
                sys.exit(2)
    dst_root.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(current, indent=2) + "\n")


# Human-readable GM program names — used only for log clarity. Source:
# https://en.wikipedia.org/wiki/General_MIDI#Program_change_events
GM_NAMES = {
    0: "Acoustic Grand Piano",
    24: "Acoustic Guitar (Nylon)",
    48: "String Ensemble 1",
    52: "Choir Aahs",
    56: "Trumpet",
    60: "French Horn",
    73: "Flute",
    80: "Lead 1 (Square)",
    89: "Pad 2 (Warm)",
}


def parse_programs_arg(s: str) -> list[int]:
    """Parse a comma-separated GM program list like '0,48,73' into sorted ints.

    Validates each value is in [0, 127] (GM range). Duplicates are
    collapsed. Raises ValueError on malformed input or out-of-range
    programs.
    """
    raw = [p.strip() for p in s.split(",") if p.strip()]
    if not raw:
        raise ValueError("--programs must be a non-empty comma-separated list")
    progs: set[int] = set()
    for p in raw:
        try:
            n = int(p)
        except ValueError:
            raise ValueError(f"--programs: '{p}' is not an integer") from None
        if not (0 <= n <= 127):
            raise ValueError(f"--programs: {n} is out of GM range [0, 127]")
        progs.add(n)
    return sorted(progs)


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
            # Cross-product stems carry their explicit program in the
            # ``_p<program>`` suffix (parsed["program"]). Use that when
            # present; fall back to SCHEDULE for schedule-mode files.
            # Without this branch, verify_dir would compare cross-product
            # MIDIs against the wrong SCHEDULE-derived expected program.
            expected = parsed.get("program")
            if expected is None:
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
        default="data/VGMIDITVar-schedule/midi",
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
    ap.add_argument(
        "--mode",
        choices=("schedule", "cross-product"),
        default="schedule",
        help="schedule (default, legacy leitmotif): one output per MIDI, "
        "program by variation idx via SCHEDULE. cross-product (timbre "
        "variant): N outputs per MIDI, one per program in --programs.",
    )
    ap.add_argument(
        "--programs",
        type=str,
        default=None,
        help="Comma-separated GM program list (0..127), required for "
        "--mode cross-product. Example: '0,24,48,52,60,73,80,89' for "
        "Piano/Guitar/Strings/Choir/Horn/Flute/SquareLead/WarmPad.",
    )
    args = ap.parse_args()

    dst_root = Path(args.dst_midi_dir)

    if args.verify:
        sys.exit(verify_dir(dst_root))

    # Parse + validate cross-product args before any I/O.
    cross_product_programs: list[int] | None = None
    if args.mode == "cross-product":
        if not args.programs:
            log.error("--mode cross-product requires --programs (e.g. '0,48,73')")
            sys.exit(1)
        try:
            cross_product_programs = parse_programs_arg(args.programs)
        except ValueError as e:
            log.error("%s", e)
            sys.exit(1)
        read_or_init_programs_list(dst_root, cross_product_programs, args.force)
    else:
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
    log.info("  mode         : %s", args.mode)
    if cross_product_programs:
        named = ", ".join(f"{p} ({GM_NAMES.get(p, f'GM{p}')})" for p in cross_product_programs)
        log.info("  programs     : %s", named)
    else:
        log.info("  schedule     : %s", SCHEDULE)
    if args.max_files:
        log.info("  max-files    : %d (pilot mode)", args.max_files)
    log.info("─" * 60)

    # Precompute the full work plan so tqdm can show an accurate total +
    # ETA. The (src, prog, dst) tuples are also useful for the cross-product
    # case where one src expands into N outputs.
    plan: list[tuple[Path, int, Path]] = []
    for split in ("train", "test"):
        d = src_root / split
        if not d.exists():
            log.warning("split missing in source: %s", d)
            continue
        for src in sorted(d.glob("*.mid")):
            parsed = _parse_filename(src.stem)
            if parsed is None:
                log.warning("skip unrecognised filename: %s", src.name)
                continue
            # Schedule mode: one target per src by idx → original filename.
            # Cross-product mode: N targets per src, one per program →
            # filename with _p<program> suffix.
            if cross_product_programs:
                for p in cross_product_programs:
                    plan.append((src, p, dst_root / split / f"{src.stem}_p{p}.mid"))
            else:
                target = target_program_for_idx(parsed["idx"])
                plan.append((src, target, dst_root / split / src.name))
            if args.max_files is not None and len(plan) >= args.max_files:
                break
        if args.max_files is not None and len(plan) >= args.max_files:
            break

    n_done = 0
    n_skipped = 0
    n_failed = 0
    for src, prog, dst in tqdm(plan, desc="rewrite", unit="midi", smoothing=0.05):
        if dst.exists():
            n_skipped += 1
            continue
        try:
            rewrite_midi(src, dst, prog)
        except Exception as e:
            log.warning("rewrite failed for %s (prog=%d): %s", src.name, prog, e)
            n_failed += 1
            continue
        n_done += 1

    log.info("")
    log.info("=" * 60)
    log.info(" Rewrote %d MIDIs (%d skipped existing, %d failed)", n_done, n_skipped, n_failed)
    if cross_product_programs:
        log.info(" Programs: %s", cross_product_programs)
    else:
        log.info(" Schedule hash: %s", schedule_hash())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
