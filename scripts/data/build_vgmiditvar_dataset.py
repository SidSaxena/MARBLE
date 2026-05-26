#!/usr/bin/env python3
"""
scripts/data/build_vgmiditvar_dataset.py
───────────────────────────────────
Render the Variation-Transformer VGMIDI-TVar (and optionally POP909-TVar)
MIDI dataset to audio and build a MARBLE-compatible JSONL.

Source
------
https://github.com/ChenyuGAO-CS/Variation-Transformer-Data-and-Model

Filename convention
-------------------
    {piece_id}_{section}_{idx}.mid
        idx == 0     →  theme
        idx >= 1     →  variation idx of that theme
The (piece_id, section) pair identifies a "work" for retrieval purposes —
the theme and all its variations share the same work_id.

Rendering pipeline
------------------
MIDI → fluidsynth + SoundFont → 44.1 kHz mono WAV → ffprobe metadata.

To avoid the encoder latching onto a single SoundFont's timbre signature
(a real concern when probing on synthesised audio), the script rotates
through a list of SoundFonts deterministically per piece — multiple
renders per theme give the encoder variety without breaking reproducibility.

Prerequisites
-------------
  fluidsynth   - audio synthesis from MIDI
    macOS    : brew install fluid-synth
    Linux    : sudo apt install fluidsynth
    Windows  : winget install FluidSynth.FluidSynth

  ffprobe      - metadata extraction (comes with ffmpeg, already required by
                 the SHS-100K downloader)

  One or more SoundFont (.sf2) files.  Defaults to a single SoundFont
  searched at $FLUIDR3_SF2 if no explicit list provided.  Recommended:

    - SGM-V2.01 (https://archive.org/details/SGM-V2.01)
    - GeneralUser GS (https://schristiancollins.com/generaluser.php)
    - FluidR3_GM (often packaged with fluidsynth)

Usage
-----
    # Render the test split of VGMIDI-TVar (default)
    python scripts/data/build_vgmiditvar_dataset.py \\
        --midi-zip data/source/VGMIDI-TVar.zip \\
        --audio-dir data/VGMIDITVar/audio \\
        --data-dir  data/VGMIDITVar \\
        --soundfont /path/to/SGM-V2.01.sf2

    # Use multiple SoundFonts rotated per-piece (recommended)
    python scripts/data/build_vgmiditvar_dataset.py \\
        --soundfont sf2/SGM-V2.01.sf2 \\
        --soundfont sf2/GeneralUser-GS.sf2 \\
        --soundfont sf2/FluidR3_GM.sf2

    # Rebuild JSONL from already-rendered audio (no re-rendering)
    python scripts/data/build_vgmiditvar_dataset.py --skip-render

    # Render POP909-TVar instead of VGMIDI-TVar
    python scripts/data/build_vgmiditvar_dataset.py \\
        --midi-zip data/source/POP909-TVar.zip \\
        --audio-dir data/POP909TVar/audio --data-dir data/POP909TVar
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import md5
from pathlib import Path

from tqdm.auto import tqdm

log = logging.getLogger(__name__)

# Filename pattern handles both POP909-TVar and VGMIDI-TVar conventions:
#   POP909-TVar:  052_A_0.mid                   piece=052, section=A, idx=0
#   VGMIDI-TVar:  e0_real_Other games_NES_Monster Party_Title Screen_A_1.mid
#                 piece="e0_real_Other games_NES_Monster Party_Title Screen", section=A, idx=1
#   VGMIDITVar-timbre (cross-product):
#                  052_A_0_p48.mid               piece=052, section=A, idx=0, program=48
# The trailing  _<section>_<idx>(_p<program>)?.mid  is what identifies the
# variation. The optional `_p<program>` suffix is the cross-product extension —
# when present, the program is written to the JSONL's `gm_program` field; when
# absent, the program is determined by other means (--instrument-map JSON
# loaded sidecar, or omitted entirely for the single-SF baseline variant).
# Everything before the structural tail is the piece identifier.
_FILENAME_RE = re.compile(
    r"^(?P<piece>.+)_(?P<section>[A-Z]+)_(?P<idx>\d+)(?:_p(?P<program>\d+))?\.mid$"
)


# ── MIDI extraction ──────────────────────────────────────────────────────────

# Characters illegal in Windows file names. We sanitize unconditionally
# (not just on Windows) so JSONLs are portable across platforms — a
# dataset built on Linux and one built on Windows produce identical
# filenames, audio paths, and work IDs.
_FS_ILLEGAL_RE = re.compile(r'[<>:"|?*\x00-\x1f]')


def _safe_filename(name: str) -> str:
    """Return a cross-platform-safe version of `name` (a filename component,
    not a full path). Replaces Windows-illegal chars with `-`, strips
    Windows-rejected trailing dots/spaces, collapses runs of `-`.

    The replacement char is `-` (NOT `_`) because the downstream
    _FILENAME_RE regex uses `_` as structural separator between piece /
    section / index — `?` → `_` could create ambiguity.
    """
    cleaned = _FS_ILLEGAL_RE.sub("-", name)
    # Windows rejects names with trailing dots or spaces
    cleaned = cleaned.rstrip(" .")
    # Collapse multiple consecutive dashes (cosmetic only)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned or "_unnamed"


def _extract_midi_zip(zip_path: Path, dest: Path) -> dict[str, list[Path]]:
    """Extract MIDI files from the source zip, return {split: [paths]}.

    The zip's internal layout is::

        29thSep2023_theme_var_extracted_for_training/
            train/*.mid
            test/*.mid

    Filenames are sanitized through `_safe_filename()` so Windows-illegal
    characters (`<>:"|?*`) don't break extraction. Sanitization is
    deterministic + applied on every platform for portable JSONLs.
    """
    dest.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[Path]] = {"train": [], "test": []}
    n_sanitized = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".mid"):
                continue
            if "__MACOSX" in name or "/._" in name:
                continue
            parts = Path(name).parts
            split = next((p for p in parts if p in ("train", "test")), None)
            if split is None:
                continue
            orig_name = Path(name).name
            safe_name = _safe_filename(orig_name)
            if safe_name != orig_name:
                n_sanitized += 1
            out_path = dest / split / safe_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if not out_path.exists():
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            splits[split].append(out_path)
    msg = f"  extracted: train={len(splits['train']):,}  test={len(splits['test']):,}  MIDI files"
    if n_sanitized:
        msg += f"  ({n_sanitized} filename(s) sanitized for cross-platform safety)"
    log.info(msg)
    return splits


# ── parsing filenames → work IDs ─────────────────────────────────────────────


def _parse_filename(stem: str) -> dict | None:
    """Parse  '052_A_0'  →  {'piece': '052', 'section': 'A', 'idx': 0}.

    Cross-product timbre stems (with ``_p<program>`` suffix) yield an
    additional ``program`` key:
      '052_A_0_p48' →  {'piece': '052', 'section': 'A', 'idx': 0, 'program': 48}

    The ``program`` key is None (not in dict) on stems without a suffix.
    """
    m = _FILENAME_RE.match(stem + ".mid")
    if not m:
        return None
    out = {
        "piece": m.group("piece"),
        "section": m.group("section"),
        "idx": int(m.group("idx")),
    }
    prog = m.group("program")
    if prog is not None:
        out["program"] = int(prog)
    return out


def _work_id_for(piece: str, section: str) -> int:
    """Stable 9-digit-ish integer derived from (piece, section).

    We avoid relying on Python's hash() because it's salted per process.
    md5 is plenty for the small label space (a few hundred works).
    """
    h = md5(f"{piece}_{section}".encode()).hexdigest()
    # Take first 8 hex chars → fits in int32
    return int(h[:8], 16)


# ── audio rendering ──────────────────────────────────────────────────────────


def _render_midi(
    midi_path: Path,
    audio_path: Path,
    soundfont: Path,
    sample_rate: int = 44100,
) -> bool:
    """Render one MIDI to mono 44.1 kHz WAV via fluidsynth.  Returns True on success."""
    if audio_path.exists() and audio_path.stat().st_size > 1024:
        return True
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "fluidsynth",
        "-ni",  # no interactive, immediate exit
        "-g",
        "1.0",  # gain
        "-r",
        str(sample_rate),
        "-T",
        "wav",  # write WAV
        "-F",
        str(audio_path),
        str(soundfont),
        str(midi_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        log.warning(f"  render failed: {midi_path.name}: {e}")
        return False
    if r.returncode != 0 or not audio_path.exists():
        log.warning(f"  render failed: {midi_path.name}: rc={r.returncode}  {r.stderr[:120]}")
        return False
    return True


def _convert_wav_to_flac(wav_path: Path, flac_path: Path, compression: int = 5) -> bool:
    """Convert WAV → FLAC via ffmpeg (lossless), then delete the WAV.

    Returns True on success, False on failure (in which case the WAV is
    LEFT INTACT — caller can decide to fall back to it or skip the record).
    On success, only the FLAC remains on disk; per-file peak disk during
    conversion is one extra ~1.5 MB while ffmpeg writes the FLAC.

    Used when --audio-format=flac to keep disk usage to ~half the WAV
    baseline without ever materialising the full WAV dataset.

    Args:
        wav_path: source WAV (will be deleted on success)
        flac_path: destination FLAC
        compression: ffmpeg FLAC compression level [0..12]. Default 5 is
            the FLAC reference default. Higher = smaller files at ~2-3x
            CPU cost; 8 typically saves ~5-10% more vs 5.
    """
    if flac_path.exists() and flac_path.stat().st_size > 512:
        # Already converted (resume safety) — drop the WAV if it's still here.
        if wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass
        return True
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav_path),
        "-c:a",
        "flac",
        "-compression_level",
        str(compression),
        str(flac_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        log.warning(f"  flac convert failed: {wav_path.name}: {e}")
        return False
    if r.returncode != 0 or not flac_path.exists() or flac_path.stat().st_size <= 512:
        log.warning(f"  flac convert failed: {wav_path.name}: rc={r.returncode}  {r.stderr[:120]}")
        # Leave the WAV behind so caller can either retry or fall back.
        return False
    # Conversion succeeded — drop the WAV to keep peak disk under control.
    try:
        wav_path.unlink()
    except OSError as e:
        log.warning(f"  could not delete WAV after FLAC convert: {wav_path.name}: {e}")
        # Not fatal — the FLAC is valid; we just leak the WAV on disk.
    return True


def _pick_soundfont(stem: str, soundfonts: list[Path]) -> Path:
    """Deterministic rotation: index = hash(stem) mod N."""
    if len(soundfonts) == 1:
        return soundfonts[0]
    return soundfonts[int(md5(stem.encode("utf-8")).hexdigest()[:8], 16) % len(soundfonts)]


def _stable_soundfont_id(sf_path: Path) -> int:
    """Order-independent deterministic int ID for a soundfont.

    Hashes the file STEM (basename without extension) so the same
    SF gets the same int across runs regardless of:
      - Where in the --soundfont CLI list it appeared
      - Which directory the .sf2 file lives in
      - Whether you pass --soundfont X Y or --soundfont Y X

    Required for ``--skip-render`` JSONL regeneration: if the user
    reordered the SF list between the original render and the JSONL
    rewrite, a positional index would silently mismatch the rendered
    audio's actual SF and corrupt cross-soundfont MAP labels. A
    content-based hash is stable across such reorderings.

    Returns a 32-bit unsigned int (md5[:8] hex → int).
    """
    return int(md5(sf_path.stem.encode("utf-8")).hexdigest()[:8], 16)


# ── metadata via ffprobe ─────────────────────────────────────────────────────


def _ffprobe_info(path: Path) -> tuple[int, int, int] | None:
    """Return (sample_rate, num_samples, channels), or None on failure."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    sr, channels, stream_dur = 0, 1, None
    for s in data.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        try:
            sr = int(s["sample_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        channels = int(s.get("channels") or 1)
        stream_dur = s.get("duration")
        break
    if sr == 0:
        return None
    dur = 0.0
    for raw in [stream_dur, data.get("format", {}).get("duration")]:
        if raw is None:
            continue
        try:
            dur = float(raw)
            if dur > 0:
                break
        except (ValueError, TypeError):
            pass
    if dur <= 0:
        return None
    return sr, int(sr * dur), channels


# ── main ──────────────────────────────────────────────────────────────────────


def _process_one(
    midi_path: Path,
    split: str,
    audio_dir: Path,
    soundfonts: list[Path],
    skip_render: bool,
    instrument_schedule: dict[int, int] | None = None,
    audio_format: str = "wav",
) -> dict | None:
    parsed = _parse_filename(midi_path.stem)
    if parsed is None:
        log.debug(f"  unexpected filename: {midi_path.name}")
        return None

    # Final on-disk audio extension depends on --audio-format. WAV is the
    # render output; FLAC is produced by a per-file convert+delete step
    # right after render (peak per-file disk stays at ~1 WAV worth, never
    # the whole dataset).
    final_ext = "flac" if audio_format == "flac" else "wav"
    wav_path = audio_dir / f"{midi_path.stem}.wav"
    final_path = audio_dir / f"{midi_path.stem}.{final_ext}"

    # Determine soundfont up front so the JSONL record can carry a stable
    # identifier (used by CoverRetrievalTask for cross-soundfont MAP on the
    # multisf variant). Even with --skip-render we know which SF was picked
    # because _pick_soundfont is a deterministic hash on the file stem.
    # Both names are initialised unconditionally to avoid UnboundLocalError
    # when soundfonts=[] reaches the render branch.
    sf: Path | None = None
    sf_id: int | None = None
    if soundfonts:
        sf = _pick_soundfont(midi_path.stem, soundfonts)
        # Content-based hash, NOT positional index — survives reordering of
        # --soundfont CLI args between the original render and a later
        # --skip-render JSONL regen. See _stable_soundfont_id docstring.
        sf_id = _stable_soundfont_id(sf)
    if not skip_render:
        if sf is None:  # no soundfonts configured (shouldn't happen in render mode)
            return None
        # Skip the whole render+convert if the FINAL audio is already on
        # disk. Without this guard, --audio-format=flac runs always re-
        # render every file: ``_render_midi`` looks at ``wav_path`` which
        # was deleted after the previous FLAC conversion, so the skip
        # check there always fails. Check final_path here instead so
        # already-converted FLACs are honoured. The 1024-byte threshold
        # matches ``_render_midi`` — guards against half-written stubs.
        if final_path.exists() and final_path.stat().st_size > 1024:
            pass  # fall through to ffprobe + record build
        else:
            # Fluidsynth only writes WAV — render WAV first, then optionally
            # transcode to FLAC + delete the WAV in the same worker call so
            # peak disk only ever holds one WAV per concurrent worker.
            ok = _render_midi(midi_path, wav_path, sf)
            if not ok:
                return None
            if final_ext == "flac":
                if not _convert_wav_to_flac(wav_path, final_path):
                    # Conversion failed; the WAV is still on disk per
                    # _convert_wav_to_flac's contract. Skip this record so
                    # we don't write a JSONL pointer to a file that may not
                    # match the final-ext promise. The orphan WAV is small;
                    # the user can rerun to retry.
                    return None
    elif not final_path.exists():
        # --skip-render path: the on-disk file MUST already be the
        # configured final_ext. If --audio-format=flac but only .wav
        # exists, the user needs to either convert manually first or
        # rerun without --skip-render. Don't silently substitute.
        return None

    info = _ffprobe_info(final_path)
    if info is None:
        log.warning(f"  ffprobe failed: {final_path.name}")
        return None
    sr, n_samples, channels = info

    record = {
        # Forward-slash separators for cross-OS portability. See
        # marble/utils/path_compat.py.
        "audio_path": final_path.as_posix(),
        "work_id": _work_id_for(parsed["piece"], parsed["section"]),
        "variation": parsed["idx"],
        "piece_id": parsed["piece"],
        "section": parsed["section"],
        "split": split,
        "sample_rate": sr,
        "num_samples": n_samples,
        "channels": channels,
        "duration": round(n_samples / sr, 3),
    }
    # gm_program selection priority:
    #   1. Filename-encoded program (e.g. ``foo_A_0_p48.mid``) — used by
    #      the cross-product timbre variant where each (MIDI, program) pair
    #      is its own file. This is the most explicit + reliable source.
    #   2. --instrument-map JSON via ``instrument_schedule`` — legacy
    #      leitmotif variant, program inferred from variation idx.
    #   3. Neither → no ``gm_program`` field (single-SF baseline).
    if "program" in parsed:
        record["gm_program"] = int(parsed["program"])
    elif instrument_schedule:
        idx = parsed["idx"]
        # Mirror the rewriter's cycling behaviour: idx ≥ len(schedule) cycles
        # modulo len(schedule).
        record["gm_program"] = instrument_schedule[idx % len(instrument_schedule)]
    # Only write soundfont_id in true multi-SF mode (where it's a
    # meaningful per-item axis). Single-SF datasets get no field —
    # CoverRetrievalTask's per-condition block then skips silently.
    if sf_id is not None and len(soundfonts) > 1:
        record["soundfont_id"] = sf_id
    return record


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description="Render VGMIDI-TVar MIDI to audio and build JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--midi-zip",
        default="data/source/VGMIDI-TVar.zip",
        help="Path to the source zip from the Variation-Transformer repo (default: %(default)s)",
    )
    ap.add_argument(
        "--audio-dir",
        default="data/VGMIDITVar/audio",
        help="Where rendered WAVs go (default: %(default)s)",
    )
    ap.add_argument(
        "--data-dir",
        default="data/VGMIDITVar",
        help="Where the JSONL is written (default: %(default)s)",
    )
    ap.add_argument(
        "--soundfont",
        action="append",
        default=[],
        help="Path to a .sf2 SoundFont.  Pass multiple --soundfont "
        "flags for SoundFont rotation (recommended).",
    )
    ap.add_argument(
        "--workers", type=int, default=4, help="Parallel rendering workers (default: 4)"
    )
    ap.add_argument(
        "--skip-render",
        action="store_true",
        help="Don't render; rebuild JSONL from existing WAVs only.",
    )
    ap.add_argument(
        "--skip-extract",
        action="store_true",
        help="Don't extract MIDIs from the zip; assume they already exist "
        "under --midi-extract-dir/{train,test}/. Use this when feeding "
        "pre-rewritten MIDIs (e.g. from rewrite_vgmidi_programs.py for "
        "the leitmotif variant). Implied by --skip-render.",
    )
    ap.add_argument(
        "--instrument-map",
        default=None,
        help="Optional path to an instrument_map.json (written by "
        "rewrite_vgmidi_programs.py). When set, each JSONL record gets "
        "a `gm_program` field derived from its variation index via the "
        "map's schedule. Used by the VGMIDITVar-leitmotif analysis.",
    )
    ap.add_argument(
        "--allow-overwrite-default-dir",
        action="store_true",
        help="Override the multi-SoundFont safety guard. By default, "
        "passing multiple --soundfont flags while writing into the "
        "default `data/VGMIDITVar/audio` directory is rejected (the "
        "no-overwrite policy would silently skip every file and leave "
        "the new SoundFonts unused). Set this flag to acknowledge that "
        "behaviour explicitly.",
    )
    ap.add_argument(
        "--midi-extract-dir",
        default=None,
        help="Where to extract the MIDI files (default: <data-dir>/midi)",
    )
    ap.add_argument(
        "--audio-format",
        choices=("wav", "flac"),
        default="wav",
        help="Output audio format. ``wav`` (default) is fluidsynth's native "
        "output. ``flac`` adds a per-file FLAC convert+delete step right "
        "after render — peak per-worker disk stays at one WAV. FLAC is "
        "lossless (bit-perfect decode); ~55%% smaller than WAV for "
        "MIDI-rendered audio, zero impact on encoder behaviour. The JSONL "
        "audio_path field uses the chosen extension. On --skip-render, the "
        "on-disk files must already match this extension (no auto-substitution).",
    )
    ap.add_argument(
        "--flac-compression",
        type=int,
        default=5,
        help="FLAC compression level [0..12] when --audio-format=flac. "
        "Default 5 is the FLAC reference default (~55%% of WAV size). "
        "Level 8 saves an additional 5-10%% at ~2-3x CPU cost. Ignored "
        "when --audio-format=wav.",
    )
    args = ap.parse_args()
    if args.audio_format == "flac" and shutil.which("ffmpeg") is None and not args.skip_render:
        log.error(
            "--audio-format=flac requires ffmpeg on PATH. Install ffmpeg "
            "(e.g. `winget install Gyan.FFmpeg` / `brew install ffmpeg` / "
            "`apt install ffmpeg`) or use --audio-format=wav."
        )
        sys.exit(1)
    if not (0 <= args.flac_compression <= 12):
        log.error("--flac-compression must be in [0, 12]; got %d", args.flac_compression)
        sys.exit(1)

    # ── Validate dependencies ────────────────────────────────────────────────
    if not args.skip_render:
        for tool in ("fluidsynth", "ffprobe"):
            if shutil.which(tool) is None:
                log.error(
                    f"{tool} not found on PATH.  See the script header for install instructions."
                )
                sys.exit(1)
        soundfonts = [Path(p) for p in args.soundfont]
        if not soundfonts:
            default_sf = Path(__import__("os").environ.get("FLUIDR3_SF2", ""))
            if default_sf.exists():
                soundfonts = [default_sf]
            else:
                log.error(
                    "No --soundfont provided and $FLUIDR3_SF2 is unset.  "
                    "Download SGM-V2.01 (free) and pass --soundfont path/to/SGM-V2.01.sf2"
                )
                sys.exit(1)
        for sf in soundfonts:
            if not sf.exists():
                log.error(f"SoundFont not found: {sf}")
                sys.exit(1)
        log.info(f"SoundFonts ({len(soundfonts)}): {[s.name for s in soundfonts]}")
    else:
        soundfonts = []

    # ── Extract MIDI from zip (or discover from pre-extracted dir) ───────────
    midi_extract_dir = Path(args.midi_extract_dir or Path(args.data_dir) / "midi")
    midi_zip = Path(args.midi_zip)
    # Skip the zip-extract step when:
    #   • --skip-extract was passed explicitly (e.g. feeding pre-rewritten
    #     MIDIs from rewrite_vgmidi_programs.py for the leitmotif variant), OR
    #   • --skip-render was passed (the existing legacy behaviour — no point
    #     extracting if we're not rendering).
    if args.skip_render or args.skip_extract:
        # Discover by walking the extract dir.
        splits = {"train": [], "test": []}
        for split in splits:
            d = midi_extract_dir / split
            if d.exists():
                splits[split] = sorted(d.glob("*.mid"))
        log.info(
            f"  found extracted MIDI: train={len(splits['train']):,}  test={len(splits['test']):,}"
        )
    else:
        if not midi_zip.exists():
            log.error(f"MIDI zip not found: {midi_zip}")
            sys.exit(1)
        log.info(f"Extracting {midi_zip} → {midi_extract_dir}")
        splits = _extract_midi_zip(midi_zip, midi_extract_dir)

    # ── Optional instrument schedule (for the leitmotif variant) ─────────────
    instrument_schedule: dict[int, int] | None = None
    if args.instrument_map:
        imap_path = Path(args.instrument_map)
        if not imap_path.exists():
            log.error(f"--instrument-map path does not exist: {imap_path}")
            sys.exit(1)
        try:
            imap_data = json.loads(imap_path.read_text())
            raw = imap_data.get("schedule", imap_data)
            instrument_schedule = {int(k): int(v) for k, v in raw.items()}
        except Exception as e:
            log.error(f"Failed to parse instrument map {imap_path}: {e}")
            sys.exit(1)
        log.info(f"  instrument map    : {imap_path} → {instrument_schedule}")

    audio_dir = Path(args.audio_dir)
    data_dir = Path(args.data_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Render plan + safety preamble ────────────────────────────────────────
    # Print a one-page summary BEFORE the slow render so misconfiguration is
    # caught in seconds rather than after waiting an hour for an "every file
    # already existed → all skipped" no-op run.
    n_existing = sum(
        1
        for ps in splits.values()
        for p in ps
        if (audio_dir / f"{p.stem}.{args.audio_format}").exists()
        and (audio_dir / f"{p.stem}.{args.audio_format}").stat().st_size > 1024
    )
    n_to_render = sum(len(v) for v in splits.values()) - n_existing
    log.info("")
    log.info("─" * 60)
    log.info("Render plan")
    log.info("─" * 60)
    log.info(f"  audio-dir         : {audio_dir}")
    log.info(f"  data-dir (jsonl)  : {data_dir / 'VGMIDITVar.jsonl'}")
    log.info(
        f"  audio-format      : {args.audio_format}"
        + (f" (compression={args.flac_compression})" if args.audio_format == "flac" else "")
    )
    log.info(f"  SoundFonts ({len(soundfonts)}): {[s.name for s in soundfonts]}")
    log.info(f"  MIDIs total       : {sum(len(v) for v in splits.values()):,}")
    log.info(f"  already rendered  : {n_existing:,} (will be skipped)")
    log.info(f"  to render now     : {n_to_render:,}")
    log.info("─" * 60)

    # Footgun guard — multiple --soundfont flags but the default single-SF
    # data-dir.  The script's no-overwrite policy means re-running with new
    # SoundFonts but the old data-dir silently reuses the previous renders
    # and produces a JSONL pointing at audio that doesn't reflect the new
    # SoundFonts at all.
    multi_sf_into_single_sf_dir = (
        len(soundfonts) >= 2
        and str(audio_dir).rstrip("/").endswith("VGMIDITVar/audio")
        and not args.allow_overwrite_default_dir
        and n_existing > 0
    )
    if multi_sf_into_single_sf_dir:
        log.error(
            "Aborting: you passed %d SoundFonts but %d WAVs already exist in the "
            "default single-SF directory `%s`. These would all be skipped, leaving "
            "your new SoundFonts unused.\n"
            "  Fix: render into a separate dir, e.g.\n"
            "    --data-dir data/VGMIDITVar-multisf --audio-dir data/VGMIDITVar-multisf/audio\n"
            "  Or, to deliberately overwrite-in-place: pass --allow-overwrite-default-dir "
            "AND first delete the existing WAVs.",
            len(soundfonts),
            n_existing,
            audio_dir,
        )
        sys.exit(2)

    # ── Render + parse ───────────────────────────────────────────────────────
    records: list[dict] = []
    total = sum(len(v) for v in splits.values())
    n_done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for split, paths in splits.items():
            for midi_path in paths:
                futs[
                    pool.submit(
                        _process_one,
                        midi_path,
                        split,
                        audio_dir,
                        soundfonts,
                        args.skip_render,
                        instrument_schedule,
                        args.audio_format,
                    )
                ] = (midi_path, split)
        # Show file-level progress with ETA. smoothing=0.05 averages over a
        # longer window so the rate is stable for the long-running render.
        pbar = tqdm(
            as_completed(futs),
            total=total,
            desc="render",
            unit="midi",
            smoothing=0.05,
        )
        for fut in pbar:
            n_done += 1
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            pbar.set_postfix(ok=len(records), refresh=False)

    if not records:
        log.error("No records — nothing to write")
        sys.exit(1)

    # Sort for deterministic JSONL output
    records.sort(key=lambda r: (r["split"], r["piece_id"], r["section"], r["variation"]))

    # ── Write JSONL (single file with split field) ───────────────────────────
    out = data_dir / "VGMIDITVar.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_works = len({(r["piece_id"], r["section"]) for r in records})
    n_train = sum(1 for r in records if r["split"] == "train")
    n_test = sum(1 for r in records if r["split"] == "test")
    log.info("")
    log.info("=" * 60)
    log.info(f" Wrote {len(records):,} entries → {out}")
    log.info(f"   train: {n_train:,}  test: {n_test:,}")
    log.info(f"   unique works (piece+section): {n_works:,}")
    log.info(f"   avg variations per work: {len(records) / n_works:.1f}")
    if not args.skip_render and n_existing > 0:
        log.info(f"   pre-existing WAVs reused : {n_existing:,}")
        log.info(f"   freshly rendered now     : {len(records) - n_existing:,}")
        if len(records) - n_existing == 0:
            log.warning(
                "   ⚠ NO new audio was rendered — every WAV already existed. "
                "If you intended new SoundFonts to take effect, render into a "
                "different --audio-dir or delete the existing WAVs first."
            )
    log.info("=" * 60)
    log.info("Next: configure a layer sweep with this JSONL")


if __name__ == "__main__":
    main()
