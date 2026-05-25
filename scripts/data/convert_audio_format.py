#!/usr/bin/env python3
"""
scripts/data/convert_audio_format.py
────────────────────────────────────
Universal audio-format converter for MARBLE datasets.

Walks a source directory tree, runs ``ffmpeg`` per file with optional
resample / channel-downmix / bit-depth changes, and optionally rewrites
matching ``audio_path`` fields in one or more JSONL files to the new
extension. Per-file parallelism, idempotent, atomic.

Replaces two older single-purpose converters:
  - ``scripts/data/convert_audio_to_wav.py`` (HookTheory MP3 → 24 kHz mono WAV)
  - ``scripts/data/convert_shs100k_to_flac.py`` (SHS100K .m4a → .flac + JSONL rewrite)

Supported output formats
────────────────────────
  - ``wav``  — PCM (lossless, uncompressed). Smallest decode overhead.
  - ``flac`` — Lossless compression. ~40-55% of WAV size.
  - ``mp3``  — Lossy. Use only for archival / transfer, not encoder input.
  - ``ogg``  — Lossy Vorbis. Same warning as MP3.

Input formats: anything ffmpeg can decode (wav, flac, mp3, m4a, ogg,
opus, aiff, …). Pass ``--input-ext`` to filter; default is to autodetect
common audio extensions inside ``--src``.

Use-case recipes
────────────────
1. HookTheory MP3 corpus → 24 kHz mono WAV (zero-decode hot path)::

    uv run python scripts/data/convert_audio_format.py \\
        --src data/HookTheory/audio --dst data/HookTheory/audio_wav \\
        --input-ext .mp3 --to wav --sample-rate 24000 --channels 1

2. SHS100K m4a → FLAC in-place + rewrite JSONL (libsndfile compat)::

    uv run python scripts/data/convert_audio_format.py \\
        --src data/SHS100K/audio --in-place \\
        --input-ext .m4a --to flac \\
        --jsonl data/SHS100K/SHS100K.train.jsonl,data/SHS100K/SHS100K.val.jsonl

3. VGMIDITVar WAV → FLAC in-place (lossless ~55% savings)::

    uv run python scripts/data/convert_audio_format.py \\
        --src data/VGMIDITVar/audio --in-place \\
        --input-ext .wav --to flac \\
        --jsonl data/VGMIDITVar/VGMIDITVar.jsonl

4. NSynth WAV → FLAC in-place (lossless ~40% savings)::

    uv run python scripts/data/convert_audio_format.py \\
        --src data/NSynth --in-place \\
        --input-ext .wav --to flac \\
        --jsonl data/NSynth/NSynth.train.jsonl,data/NSynth/NSynth.val.jsonl,data/NSynth/NSynth.test.jsonl

5. WAV → 320 kbps MP3 archive for transfer (lossy, do NOT use as encoder input)::

    uv run python scripts/data/convert_audio_format.py \\
        --src data/somecorpus --dst data/somecorpus-mp3 \\
        --input-ext .wav --to mp3 --mp3-bitrate 320

Idempotency
───────────
Skips any destination that already exists with non-zero size unless
``--force``. Each conversion writes to ``<dst>.tmp.<pid>`` first and
moves to ``<dst>`` on success — no half-written outputs on crash.

In-place mode
─────────────
When ``--in-place`` is set, the destination is the same dir as the
source. After successful conversion of ``foo.wav`` to ``foo.flac``,
the original ``foo.wav`` is deleted. Per-file peak disk overhead =
one in-flight conversion (~one input file's worth).

JSONL rewriting
───────────────
Each path in ``--jsonl`` (comma-separated) is loaded, every record's
``audio_path`` field whose extension matches the source extension is
swapped to the destination extension, and the file is rewritten
atomically (via ``<jsonl>.tmp``). Other fields and records without a
matching extension are passed through unchanged. Uses
``marble.utils.path_compat.load_jsonl`` so Windows-generated JSONLs
with backslash paths are normalised on the read side.

ONLY ``audio_path`` is rewritten — cached metadata fields
(``sample_rate``, ``num_samples``, ``duration``, ``channels``) are NOT
refreshed. If your JSONL carries those fields (HookTheory, NSynth,
VGMIDITVar, …) and you've changed sample rate / channels / bit depth
during conversion, re-run ``scripts/data/cache_audio_info_in_jsonl.py
--force`` afterwards.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

# Extensions we consider "audio" when --input-ext is not specified.
# Order doesn't matter — we only filter by membership.
_DEFAULT_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac", ".aiff")


# ── ffmpeg command construction ──────────────────────────────────────────────


def _build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    *,
    to_fmt: str,
    sample_rate: int | None,
    channels: int | None,
    bit_depth: int,
    flac_compression: int,
    mp3_bitrate: int,
    ogg_quality: int,
) -> list[str]:
    """Build the ffmpeg command for one conversion.

    Output codec / container is determined by ``to_fmt``; format-specific
    quality flags only apply to their corresponding ``to_fmt``.
    """
    cmd: list[str] = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",  # overwrite (we'll have already checked dst doesn't exist)
        "-i",
        str(src),
        # Drop any non-audio streams (SHS100K m4a often carries an album-art
        # video track; without -vn ffmpeg complains it can't write video to
        # an audio-only container). No-op on streams without video.
        "-vn",
        # Strip source metadata for deterministic, smaller outputs.
        "-map_metadata",
        "-1",
    ]
    # Optional resample / downmix — apply BEFORE codec flags so ffmpeg
    # encodes the resampled stream.
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    # Codec selection per output format. Always pass an explicit ``-f``
    # container so ffmpeg doesn't try to infer from the dst extension —
    # important because we write to ``<dst>.tmp.<pid>`` for atomicity.
    if to_fmt == "wav":
        # PCM int (16 / 24 / 32 bit). 16 is the default — sufficient for
        # all encoder inputs we care about, half the size of 24-bit.
        codec_map = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
        cmd += ["-c:a", codec_map[bit_depth], "-f", "wav"]
    elif to_fmt == "flac":
        cmd += ["-c:a", "flac", "-compression_level", str(flac_compression), "-f", "flac"]
    elif to_fmt == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", f"{mp3_bitrate}k", "-f", "mp3"]
    elif to_fmt == "ogg":
        # libvorbis -q:a [0..10]; 6 ≈ 192 kbps avg
        cmd += ["-c:a", "libvorbis", "-q:a", str(ogg_quality), "-f", "ogg"]
    else:
        raise ValueError(f"unsupported --to format: {to_fmt}")
    cmd.append(str(dst))
    return cmd


# ── per-file convert worker ──────────────────────────────────────────────────


def _convert_one(
    src: Path,
    dst: Path,
    *,
    to_fmt: str,
    sample_rate: int | None,
    channels: int | None,
    bit_depth: int,
    flac_compression: int,
    mp3_bitrate: int,
    ogg_quality: int,
    delete_source: bool,
    force: bool,
) -> tuple[Path, bool, str]:
    """Convert one file. Returns (src, ok, message).

    Atomic: writes to ``dst.tmp.<pid>`` then renames to ``dst``. On
    failure the temp file is removed and ``dst`` is never touched.

    Idempotent: if ``dst`` already exists with non-zero size, returns
    ``(src, True, "already")`` without invoking ffmpeg. If
    ``delete_source`` is set and ``src != dst``, the now-redundant
    source is removed (matches in-place resume semantics).
    """
    if not src.exists():
        return src, False, "src missing"
    if dst.exists() and dst.stat().st_size > 0 and not force:
        # Resume / idempotency path. Drop the stale source if in-place
        # delete is requested.
        if delete_source and src != dst:
            try:
                src.unlink()
            except OSError:
                pass
        return src, True, "already"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp.{os.getpid()}")
    cmd = _build_ffmpeg_cmd(
        src,
        tmp,
        to_fmt=to_fmt,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        flac_compression=flac_compression,
        mp3_bitrate=mp3_bitrate,
        ogg_quality=ogg_quality,
    )
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return src, False, "timeout"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return src, False, f"exception: {e}"

    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        msg = (r.stderr or "")[:160].strip() or f"rc={r.returncode}"
        return src, False, msg

    # Atomic rename. On Windows os.replace overwrites; same on POSIX.
    try:
        os.replace(tmp, dst)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        return src, False, f"rename: {e}"

    if delete_source and src != dst:
        try:
            src.unlink()
        except OSError as e:
            log.warning("converted %s but could not delete source: %s", src.name, e)
    return src, True, "ok"


# ── JSONL rewriting ──────────────────────────────────────────────────────────


def _rewrite_jsonl(jsonl_path: Path, old_ext: str, new_ext: str) -> tuple[int, int]:
    """Rewrite a JSONL file's ``audio_path`` field, swapping ``old_ext`` → ``new_ext``.

    Idempotent: records whose ``audio_path`` doesn't end with ``old_ext``
    are passed through unchanged. Atomic: writes to ``<jsonl>.tmp`` then
    renames.

    Uses ``marble.utils.path_compat.load_jsonl`` so Windows-generated
    JSONLs (with backslash paths) are normalised to forward slashes on
    read — preventing the recurring cross-OS portability bug.

    Returns ``(n_rewritten, n_total)``.
    """
    try:
        from marble.utils.path_compat import load_jsonl
    except ImportError:
        # Fallback if MARBLE isn't importable (e.g. running this script
        # from outside the venv). Still works, just doesn't get the
        # backslash-to-forward-slash normalisation.
        def load_jsonl(path):
            with open(path) as f:
                return [json.loads(line) for line in f if line.strip()]

    records = load_jsonl(jsonl_path)
    n_rewritten = 0
    for r in records:
        p = r.get("audio_path")
        if p and p.endswith(old_ext):
            r["audio_path"] = p[: -len(old_ext)] + new_ext
            n_rewritten += 1
    tmp = jsonl_path.with_name(jsonl_path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, jsonl_path)
    return n_rewritten, len(records)


# ── file discovery ───────────────────────────────────────────────────────────


def _discover_files(src_root: Path, input_ext: str | None) -> list[Path]:
    """Return all files under ``src_root`` matching ``input_ext``.

    If ``input_ext`` is None, returns any file whose extension is in
    :data:`_DEFAULT_AUDIO_EXTS`. The walk is recursive — nested split
    dirs (e.g. ``nsynth-train/audio/*.wav``) are picked up.
    """
    if not src_root.exists():
        raise FileNotFoundError(f"--src does not exist: {src_root}")
    if not src_root.is_dir():
        # Single-file convenience: --src points at one file. Just return it.
        return [src_root]
    ext = input_ext.lower() if input_ext else None
    out: list[Path] = []
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        if ext is not None:
            if p.suffix.lower() != ext:
                continue
        else:
            if p.suffix.lower() not in _DEFAULT_AUDIO_EXTS:
                continue
        out.append(p)
    return sorted(out)


def _get_nested(d: dict, dotted_key: str):
    """Resolve a dotted key like ``youtube.id`` from a nested dict.

    Returns ``None`` if any key along the path is missing.
    """
    cur = d
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _discover_files_from_jsonl(
    jsonl_paths: list[Path],
    src_root: Path,
    input_ext: str,
    id_key: str,
) -> list[Path]:
    """Resolve work list from JSONL records.

    For each record, extract ``id_key`` (dotted) and resolve to
    ``src_root / <id><input_ext>``. Deduplicates across all JSONLs.
    Skips records whose key is missing.
    """
    seen: set[str] = set()
    for jp in jsonl_paths:
        if not jp.exists():
            raise FileNotFoundError(f"--from-jsonl path does not exist: {jp}")
        with open(jp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                aid = _get_nested(json.loads(line), id_key)
                if aid is None:
                    continue
                seen.add(str(aid))
    return sorted(src_root / f"{aid}{input_ext}" for aid in seen)


def _resolve_dst_path(
    src: Path, src_root: Path, dst_root: Path, new_ext: str, in_place: bool
) -> Path:
    """Compute the destination path for one source file.

    In-place mode: dst is in the same directory, only the extension
    changes.

    Separate dst_root mode: preserve the source's relative path under
    src_root, swap to dst_root, swap extension.
    """
    if in_place:
        return src.with_suffix(new_ext)
    rel = src.relative_to(src_root)
    return (dst_root / rel).with_suffix(new_ext)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── input / output paths ──────────────────────────────────────────────
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source directory (recursive walk) OR a single audio file.",
    )
    dst_group = ap.add_mutually_exclusive_group(required=True)
    dst_group.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination directory. Mirrors --src's relative layout. "
        "Mutually exclusive with --in-place.",
    )
    dst_group.add_argument(
        "--in-place",
        action="store_true",
        help="Write outputs alongside sources (only the extension changes). "
        "Sources are DELETED after successful conversion unless --keep-source. "
        "Mutually exclusive with --dst.",
    )

    # ── format filters ────────────────────────────────────────────────────
    ap.add_argument(
        "--input-ext",
        default=None,
        help="Only convert files with this extension (e.g. '.mp3', '.m4a', "
        "'.wav'). Default: any of " + ", ".join(_DEFAULT_AUDIO_EXTS) + ".",
    )
    ap.add_argument(
        "--to",
        choices=("wav", "flac", "mp3", "ogg"),
        required=True,
        help="Output format. wav/flac are lossless; mp3/ogg are lossy "
        "(archive use only — do NOT feed mp3 to encoders for research).",
    )

    # ── optional resample / downmix ───────────────────────────────────────
    ap.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Resample to this rate (Hz). Default: preserve source rate.",
    )
    ap.add_argument(
        "--channels",
        type=int,
        default=None,
        choices=(1, 2),
        help="Downmix to this channel count. 1 = mono (channel-mixed), "
        "2 = stereo. Default: preserve source channels.",
    )

    # ── format-specific quality flags ─────────────────────────────────────
    ap.add_argument(
        "--bit-depth",
        type=int,
        choices=(16, 24, 32),
        default=16,
        help="PCM bit depth for --to wav. Default 16 (CD quality).",
    )
    ap.add_argument(
        "--flac-compression",
        type=int,
        default=5,
        help="FLAC compression level [0..12]. Default 5 (FLAC reference). "
        "Higher = smaller files at ~2-3x CPU cost.",
    )
    ap.add_argument(
        "--mp3-bitrate",
        type=int,
        default=192,
        help="MP3 bitrate in kbps. Default 192. Use 320 for archive quality.",
    )
    ap.add_argument(
        "--ogg-quality",
        type=int,
        default=6,
        choices=range(0, 11),
        metavar="[0..10]",
        help="OGG Vorbis quality [0..10]. Default 6 (~192 kbps avg).",
    )

    # ── JSONL-driven work list (smoke / subset mode) ──────────────────────
    ap.add_argument(
        "--from-jsonl",
        type=Path,
        action="append",
        default=None,
        help="Repeatable. Instead of walking --src, build the work list from "
        "audio IDs in each JSONL (resolved via --id-key + --input-ext). "
        "Each <id> resolves to <src>/<id><input-ext>. Requires --input-ext.",
    )
    ap.add_argument(
        "--id-key",
        default="youtube.id",
        help="Dotted JSONL key whose value is the audio file's stem "
        "(default: youtube.id). Only used with --from-jsonl.",
    )

    # ── JSONL rewriting (audio_path extension swap) ───────────────────────
    ap.add_argument(
        "--jsonl",
        default=None,
        help="Comma-separated JSONL paths whose ``audio_path`` fields should "
        "be rewritten to use the new extension after conversion completes.",
    )

    # ── parallelism + safety ──────────────────────────────────────────────
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel conversion workers. Default 4.",
    )
    ap.add_argument(
        "--keep-source",
        action="store_true",
        help="Don't delete source files. Default behaviour: with --in-place, "
        "delete sources after success (otherwise --in-place would leave "
        "duplicates).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files. Default: skip files "
        "where the destination already exists with non-zero size.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion plan and exit without invoking ffmpeg.",
    )
    ap.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Convert at most this many files (smoke / pilot mode).",
    )

    args = ap.parse_args()

    # ── Validate environment ───────────────────────────────────────────────
    if shutil.which("ffmpeg") is None and not args.dry_run:
        log.error(
            "ffmpeg not on PATH. Install ffmpeg (`brew install ffmpeg` / "
            "`winget install Gyan.FFmpeg` / `apt install ffmpeg`) or pass "
            "--dry-run."
        )
        return 1
    new_ext = "." + args.to
    if args.input_ext and not args.input_ext.startswith("."):
        args.input_ext = "." + args.input_ext

    # delete_source policy: --keep-source overrides; in-place defaults to
    # delete; separate-dst defaults to keep (don't touch user's source tree).
    if args.keep_source:
        delete_source = False
    elif args.in_place:
        delete_source = True
    else:
        delete_source = False

    # ── Discover files ────────────────────────────────────────────────────
    if args.from_jsonl:
        if not args.input_ext:
            log.error("--from-jsonl requires --input-ext (used to resolve <id><ext>)")
            return 1
        if not args.src.is_dir():
            log.error("--from-jsonl requires --src to be a directory")
            return 1
        files = _discover_files_from_jsonl(args.from_jsonl, args.src, args.input_ext, args.id_key)
    else:
        files = _discover_files(args.src, args.input_ext)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        log.warning("No matching files found under %s", args.src)
        return 0

    # ── Build src→dst map ─────────────────────────────────────────────────
    src_root = args.src if args.src.is_dir() else args.src.parent
    dst_root = src_root if args.in_place else args.dst
    if dst_root is None:
        log.error("--dst is required unless --in-place is set")
        return 1
    pairs = [
        (src, _resolve_dst_path(src, src_root, dst_root, new_ext, args.in_place)) for src in files
    ]

    # ── Render plan ───────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Audio format conversion plan")
    log.info("─" * 60)
    log.info("  --src              : %s", args.src)
    log.info(
        "  --dst              : %s%s",
        dst_root,
        " (in-place)" if args.in_place else "",
    )
    log.info("  --input-ext        : %s", args.input_ext or "<auto-detect>")
    log.info(
        "  --to               : %s (%s)",
        args.to,
        {
            "wav": f"{args.bit_depth}-bit PCM",
            "flac": f"compression level {args.flac_compression}",
            "mp3": f"{args.mp3_bitrate} kbps",
            "ogg": f"quality {args.ogg_quality}",
        }[args.to],
    )
    if args.sample_rate:
        log.info("  --sample-rate      : %d Hz", args.sample_rate)
    if args.channels:
        log.info(
            "  --channels         : %d (%s)",
            args.channels,
            "mono" if args.channels == 1 else "stereo",
        )
    log.info("  --workers          : %d", args.workers)
    log.info("  delete source      : %s", delete_source)
    log.info("  files to convert   : %d", len(pairs))
    log.info("─" * 60)

    if args.dry_run:
        for i, (src, dst) in enumerate(pairs[:10]):
            log.info("  [%d] %s → %s", i, src, dst)
        if len(pairs) > 10:
            log.info("  ... (+%d more)", len(pairs) - 10)
        return 0

    # ── Convert in parallel ───────────────────────────────────────────────
    n_ok = 0
    n_skipped = 0
    n_failed = 0
    failures: list[tuple[Path, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(
                _convert_one,
                src,
                dst,
                to_fmt=args.to,
                sample_rate=args.sample_rate,
                channels=args.channels,
                bit_depth=args.bit_depth,
                flac_compression=args.flac_compression,
                mp3_bitrate=args.mp3_bitrate,
                ogg_quality=args.ogg_quality,
                delete_source=delete_source,
                force=args.force,
            ): src
            for src, dst in pairs
        }
        for i, fut in enumerate(as_completed(futs), 1):
            src, ok, msg = fut.result()
            if ok:
                if msg == "already":
                    n_skipped += 1
                else:
                    n_ok += 1
            else:
                n_failed += 1
                failures.append((src, msg))
            if i % 100 == 0 or i == len(pairs):
                log.info(
                    "  [%5d/%5d] ok=%d  skipped=%d  failed=%d",
                    i,
                    len(pairs),
                    n_ok,
                    n_skipped,
                    n_failed,
                )

    log.info("")
    log.info("=" * 60)
    log.info(
        " Converted %d (+%d skipped, %d failed) of %d files",
        n_ok,
        n_skipped,
        n_failed,
        len(pairs),
    )
    if failures:
        log.warning(" First few failures:")
        for src, msg in failures[:5]:
            log.warning("   - %s: %s", src.name, msg)

    # ── Rewrite JSONL audio_path fields ───────────────────────────────────
    if args.jsonl and (n_ok > 0 or n_skipped > 0):
        jsonl_paths = [Path(p.strip()) for p in args.jsonl.split(",") if p.strip()]
        old_ext = args.input_ext or pairs[0][0].suffix
        log.info("")
        log.info(" Rewriting JSONL audio_path fields: %s → %s", old_ext, new_ext)
        for jp in jsonl_paths:
            if not jp.exists():
                log.warning("  ! %s does not exist; skipping", jp)
                continue
            n_rewritten, n_total = _rewrite_jsonl(jp, old_ext, new_ext)
            log.info(
                "  %s : %d/%d records rewritten",
                jp,
                n_rewritten,
                n_total,
            )
    log.info("=" * 60)

    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
