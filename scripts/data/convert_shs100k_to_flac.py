#!/usr/bin/env python3
"""
scripts/data/convert_shs100k_to_flac.py
───────────────────────────────────────
Convert SHS100K audio (downloaded as m4a/webm/opus by ``download_shs100k.py``
or pulled from the ``marble-data`` Modal volume) into mono 24 kHz FLAC,
then rewrite the JSONL so ``audio_path`` and metadata point at the new files.

Why
---
* The m4a/webm files from yt-dlp are stereo at whatever sample rate the
  YouTube encoder picked (often 44.1 kHz).  Every probe encoder we use
  resamples to ≤24 kHz mono at run time — decoding the source via the m4a
  codec on Windows is slow and occasionally flaky.
* FLAC is lossless, decodes uniformly on every platform via soundfile, and
  at 24 kHz mono is ~40 % larger than the m4a source — a worthwhile trade
  for SHS100K-sized sweeps.

Pipeline
--------
1. Walk ``--audio-dir`` for files matching the JSONL's listed ``audio_path``.
2. For each file, run ffmpeg ``-ac 1 -ar 24000 -c:a flac -compression_level 5``
   to write a sibling ``.flac``.
3. ffprobe the new file, then rewrite the JSONL with updated
   ``audio_path`` / ``sample_rate`` / ``num_samples`` / ``channels`` /
   ``duration``.
4. Optionally delete the source (``--delete-source``); off by default.

Usage
-----
  python scripts/data/convert_shs100k_to_flac.py \\
      --jsonl data/SHS100K/SHS100K.test.jsonl \\
      --audio-dir data/SHS100K/audio \\
      --workers 12

  # Convert and free disk by removing the m4a sources once converted:
  python scripts/data/convert_shs100k_to_flac.py --delete-source ...

The script is idempotent: pre-existing .flac files are reused (their
metadata is re-probed so the JSONL is always rebuilt from disk truth).
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

_SOURCE_EXTS = {".m4a", ".webm", ".opus", ".ogg", ".mp3", ".wav"}


def _ffprobe(path: Path, retries: int = 3, retry_sleep: float = 0.3) -> tuple[int, int, int] | None:
    """Probe with retries to absorb Windows FS-flush races after ffmpeg writes."""
    import time as _time

    for attempt in range(retries):
        info = _ffprobe_once(path)
        if info is not None:
            return info
        if attempt < retries - 1:
            _time.sleep(retry_sleep)
    return None


def _ffprobe_once(path: Path) -> tuple[int, int, int] | None:
    """Return (sample_rate, num_frames, channels), or None on failure."""
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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    sr = 0
    ch = 1
    dur_s: str | None = None
    for s in data.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        try:
            sr = int(s["sample_rate"])
        except (KeyError, ValueError, TypeError):
            continue
        ch = int(s.get("channels") or 1)
        dur_s = s.get("duration")
        break
    if sr == 0:
        return None
    dur = 0.0
    for raw in [dur_s, data.get("format", {}).get("duration")]:
        if raw is None:
            continue
        try:
            dur = float(raw)
            if dur > 0:
                break
        except (ValueError, TypeError):
            continue
    if dur <= 0:
        return None
    return sr, int(round(sr * dur)), ch


def _convert_one(
    src: Path, target_sr: int, delete_source: bool
) -> tuple[Path, Path | None, tuple[int, int, int] | None, str | None]:
    """Convert one file → sibling .flac. Returns (src, dst, probe_info, err)."""
    dst = src.with_suffix(".flac")

    # Idempotent: keep existing flac if it probes ok.
    if dst.exists() and dst.stat().st_size > 1024:
        info = _ffprobe(dst)
        if info is not None:
            if delete_source and src.exists() and src.suffix.lower() != ".flac":
                try:
                    src.unlink()
                except OSError:
                    pass
            return src, dst, info, None

    # In-process decode via PyAV (libavcodec bound in-proc) + encode flac via
    # soundfile.  We deliberately avoid spawning a separate ffmpeg.exe per file
    # because on Windows the desktop heap / session handle table is exhausted
    # after a few hundred spawns and subsequent CreateProcess calls fail with
    # STATUS_DLL_INIT_FAILED (rc=0xC0000142 = 3221225794).
    import av  # noqa: PLC0415 — lazy import so the script imports without PyAV
    import av.audio.resampler  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    tmp = dst.with_suffix(".flac.part")
    try:
        with av.open(str(src)) as container:
            astreams = [s for s in container.streams if s.type == "audio"]
            if not astreams:
                return src, None, None, "no audio stream"
            stream = astreams[0]
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=target_sr
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray())
            for out in resampler.resample(None):  # flush
                chunks.append(out.to_ndarray())
        if not chunks:
            return src, None, None, "decoded zero frames"
        # Each chunk shape: (1, n) for mono s16; concat along sample axis.
        arr = np.concatenate(chunks, axis=1).squeeze(0)
        # Defensive: catch any AudioResampler pathology that returns a
        # multi-channel layout despite ``layout="mono"`` (we've never seen
        # this in practice but the assertion exits cleanly via the
        # exception handler below — the failed file is dropped from the
        # rewritten JSONL).
        assert arr.ndim == 1, f"expected mono after resample, got shape {arr.shape}"
        sf.write(str(tmp), arr, target_sr, subtype="PCM_16", format="FLAC")
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return src, None, None, f"av/sf error: {type(e).__name__}: {str(e)[:160]}"

    # Atomic rename so partial files never look like completed ones.
    try:
        if dst.exists():
            dst.unlink()
        tmp.rename(dst)
    except OSError as e:
        return src, None, None, f"rename failed: {e}"

    info = (target_sr, int(arr.shape[0]), 1)

    if delete_source:
        try:
            src.unlink()
        except OSError:
            pass

    return src, dst, info, None


def _candidate_source(audio_dir: Path, stem: str) -> Path | None:
    """Find the source file for a given yt-dlp stem (any supported ext)."""
    for ext in _SOURCE_EXTS:
        p = audio_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Also tolerate already-converted flac (idempotent reruns).
    p = audio_dir / f"{stem}.flac"
    if p.exists():
        return p
    return None


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--jsonl", required=True, help="Input JSONL (e.g. data/SHS100K/SHS100K.test.jsonl)"
    )
    ap.add_argument(
        "--audio-dir",
        default=None,
        help="Directory containing source audio. Default: dirname(jsonl)/audio",
    )
    ap.add_argument(
        "--target-sr", type=int, default=24000, help="Output flac sample rate (default 24000)"
    )
    ap.add_argument("--workers", type=int, default=12, help="Parallel ffmpeg workers (default 12)")
    ap.add_argument(
        "--delete-source", action="store_true", help="Remove m4a after successful convert"
    )
    ap.add_argument(
        "--out-jsonl", default=None, help="Output JSONL path (default: overwrite --jsonl)"
    )
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        log.error("ffmpeg/ffprobe not on PATH")
        return 1

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        log.error(f"JSONL not found: {jsonl_path}")
        return 1

    audio_dir = Path(args.audio_dir) if args.audio_dir else jsonl_path.parent / "audio"
    if not audio_dir.is_dir():
        log.error(f"Audio dir not found: {audio_dir}")
        return 1

    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Loaded {len(records):,} records from {jsonl_path}")

    # Resolve each record's source file (m4a/webm/etc.) by yt-dlp stem.
    work: list[tuple[dict, Path]] = []
    missing = 0
    for rec in records:
        stem = rec.get("youtube_id") or Path(rec["audio_path"]).stem
        src = _candidate_source(audio_dir, stem)
        if src is None:
            missing += 1
            continue
        work.append((rec, src))
    if missing:
        log.warning(f"{missing} records had no source file on disk; they will be dropped")

    log.info(
        f"Converting {len(work):,} files → {args.target_sr} Hz mono FLAC ({args.workers} workers)"
    )

    done = 0
    failures: list[tuple[Path, str]] = []
    new_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_convert_one, src, args.target_sr, args.delete_source): (rec, src)
            for (rec, src) in work
        }
        for fut in as_completed(futs):
            rec, src = futs[fut]
            try:
                _, dst, info, err = fut.result()
            except Exception as e:
                err, dst, info = str(e), None, None
            done += 1
            if dst is None or info is None:
                failures.append((src, err or "unknown error"))
                if len(failures) <= 20:
                    log.warning(f"  FAIL {src.name}: {(err or 'unknown')[:300]}")
            else:
                sr, n_frames, ch = info
                new = dict(rec)
                new["audio_path"] = dst.as_posix()
                new["sample_rate"] = sr
                new["num_samples"] = n_frames
                new["channels"] = ch
                new["duration"] = round(n_frames / sr, 3)
                new_records.append(new)
            if done % 200 == 0 or done == len(work):
                pct = 100 * done // max(1, len(work))
                log.info(
                    f"  [{pct:3d}%] {done:>5}/{len(work)}  ok={len(new_records)}  failed={len(failures)}"
                )

    new_records.sort(key=lambda r: (r.get("work_id", 0), r.get("performance_id", 0)))
    out = Path(args.out_jsonl) if args.out_jsonl else jsonl_path
    tmp = out.with_suffix(out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in new_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # ``os.replace`` is atomic on POSIX but can fail on Windows if the
    # destination is open in another process (e.g. a sweep reading the
    # JSONL concurrently). Retry once after a brief pause; if the second
    # attempt fails, leave the .tmp beside the original so the caller can
    # recover manually rather than ending up with a half-written file.
    import time as _time

    for attempt in range(2):
        try:
            os.replace(tmp, out)
            break
        except OSError as e:
            if attempt == 0:
                log.warning(f"os.replace failed ({e}); retrying in 0.5 s")
                _time.sleep(0.5)
            else:
                log.error(
                    f"os.replace failed twice ({e}). Tmp left at {tmp}; "
                    f"rename manually or re-run convert."
                )
                return 1

    log.info(f"Wrote {len(new_records):,} records → {out}")
    if failures:
        log.warning(f"{len(failures)} failures:")
        for p, err in failures[:10]:
            log.warning(f"  {p.name}: {err}")
        if len(failures) > 10:
            log.warning(f"  ... and {len(failures) - 10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
