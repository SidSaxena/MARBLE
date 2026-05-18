#!/usr/bin/env python3
"""
scripts/data/download_ninsheetmusic.py
──────────────────────────────────────
Playwright-based downloader for NinSheetMusic files.

NinSheetMusic (https://www.ninsheetmusic.org/) blocks every non-browser
HTTP request with a 403 — User-Agent / Referer / cookie spoofing all
fail. The site requires a real browser session (Cloudflare + custom
challenge). Playwright drives an actual Chromium that handles whatever
the site's challenge response expects, so downloads go through.

Reads a pieces.csv with columns ``piece_id, ..., url_pdf, url_mid,
url_mus`` (the format from the
``ShxLuo-Saxon/supermario-structure-annotation`` repo, but works for
any CSV with those columns). Saves files as ``<piece_id>.<ext>`` in
``--out-dir``, ready to feed straight into
``scripts/data/build_supermario_dataset.py --midi-source-dir``.

Prerequisites
-------------
  uv sync --extra ninsheetmusic       # installs playwright
  uv run playwright install chromium  # ~200 MB; one-time

Usage
-----
  # Pilot: 5 MIDIs
  uv run python scripts/data/download_ninsheetmusic.py \\
      --csv data/SuperMarioStructure/_upstream/supermario-structure-annotation/metadata/pieces.csv \\
      --out-dir data/SuperMarioStructure/midi_user \\
      --kind mid --max-pieces 5

  # Full MIDI download (554 files)
  uv run python scripts/data/download_ninsheetmusic.py \\
      --csv data/SuperMarioStructure/_upstream/supermario-structure-annotation/metadata/pieces.csv \\
      --out-dir data/SuperMarioStructure/midi_user \\
      --kind mid

  # All three kinds (mid + pdf + mus)
  uv run python scripts/data/download_ninsheetmusic.py \\
      --csv .../pieces.csv --out-dir ./nsm_files --kind all

  # Show the browser (debug mode)
  uv run python scripts/data/download_ninsheetmusic.py \\
      --csv .../pieces.csv --out-dir ./out --no-headless

Idempotency
-----------
Existing output files (size > 100 bytes) are skipped. Re-run after a
crash / interruption to resume.

Politeness
----------
Default 1.0 s delay between downloads. Default 1 worker. NinSheetMusic
is a free fan site — don't hammer it. The full 554-file MIDI download
takes ~15–30 min single-threaded.
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

_KIND_TO_EXT = {"mid": "mid", "pdf": "pdf", "mus": "mus"}
_KIND_TO_URL_COL = {"mid": "url_mid", "pdf": "url_pdf", "mus": "url_mus"}


# ── Pieces CSV parsing ────────────────────────────────────────────────────────


def _resolve_csv_path(raw: Path) -> Path:
    """Accept either a CSV file path or a directory containing pieces.csv.

    Friendlier than failing with a cryptic PermissionError when the user
    passes the parent dir by mistake.
    """
    if raw.is_dir():
        candidate = raw / "pieces.csv"
        if candidate.exists():
            log.info(f"  --csv pointed at a directory; resolved to: {candidate}")
            return candidate
        log.error(
            f"--csv is a directory ({raw}) and contains no pieces.csv. "
            f"Either pass the full path to the CSV file, or place it under "
            f"this directory as 'pieces.csv'."
        )
        sys.exit(1)
    if not raw.exists():
        log.error(
            f"--csv does not exist: {raw}. Expected a path to a CSV file with "
            f"columns: piece_id, url_pdf, url_mid, url_mus (the format from the "
            f"upstream supermario-structure-annotation repo)."
        )
        sys.exit(1)
    return raw


def _parse_pieces_csv(csv_path: Path, kinds: list[str]) -> list[tuple[str, dict[str, str]]]:
    """Parse pieces.csv → [(piece_id, {kind: url})].

    Skips rows missing piece_id or where all selected kind URLs are blank.
    """
    csv_path = _resolve_csv_path(csv_path)
    out: list[tuple[str, dict[str, str]]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("piece_id") or "").strip()
            if not pid:
                continue
            urls: dict[str, str] = {}
            for kind in kinds:
                col = _KIND_TO_URL_COL[kind]
                url = (row.get(col) or "").strip()
                if url:
                    urls[kind] = url
            if urls:
                out.append((pid, urls))
    return out


# ── Single-download helper (sync Playwright) ──────────────────────────────────


def _download_one(context, url: str, dst: Path, timeout_ms: int = 45_000) -> bool:
    """Download ``url`` via the Playwright browser context's HTTP client.

    Uses ``context.request.get`` rather than ``page.goto`` + expect_download
    because the latter trips Cloudflare's headless-navigation detection
    even with stealth tweaks — the navigation never resolves to a download.
    ``context.request.get`` inherits the browser context's cookies
    (including ``cf_clearance`` from the homepage pre-warm) and TLS
    fingerprint, so it appears to NSM as the same Chromium that just
    completed the JS challenge, but skips the navigation event entirely.

    Returns True on success.
    """
    if dst.exists() and dst.stat().st_size > 100:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = context.request.get(url, timeout=timeout_ms)
    except Exception as e:
        log.debug(f"    request error for {url}: {type(e).__name__}: {e}")
        return False
    if response.status != 200:
        log.debug(f"    HTTP {response.status} for {url}")
        return False
    body = response.body()
    if len(body) < 100:
        log.debug(f"    response body too small ({len(body)} bytes) for {url}")
        return False
    # Sanity: the first bytes should NOT be "<!DOCTYPE" or "<html" — that
    # would mean Cloudflare served the challenge page despite cookies.
    head = body[:16].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        log.debug(f"    received HTML (likely Cloudflare challenge) for {url}")
        return False
    dst.write_bytes(body)
    return dst.exists() and dst.stat().st_size > 100


# ── Main ──────────────────────────────────────────────────────────────────────


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
        "--csv",
        required=True,
        help="pieces.csv with piece_id + url_{mid,pdf,mus} columns",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory; files saved as <piece_id>.<ext>",
    )
    ap.add_argument(
        "--kind",
        choices=("mid", "pdf", "mus", "all"),
        default="mid",
        help="Which file types to download (default: mid only)",
    )
    ap.add_argument(
        "--max-pieces",
        type=int,
        default=None,
        help="Pilot mode: download at most this many pieces total",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between downloads (be kind to NSM — default 1.0)",
    )
    ap.add_argument(
        "--retry",
        type=int,
        default=2,
        help="Retries per file on failure (default: 2 → 3 total attempts)",
    )
    ap.add_argument(
        "--timeout-ms",
        type=int,
        default=45_000,
        help="Per-download timeout in milliseconds (default: 45 s)",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run with a hidden browser. Default is HEADED (visible browser) "
        "because NinSheetMusic's Cloudflare anti-bot reliably detects and "
        "blocks headless Chromium (we tested with both stock Playwright + "
        "stealth and patchright; both got stuck on 'Just a moment...'). "
        "Use --headless only if you have a persistent-context-dir from a "
        "prior headed run that already has cf_clearance.",
    )
    ap.add_argument(
        "--persistent-context-dir",
        default=None,
        help="Path to a persistent Chromium user-data-dir. Cookies + "
        "Cloudflare clearance survive across runs. First run: omit this "
        "(or use --no-headless) — Cloudflare may challenge. Subsequent "
        "runs: pass the same dir to reuse the cached session and avoid "
        "re-challenges. Recommended: '<data-dir>/.playwright_profile'.",
    )
    args = ap.parse_args()

    # Lazy import — playwright is an optional extra. Prefer `patchright`
    # (drop-in fork with better anti-detection) when available; fall back
    # to stock playwright otherwise.
    try:
        from patchright.sync_api import sync_playwright

        _backend = "patchright"
    except ImportError:
        try:
            from playwright.sync_api import sync_playwright

            _backend = "playwright"
        except ImportError:
            log.error(
                "Neither patchright nor playwright is installed. Run:\n"
                "    uv sync --extra ninsheetmusic\n"
                "    uv run playwright install chromium\n"
                "    # Optional but recommended:\n"
                "    uv pip install patchright && uv run patchright install chromium\n"
                "Then re-run this script."
            )
            sys.exit(1)
    log.info(f"Using browser backend: {_backend}")

    # Resolve kinds list
    if args.kind == "all":
        kinds = ["mid", "pdf", "mus"]
    else:
        kinds = [args.kind]

    # Parse CSV
    csv_path = Path(args.csv)
    pieces = _parse_pieces_csv(csv_path, kinds)
    if args.max_pieces:
        pieces = pieces[: args.max_pieces]
    if not pieces:
        log.error(
            f"No pieces parsed from {csv_path}. Check the CSV has piece_id and "
            f"url_{kinds[0]} columns."
        )
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan preamble — show counts BEFORE the slow loop
    n_targets = sum(len(urls) for _, urls in pieces)
    n_already = sum(
        1
        for pid, urls in pieces
        for kind in urls
        if (out_dir / f"{pid}.{_KIND_TO_EXT[kind]}").exists()
        and (out_dir / f"{pid}.{_KIND_TO_EXT[kind]}").stat().st_size > 100
    )
    n_todo = n_targets - n_already
    log.info("")
    log.info("─" * 60)
    log.info("Download plan")
    log.info("─" * 60)
    log.info(f"  csv          : {csv_path}")
    log.info(f"  out-dir      : {out_dir}")
    log.info(f"  pieces       : {len(pieces):,}")
    log.info(f"  kinds        : {', '.join(kinds)}")
    log.info(f"  total files  : {n_targets:,}  (already present: {n_already:,})")
    log.info(f"  to download  : {n_todo:,}")
    log.info(f"  delay        : {args.delay:.1f} s between downloads")
    log.info(f"  headless     : {args.headless}")
    log.info(f"  timeout      : {args.timeout_ms / 1000:.0f} s per file")
    log.info(f"  retries      : {args.retry}")
    log.info("─" * 60)

    if n_todo == 0:
        log.info("All files already present — nothing to do.")
        sys.exit(0)

    # Estimate wall-clock
    est_min = (n_todo * (args.delay + 3.0)) / 60.0
    log.info(f"  ~{est_min:.1f} min estimated wall-clock at {args.delay:.1f} s delay")
    log.info("")

    # SIGINT handler: graceful shutdown via flag
    _stop = {"set": False}

    def _sigint(_sig, _frame):
        log.warning("\nInterrupted — finishing current download then exiting.")
        _stop["set"] = True

    signal.signal(signal.SIGINT, _sigint)

    n_ok = 0
    n_fail = 0
    n_skipped = 0

    # Stealth: patch navigator.webdriver and a handful of other bot tells.
    # Defensive import — if playwright-stealth isn't installed we still
    # work, just with worse anti-detection.
    try:
        from playwright_stealth import Stealth

        stealth = Stealth()
        has_stealth = True
    except ImportError:
        stealth = None
        has_stealth = False
        log.warning(
            "playwright-stealth not installed — running without anti-detection "
            "patches. If downloads keep failing with HTML responses, install via "
            "`uv sync --extra ninsheetmusic` and retry."
        )

    with sync_playwright() as p:
        # Chromium launch flags that reduce automation fingerprint.
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ]
        ua = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        ctx_kwargs = {
            "accept_downloads": True,
            "user_agent": ua,
            "viewport": {"width": 1280, "height": 800},
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "extra_http_headers": {
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
                ),
            },
        }

        if args.persistent_context_dir:
            # Persistent context: cookies (cf_clearance especially) survive
            # across runs. Single-step launch returns a Context directly,
            # no separate browser handle. Use this after a successful
            # bootstrap run so Cloudflare's challenge doesn't re-fire each
            # invocation.
            from pathlib import Path as _P

            profile = _P(args.persistent_context_dir)
            profile.mkdir(parents=True, exist_ok=True)
            log.info(f"Using persistent context: {profile}")
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                headless=args.headless,
                args=launch_args,
                **ctx_kwargs,
            )
            browser = None
        else:
            browser = p.chromium.launch(headless=args.headless, args=launch_args)
            context = browser.new_context(**ctx_kwargs)

        if has_stealth:
            try:
                stealth.apply_stealth_sync(context)
            except Exception as e:
                log.warning(f"  stealth patch failed: {e}; continuing without")
        page = context.new_page()

        # Pre-warm: visit the homepage so Cloudflare's JS challenge fires
        # and stores cf_clearance in the context's cookie jar. All
        # subsequent context.request.get() calls inherit that cookie.
        try:
            log.info("Pre-warming browser session (visiting NSM homepage) …")
            page.goto("https://www.ninsheetmusic.org/", timeout=args.timeout_ms)
            page.wait_for_load_state("domcontentloaded", timeout=15_000)
            time.sleep(5.0)  # let Cloudflare's JS challenge finish
            # Try to read the title to confirm we actually got the homepage,
            # not the challenge page
            title = page.title()
            log.info(f"  page title: {title!r}")
            if "Just a moment" in title or "Attention" in title:
                log.warning("  Cloudflare challenge page still showing — extending wait …")
                time.sleep(10.0)
        except Exception as e:
            log.warning(f"  pre-warm warning ({type(e).__name__}: {e}); continuing")

        for i, (pid, urls) in enumerate(pieces, 1):
            if _stop["set"]:
                break
            for kind, url in urls.items():
                if _stop["set"]:
                    break
                ext = _KIND_TO_EXT[kind]
                dst = out_dir / f"{pid}.{ext}"
                if dst.exists() and dst.stat().st_size > 100:
                    n_skipped += 1
                    continue

                ok = False
                for attempt in range(args.retry + 1):
                    if _download_one(context, url, dst, timeout_ms=args.timeout_ms):
                        ok = True
                        break
                    if attempt < args.retry:
                        # Exponential-ish backoff
                        backoff = args.delay * (2**attempt)
                        log.debug(
                            f"    retry {attempt + 1}/{args.retry} for {pid}.{ext} after {backoff:.1f}s"
                        )
                        time.sleep(backoff)

                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    log.warning(f"  [{pid}.{ext}] FAILED after {args.retry + 1} attempts: {url}")

                time.sleep(args.delay)

            if i % 25 == 0 or i == len(pieces):
                log.info(
                    f"  [{i:>4}/{len(pieces)}]  ok={n_ok}  skipped={n_skipped}  failed={n_fail}"
                )

        # Persistent context owns its browser; just close the context.
        # Non-persistent path: close the browser (which closes the context too).
        if browser is None:
            context.close()
        else:
            browser.close()

    log.info("")
    log.info("=" * 60)
    log.info(f" Done — ok={n_ok}, skipped (already present)={n_skipped}, failed={n_fail}")
    log.info("=" * 60)
    if n_fail > 0:
        log.info(
            "Re-run with the same args to retry only the failed files "
            "(existing files are skipped automatically)."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
