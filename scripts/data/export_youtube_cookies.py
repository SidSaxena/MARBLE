#!/usr/bin/env python3
"""
scripts/data/export_youtube_cookies.py
──────────────────────────────────────
Export YouTube cookies from a local browser to a Netscape-format
``cookies.txt`` file that yt-dlp's ``--cookies`` flag can consume
reliably across parallel workers.

Why this helper exists
----------------------
The "obvious" export command we used to document:

    python -m yt_dlp --cookies-from-browser firefox \\
                     --cookies cookies.txt --skip-download URL

is broken in current yt-dlp (2024+) because the ``--cookies FILE`` flag
is **bidirectional** — yt-dlp reads the file at startup AND writes to
it at exit. If the file already exists from a previous run AND is
malformed (empty, partial write, wrong format from an interrupted
export), yt-dlp errors on the READ step with:

    ERROR: 'cookies.txt' does not look like a Netscape format cookies file

…before it ever gets to writing fresh cookies. So once you've had ONE
failed export, subsequent attempts using that command also fail. The
"obvious" command can never recover from its own first failure.

This helper avoids the chicken-and-egg problem entirely:
  1. Deletes any stale ``cookies.txt`` first.
  2. Uses yt-dlp's Python API to load cookies into memory from the
     browser.
  3. Explicitly saves the cookie jar via ``MozillaCookieJar.save()``,
     which writes proper Netscape format.

No --skip-download dummy URL, no risk of a half-written file.

Usage
-----
    # Default: export from Firefox to ./cookies.txt
    uv run python scripts/data/export_youtube_cookies.py

    # Specify a different browser:
    uv run python scripts/data/export_youtube_cookies.py --browser chrome

    # Custom output path:
    uv run python scripts/data/export_youtube_cookies.py \\
        --browser firefox --output /path/to/my-cookies.txt

Then pass the file to any of the downloaders:
    uv run python scripts/data/download_shs100k.py --cookies-file cookies.txt ...
    uv run python scripts/data/build_hxmsa_dataset.py --cookies-file cookies.txt ...
    uv run yt-dlp --cookies cookies.txt ...

Re-run this helper when cookies expire (typically a few weeks for YouTube
sign-in cookies). The fresh file always replaces the stale one cleanly.

Browser support
---------------
yt-dlp can read from: brave, chrome, chromium, edge, firefox, opera,
safari, vivaldi, whale.

On Windows + macOS the relevant cookie store is locked while the
browser is running, but Firefox uses SQLite WAL mode (concurrent
reads OK) while Chromium variants don't — close Chrome/Edge before
running this helper to avoid a "database is locked" error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--browser",
        default="firefox",
        choices=[
            "brave",
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "opera",
            "safari",
            "vivaldi",
            "whale",
        ],
        help="Browser to extract cookies from (default: firefox). Close "
        "Chromium-based browsers before running — they lock their "
        "cookie SQLite DB. Firefox uses WAL mode and can stay open.",
    )
    ap.add_argument(
        "--output",
        default="cookies.txt",
        help="Output cookies.txt path (default: ./cookies.txt). "
        "Will be OVERWRITTEN if it already exists.",
    )
    ap.add_argument(
        "--container",
        default=None,
        help="Firefox container name (advanced). Pass None for the "
        "default container. Only Firefox supports containers.",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="Browser profile name (advanced). Defaults to the browser's standard profile.",
    )
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    out_path = Path(args.output).expanduser().resolve()

    # Delete any stale file first so a malformed leftover can't trip
    # us up. This is the whole point of the helper.
    if out_path.exists():
        try:
            out_path.unlink()
            print(f"  removed stale {out_path}")
        except OSError as e:
            print(f"  ERROR: could not remove {out_path}: {e}", file=sys.stderr)
            return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Import yt-dlp lazily so the --help flow doesn't pay the import cost.
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        print(
            "ERROR: yt-dlp not installed. Run `uv sync` or `pip install yt-dlp`, then retry.",
            file=sys.stderr,
        )
        return 1

    # yt-dlp's cookiesfrombrowser param is a tuple of
    # (browser, profile, container, keyring). None for any slot uses
    # the default.
    cookiesfrombrowser = (args.browser, args.profile, args.container, None)
    ydl_opts = {
        "cookiesfrombrowser": cookiesfrombrowser,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            jar = ydl.cookiejar
            # YoutubeDLCookieJar inherits from http.cookiejar.MozillaCookieJar
            # which has .save() writing proper Netscape format with the
            # canonical 3-line header. ignore_discard + ignore_expires
            # ensure session cookies and expired (but still useful) cookies
            # get written too — yt-dlp's filter is more permissive than the
            # default MozillaCookieJar.
            n_cookies = sum(1 for _ in jar)
            if n_cookies == 0:
                print(
                    f"  WARNING: 0 cookies loaded from {args.browser}. "
                    f"Are you signed into YouTube in that browser?",
                    file=sys.stderr,
                )
            jar.save(str(out_path), ignore_discard=True, ignore_expires=True)
    except Exception as e:
        # Common failure modes:
        #   - Chrome/Edge running and holding their SQLite cookie DB
        #     ("database is locked")
        #   - Browser not installed / no profile found
        #   - Permission denied on the browser's data dir
        print(f"  ERROR during cookie extraction: {type(e).__name__}: {e}", file=sys.stderr)
        if "database is locked" in str(e).lower():
            print(
                "  HINT: close the browser before running this helper. "
                "Chromium-based browsers (Chrome/Edge/Brave/...) lock "
                "their cookie DB while running. Firefox should be OK.",
                file=sys.stderr,
            )
        return 2

    # Quick post-flight verification that the file we wrote is what
    # yt-dlp will accept on re-read.
    with open(out_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line.startswith("# Netscape HTTP Cookie File"):
        print(
            f"  ERROR: wrote {out_path} but it doesn't start with the "
            f"Netscape header — got: {first_line!r}. Please re-run.",
            file=sys.stderr,
        )
        return 3

    print(f"  wrote {n_cookies:,} cookies to {out_path}")
    print(f"  use it with:  --cookies-file {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
