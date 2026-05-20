#!/usr/bin/env python3
"""
scripts/data/download_ninsheetmusic_url.py
──────────────────────────────────────────
Bulk-download every sheet (mid + pdf + mus) linked from a single
NinSheetMusic browse URL.

Sibling of ``download_ninsheetmusic.py`` (which takes a pieces.csv);
this one takes ``--listing-url`` instead and scrapes the page itself.
Use cases:

  - A specific series:    /browse/series/SuperMarioBros
  - A console:            /browse/console/NES
  - An arranger:          /browse/arranger/12345
  - "Piano Four Hands":   /browse/category/piano-four-hands

The script visits the URL, finds every ``/download/<kind>/<piece_id>``
link on the rendered page, dedups by piece_id, and saves files as
``<piece_id>_<title-slug>.<ext>`` (same convention as the sibling
script, so files merge cleanly into shared output dirs).

Prerequisites
-------------
  uv sync --extra ninsheetmusic
  uv run patchright install chromium     # or: playwright install chromium

Usage
-----
  # Smoke test — 3 pieces from the SuperMarioBros series page
  uv run python scripts/data/download_ninsheetmusic_url.py \\
      --listing-url https://www.ninsheetmusic.org/browse/series/SuperMarioBros \\
      --out-dir /tmp/nsm_test --kind all --max-pieces 3

  # Full series download, all three kinds
  uv run python scripts/data/download_ninsheetmusic_url.py \\
      --listing-url https://www.ninsheetmusic.org/browse/series/Pokemon \\
      --out-dir data/Pokemon --kind all

Cloudflare workflow
-------------------
Same as the sibling script — patchright is the default, persistent
context lives at ``<out-dir>/.playwright_profile`` so cf_clearance
survives across runs. First-run Cloudflare challenge: click the
Turnstile checkbox once when prompted (headed mode is the default).
Subsequent runs reuse the cached session.

Recursion / hub pages
---------------------
Out of scope. If you point this at ``/browse/series`` (top-level
hub that lists series, not pieces), it will find zero downloadable
pieces and exit. Pick a specific series / console / arranger page.

Shared with download_ninsheetmusic.py — update both when changing
_download_one / browser-launch block / _slugify.
"""

from __future__ import annotations

import argparse
import logging
import re
import signal
import sys
import time
import unicodedata
from pathlib import Path

log = logging.getLogger(__name__)

_KIND_TO_EXT = {"mid": "mid", "pdf": "pdf", "mus": "mus"}
_NSM_BASE = "https://www.ninsheetmusic.org"
# /download/<kind>/<numeric_id> — Cloudflare-protected; only fetchable
# via the same browser context that solved the challenge.
_DOWNLOAD_LINK_RE = re.compile(r"^/download/(mid|pdf|mus)/(\d+)$")

# Cap slug length so filenames stay well under the 255-byte limit even with
# piece_id prefix and extension.
_SLUG_MAX = 80


def _slugify(title: str) -> str:
    """Filesystem-safe slug from an NSM title.

    Strategy: NFKD-normalize → strip non-ASCII → collapse anything
    that isn't [a-zA-Z0-9] into single dashes → trim → cap length.
    Keeps capitalisation. Verbatim from download_ninsheetmusic.py.
    """
    if not title:
        return ""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_title).strip("-")
    return slug[:_SLUG_MAX].rstrip("-")


def _target_path(out_dir: Path, pid: str, slug: str, ext: str) -> Path:
    """Canonical destination path. Verbatim from download_ninsheetmusic.py."""
    stem = f"{pid}_{slug}" if slug else pid
    return out_dir / f"{stem}.{ext}"


def _find_existing(out_dir: Path, pid: str, slug: str, ext: str) -> Path | None:
    """Idempotency check. Verbatim from download_ninsheetmusic.py."""
    new_style = _target_path(out_dir, pid, slug, ext)
    if new_style.exists() and new_style.stat().st_size > 100:
        return new_style
    legacy = out_dir / f"{pid}.{ext}"
    if legacy != new_style and legacy.exists() and legacy.stat().st_size > 100:
        return legacy
    return None


def _download_one(context, url: str, dst: Path, timeout_ms: int = 45_000) -> bool:
    """Download ``url`` via the browser context's HTTP client.

    Uses ``context.request.get`` rather than ``page.goto`` +
    expect_download because the latter trips Cloudflare's headless-
    navigation detection. ``context.request.get`` inherits the
    context's cookies (including ``cf_clearance``) and TLS
    fingerprint, so it looks like the same Chromium that just
    completed the JS challenge.

    Verbatim from download_ninsheetmusic.py.
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
    head = body[:16].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        log.debug(f"    received HTML (likely Cloudflare challenge) for {url}")
        return False
    dst.write_bytes(body)
    return dst.exists() and dst.stat().st_size > 100


# ── Listing-page scraping ─────────────────────────────────────────────────────


def _scrape_listing_page(page, listing_url: str) -> list[tuple[str, str]]:
    """Visit a NSM browse URL; return ``[(piece_id, title), ...]``.

    Strategy: NSM renders each sheet as ``<li class="tableRow sheetRow"
    id="sheetNNNN">`` containing ``<div class="sheetRow-title">`` and
    download links ``/download/{mid,pdf,mus}/NNNN``. We extract piece_id
    from the row's ``id=sheetNNNN`` attribute and title from the
    ``.sheetRow-title`` div.

    Fallback selectors are included for robustness if NSM tweaks
    class names — we look for any ``[id^="sheet"]`` row, OR any
    element containing a ``/download/*/<id>`` link if no rows exist.
    """
    log.info(f"Visiting listing page: {listing_url}")
    page.goto(listing_url, wait_until="domcontentloaded")
    # Wait actively for the challenge / loader to clear before scraping.
    for _ in range(20):
        t = page.title()
        if "Just a moment" not in t and "Loading" not in t and "Not Found" not in t:
            break
        time.sleep(1.5)
    log.info(f"  page title: {page.title()!r}")

    raw = page.evaluate(r"""
    () => {
        // Primary: well-formed sheetRow elements
        const rows = Array.from(document.querySelectorAll('li[id^="sheet"], div[id^="sheet"]'));
        if (rows.length > 0) {
            return rows.map(li => {
                const m = li.id.match(/^sheet(\d+)$/);
                if (!m) return null;
                const pid = m[1];
                let title = "";
                const titleEl = li.querySelector('.sheetRow-title');
                if (titleEl) {
                    title = titleEl.textContent.trim();
                } else {
                    // Fallback: any header or first non-link text node
                    const h = li.querySelector('h1,h2,h3,h4,strong');
                    if (h) title = h.textContent.trim();
                }
                return {pid, title};
            }).filter(x => x !== null);
        }
        // Fallback: scan all anchors for /download/<kind>/<id> and walk
        // up to find a title. Used if NSM changes class names.
        const seen = new Map();
        const re = /^\/download\/(mid|pdf|mus)\/(\d+)$/;
        document.querySelectorAll('a[href]').forEach(a => {
            const m = a.getAttribute('href').match(re);
            if (!m) return;
            const pid = m[2];
            if (seen.has(pid)) return;
            let row = a;
            for (let i = 0; i < 6 && row.parentElement; i++) {
                row = row.parentElement;
                if (row.id && row.id.startsWith('sheet')) break;
                if (row.querySelector('.sheetRow-title, h1, h2, h3, h4')) break;
            }
            const titleEl = row.querySelector('.sheetRow-title, h1, h2, h3, h4');
            const title = titleEl ? titleEl.textContent.trim() : "";
            seen.set(pid, {pid, title});
        });
        return Array.from(seen.values());
    }
    """)

    # Dedupe (defensive — the primary path already returns unique pids)
    seen: dict[str, str] = {}
    for item in raw:
        pid = item["pid"]
        title = item["title"] or ""
        if pid not in seen or (title and not seen[pid]):
            seen[pid] = title

    pieces = sorted(seen.items(), key=lambda kv: int(kv[0]))
    log.info(f"  found {len(pieces)} unique pieces on the page")
    for pid, title in pieces[:3]:
        log.info(f"    sample: pid={pid}  title={title!r}")
    if len(pieces) > 3:
        log.info(f"    … and {len(pieces) - 3} more")
    return pieces


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
        "--listing-url",
        required=True,
        help="NSM browse URL (series / console / arranger / category page) "
        "to scrape for piece links",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory; files saved as <piece_id>_<title-slug>.<ext>",
    )
    ap.add_argument(
        "--kind",
        choices=("mid", "pdf", "mus", "all"),
        default="all",
        help="Which file types to download (default: all)",
    )
    ap.add_argument(
        "--max-pieces",
        type=int,
        default=None,
        help="Pilot mode: download at most this many pieces total",
    )
    ap.add_argument("--delay", type=float, default=1.0, help="Seconds between downloads")
    ap.add_argument("--retry", type=int, default=2, help="Retries per file on failure")
    ap.add_argument(
        "--timeout-ms",
        type=int,
        default=45_000,
        help="Per-download timeout in milliseconds",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run with a hidden browser. Default is HEADED — NSM's "
        "Cloudflare Turnstile reliably blocks headless even with patchright.",
    )
    ap.add_argument(
        "--persistent-context-dir",
        default=None,
        help="Persistent profile path; defaults to <out-dir>/.playwright_profile",
    )
    ap.add_argument(
        "--no-persistent-context",
        action="store_true",
        help="Force an ephemeral context (debug only — cf_clearance won't survive)",
    )
    ap.add_argument(
        "--channel",
        default=None,
        choices=["chrome", "msedge", "chromium", "chrome-beta", "msedge-beta"],
        help="System-installed Chromium variant. 'chrome' recommended for "
        "Cloudflare — system binaries are much harder to fingerprint.",
    )
    ap.add_argument(
        "--firefox",
        action="store_true",
        help="Use Firefox (forces stock playwright; patchright is Chromium-only)",
    )
    ap.add_argument(
        "--no-proxy",
        action="store_true",
        help="Force direct connection, bypassing system proxy",
    )
    ap.add_argument(
        "--proxy-server",
        default=None,
        help="Explicit proxy URL (e.g. http://proxy.corp:8080)",
    )
    args = ap.parse_args()

    if args.no_proxy and args.proxy_server:
        log.error("--no-proxy and --proxy-server are mutually exclusive.")
        sys.exit(1)

    # Lazy import: prefer patchright unless --firefox forces stock.
    sync_playwright = None
    _backend = None
    use_patchright = False
    if not args.firefox:
        try:
            from patchright.sync_api import sync_playwright as _sp

            sync_playwright = _sp
            _backend = "patchright"
            use_patchright = True
        except ImportError:
            pass
    if sync_playwright is None:
        try:
            from playwright.sync_api import sync_playwright as _sp

            sync_playwright = _sp
            _backend = "playwright"
        except ImportError:
            log.error(
                "Neither patchright nor playwright is installed. Run:\n"
                "    uv sync --extra ninsheetmusic\n"
                "    uv run playwright install chromium\n"
            )
            sys.exit(1)
    log.info(f"Using browser backend: {_backend}")

    # Resolve kinds
    if args.kind == "all":
        kinds = ["mid", "pdf", "mus"]
    else:
        kinds = [args.kind]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan preamble
    log.info("")
    log.info("─" * 60)
    log.info("Download plan")
    log.info("─" * 60)
    log.info(f"  listing-url  : {args.listing_url}")
    log.info(f"  out-dir      : {out_dir}")
    log.info(f"  kinds        : {', '.join(kinds)}")
    log.info(f"  max-pieces   : {args.max_pieces if args.max_pieces else 'all'}")
    log.info(f"  delay        : {args.delay:.1f} s between downloads")
    log.info(f"  headless     : {args.headless}")
    log.info(f"  channel      : {args.channel or 'bundled'}")
    log.info("─" * 60)

    # SIGINT
    _stop = {"set": False}

    def _sigint(_sig, _frame):
        log.warning("\nInterrupted — finishing current download then exiting.")
        _stop["set"] = True

    signal.signal(signal.SIGINT, _sigint)

    n_ok = 0
    n_fail = 0
    n_skipped = 0

    # Stealth: only for stock playwright (patchright handles it natively).
    stealth = None
    has_stealth = False
    if not use_patchright:
        try:
            from playwright_stealth import Stealth

            stealth = Stealth()
            has_stealth = True
        except ImportError:
            log.warning("playwright-stealth not installed — stock playwright lacks stealth")
    else:
        log.info("  stealth: patchright handles it natively")

    with sync_playwright() as p:
        engine = p.firefox if args.firefox else p.chromium

        if use_patchright:
            ctx_kwargs: dict = {
                "accept_downloads": True,
                "no_viewport": True,
            }
            launch_args: list[str] = []
        else:
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
            }
            launch_args = [
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]

        launch_extra: dict = {}
        if not args.firefox and args.channel:
            launch_extra["channel"] = args.channel
        if args.no_proxy:
            launch_extra["proxy"] = {"server": "direct://"}
            log.info("  proxy: bypassed (--no-proxy)")
        elif args.proxy_server:
            launch_extra["proxy"] = {"server": args.proxy_server}

        engine_args = launch_args if not args.firefox else []

        use_persistent = not args.no_persistent_context
        if use_persistent:
            profile_path = (
                Path(args.persistent_context_dir)
                if args.persistent_context_dir
                else out_dir / ".playwright_profile"
            )
            profile_path.mkdir(parents=True, exist_ok=True)
            log.info(f"  persistent profile: {profile_path}")
            context = engine.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=args.headless,
                args=engine_args,
                **launch_extra,
                **ctx_kwargs,
            )
            browser = None
        else:
            log.warning("  persistent context disabled")
            browser = engine.launch(headless=args.headless, args=engine_args, **launch_extra)
            context = browser.new_context(**ctx_kwargs)

        if has_stealth:
            try:
                stealth.apply_stealth_sync(context)
            except Exception as e:
                log.warning(f"  stealth patch failed: {e}")
        page = context.new_page()

        # Pre-warm: visit homepage to clear Cloudflare BEFORE scraping
        # the listing page. Cloudflare's bot score builds from cookie
        # history — landing directly on a deep page can fire a stricter
        # challenge.
        try:
            log.info("Pre-warming browser session (visiting NSM homepage) …")
            page.goto(_NSM_BASE + "/", timeout=args.timeout_ms)
            page.wait_for_load_state("domcontentloaded", timeout=15_000)
            challenge_re = ("Just a moment", "Attention", "Checking")
            cleared = False
            for _ in range(30):
                title = page.title()
                cookies = context.cookies(_NSM_BASE + "/")
                has_cf = any(c.get("name") == "cf_clearance" for c in cookies)
                if has_cf and not any(p in title for p in challenge_re):
                    cleared = True
                    log.info(f"  cleared Cloudflare: title={title!r}")
                    break
                time.sleep(2.0)
            if not cleared:
                log.warning("  Cloudflare not cleared in 60 s — try solving manually now")
                for _ in range(30):
                    cookies = context.cookies(_NSM_BASE + "/")
                    if any(c.get("name") == "cf_clearance" for c in cookies):
                        log.info("  cf_clearance appeared — proceeding")
                        cleared = True
                        break
                    time.sleep(2.0)
            if not cleared:
                log.error("  Cloudflare never cleared. Aborting.")
                sys.exit(2)
        except Exception as e:
            log.warning(f"  pre-warm warning ({type(e).__name__}: {e}); continuing")

        # Scrape the listing page in the now-warmed context.
        pieces = _scrape_listing_page(page, args.listing_url)
        if not pieces:
            log.error(
                "No pieces found on the listing page. Either:\n"
                "  - The URL is a top-level hub (e.g. /browse/series) that "
                "lists series, not pieces. Pick a specific series page.\n"
                "  - The page structure changed and the scraper's selectors "
                "no longer match. Inspect the rendered HTML and update "
                "_scrape_listing_page in this script."
            )
            sys.exit(3)
        if args.max_pieces:
            pieces = pieces[: args.max_pieces]
            log.info(f"  --max-pieces limit: keeping first {len(pieces)} pieces")

        # Estimate / preflight
        n_targets = len(pieces) * len(kinds)
        n_already = sum(
            1
            for pid, title in pieces
            for kind in kinds
            if _find_existing(out_dir, pid, _slugify(title), _KIND_TO_EXT[kind]) is not None
        )
        n_todo = n_targets - n_already
        log.info(f"  total files  : {n_targets:,}  (already present: {n_already:,})")
        log.info(f"  to download  : {n_todo:,}")
        est_min = (n_todo * (args.delay + 3.0)) / 60.0
        log.info(f"  ~{est_min:.1f} min estimated wall-clock")

        if n_todo == 0:
            log.info("All files already present — nothing to do.")
            if browser is None:
                context.close()
            else:
                browser.close()
            sys.exit(0)

        for i, (pid, title) in enumerate(pieces, 1):
            if _stop["set"]:
                break
            slug = _slugify(title)
            for kind in kinds:
                if _stop["set"]:
                    break
                ext = _KIND_TO_EXT[kind]
                if _find_existing(out_dir, pid, slug, ext) is not None:
                    n_skipped += 1
                    continue
                dst = _target_path(out_dir, pid, slug, ext)
                url = f"{_NSM_BASE}/download/{kind}/{pid}"
                ok = False
                for attempt in range(args.retry + 1):
                    if _download_one(context, url, dst, timeout_ms=args.timeout_ms):
                        ok = True
                        break
                    if attempt < args.retry:
                        backoff = args.delay * (2**attempt)
                        time.sleep(backoff)
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    log.warning(f"  [{dst.name}] FAILED after {args.retry + 1} attempts: {url}")
                time.sleep(args.delay)

            if i % 10 == 0 or i == len(pieces):
                log.info(
                    f"  [{i:>4}/{len(pieces)}]  ok={n_ok}  skipped={n_skipped}  failed={n_fail}"
                )

        if browser is None:
            context.close()
        else:
            browser.close()

    log.info("")
    log.info("=" * 60)
    log.info(f" Done — ok={n_ok}, skipped={n_skipped}, failed={n_fail}")
    log.info("=" * 60)
    if n_fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
