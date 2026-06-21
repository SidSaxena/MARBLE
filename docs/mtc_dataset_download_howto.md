# Downloading the Meertens Tune Collections (MTC) — how-to

The MTC datasets are hosted on **SURFdrive** (SURF's academic Nextcloud), fronted by a
session-gated PHP page on liederenbank.nl. This is the working download path (worked out
2026-06-21) so any MTC collection can be re-downloaded directly. License: **CC BY-NC-SA 3.0**
(non-commercial; research/thesis use OK).

---

## TL;DR — direct public download links

The files live on SURFdrive **public shares**. Once you have a share **token**, the download
URL is `https://surfdrive.surf.nl/s/<TOKEN>/download` — **public: no login, no form, no
cookie.** curl / wget / browser all work. Tokens resolved 2026-06-21:

| Collection | Size | File | Direct link (`…/s/<TOKEN>/download`) |
|---|---|---|---|
| **MTC-ANN-2.0.1** (annotated — kern + motif/tune-family GT) | 56 MB | mtc-ann-2.0.1.tgz | https://surfdrive.surf.nl/s/tIJnKK6NpUeOM4y/download |
| MTC-FS-1.0 (vocal folk) | 246 MB | mtc-fs-1.0.tgz | https://surfdrive.surf.nl/s/9RTUBFw431nDCcZ/download |
| MTC-INST-1.0 (instrumental) | 126 MB | mtc-inst-1.0.tgz | https://surfdrive.surf.nl/s/wyWDltl1qy7dXIk/download |
| MTC-LC-1.0 | 1 MB | mtc-lc-1.0.tgz | https://surfdrive.surf.nl/s/BkrnC5t7au7uwKq/download |
| MTC-FS-INST-2.0 (18.6k melodies) | 859 MB | MTC-FS-INST-2.0.tgz | https://surfdrive.surf.nl/s/kwUXQF8rCEZ446q/download |
| MTC-OGLAUDIO-1.0 part 1 | 3.78 GB | mtc-oglaudio-part1-1.0.tgz | https://surfdrive.surf.nl/s/s6AgesanRx2GlOE/download |
| MTC-OGLAUDIO-1.0 part 2 | 3.86 GB | mtc-oglaudio-part2-1.0.tgz | https://surfdrive.surf.nl/s/KrCJ4CRgTJMVTuU/download |
| MTC-OGLAUDIO-1.0 part 3 | 3.67 GB | mtc-oglaudio-part3-1.0.tgz | https://surfdrive.surf.nl/s/ZYfxRlIUSBnixZZ/download |
| MTC-OGLAUDIO-1.0 part 4 | 3.85 GB | mtc-oglaudio-part4-1.0.tgz | https://surfdrive.surf.nl/s/FFcdUEqHmDAB5GW/download |
| MTC-OGLAUDIO metadata | tiny | mtc-oglaudio-metadata-1.0.tgz | https://surfdrive.surf.nl/s/n8QvFc2v3ZA64d7/download |
| MTC-OGLSCANS-1.0 (page scans) | 1.53 GB | mtc-oglscans-1.0.tgz | https://surfdrive.surf.nl/s/W0nAkVv1VfCB5bZ/download |

```bash
curl -L "https://surfdrive.surf.nl/s/tIJnKK6NpUeOM4y/download" -o mtc-ann-2.0.1.tgz
tar -xzf mtc-ann-2.0.1.tgz       # -> MTC-ANN-2.0.1/{krn,mid,ly,wce,metadata,...}
```

> ⚠️ Tokens are SURFdrive share IDs. They've been stable, but if a share is ever rotated,
> re-resolve it via the process below.

---

## The full process (how those tokens were obtained)

liederenbank's download UI is a thin gate in front of SURFdrive. Three steps:

### 1. Submit the form ONCE (sets a session)
`https://www.liederenbank.nl/mtc/download.php` is a POST form — `name` (required),
`email`/`comments` (optional). Submitting it over **https** sets a PHP **session cookie**
that authorizes step 2. Submit it **once** (session ~30 min); don't re-submit (each submit
re-registers your name/email with them).

```bash
curl -sSL -c cj.txt "https://www.liederenbank.nl/mtc/download.php" -o /dev/null      # grab session cookie
curl -sS -L --post301 --post302 -b cj.txt -c cj.txt -X POST \
  "https://www.liederenbank.nl/mtc/download.php" \
  --data-urlencode "name=YOUR NAME" \
  --data-urlencode "email=you@example.org" \
  --data-urlencode "comments=academic research" \
  --data-urlencode "submit=Download" \
  -o downloads_page.html
```
(The **http** endpoint 301-redirects to **https** for the POST — use `--post301 --post302`
so curl keeps the POST body, or just POST straight to the https URL.) The returned page
lists `serve.php?col=<COL>` links for every collection.

### 2. Resolve a collection → SURFdrive token
`serve.php?col=<COL>` **with the session cookie** 302-redirects to a SURFdrive share.
Capture the `Location`:
```bash
curl -sS -o /dev/null -D - -b cj.txt \
  "https://www.liederenbank.nl/mtc/serve.php?col=mtc-ann-2.0.1" | grep -i '^location:'
#  -> location: https://surfdrive.surf.nl/files/index.php/s/tIJnKK6NpUeOM4y
```
Collection ids (`col=`): `mtc-ann-2.0.1`, `mtc-ann-2.0`, `mtc-ann-1.1`, `mtc-fs`,
`mtc-inst`, `mtc-lc`, `mtc-fs-inst`, `mtc-oglaudio-1`…`-4`, `mtc-oglaudio-metadata`,
`mtc-oglscans`.

### 3. Fix the URL + download (public, no session)
The redirect hands you the **legacy** ownCloud path `…/files/index.php/s/<TOKEN>` which is
now **dead (404)**. The **canonical** Nextcloud path is `…/s/<TOKEN>`, and the direct
download is `…/s/<TOKEN>/download`:
```bash
curl -L "https://surfdrive.surf.nl/s/<TOKEN>/download" -o file.tgz
```
No cookie/login needed at this step — it's a public share (filename comes from the
`Content-Disposition` header).

---

## The two gotchas that cost time (so you don't repeat them)

1. **Old vs new SURFdrive path.** `serve.php` hands you the dead legacy
   `/files/index.php/s/<TOKEN>`; you must use **`/s/<TOKEN>/download`** (drop
   `files/index.php`). The legacy path 404s with a JS-required SURFdrive page.
2. **The session is for *resolving*, not *downloading*.** You only need the liederenbank
   session cookie to turn `col` → `token` (step 2). The SURFdrive download (step 3) is
   public — so the `/s/<TOKEN>/download` links in the TL;DR table are **durable direct
   links** you can save and reuse without ever touching the form again.

(If you'd rather not script it at all: submit the form in a browser, then on the returned
page right-click the dataset link → the browser will follow `serve.php` → SURFdrive → a
normal download. The scripted path above is only needed for headless/remote downloads.)
