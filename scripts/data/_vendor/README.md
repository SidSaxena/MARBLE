# Vendored third-party scripts

Scripts here are **third-party code vendored verbatim** so the MARBLE
pipelines can depend on them without making them PyPI-installable
themselves. They retain their original upstream license headers
unmodified.

Treat the contents as read-only. If a script needs updating, replace
the whole file from upstream and update the version note below.

## `xml2abc.py` — MusicXML → ABC converter

- **Upstream:** <https://wim.vree.org/svgParse/xml2abc.html>
- **Author:** W.G. Vree (2012–2025), with contributions from many others.
- **License:** Lesser GNU General Public License (LGPL), per the file
  header. We invoke it as a subprocess only (no linking, no
  modification), which keeps the rest of MARBLE under its own license.
- **Version:** 174 (Sep 2024 release, retrieved May 2026 from
  `xml2abc.py-174.zip`).
- **Used by:** `scripts/data/build_supermario_dataset.py` when
  `--build-abc` is passed, and `scripts/data/convert_mxl_to_abc.py`.

### Why vendored?

It is not on PyPI (a search for `xml2abc`, `xml2abc-py`, `wim-xml2abc`,
`musicxml2abc` all returned nothing as of May 2026), so users would
otherwise need to manually download it from the upstream zip. Vendoring
makes the build script work out-of-the-box once `music21` and a working
Python interpreter are available.

### Why not a Python module?

xml2abc is structured as a CLI script with all logic packed under an
`if __name__ == '__main__':` block that depends on `optparse` and
file I/O conventions. Importing it as a library would require
refactoring the LGPL'd code — easier to call as a subprocess and let
the upstream stay byte-identical.
