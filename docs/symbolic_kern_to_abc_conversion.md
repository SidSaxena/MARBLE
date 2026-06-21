# Feeding score-native data to CLaMP3: **kern → MusicXML → interleaved-ABC (not MIDI→MTF)

**Status:** research / recommendation (web-sourced 2026-06-21). Decides how to feed
MTC-ANN (and ideally the rest of the symbolic-motif line) into CLaMP3 so we preserve
notation instead of throwing it away through a lossy MIDI path.

## TL;DR

- CLaMP3's symbolic encoder **M3 ingests two representations**: voice-interleaved
  **ABC** (bar-segmented, *notation*: key, pitch spelling, meter, beaming, slurs,
  phrase marks) and **MTF** (lossless MIDI-as-text, message-segmented, *performance*:
  exact timing/dynamics). They are different views, not better/worse in the abstract.
- **For cross-piece motif retrieval, the ABC path is the right one** — ABC patches are
  **bar-aligned** (motifs live in bars) and carry score semantics that motif identity
  depends on; and, practically, the source MIDI we'd otherwise use is often already
  **lossy** (MTC-ANN's shipped occurrence MIDIs are time-zeroed — the same flattening
  that produced JKUPDD's byte-duplicate inflation).
- **MTC-ANN ships Humdrum `**kern` + MIDI.** Going `**kern → MusicXML → interleaved-ABC`
  preserves key/spelling/meter/phrase that `**kern → MIDI → MTF` discards. Use it.
- **Best kern→MusicXML tool: `converter21`** (Greg Chapman). Its Humdrum reader is
  **derived from Craig Sapp's `humlib`** (the reference Humdrum library, same engine as
  Verovio), and it **registers into music21** so the rest of the score-slicing/export
  pipeline is plain music21. Pip-installable, actively maintained (v4.0.1, Feb 2026).
  Higher fidelity than music21's built-in Humdrum parser.

## Why not the obvious alternatives

| Path | Verdict |
|---|---|
| `**kern → MIDI → MTF` (current default for BPS/JKUPDD) | Loses key, spelling, meter, phrase; MIDI message-segmentation isn't bar-aligned. Worst for structural/notation-sensitive motif identity. |
| music21 built-in Humdrum → MusicXML | Works, but music21's native `**kern` importer is **incomplete/lossy** — `converter21` exists specifically to replace it. |
| **Verovio** Python toolkit | Excellent `**kern` reader, but its Python toolkit **outputs only MEI/SVG/MIDI/Humdrum — no MusicXML**. Would need `**kern → MEI → MusicXML` (extra hop). Not the clean bridge. |
| `converter21` `**kern → MusicXML` | **Recommended.** humlib-grade reading + music21 MusicXML writer + music21 integration for slicing. |
| direct `**kern → ABC` (hum2abc / music21 ABC writer) | Produces *standard* ABC, **not CLaMP3's interleaved-ABC** (its specific bar-aligned multi-voice format). MusicXML is the intermediate the CLaMP3 preprocessing already consumes — go through it. |

> Translation between symbolic formats is **asymmetric** (A→B can lose more than B→A);
> `**kern ↔ MusicXML` is a high-overlap pair (both are full notation), whereas anything
> → MIDI is a notation→performance projection that *cannot* round-trip key/spelling.

## Recommended pipeline (MTC-ANN)

```
**kern  --converter21-->  MusicXML  --(existing leitmotifs encode_musicxml: xml2abc
        register() into                + abc_pipeline)-->  interleaved ABC  -->  CLaMP3 M3 (L7)
        music21
```

- **Whole-melody (tune-family retrieval):** convert each melody `**kern → MusicXML`, feed
  the existing `encode_musicxml` path (it already does MusicXML → interleaved ABC → CLaMP3).
- **Motif-fragment (motif-class retrieval):** slice the motif span first. The
  Annotated-Motifs ground truth is a per-note label (which notes ∈ which motif); use
  music21 (with converter21 registered) to parse the `**kern`, select the note/measure
  range, and `.write('musicxml')` the fragment → ABC path. Fragments need care (partial
  bars, clef/key context carried into the slice) — music21 handles measure/voice slicing.
- MTC-ANN is **monophonic**, so "voice-interleaving" is trivial (one voice).

CLI / API:
```bash
pip install converter21
python3 -m converter21 -f humdrum -t musicxml in.krn out.xml      # CLI
```
```python
import converter21, music21
converter21.register()                       # swap in the humlib-grade Humdrum converter
s = music21.converter.parse('in.krn')        # high-fidelity parse
s.write('musicxml', fp='out.xml')            # or slice s first for a motif fragment
```

## Broader implication (worth an A/B)

BPS-Motif and JKUPDD currently run through **MIDI→MTF**. If their source scores exist as
`**kern`/MusicXML, an **ABC path could lift all of them** — the same notation-preservation
argument applies, and the bar-aligned ABC segmentation is more structurally meaningful for
motif retrieval. Recommended: once the MTC-ANN ABC path exists, **A/B ABC-vs-MTF on a set we
already have numbers for** (BPS-Motif), at L7, before committing the whole line to ABC. Note
ABC will *not* fix JKUPDD's byte-duplicate issue (identical notation → identical ABC too) —
that's a dataset-dedup problem, separate from representation.

## Follow-up Q&A (2026-06-21)

**Is there loss in `**kern → MusicXML → interleaved-ABC`?** Some at each hop, but far
less than `→ MIDI → MTF`:
- `**kern → MusicXML` (converter21/humlib): **high fidelity** for musical content —
  pitch+spelling, duration, key, meter, accidentals, beams, ties, slurs, articulations,
  lyrics map cleanly. What doesn't carry: Humdrum-specific *analytical* spines / comments
  (not notation we feed CLaMP3 anyway).
- `MusicXML → interleaved-ABC` (CLaMP3 xml2abc): ABC is **less expressive** than MusicXML,
  so this hop reduces to ABC's vocabulary (some ornaments/layout drop). But this is
  **intrinsic** — ABC is the representation M3 was *trained on*, so it's the intended input
  ceiling regardless of path. Net: this is the highest-notation route *into* CLaMP3.

**Is there a direct `**kern → ABC`?** Yes — **`hum2abc`** (Craig Sapp, Humdrum Extras).
But two reasons to still go via MusicXML: (1) it's **lossy** (the man page notes lyrics are
dropped), and (2) it emits *standard* ABC, **not CLaMP3's interleaved-ABC dialect** — feeding
off-distribution ABC to M3 risks worse embeddings than its trained format. The "extra hop"
through MusicXML → CLaMP3's own xml2abc buys **distribution-match**, which matters more than
saving a conversion.

**Do BPS-Motif / JKUPDD support an ABC path, or only MIDI?** The MARBLE tasks **confirmedly
use MIDI→MTF** (`BPSMotif/datamodule.py` → `midi_to_mtf`; `build_jkupdd_retrieval.py` globs
`*.mid*`). The *distributions* are MIDI-centric:
- **JKUPDD** ships **point-sets (lisp/CSV)** + MIDI; the **MIDI is explicitly "for
  verification, discouraged"** (wrong bar position, extra trailing note). Its CSV point-sets
  carry **morphetic pitch** (spelled) — already richer than the MIDI. No kern/MusicXML in the
  distribution, **but the pieces are "mainly from KernScores"** → score-native `**kern`
  exists upstream.
- **BPS-Motif** ships MIDI windows; the underlying **Beethoven sonatas are a KernScores
  `**kern` collection** → score-native exists upstream.

So an ABC path for BPS/JKUPDD is **achievable but not free**: re-source the `**kern` from
KernScores and **re-align the motif/pattern annotations** (which live in MIDI-time / point-set
coordinates) onto the score before slicing → MusicXML → ABC. The alignment is the hard part.
**MTC-ANN is the cheap case** (ships `**kern` directly, annotations already note-indexed).
Cheaper interim win for JKUPDD specifically: rebuild from the **CSV point-sets** (spelled
pitch, no MIDI bar/echo artifacts) instead of the discouraged MIDI — also sidesteps the
byte-duplicate/time-zero pathology.

## Risks / to verify on a sample before building

1. **converter21 fidelity on MTC-ANN's specific `**kern`** (phrase `{ }` markers, fermatas,
   lyrics, tune-family ref-records) — high confidence (humlib-derived) but confirm on 2–3 files.
2. **Fragment slicing → valid MusicXML/ABC** — partial measures and key/clef context must be
   carried into the motif slice or the ABC will misrepresent the fragment.
3. **New dependency** (`converter21` + music21) in the extraction env; pin a version.

## Sources

- [CLaMP 3 paper (M3: interleaved-ABC + MTF)](https://arxiv.org/html/2502.10362v1) ·
  [CLaMP3 repo](https://github.com/sanderwood/clamp3)
- [converter21 (humlib-derived Humdrum↔MusicXML, music21-extending)](https://github.com/gregchapman-dev/converter21) ·
  [PyPI](https://pypi.org/project/converter21/)
- [Verovio toolkit I/O formats (no MusicXML output)](https://book.verovio.org/toolkit-reference/input-formats.html)
- [Humdrum ↔ MusicXML/MEI/Lilypond translation-loss study (MEC2019)](https://music-encoding.org/conference/abstracts/abstracts_mec2019/)
