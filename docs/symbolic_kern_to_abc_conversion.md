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
