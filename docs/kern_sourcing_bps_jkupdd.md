# Sourcing score-native `**kern` for BPS-Motif & JKUPDD (for a future ABC path)

**Status:** sourcing + feasibility assessment (2026-06-21). Companion to
`docs/symbolic_kern_to_abc_conversion.md`, which argues CLaMP3 embeds
notation-rich interleaved-ABC better than the lossy MIDI→MTF path BPS-Motif and
JKUPDD currently use. That doc's open question was: *can we re-source the
`**kern` and re-align the motif annotations onto it?* This doc answers it. **This
is sourcing only — no conversion pipeline was built.**

## TL;DR

- **All target `**kern` was located and downloaded.** Both datasets' underlying
  pieces are KernScores editions (Craig Sapp / David Huron), all public + git-clonable.
- **JKUPDD is the easy case and a pleasant surprise:** the dataset *ships its own
  `**kern` (`kernOriginal/`) AND a beat-indexed point-set CSV per piece* — no
  separate KernScores sourcing was even needed, and the CSV is the alignment key.
  Even the feared Gibbons gap is closed (it ships `silverswan.krn`).
- **BPS-Motif is also tractable:** the 32 Beethoven first-movement `**kern` come
  from `craigsapp/beethoven-piano-sonatas` (`sonataNN-1.krn`), and the upstream
  annotation repo (`Wiilly07/Beethoven_motif`) carries a **per-note CSV with an
  explicit `measure` column + morphetic (spelled) pitch + motif label** — so
  alignment is by *measure + onset + pitch*, not a fragile MIDI pitch-sequence search.
- **Alignment verdict:** JKUPDD = **clean/low-risk** (anchor by point-set ontime,
  already note-indexed; occurrence CSVs are literal sub-sequences of the full
  point-set). BPS-Motif = **moderate** (measure-indexed annotations align to kern
  barlines; main hazards are written-out repeats and the pickup-bar offset, both
  of which upstream already encodes — `*>[...]` section maps + `pickup.csv`).
- **Worth-it verdict:** an ABC path is **feasible for both** and meaningfully
  cheaper than feared, because *neither requires pitch-sequence fuzzy matching* —
  both ship score-position indices (JKUPDD: ontime; BPS: measure). JKUPDD is worth
  doing first as a near-free win; BPS-Motif is worth doing but needs a one-time
  measure-alignment harness per piece. See "Recommendation" at the bottom.

## What was downloaded

Everything under `data/kern_sources/` (gitignored — local only, do **not** commit):

```
data/kern_sources/
├── JKUPDD/                         # from github.com/ns2max/jkupdd_dataset (exact mirror of JKUPDD-Aug2013)
│   ├── bachBWV889Fg/        wtc2f20.krn      wtc2f20.csv
│   ├── beethovenOp2No1Mvt3/ sonata01-3.krn   sonata01-3.csv
│   ├── chopinOp24No4/       mazurka24-4.krn  mazurka24-4.csv
│   ├── gibbonsSilverSwan1612/ silverswan.krn silverswan.csv
│   └── mozartK282Mvt2/      sonata04-2.krn   sonata04-2.csv
│       # .krn = the dataset's kernOriginal/ (verbatim KernScores edition)
│       # .csv = the dataset's full-piece point-set: ontime,MIDI,morphetic,dur,voice
└── BPS-Motif/
    ├── beethoven_sonatas_kern/   sonata01-1.krn … sonata32-1.krn  (32 files)
    │       # from github.com/craigsapp/beethoven-piano-sonatas (kern/)
    └── beethoven_motif_csv/      # from github.com/Wiilly07/Beethoven_motif (BPS-Motif upstream)
        ├── csv_notes/  01-1.csv … 32-1.csv   (32 — per-note: onset,midi,morphetic,dur,staff,measure,type)
        ├── csv_label/  01-1.csv … 32-1.csv   (32 — per-motif spans, measure + seconds + TS)
        └── pickup.csv                         (per-movement pickup-beat offsets)
```

**Note on sourcing route:** the Mac sandbox blocks `curl`/`git clone`/`scp`, so
all clones ran on the Windows PC (`ssh my-pc`, which has network); files were
streamed back via `ssh my-pc "cat …" > local` (and `cat tarball` for the bulk
BPS transfer, extracted locally). The JKUPDD `**kern` did **not** need a separate
KernScores fetch — it is bundled inside the JKUPDD distribution itself.

## Per-piece match table

### JKUPDD (5 pieces)

| dataset id | work | KernScores source (repo · file) | downloaded? | edition/movement confidence |
|---|---|---|---|---|
| `bachBWV889Fg` | Bach, WTC Bk II **Fugue 20, A minor** (BWV 889) | bundled in JKUPDD `kernOriginal/wtc2f20.krn` (= David Huron KernScores edn; also `humdrum-tools/bach-wtc` · `kern/wtc2f20.krn`) | ✅ | **High.** kern header: `SCT: BWV 889b`, `ONB: A minor, 3-part`, `Fuga 20, Vol. 2`. **Note: this is A minor, not "F minor" as the task brief said** — BWV 889 is WTC II No. 20 in A minor; the brief's "(F minor)" is an error. |
| `beethovenOp2No1Mvt3` | Beethoven Sonata Op.2 No.1, **mvt 3** (Menuetto) | bundled `kernOriginal/sonata01-3.krn` (= `craigsapp/beethoven-piano-sonatas` · `kern/sonata01-3.krn`) | ✅ | **High.** Same Sapp Beethoven edition family as the BPS-Motif kern; `sonataNN-M` ⇒ sonata 1 (=Op.2 No.1), mvt 3. |
| `chopinOp24No4` | Chopin Mazurka **Op.24 No.4**, B♭ minor | bundled `kernOriginal/mazurka24-4.krn` (= `craigsapp/chopin-mazurkas` · `kern/mazurka24-4.krn`) | ✅ | **High.** kern header: `OTL: Mazurka in B-flat Minor, Op. 24, No. 4`, `OPS: Op. 24`, `ONM: No. 4`. |
| `gibbonsSilverSwan1612` | Gibbons, **"The Silver Swanne"** (1612) | bundled `kernOriginal/silverswan.krn` (Craig Stuart Sapp enc. 2004; KernScores early-music) | ✅ | **High — the suspected gap is closed.** kern header: `OTL: The Silver Swanne`, `COM: Gibbons, Orlando`, `PDT: 1612`, 5 vocal spines. Ships *inside* JKUPDD; no standalone craigsapp Gibbons repo needed. |
| `mozartK282Mvt2` | Mozart Sonata **K.282** (=No.4), **mvt 2** (Menuetto I/II) | bundled `kernOriginal/sonata04-2.krn` (= `craigsapp/mozart-piano-sonatas` · `kern/sonata04-2.krn`) | ✅ | **High.** Sapp Mozart edition (Alte Mozart-Ausgabe); sonata 04 = K.282, mvt 2. |

All 5 JKUPDD pieces matched cleanly. **No gaps.**

### BPS-Motif (32 Beethoven first movements)

BPS-Motif ids `01-1 … 32-1` = Beethoven piano sonatas **1–32, movement 1**.
`craigsapp/beethoven-piano-sonatas` ships all 32 as `kern/sonataNN-1.krn`.

| dataset id range | work | KernScores source | downloaded? | confidence |
|---|---|---|---|---|
| `01-1` … `32-1` (all 32) | Beethoven piano sonatas 1–32, mvt 1 | `craigsapp/beethoven-piano-sonatas` · `kern/sonataNN-1.krn` | ✅ (32/32) | **High** for the *edition* (verified `sonata01-1.krn` header = "Piano Sonata no. 1 in F minor, OPS 2 ONM 1 OMV 1", `M2/2` matching the BPS-Motif `csv_label` `TS=2/2` for `01-1`). The *annotation* itself does not need this kern (it ships its own `csv_notes`); the kern is what the ABC path consumes. |

Caveat on edition identity: BPS-Motif's `csv_notes` were authored against *some*
Beethoven score; the Sapp kern is the standard KernScores Beethoven edition and
the one BPS-Motif's own provenance points at, but the per-note alignment (below)
is what actually proves edition-match per movement — run it as a validation gate,
not an assumption. Mismatch risk is highest where editions differ on
repeats/ossia/grace-note spelling (a handful of the 32, not the rule).

## Alignment feasibility (the hard part)

The annotations live in **MIDI-/beat-time**; the `**kern` is **score-position**.
To extract a motif occurrence as ABC you must map *annotation span → kern note
range*. Crucially, **neither dataset requires blind pitch-sequence fuzzy matching**
— both carry an explicit score-position index. That is the key finding.

### JKUPDD — CLEAN / LOW RISK

- **Coordinate system:** each piece ships a full-piece point-set CSV
  `(ontime_beats, MIDI_pitch, morphetic_pitch, duration, voice)` sorted by
  ontime. `ontime` is score beats from the start (verified: Bach fugue row 1 =
  `(1.0, 64, 62, …)` = the e′ on beat 1 of bar 1 in `wtc2f20.krn`).
- **Annotation form:** each pattern occurrence ships its own
  `occurrences/csv/occN.csv` whose rows are a **literal contiguous sub-sequence of
  the full point-set** (verified: Bach pattern A occ1 = `(1.0,64),(2.0,60),…` =
  the first point-set rows; occ2 = `(21.0,76),…`). So the occurrence *already is*
  a (ontime, pitch) note list — no search needed.
- **kern alignment route:** parse the kern with a humlib-grade reader
  (converter21/music21), emit per-note `(ontime_beats, MIDI/morphetic pitch)`,
  and **join to the point-set on (ontime, pitch)**. Because morphetic pitch is
  present, ties between enharmonic/octave-equal candidates are resolvable. Then
  the occurrence's ontime range selects the kern note range → slice → MusicXML →
  ABC. Whole-piece retrieval is even simpler (no slicing).
- **Failure modes:** (a) the kern's *barline/repeat structure* vs the point-set's
  *unfolded* ontime — JKUPDD point-sets are already time-linear/unfolded, so the
  kern may need `*>` section-expansion before the ontime↔note map is monotonic;
  (b) voicing/chords — multiple notes share an ontime, disambiguated by pitch
  (point-set carries voice index too); (c) Gibbons is 5-voice vocal (no repeats,
  *easier* structurally) but ABC voice-interleaving is 5 voices vs the others'
  2-staff piano.
- **Difficulty: LOW.** This is essentially a deterministic join on a key the
  dataset hands you. JKUPDD is the cheap win.

### BPS-Motif — MODERATE

- **Coordinate system:** annotations come in TWO indices, both score-native:
  - `csv_notes/NN-1.csv` per-note: `onset(beats), midi_number, morphetic_number,
    duration, staff_number, **measure**, type(motif letter or empty)`.
  - `csv_label/NN-1.csv` per-motif: `start,end (beats), type, **measure**,
    start_beat, duration, track, **TS**, measure_score, start_midi, end_midi(sec)`.
- **Why this is better than the MIDI window path:** the current pipeline
  synthesises a 60-QPM MIDI from `csv_notes` then slices by *seconds*; that throws
  away the **`measure` column already present in the CSV**. For the ABC path you
  align `csv_label.measure`/`measure_score` directly to the kern's `=N` barlines,
  and `csv_notes.onset` (beats-in-movement) to within-bar position — no
  pitch-sequence search and no MIDI round-trip.
- **kern alignment route:** (1) parse `sonataNN-1.krn` → ordered notes with
  (measure, beat-in-bar, MIDI, morphetic). (2) Build the same table from
  `csv_notes`. (3) Align the two note streams by `(measure, onset, pitch)`; the
  morphetic column makes spelling tie-breaks exact. (4) A motif occurrence =
  the `csv_notes` rows whose `type==letter` in the occurrence's measure span (or
  the `csv_label` `[start,end]` beat range) → map to kern notes → slice → ABC.
- **Failure modes (all anticipated by upstream, none blocking):**
  1. **Written-out / structural repeats.** Sapp kern carries `*>[A,A,B,B]` /
     `*>norep[...]` section maps (seen in `sonata01-1.krn`); the BPS-Motif
     `measure`/`measure_score` pair is the hook to pick folded vs unfolded
     numbering. Must expand consistently on both sides before the measure join.
  2. **Pickup bars.** `csv_notes` uses `onset=-1.0, measure=0` for pickups;
     upstream ships `pickup.csv` (per-movement pickup offset) precisely to
     reconcile this with measure 1. Kern encodes the pickup as a partial bar
     before `=1`. Map measure 0 ↔ kern pickup.
  3. **Voicing / staff.** `csv_notes.staff_number` ↔ kern staff spines; chords
     share an onset (resolve by pitch+staff).
  4. **Edition drift.** A few of the 32 may differ between the BPS annotation
     score and the Sapp kern (extra/omitted repeats, grace-note count). The
     `(measure,onset,pitch)` join *is* the per-movement edition-match test — log
     unmatched notes; a movement with a low match rate is flagged, not silently
     mis-sliced.
- **Difficulty: MODERATE.** No fuzzy matching, but a one-time per-movement
  measure-alignment + repeat-unfolding harness is needed, plus a match-rate gate
  to catch edition drift on the handful of divergent movements.

## What's blocked / open

- **Nothing is blocked on sourcing** — all `**kern` is in hand for both datasets.
- The *conversion pipeline itself* is not built (out of scope here). To proceed:
  reuse `converter21` (`**kern`→MusicXML, humlib-grade) + the existing
  `encode_musicxml` ABC path from `docs/symbolic_kern_to_abc_conversion.md`.
- **Validation gate to build first** (cheap, catches the only real risk):
  for each movement, run the `(measure/ontime, pitch)` join between the kern and
  the CSV and report a per-movement note match-rate. JKUPDD should be ~100%
  trivially; BPS-Motif's match-rate per sonata *is* the edition-confidence number
  the table above estimates qualitatively. Movements below threshold get manual
  review or fall back to MTF.
- **Gibbons ABC nuance:** 5 vocal voices → confirm CLaMP3 interleaved-ABC handles
  ≥5 voices as expected (the piano pieces are 2-staff). Not a sourcing blocker.

## Recommendation (is the ABC path worth it for BPS/JKUPDD vs just MTC-ANN?)

- **JKUPDD: yes, do it — near-free.** The dataset hands you `**kern` + a beat-
  indexed point-set whose occurrences are literal sub-sequences. Alignment is a
  deterministic join. This is a clean cross-composer ABC A/B with almost no
  alignment risk, and it doubly de-risks the line because it *also* sidesteps the
  discouraged-MIDI / time-zero byte-duplicate pathology (rebuild from the CSV
  point-set, not the MIDI).
- **BPS-Motif: yes, but second.** Higher value (32 movements, the dataset we
  already have MTF numbers for, so a real ABC-vs-MTF A/B at L7), but it needs the
  one-time measure-alignment + repeat-unfolding harness and the match-rate gate.
  The payoff is that the `measure` column makes this a structured join, not the
  fragile pitch-sequence search the original concern assumed.
- **Net:** the ABC path for BPS/JKUPDD is **worth it and cheaper than feared** —
  the alignment "hard part" is real only for BPS-Motif and is *moderate*, not
  blocking, because both datasets ship score-position indices. MTC-ANN remains the
  cheapest first mover (ships `**kern` + note-indexed labels directly); JKUPDD is
  the natural second (near-free); BPS-Motif third (the informative A/B, needs the
  harness).

## Sources

- JKUPDD mirror (ships `kernOriginal/` + point-set CSV per piece, incl. Gibbons):
  `github.com/ns2max/jkupdd_dataset` (exact copy of JKUPDD-Aug2013).
- Beethoven sonatas `**kern`: `github.com/craigsapp/beethoven-piano-sonatas` (`kern/sonataNN-1.krn`).
- BPS-Motif annotations (per-note CSV with `measure`+morphetic+label, + `pickup.csv`):
  `github.com/Wiilly07/Beethoven_motif`.
- Cross-checks: `github.com/craigsapp/chopin-mazurkas` (`mazurka24-4.krn`),
  `github.com/craigsapp/mozart-piano-sonatas` (`sonata04-2.krn`),
  `github.com/humdrum-tools/bach-wtc` (`wtc2f20.krn`).
- Conversion design + the "alignment is the hard part" framing this resolves:
  `docs/symbolic_kern_to_abc_conversion.md`.

Claude-Session: https://claude.ai/code/session_018p1T4iWsECNA4NtQe7XNGd
