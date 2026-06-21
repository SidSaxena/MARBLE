# MTC-ANN (Meertens Tune Collection — Annotated) — scoping for the next symbolic-motif benchmark

**Status:** scoping / assessment only. No code was written and no data downloaded for
this doc. It decides *whether and how* to add MTC-ANN as the next cross-texture
benchmark in the CLaMP3-symbolic layer-sweep line of work
([roadmap §2.2](symbolic_motif_benchmark_roadmap.md)).

**Why now:** the layer sweep showed the CLaMP3-symbolic motif signal peaks mid-network
(**L6 MNID / L7 Retrieval**) and the final projected layer **L12 is the weakest**
([MNID](bps_motif_mnid_clamp3_layersweep.md),
[Retrieval](bps_motif_retrieval_clamp3_layersweep.md)) — but that was measured on
**all-Beethoven solo piano** (BPS-Motif). [JKUPDD](jkupdd_retrieval_clamp3_layersweep.md)
just confirmed the **L7 peak generalizes across 5 composers**, but JKUPDD is too
small/easy (165 windows, MAP saturates ~0.89 at *every* layer) to re-expose the L12
penalty. We need a **harder, texture-distant** confirmation. MTC-ANN is the top
candidate: **monophonic Dutch folk melody** — the single biggest texture jump available
from piano classical.

> **HONEST SOURCING CAVEAT (read first).** Web search and fetch were **blocked in the
> environment this doc was produced in**, so the external dataset facts below
> (counts, formats, license, exact annotation layers) come from prior knowledge of the
> MTC literature, **not** from a fresh fetch of liederenbank.nl / the release notes.
> Every number that needs live confirmation is tagged **[verify]**. Treat §1–§3 as a
> well-grounded prior to confirm in one afternoon of fetching, not as gospel. The
> integration analysis in §5 is grounded in the **actual MARBLE code** that was read
> (`midi_to_mtf`, `_BPSMotifSymbolicBase`, `build_jkupdd_retrieval.py`) and is firm.

---

## TL;DR

- **What it is:** MTC-ANN is the small, heavily-annotated subset of the Meertens Tune
  Collections — **~360 monophonic Dutch folk-song melodies** organized into **~26 tune
  families** [verify], with **phrase-boundary annotations** and metadata, curated at the
  Meertens Institute (van Kranenburg et al.). It is the de-facto reference set for
  **tune-family / melodic-similarity** research.
- **The relevance signal we'd use:** the cleanest, most defensible group key is the
  **tune family** (all melodies descended from one "tune" = one relevance group). This
  is a genuinely *different* relevance unit from BPS/JKUPDD (within-piece motif
  occurrence) — it is **cross-melody** similarity under heavy oral-tradition variation.
- **Motif/phrase caveat:** MTC-ANN reliably ships **phrase boundaries** and **tune-family
  labels**; whether it ships explicit **motif-occurrence** annotations (the BPS/JKUPDD
  unit) is **the key thing to verify** — my read is that motif-level relatedness in the
  MTC world is *derived* (via the related-but-separate MTC-FS-INST "motif" experiments
  and the IMA/melodic-feature tooling), **not** a first-class shipped layer of MTC-ANN.
  Do **not** assume a ready-made motif-occurrence ground truth like JKUPDD's.
- **Tokenization mismatch risk — LOW.** I checked: `midi_to_mtf` serializes the **flat
  merged MIDI event stream** (`mid.merged_track`) — it is **not** bar-aligned and **not**
  polyphony-dependent. A monophonic single-line melody is a perfectly valid, just
  shorter, event stream. The M3 patchilizer chunks the resulting MTF text into fixed
  byte patches. So "will a single-line melody tokenize?" → **yes, cleanly**. The real
  question is whether CLaMP3 (trained largely on richer scores) produces *useful
  geometry* on thin monophony, which is exactly what this benchmark would measure.
- **Recommended first task:** **retrieval-by-tune-family** — group = tune family, query =
  whole melody, relevance = same family. It is the most defensible (uses MTC-ANN's
  best-attested annotation), reuses the JKUPDD retrieval datamodule pattern almost
  verbatim, and directly stresses the L6–L7 ≫ L12 claim on a hard, distractor-rich,
  texture-distant pool. **Effort ≈ the roadmap's ~1 day is optimistic but the right
  order of magnitude — budget ~1–2 days** (format conversion + a similarity protocol
  decision are the real costs, not infra).

---

## 1. The paper(s) and what MTC-ANN is

**[verify — sourced from prior knowledge, web fetch was blocked]**

- **Canonical dataset reference.** Peter van Kranenburg, Martine de Bruin, Louis P.
  Grijp, Frans Wiering — *"The Meertens Tune Collections."* Meertens Online Reports,
  Meertens Institute, Amsterdam (the MTC release report). This is the umbrella
  description of the MTC family of corpora.
- **Annotated subset + melodic-similarity protocol.** Peter van Kranenburg,
  *A Computational Approach to Content-Based Retrieval of Folk Song Melodies* (PhD
  thesis, Utrecht University, 2010) and the associated ISMIR/CMMR papers (van
  Kranenburg, Volk, Wiering et al.) establish **tune-family classification / melodic
  similarity** as the task MTC-ANN was built to support. Expert annotation of tune
  families by Meertens folk-song scholars (the "Onder de groene linde" tradition) is
  the provenance.
- **What MTC-ANN *is*.** The **ANN(otated)** instance of the collection: a deliberately
  small, *expert-curated* set chosen so that melodies fall into a controlled number of
  **tune families** — groups of melodies that are historically/musically variants of the
  "same tune." Melodies are **monophonic vocal folk songs** transcribed from the
  Meertens *Liederenbank* (Dutch Song Database). Annotation layers (attested): **tune
  family membership**, **phrase boundaries** (folk-song phrases were manually segmented),
  lyric/strophe metadata, and per-note melodic features.

**Scale [verify]:** on the order of **~360 melodies in ~26 tune families** (the figure I
carry from the MTC-ANN melodic-similarity papers). This is *small by ML standards but
dense in within-family positives* — most families have 10–15 member melodies, which is
exactly what makes it a usable **retrieval** set (every query has multiple relevant
neighbours). **Confirm the exact melody/family counts and the family-size distribution
from the release notes — they drive whether MAP will saturate (JKUPDD problem) or
discriminate (BPS problem).**

---

## 2. The dataset — obtain, format, annotation layers

**[verify — all of §2 needs a live fetch of liederenbank.nl/mtc + the README]**

### Where to get it
- **Home:** the Meertens Tune Collections page, **`https://www.liederenbank.nl/mtc/`**
  (the MTC landing page; download links for MTC-ANN and the larger MTC-FS / MTC-FS-INST
  instances live there).
- **License/terms [verify — DO NOT skip this check].** MTC is distributed for
  **research use** under Meertens Institute terms; historically a click-through /
  research-only agreement rather than a permissive open license. **This is a build
  blocker until confirmed** — unlike JKUPDD (MIREX, freely redistributable) and BPS-Motif
  (GitHub, permissive), MTC's redistribution terms may forbid committing the raw melodies
  into the MARBLE repo. Plan to keep MTC under `data/MTC-ANN/` as a *user-provided* root
  (like JKUPDD's `--jkupdd-root`), **not** vendored.

### Format (what's actually distributed) [verify]
MTC ships melodies in several parallel encodings — my recollection of the set:
- **`**kern`** (Humdrum) — the primary symbolic encoding for the melodies.
- **MIDI** — derived MIDI per melody (this is the format MARBLE's pipeline wants).
- **MusicXML** — likely present for the ANN subset.
- **JSON / "MTC feature" files** — per-note melodic-feature sequences (pitch, contour,
  metric weight, phrase position) produced by the Meertens **melodic-feature** tooling
  (the `MTCFeatures` Python package on PyPI/GitHub by van Kranenburg is the modern
  access layer and exposes melodies as feature sequences + metadata).
- Lyric / metadata sidecars (strophe text, song catalogue IDs).

**Integration consequence:** if MIDI is shipped, we consume it directly. If only
`**kern`/MusicXML is shipped, MARBLE already has the conversion path —
`scripts/data/convert_mxl_to_midi.py` (music21) for MusicXML, and `**kern` → MIDI is a
one-liner via music21/Humdrum tools. **Either way the MIDI hop is solved infra.**

### Annotation layers (the relevance signal — the crux)
1. **Tune family** — *well attested, first-class.* Every melody is labelled with its
   tune family. This is the **strongest, cleanest grouping key** and the one the dataset
   was designed around.
2. **Phrase boundaries** — *well attested.* Manual phrase segmentation per melody. Gives
   a finer unit (a melody → ordered phrases) but **phrase boundaries are not a relevance
   key by themselves** — there's no shipped "this phrase is the same motif as that
   phrase" cross-melody linking (that's the gap below).
3. **Motif / occurrence relatedness** — **THE THING TO VERIFY.** Unlike BPS-Motif and
   JKUPDD, I am **not confident MTC-ANN ships a labelled motif-occurrence ground truth**
   (i.e. "pattern P occurs at these spans in these melodies"). The MTC line of work
   treats *melodic similarity at the whole-tune / tune-family level*; motif/phrase
   pattern discovery exists in adjacent Meertens work but may be **derived**, not a
   shipped annotation. **If you need a JKUPDD-style motif-occurrence task, confirm this
   exists before promising it — otherwise the realistic MTC-ANN task is family-level (or
   phrase-level) retrieval, not motif-occurrence retrieval.**

---

## 3. Evaluation metrics — how MTC-ANN is scored in the literature

**[verify against the van Kranenburg similarity papers]**

The MTC literature evaluates two intertwined things:

1. **Tune-family classification accuracy.** Given a melody, predict its tune family
   (often **leave-one-out k-NN** in some melodic-similarity space). Reported as
   **classification accuracy** (and sometimes per-family). This is the headline MTC-ANN
   metric in van Kranenburg's similarity work — the dataset's *raison d'être* is "can a
   similarity measure recover the expert tune-family grouping?"
2. **Melodic-similarity retrieval.** Rank all other melodies by similarity to a query;
   score with **MAP / precision-recall / R-precision**, relevance = same tune family.
   This is the **MIREX "Symbolic Melodic Similarity" lineage** (MIREX 2005–2007 used a
   related Meertens/RISM-style ground truth and scored with MAP-style measures). This
   framing is a **drop-in match for MARBLE's `CoverRetrievalTask`** (same-`work_id`
   relevance, MAP + recall@K), which is exactly what BPS/JKUPDD already use.

Standard protocol notes to mirror:
- **Leave-one-out / no fixed train-test split** is common for MTC-ANN classification
  (the set is small). For a zero-shot CLaMP3 retrieval probe this is *ideal* — no folds
  needed, just like JKUPDD (`max_epochs: 0`, one pool).
- **Family-balanced** scoring matters: report MAP per family or macro-averaged so a few
  large families don't dominate.

**Where MTC bites harder than JKUPDD:** relevance is **cross-melody** under real
oral-tradition variation (ornamentation, rhythmic stretching, transposition, melodic
drift across centuries of oral transmission). That is a *much* harder invariance than
"same notated pattern re-stated in the same movement." If MAP does **not** saturate here
(unlike JKUPDD), MTC-ANN re-exposes the layer gap the way BPS-Motif does — which is the
entire point of adding it.

---

## 4. How MTC-ANN compares to our existing tasks

| axis | **BPS-Motif** | **JKUPDD** | **MTC-ANN (proposed)** |
|---|---|---|---|
| texture | polyphonic solo piano | polyphonic (5 composers) | **monophonic vocal melody** |
| domain | classical (Beethoven) | classical/Baroque/Renaissance | **Dutch folk (oral tradition)** |
| unit of relevance | within-piece **motif occurrence** `(piece, letter)` | within-piece **pattern occurrence** `(piece, annotator, pattern)` | **tune family** (cross-melody) — or phrase, if motif GT absent |
| relevance is… | literal/near-literal re-statement in one movement | literal/near-literal re-statement in one piece | **variant melodies of the same tune** under heavy oral variation |
| size | 263 motifs / 4,944 occ / 32 movements | 32 patterns / **165 occ** / 5 pieces | **~360 melodies / ~26 families** [verify] |
| CV | movement-level **5-fold** | **none** (one pool) | **none / leave-one-out** (zero-shot, one pool) |
| difficulty / saturation | hard — long occurrence tails, recall@100 ≈ 0.61, **L12 collapses −20%** | **easy** — MAP ~0.89 every layer, recall@50 = 1.0, **L12 only −1.7%** | **unknown but expected HARD** — cross-melody folk variation; the desired re-test of the L12 penalty |
| what it tests | occurrence-tail depth + layer penalty | cross-composer breadth (peak location) | **cross-texture breadth + does the layer penalty survive a hard, distant pool** |
| MARBLE framing | Retrieval (`CoverRetrievalTask`) + MNID | Retrieval (`CoverRetrievalTask`) | **Retrieval (`CoverRetrievalTask`)** ✅ — and optionally MNID-style family classification |

**Which MARBLE framing fits.** A **Retrieval** task is the natural and lowest-risk fit:
group = **tune family**, score same-family-ranked-high via the existing
`CoverRetrievalTask` (MAP, recall@K, mrr). It is structurally identical to JKUPDD's
datamodule — only the grouping key changes from `(piece, annotator, pattern)` to
`(tune_family)`. A secondary **MNID-style classification** (predict the tune family with
a frozen-encoder + small head, the BPS-MNID pattern) is also available and would give a
*supervised separability* read to pair with the zero-shot retrieval read — but it needs
a train/val/test split over a small set, so retrieval-first is cleaner.

**Mismatch / risk flags (be explicit):**
- **Unit ambiguity — the biggest design decision.** BPS/JKUPDD relevance is a
  *sub-melody occurrence window*. MTC-ANN's first-class relevance is a *whole melody's
  family*. These are **not the same task** — MTC-ANN-by-family measures *melodic
  similarity / cover-song-like retrieval*, not *within-piece motif discovery*. That is
  **fine and arguably better** for the breadth question (it's a genuinely different,
  harder invariance), **but the doc/thesis must not conflate "MTC-ANN retrieval" with
  "motif-occurrence retrieval"** — it's a tune-family / melodic-similarity task. If a
  true motif-occurrence task is wanted, it depends on §2.3's unverified motif GT.
- **Whole melody vs window length.** BPS/JKUPDD windows are short motif spans; MTC-ANN
  melodies are whole folk songs (tens of seconds). They will produce **longer MTF
  patch sequences** — confirm they fit under M3's `PATCH_LENGTH = 512` cap (the
  `_tokenise` path truncates beyond it). Folk melodies are short enough that this is
  *probably* fine, but **a long-melody truncation check is part of the build**. (If you
  also build a phrase-level task, phrases are short and dodge this entirely.)
- **Monophonic geometry, not tokenization.** Re-stated from the TL;DR because it's the
  load-bearing risk reframe: tokenization is **safe** (event-stream `midi_to_mtf`), but
  CLaMP3 was trained predominantly on richer scores. The open empirical question MTC-ANN
  answers is whether the **L6–L7 geometry stays useful on thin monophony** — that's a
  feature of the experiment, not a blocker.

---

## 5. Integration assessment for MARBLE

The integration template is **already proven** by JKUPDD: a build script that enumerates
items into a JSONL with a `group` relevance key, plus a datamodule that subclasses
`_BPSMotifSymbolicBase`. MTC-ANN slots into the same shape.

**What a build script + datamodule need:**

1. **`scripts/data/build_mtc_ann_retrieval.py`** — parse the MTC-ANN root, and for each
   melody emit a JSONL record `{midi_path, tune_family, group: "<family>", song_id,
   split: "test"}`. Mirrors `build_jkupdd_retrieval.py` almost line-for-line; the only
   real logic is reading the tune-family label out of the MTC metadata (or via the
   `MTCFeatures` package) and the format hop:
   - if MIDI ships → copy/enumerate directly;
   - if `**kern`/MusicXML → MIDI via music21 (`convert_mxl_to_midi.py` already exists for
     MusicXML; `**kern` → MIDI is the same one-liner). **This is solved infra.**
2. **`marble/tasks/MTCANNRetrieval/datamodule.py`** — a ~30-line clone of
   `JKUPDDRetrieval/datamodule.py`: subclass `_BPSMotifSymbolicBase`, return
   `(patches, work_id, midi_path, clip_id)`, with `work_id = hash(tune_family)` (reuse
   JKUPDD's `_work_id` SHA-1→int helper verbatim). **`midi_to_mtf` needs no change** —
   verified it's texture-agnostic.
3. **A config** `configs/probe.CLaMP3-symbolic-layers.MTCANNRetrieval.yaml` — copy the
   JKUPDD layer-sweep config, swap the task. Then `run_sweep_local.py` + a
   `mtc_ann_retrieval_summary.py` (clone of `jkupdd_retrieval_summary.py`).

**Realistic effort — the roadmap guessed ~1 day; correct it to ~1–2 days.** The *code* is
genuinely ~1 day (it's a JKUPDD clone). What the 1-day estimate omits:
- **License/provenance check** (could be a hard blocker or a click-through — must be done
  first; non-trivial calendar time, even if low effort).
- **A similarity-protocol decision**: tune-family vs phrase vs (if it exists) motif — this
  is the design call that makes the task meaningful, not mechanical.
- **The format/metadata-parse step** is slightly fiddlier than JKUPDD's ready-made
  per-occurrence MIDIs (MTC ships whole melodies + a metadata table to join family
  labels), and the long-melody truncation check.

**Top 3 risks / unknowns (ranked):**
1. **License / redistribution [highest].** May forbid vendoring the data → must run as a
   user-provided root. Resolve before any build. (Verify on liederenbank.nl.)
2. **Does a motif-occurrence ground truth exist? [high].** If we want a JKUPDD-style
   *motif* task, it hinges on this. My prior: **no first-class motif-occurrence layer in
   MTC-ANN** → default to **tune-family retrieval**, which is well-attested. Don't promise
   motif-occurrence until confirmed.
3. **Will MAP saturate or discriminate? [medium].** If MTC-ANN turns out *easy* (like
   JKUPDD) it won't re-expose the L12 penalty and adds breadth but not the hard re-test
   we wanted. The cross-melody oral-variation relevance makes this *unlikely* — it should
   be hard — but it's the empirical bet the whole add is making. (Family-size
   distribution from §1 is the early tell.)

---

## 6. Recommendation

**Build MTC-ANN as a `CoverRetrievalTask` keyed on tune family** (retrieval-by-tune-family)
as the **first** MTC-ANN task, then run the 13-layer CLaMP3-symbolic sweep on it.

**Why this over the alternatives:**
- **vs retrieval-by-phrase/motif:** tune-family is the **best-attested, first-class**
  MTC-ANN annotation; phrase/motif relevance is finer but its cross-melody linking is
  **unconfirmed** and may not ship. Don't bet the build on an annotation we couldn't
  verify.
- **vs tune-family classification (MNID-style):** retrieval is **zero-shot** (no
  train/val/test split needed on a small set, exactly like JKUPDD), reuses
  `CoverRetrievalTask` and the existing summary/plot tooling, and reads the **same ruler**
  (MAP, recall@K) as BPS/JKUPDD so the layer curves are directly comparable. Classification
  is a good *second* task once retrieval lands.
- **It directly serves the motivating question:** monophonic folk, cross-melody variation
  → the **hardest, most texture-distant** test of "does L6–L7 ≫ L12 survive?" If the L12
  penalty re-appears here the way it does on BPS (and unlike JKUPDD), the architectural
  claim is locked across *texture*, not just *composer*. If it doesn't, we learn the
  penalty is genuinely difficulty-bound — also a publishable, thesis-relevant result.

**Sequencing:** do the **license check + count/format/annotation verification fetch first**
(one afternoon on liederenbank.nl + the `MTCFeatures` repo), *then* the ~1-day JKUPDD-clone
build, *then* the sweep. Bundle MTC-ANN with the other breadth checks **before** spending
LoRA budget (roadmap §4 step 2), since it's the cheapest place to discover the layer claim
is texture-bound.

---

## 7. Cross-links

- [`symbolic_motif_benchmark_roadmap.md`](symbolic_motif_benchmark_roadmap.md) §2.2 — the
  preliminary MTC-ANN take this doc verifies/corrects (corrections: effort ~1–2d not ~1d;
  **motif-occurrence GT is unconfirmed → default to tune-family**; tokenization risk
  reframed as *geometry* risk, not a *format* risk).
- [`jkupdd_retrieval_clamp3_layersweep.md`](jkupdd_retrieval_clamp3_layersweep.md) — the
  integration template (datamodule + build-script pattern) MTC-ANN clones; and the
  saturated-pool behaviour MTC-ANN is meant to *escape*.
- [`bps_motif_retrieval_clamp3_layersweep.md`](bps_motif_retrieval_clamp3_layersweep.md) /
  [`bps_motif_mnid_clamp3_layersweep.md`](bps_motif_mnid_clamp3_layersweep.md) — the L6–L7
  peak / L12 collapse this benchmark stress-tests on a new texture.
- Code touched to ground §5:
  `marble/tasks/JKUPDDRetrieval/datamodule.py`,
  `marble/tasks/BPSMotif/datamodule.py` (`_BPSMotifSymbolicBase`),
  `marble/encoders/CLaMP3/midi_util.py` (`midi_to_mtf`, texture-agnostic),
  `scripts/data/build_jkupdd_retrieval.py`, `scripts/data/convert_mxl_to_midi.py`.
