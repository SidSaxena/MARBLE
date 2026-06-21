# JKUPDD motif retrieval — score-native **ABC** vs lossy **MIDI→MTF**

**Question.** CLaMP3-symbolic's M3 encoder was trained on two text views of
music: **MTF** (a lossless serialisation of MIDI *performance* — exact
ticks/velocities, message-segmented) and **interleaved ABC** (bar-segmented
*notation* — key, pitch spelling, meter, beaming, slurs, multi-voice). The
current JKUPDD/BPS-Motif pipelines feed MTF, produced by a lossy
`**kern → MIDI → MTF` round-trip. Does feeding **score-native ABC** instead —
built directly from the piece `**kern`, preserving the notation the MIDI
round-trip throws away — improve cross-piece motif retrieval?

> **⚠️ This document was corrected after an adversarial audit.** The original
> conclusion ("ABC is decisively *worse*, MAP 0.527 vs 0.825, and shows no
> depth peak") was a **content confound**, not a representation effect: the ABC
> arm embedded the *whole polyphonic measure span* (all voices,
> accompaniment-laden) while the MTF arm embedded the *isolated motif*. The two
> arms were embedding **different musical objects**. After rebuilding ABC at
> **note level** (only the matched motif notes, single-voice, re-zeroed — the
> same object MTF embeds), the result **flips**: content-matched ABC **matches
> and slightly beats** MTF and **reproduces a depth peak**. The original
> (confounded) numbers are preserved below for the record; see
> [The content confound](#the-content-confound-what-went-wrong) and
> [Corrected result](#corrected-result--note-level-abc-vs-mtf-identical-66-occurrence-pool).

**Answer (corrected): yes — content-matched note-level ABC matches-to-beats MTF,
and the depth peak returns.** On the identical 66-occurrence pool, note-level
ABC scores **MAP 0.862 at its best layer (L11) vs MTF's 0.825 (L7), Δ = +0.037**,
wins at 11 of 13 layers, and shows a **clear rising depth profile** peaking deep
in the stack (the exact mid-stack mechanism the original confounded run claimed
ABC *destroyed*). For CLaMP3 motif retrieval, notation-preserving ABC is **not**
worse than the lossy-MIDI MTF path — it is competitive-to-better, **once the two
arms embed the same notes**.

## The content confound (what went wrong)

The first version of this A/B built each ABC occurrence with
`score.measures(lo, hi)` — the **whole polyphonic measure span, every voice**
covering the motif. The MTF arm, by contrast, embedded JKUPDD's shipped
per-occurrence MIDI = the **isolated motif** (the notes of that occurrence,
time-zeroed). So the two "arms" were not the same musical object at all:

| | ABC arm (old) | MTF arm |
|---|---|---|
| what was embedded | whole 1–3-bar **polyphonic** measure span, all voices + accompaniment | the **isolated motif** (occurrence notes only) |
| notes per occurrence vs motif notes | **≈ 6× on mean, up to 15.8×** | ≈ 1.1× |
| example (Bach `bruhn A occ1`, an 8-note motif) | 3-bar / 2-staff ABC, **24–99 note-heads** | 8 notes |

The earlier doc even flagged ABC was "coarser… median 21 patches" and reasoned
about token counts — but the real story was simpler and worse: ABC was carrying
**the surrounding accompaniment**, so it was being asked to retrieve a *different
and harder thing* (a bar of polyphony) than MTF (a clean motif), and unfairly
lost.

**Audit evidence (what was and wasn't broken).** An adversarial audit confirmed:

- **Alignment is CORRECT** — the point-set→kern join resolves the *right* motif
  notes (piece-level kern↔point-set pitch-agreement 1.000 Bach / 0.994 Chopin /
  0.999 Mozart; the matched notes are the motif notes).
- **Tokenisation is clean** — converter21 → MusicXML → xml2abc → abctoolkit
  interleave produces well-formed ABC (correct `K:`/`M:`/`Q:`, accidentals,
  voices); no embedding collapse.
- **The defect was purely the slice granularity** — `score.measures()` grabs all
  voices in the bar range, so the motif's 8 notes were embedded inside ~85 notes
  of texture. The alignment found the right notes; the build then **threw the
  isolation away** by slicing whole measures.
- **A matched-content control settled it.** On whole-piece monophonic MTC-ANN
  (where ABC and MTF embed the *same* single-line melody — no accompaniment to
  confound), **ABC beats MTF (0.783 vs 0.642 @ L7), with a depth peak**. So ABC
  is a good representation; JKUPDD's ABC arm was simply confounded. (MTC-ANN is
  out of scope for *this* doc — recorded here only as the control that motivated
  the fix.)

## The fix — note-level ABC (`_build_motif_abc`)

`scripts/data/build_jkupdd_abc.py` now reconstructs **only the occurrence's
matched notes** as a **single-voice** music21 stream and runs `score_to_abc` on
*that*, instead of slicing whole measures. Per matched note it preserves:

- **pitch** — the matched point-set pitch (`kn["midi"]`; on a chord, the specific
  constituent the row matched);
- **duration** — the note's *notated* `quarterLength` from the score (the
  occurrence CSV doesn't reliably carry duration for these pieces);
- **relative onset** — each note's offset minus the motif's first onset, so the
  fragment is **re-zeroed to t=0** exactly like the MTF occurrence MIDI; inter-
  onset gaps inside the motif are preserved (the MTF MIDI has them too).

Key / meter headers are carried from the motif's first note's context so the ABC
`K:`/`M:` reflect the real tonal/metric context (pitch spelling via accidentals
is preserved regardless). The result is a short **monophonic ABC line of the
motif** — the same notes as the MTF window, not the accompaniment.

### Content parity — the gate (verified before re-running)

The build now reports ABC note-heads over (a) the matched motif notes and (b)
the MTF-window notes. Both must be ≈ 1.0 (the old build ran ≈ 6× / up to 15.8×):

| ratio | mean | max | n | offenders > 1.5× |
|---|---:|---:|---:|---:|
| ABC-heads / motif notes | **1.023** | 1.309 | 66 | **0** |
| ABC-heads / MTF-window notes | **1.020** | 1.309 | 66 | **0** |

`n_matched == n_mtf_notes` for **59/66** occurrences; the 7 small mismatches are
all Chopin and off by 1–5 notes (a grace/ornament in the MIDI not in the matched
point-set). The ≈1.02 mean is a counting artifact — the note-head regex catches a
few tie/duration tokens — **not** extra musical content. Spot-checks confirm
**all 66 fragments are single-voice** (max distinct voices across the set = 1;
zero polyphonic), short monophonic motif lines. Before vs after, same Bach motif:

```
bachBWV889Fg__bruhn__A__occ1  (an 8-note motif)
  OLD (whole-measure):  24 ABC note-heads  (3.0× the motif; 2 voices, 3 bars)
  NEW (note-level):       8 ABC note-heads  (1.0×; single voice)
                          [V:1]E2 C2 F2 ^G,2 | x3 D B,E C x |   ← the motif, monophonic
bachBWV889Fg__bruhn__A__occ2..5: OLD 85–99 heads (10–12×) → NEW 8 heads (1.0×)
```

## Setup — an apples-to-apples A/B on one occurrence pool

Both arms are zero-shot CLaMP3-symbolic per-layer sweeps (13 layers + a
mean-of-all-layers baseline, `max_epochs: 0`, no CV folds) over the **same
occurrence pool** with **identical `work_id`/relevance** (same
`(piece, annotator, pattern)` grouping, same `_work_id` hash). The only thing
that differs is the *input text* fed to the identical `M3Patchilizer.encode` →
CLaMP3 encoder:

- **MTF arm** (`JKUPDDRetrievalMatched`): each occurrence's lossy MIDI window →
  `midi_to_mtf` → patches. The existing pipeline, restricted to the aligned
  subset. (Reused unchanged — same cached embeddings as before.)
- **ABC arm** (`JKUPDDRetrievalABCnote`): each occurrence's **note-level**
  motif-only single-voice ABC (the fix above) → patches. Run under a **distinct
  task tag** (`JKUPDDRetrievalABCnote`) so its embedding cache + output dirs do
  **not** reuse the prior (confounded) `JKUPDDRetrievalABC` run.

The pool is the **66 occurrences / 15 groups** that survived ABC alignment (Bach,
Chopin, Gibbons, Mozart; Beethoven dropped — see
[Alignment](#alignment--the-docs-cleanlow-risk-claim-is-only--true-a-real-finding)) —
a strict subset of the dedup'd 78/20 MTF set the
[MTF leaderboard](jkupdd_retrieval_clamp3_layersweep.md) used. The MTF arm was
re-run on exactly these 66 (`data/JKUPDD/JKUPDDRetrieval.matched.test.jsonl`).
Its L7 MAP is **0.825**, essentially the full-78 leaderboard's **0.843** — the
subset behaves like the whole, so the ABC↔MTF gap below is a representation
effect, not a pool artifact.

## Corrected result — note-level ABC vs MTF (identical 66-occurrence pool)

Raw `test/map` (Δ = ABC − MTF; positive ⇒ ABC helps):

| layer | ABC | MTF | Δ (ABC−MTF) |
|------:|----:|----:|------------:|
| 0 | 0.6768 | 0.6980 | −0.0212 |
| 1 | 0.7305 | 0.6547 | +0.0757 |
| 2 | 0.7883 | 0.6488 | **+0.1395** |
| 3 | 0.7735 | 0.6739 | +0.0996 |
| 4 | 0.7790 | 0.7160 | +0.0630 |
| 5 | 0.7937 | 0.7494 | +0.0443 |
| 6 | 0.8144 | 0.7852 | +0.0292 |
| **7** ⭐MTF | 0.8248 | **0.8250** | −0.0002 |
| 8 | 0.8232 | 0.8036 | +0.0197 |
| 9 | 0.8541 | 0.8100 | +0.0441 |
| 10 | 0.8466 | 0.8054 | +0.0412 |
| **11** ⭐ABC | **0.8615** | 0.8076 | **+0.0539** |
| 12 | 0.8474 | 0.7995 | +0.0479 |
| **meanall** | **0.8618** | 0.7873 | **+0.0746** |

![ABC vs MTF per-layer MAP](jkupdd_abc_vs_mtf.png)

(Centered MAP tells the same story — every cell within ~0.01 of raw; full table
in `jkupdd_abc_vs_mtf_leaderboard.csv`.)

### For comparison — the OLD (confounded) numbers, NOT to be cited

For the record only — these are the *whole-measure ABC* vs MTF numbers the
original version of this doc reported. They are **invalid** (content confound):

| layer | ABC (old, confounded) | MTF | Δ (old) |
|------:|----:|----:|------------:|
| 0 ⭐old-ABC | 0.5270 | 0.6980 | −0.1710 |
| 7 ⭐MTF | 0.5001 | 0.8250 | −0.3249 |
| meanall | 0.5053 | 0.7873 | −0.2819 |

The old ABC peaked at the *surface* (L0 = 0.527) with a flat 0.48–0.53 profile
and "no mid-stack peak" — but that flatness was the encoder failing to retrieve a
*bar of polyphony*, not a property of notation. Correcting the content moves
ABC's peak from 0.527 → **0.862** (+0.335) and restores the depth profile.

## The three questions, answered concretely (corrected)

**(a) Does ABC beat MTF, and by how much at the peak?** **Yes — it wins.** ABC's
best layer (L11) is **0.862**; MTF's best (L7) is **0.825** — a **+0.037 MAP
peak gap** in ABC's favour. ABC has a **positive Δ at 11 of 13 layers** (only L0
−0.02 and L7 ≈0 are non-positive); the largest gap is **+0.14** (L2, where MTF is
deep in its surface trough). On the all-layer mean ABC leads by **+0.075**
(0.862 vs 0.787).

**(b) Does the mid-stack/depth peak reappear for ABC?** **Yes — emphatically.**
The confounded run's headline was "ABC has no interior maximum, it's flat." With
matched content ABC shows a **clear rising depth profile**: 0.677 (L0) → 0.79
(L2) → 0.81 (L6) → 0.83 (L8) → **0.862 (L11)**, then a mild last-layer taper
(L12 = 0.847). It actually peaks **deeper** than MTF (L11 vs L7) and its
deep-stack cells (L9–L12) all sit *above* MTF's peak. The notation representation
does develop the deep-layer motif-identity signal after all — the original claim
that "ABC defeats the depth mechanism" was an artifact of the confound.

**(c) Is the peak sharper or flatter?** ABC's curve spans 0.677–0.862 (range
0.185) with a genuine deep-stack maximum and a smooth climb — *more* structured
than the old flat 0.48–0.53 band. ABC's `meanall` (0.862) ≈ its best layer,
i.e. the deep layers collectively carry the signal (unlike the old run where
meanall ≈ per-layer because *no* layer was informative).

**Secondary metrics (best layer each).** recall@1 **ABC 0.289 vs MTF 0.254**;
recall@5 **ABC 0.773 vs MTF 0.758**; recall@10 ABC 0.913 vs MTF 0.925 (MTF
marginally ahead here); recall@50 = 1.000 for both (the 66-window pool has < 50
distractors per query, so @50 trivially covers every relevant — read the lower-K
recalls). ABC leads at recall@{1,5}; the arms are within noise at recall@10.

## Alignment — the doc's "clean/low-risk" claim is only ⅘ true (a real finding)

*(Unchanged by the fix — alignment was always correct; only the slice changed.)*

To build the note-level ABC we map each occurrence's JKUPDD point-set (the
per-occurrence `occurrences/csv/occN.csv`, rows `(ontime, midi)`) onto the piece
`**kern` and take the **matched notes themselves** as the ABC content. The
bridge: parse the kern with converter21, flatten to per-note `(offset, midi)`,
calibrate the constant per-piece origin shift between kern offset and point-set
ontime (Bach +1.0; the pickup pieces −1.0), and key every kern note by its
`(ontime, midi)`. Each occurrence row is then a direct lookup. We report a
**per-occurrence note match-rate** and gate on it (`--min-match-rate 0.9`).

`docs/kern_sourcing_bps_jkupdd.md` predicted JKUPDD would be **"clean / low-risk
— essentially a deterministic join."** In practice it is clean for **4 of 5
composers** but **breaks on the 5th**:

| piece | kern↔point-set pitch-agree | occ note match | status |
|---|---:|---:|---|
| Bach BWV 889 (fugue) | **1.000** | 21/21 perfect | clean (needed a `*staff` renumber) |
| Chopin Op.24/4 (mazurka) | 0.994 | 22/22 (≥0.94) | clean |
| Mozart K.282/2 (menuetto) | 0.999 | 8/8 perfect | clean |
| Gibbons *Silver Swan* | 0.816 | 15/19 (4 dropped) | mostly clean |
| **Beethoven Op.2/1 mvt-3** | **0.003** | **0/8** | **dropped entirely** |

- **Beethoven is unusable.** converter21 parses this written-out-repeat Menuetto
  into a score whose every note's offset-in-hierarchy **collapses to a single
  value** — the ontime bridge is then meaningless. The builder *detects* this
  degeneracy and **drops the piece** rather than mis-slicing. All 8 Beethoven
  occurrences are lost.
- **The point-set is the *unfolded* performance; the kern is *folded* notation.**
  For repeat-heavy pieces the ontime↔kern map is non-monotone past the first
  section, so a few occurrences straddling a repeat boundary fall below the 0.9
  gate (4 Gibbons `tomCollins` occurrences).
- **Two converter21 quirks, both handled.** (a) Bach's `wtc2f20.krn` shares a
  staff number across spines, which converter21 rejects; the builder renumbers
  to unique staves (changes layout, not pitch/rhythm). (b) Tie-straddling motifs
  can hit a music21 `duplex-maxima` MusicXML-export bug; the builder retries with
  `stripTies`.

**Net alignment outcome:** 66/78 occurrences aligned; **mean per-occurrence note
match-rate 0.967, 59/70 perfect**; 15 groups, **no singletons** (every surviving
group still has ≥2 occurrences → a valid query pool).

## Verdict (corrected)

**Score-native, note-level ABC matches-to-beats the lossy MIDI→MTF path for
cross-piece CLaMP3 motif retrieval, and reproduces the mid-stack depth peak.** On
a controlled 66-occurrence pool the content-matched ABC path leads MTF by
**+0.037 MAP at the peak** (0.862 vs 0.825) and **+0.075 on the all-layer mean**,
winning at 11 of 13 layers, with a clear deep-stack peak (L11). The original
"ABC loses by ~0.30" was a **content confound** — whole-measure ABC (all voices,
accompaniment) vs isolated-motif MTF — *not* a representation effect. Once both
arms embed the **same notes**, notation-faithful ABC is the equal-or-better
input.

**Actionable take (reversed from the original):** ABC-ifying the symbolic line is
**viable** for JKUPDD/BPS-Motif motif retrieval — **provided you slice the motif
itself, not its measures**. Use note-level (motif-only, single-voice) ABC and a
**deep layer (L9–L11)**, not the surface. The confounded conclusion ("keep MTF,
don't ABC-ify") is withdrawn; ABC is validated for fragment-level retrieval here.
The one honest caveat is that this is a **small, saturated** benchmark (below).

## Caveats

- **Thin & saturated.** 66 windows, 15 groups, 4 composers, no folds → no error
  bars. The **direction** (ABC ≥ MTF, ABC depth-peaked) is consistent across
  layers and centered/raw, but treat the **magnitudes** as directional.
  recall@50 = 1.0 both arms confirms the pool is small/easy — the discriminating
  signal is in MAP and low-K recall.
- **The content-parity fix is the load-bearing change.** The whole reversal
  hinges on embedding the motif notes, not the bar. The parity numbers
  (ABC/motif ≈ 1.0, zero offenders, all single-voice) are reported above and
  re-checked by the builder on every run — re-verify them before trusting any
  future re-run.
- **Pool is the aligned subset, not all 78.** Beethoven (8 occ) is absent — its
  kern won't parse cleanly under converter21. The A/B is honest *on the 4
  composers that align*.
- **One encoder, one task.** This is CLaMP3-symbolic on within-piece JKUPDD
  retrieval; it does not by itself settle ABC-vs-MTF for *generation*,
  *classification*, or other encoders.

## Reproduce

```bash
cd ~/developer/python/marble    # PC: /home/sid/developer/marble (WSL)
# 0. source JKUPDD groundTruth occurrence CSVs (sparse, blobless):
git clone --filter=blob:none --sparse \
    https://github.com/ns2max/jkupdd_dataset.git data/kern_sources/jkupdd_sparse
( cd data/kern_sources/jkupdd_sparse && \
  git sparse-checkout set --no-cone \
    "groundTruth/*/*/repeatedPatterns/*/*/occurrences/csv/*.csv" )
#    the per-piece **kern + full-piece point-set live under data/kern_sources/JKUPDD/

# 1. build the NOTE-LEVEL ABC set + the matched-MTF subset (mirrors dedup'd MTF
#    identity). The stats JSON includes the content-parity gate (abc_over_motif,
#    abc_over_mtf_window — both must be ≈ 1.0, zero offenders > 1.5×):
uv run python scripts/data/build_jkupdd_abc.py \
    --jkupdd-root data/kern_sources/jkupdd_sparse \
    --kern-dir    data/kern_sources/JKUPDD       # --min-match-rate 0.9 (default)

# 2. both 13-layer + meanall sweeps (zero-shot, CUDA, ~2.5 min each). Use the
#    ABCnote task tag so the cache + output dirs don't reuse the old ABC run:
.venv/bin/python scripts/sweeps/run_sweep_local.py \
    --base-config configs/probe.CLaMP3-symbolic-layers.JKUPDDRetrievalABCnote.yaml \
    --num-layers 13 --model-tag CLaMP3-symbolic --task-tag JKUPDDRetrievalABCnote \
    --accelerator gpu --skip-fit-if-no-train --concurrency 2
# (the matched-MTF arm — task-tag JKUPDDRetrievalMatched — is reused unchanged.)
# backfill wandb sweep coords (idempotent — the live callback already stamps them):
.venv/bin/python scripts/analysis/fix_wandb_runs.py --apply coords \
    --group "CLaMP3-symbolic / JKUPDDRetrievalABCnote"

# 3. aggregate the A/B + figure:
.venv/bin/python scripts/sweeps/jkupdd_abc_vs_mtf_summary.py \
    --abc-tag JKUPDDRetrievalABCnote --mtf-tag JKUPDDRetrievalMatched \
    --out-csv docs/jkupdd_abc_vs_mtf_leaderboard.csv
.venv/bin/python scripts/sweeps/plot_jkupdd_abc_vs_mtf.py \
    --csv docs/jkupdd_abc_vs_mtf_leaderboard.csv --out docs/jkupdd_abc_vs_mtf.png
```

wandb: project `marble`, groups `CLaMP3-symbolic / JKUPDDRetrievalABCnote`
(corrected note-level ABC, 14 runs) and `CLaMP3-symbolic / JKUPDDRetrievalMatched`
(MTF baseline, 14 runs). The prior `CLaMP3-symbolic / JKUPDDRetrievalABC` group
is the **confounded** whole-measure run — kept for provenance, not cited.

Claude-Session: https://claude.ai/code/session_018p1T4iWsECNA4NtQe7XNGd
