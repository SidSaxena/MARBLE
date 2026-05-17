# Music Structure Analysis — dataset survey & implementation ranking

Comprehensive survey of music structure analysis (MSA) datasets that
MARBLE could turn into encoder-benchmarking tasks. Written 2026-05-17
after surveying ISMIR / TISMIR / ArXiv 2019–2026.

**Purpose:** rank candidate datasets by impact and implementation cost
so future sweep additions can be prioritised without re-doing the
literature search.

**Status of current MARBLE structure tasks:**
- ✅ `HookTheoryStructure` — implemented, swept on 4 encoders
- 🚧 `HXMSA` (Harmonix) — implemented, queued for sweep (see
  `docs/data/hxmsa_setup.md`)
- 📝 `SuperMario Structure` — planned, runbook drafted (see
  `docs/data/supermario_setup.md`)

This survey covers everything else. Companion to
[`layer_analysis.md`](layer_analysis.md) and
[`leitmotif_findings.md`](leitmotif_findings.md).

---

## TL;DR — implementation priority ranking

| Rank | Dataset | Domain | Why this rank | Est. impl cost |
|---:|---|---|---|---|
| **1** | **SongFormBench** (Oct 2025) | General pop | Newest SOTA benchmark. 300 expert-verified tracks including 200 revised Harmonix subset. Open-sourced training code from the same team. Directly comparable to a current SOTA paper. | ~3 h |
| **2** | **SuperMario Structure** | Video game | Already planned. VGM-native labels (`loop`, `stinger`) complement HXMSA. Symbolic + audio path. High annotation agreement (95–97%). | ~5 h |
| **3** | **BPS-Motif** | Classical (Beethoven) | DIRECT relevance to our leitmotif work — 263 motifs + 4,944 occurrences in Beethoven sonatas. Symbolic, renderable via SGM-Pro. Peer-reviewed annotations. | ~6 h |
| **4** | **JSD** (Jazz) | Jazz | 340 tracks, 3000+ segments, includes solo/accompanying instrument labels. Extends genre coverage; complementary domain to pop+VGM. | ~4 h |
| **5** | **TAVERN** (Theme & Variation) | Classical | 27 theme-variation sets by Mozart and Beethoven. Direct classical analogue to our VGMIDITVar leitmotif task. | ~5 h |
| 6 | **BPSD** | Classical (Beethoven) | Multi-version Beethoven sonatas with coherent alignment. Enables a cross-version retrieval task (parallel to Covers80). | ~6 h |
| 7 | **NES-VMDB** | Video game | 474 hours from 389 NES games. Huge, with gameplay video pairing. Most ambitious VGM dataset. | ~8 h |
| 8 | **SALAMI** | Multi-genre | 1356 tracks, 2400+ multi-annotator structure annotations, 3-level hierarchy. Gold standard for boundary detection — but partially overlaps Harmonix and we already have that. | ~8 h |
| 9 | **SongFormDB** | General | 14k songs (the SongFormBench training corpus). Largest MSA corpus. Hugely diverse but supervision is heterogeneous/noisy. | ~10 h |
| 10 | **Raveform** | EDM | 1,423 DJ-mix tracks with EDM-specific structural vocabulary. Niche but only EDM-focused MSA dataset. | ~5 h |
| 11 | **Annotated Mozart Sonatas** | Classical | All Mozart piano sonatas with harmonic + phrase + cadence. Mostly harmonic analysis, not segmentation. Useful for cross-task probing. | ~4 h |
| 12 | **YM2413-MDB** | Video game (8-bit) | 669 FM-synthesis VGM songs with emotion labels. 8-bit domain is narrow but distinctive. | ~6 h |
| 13 | **OSSL** (Movie clips) | Film | 36.5 hours of public-domain movie clips with mood annotations. Mostly mood not structure. | ~4 h |
| 14 | **Mozart Texture Annotations** | Classical | 9 movements, 1164 piano-texture labels. Narrow but interesting probe of texture-specific encoder features. | ~3 h |
| 15 | **Isophonics / RWC Popular** | Western pop / J-pop | Classic benchmarks (Beatles 180 tracks; RWC-P 100 tracks). Well-trodden. Mostly historical interest now. | ~5 h |

**Recommended sequence:** finish HXMSA → SuperMario → SongFormBench →
BPS-Motif. That gives us coverage of: large pop (HXMSA), VGM (SuperMario),
SOTA general (SongFormBench), and classical motif-retrieval (BPS-Motif)
— four complementary axes in ~14 hours of implementation.

---

## Detailed dataset profiles

### Tier 1 — top priority

#### 1. SongFormBench / SongFormDB (October 2025) — most recent SOTA

| | |
|---|---|
| Source | https://github.com/ASLP-lab/SongFormer · [HF dataset](https://huggingface.co/datasets/ASLP-lab/SongFormBench) · [paper](https://arxiv.org/abs/2510.02797) |
| Domain | General pop, multi-language (English + Chinese) |
| Size | **300 expert-verified tracks** (benchmark); **14k+ tracks** (training, SongFormDB) |
| Annotation | Functional segment labels + boundaries. SongFormBench is 200 manually-revised Harmonix tracks + 100 Chinese songs. |
| Audio | Distributed (HF dataset) |
| Eval metrics | HR.5F (strict boundary F1), HR3F (relaxed), functional label accuracy |
| SOTA result | SongFormer achieves HR.5F=0.703, ACC=0.807 on SongFormBench; beats Gemini 2.5 Pro |
| **Why prioritise** | (1) Most recent published benchmark in MSA. (2) Directly comparable to a current SOTA paper. (3) Includes a Chinese-language subset — tests cross-lingual generalisation our other tasks don't. (4) The 200-track Harmonix subset overlaps HXMSA so we can sanity-check our HXMSA results against SongFormBench numbers. (5) Open-source SongFormer code provides a baseline to beat. |
| Implementation | Audio + JSONL distributed via HF — just download. Datamodule is HookTheoryStructure-style clone. ~3 hours. |
| Overlap with existing | 200 Harmonix tracks overlap HXMSA — we'd avoid those for training but use them as held-out comparison. |

#### 2. SuperMario Structure (already planned)

See [`docs/data/supermario_setup.md`](data/supermario_setup.md). Tier 1
because VGM-native labels (`loop`, `stinger`) genuinely complement
HXMSA's pop-music labels, and the symbolic + audio dual-pathway means
CLaMP3-symbolic can compete here in a way it can't on Harmonix.

#### 3. BPS-Motif — Beethoven motifs (ISMIR 2023)

| | |
|---|---|
| Source | https://github.com/Wiilly07/Beethoven_motif · [Zenodo](https://zenodo.org/records/10265277) · [paper](https://archives.ismir.net/ismir2023/paper/000032.pdf) |
| Domain | Classical (Beethoven's first movements, 32 piano sonatas) |
| Size | ~127k notes, **263 manually annotated motifs**, **4,944 motif occurrences**. Section, subsection, phrase, and motif intervals. |
| Annotation | Note-level motif marking via peer-reviewed process (7 reviewers with composition background) |
| Audio | NOT distributed; symbolic input. Audio renderable from MIDI via SGM-Pro (reuses our VGMIDI infrastructure). |
| Eval | Motif retrieval precision/recall, boundary F1 |
| **Why prioritise** | This is the closest classical-music analogue to leitmotif. The user's broader research interest IS leitmotifs in soundtracks; Beethoven motif discovery is the same problem in the classical canon. Peer-reviewed annotation quality. |
| Implementation | Symbolic renderable. Two tasks possible: (a) motif occurrence detection (frame-level), (b) motif-to-motif similarity retrieval. Start with (b) as it parallels our existing VGMIDITVar retrieval setup. ~6 hours. |
| Caveats | Smaller than Harmonix (32 pieces vs 912). Need to think carefully about train/test split — splitting Beethoven sonatas randomly may put motifs from the same sonata on both sides. |

### Tier 2 — strong candidates

#### 4. JSD (Jazz Structure Dataset, TISMIR 2022)

| | |
|---|---|
| Source | https://github.com/stefan-balke/jsd · [paper](https://transactions.ismir.net/articles/10.5334/tismir.131) |
| Domain | Jazz (340 famous recordings) |
| Size | 340 tracks, 3000+ annotated segments |
| Annotation | Structural segments + per-segment solo/accompanying instrument labels |
| Audio | Sourced from copyright-cleared archival recordings; check distribution mechanism |
| **Why prioritise** | Extends genre coverage substantially. Jazz has very different structural conventions (head-solo-trade-head, etc) from pop. Per-segment instrument labels enable a multi-task probe. |
| Implementation | ~4 hours if audio is directly available; ~8 hours if YouTube-sourced via yt-dlp. |

#### 5. TAVERN — Theme And Variation Encodings (Mozart + Beethoven)

| | |
|---|---|
| Source | Available via several musicology repositories; check for current location |
| Domain | Classical (27 theme + variation sets by Mozart and Beethoven, 1765–1810) |
| Size | 27 complete sets |
| Annotation | Roman numeral harmonic analysis + theme-variation correspondence |
| Audio | Symbolic only; need to render |
| **Why prioritise** | **Direct classical analogue to our VGMIDITVar leitmotif task.** Each theme has multiple variations, evaluated as retrieval (same query→theme matching across instrumental/figural variations). Tests whether MuQ's late-layer leitmotif advantage generalises beyond synth-VGM to actual classical theme/variation. |
| Implementation | ~5 hours. Reuse build_vgmiditvar_dataset.py renderer pattern + our existing CoverRetrievalTask. |

#### 6. BPSD — Beethoven Piano Sonata Dataset (multi-version)

| | |
|---|---|
| Source | [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.196) (2024) |
| Domain | Classical (first movements of 32 Beethoven piano sonatas) |
| Size | 32 sonatas × multiple performance versions (count TBD; likely 3–6 versions each → ~100–200 audio files) |
| Annotation | Measure positions, beats, global/local keys, chords, structural elements. All versions aligned on unified musical timeline. |
| Audio | Distributed (multi-version audio recordings) |
| **Why prioritise** | Enables a **cross-version retrieval task** for classical music — parallel to Covers80 but in the classical domain. Tests whether our encoders handle classical timbre/performance variations as well as they handle pop covers. The coherent alignment is a unique feature: same passage across versions is sample-aligned, so we can do tight similarity probes. |
| Implementation | ~6 hours. Cross-version retrieval similar to Covers80 setup. |

#### 7. NES-VMDB — NES Video-Music Database (2024)

| | |
|---|---|
| Source | https://arxiv.org/html/2404.04420v1 |
| Domain | Video game music (NES era) |
| Size | **98,940 clips, 474 hours from 389 NES games** |
| Annotation | Each clip linked to source game; paired with 15-second gameplay videos; symbolic + audio formats |
| Audio | Distributed (rendered from NES sound chip) |
| **Why prioritise** | Largest VGM dataset by a huge margin. Game-paired visual context enables future multimodal work. Symbolic-side overlap with our existing CLaMP3-symbolic pipeline. |
| Implementation | ~8 hours. Need to figure out which sub-task makes sense (game classification? genre classification?). The dataset itself doesn't ship structure annotations — would need to be paired with another annotation effort or used for a different task. |
| Caveats | 8-bit NES audio is very different from modern game music. Worth a separate probe but unlikely to transfer to orchestral leitmotifs. |

### Tier 3 — large but expensive

#### 8. SALAMI — Structural Analysis of Large Amounts of Musical Information

| | |
|---|---|
| Source | [SALAMI](https://github.com/DDMAL/salami-data-public) · `msaf-data` repo |
| Domain | Multi-genre (Western pop, jazz, blues, classical, world, live music) |
| Size | **1356 tracks, 2400+ annotations** (65.9% double-annotated) |
| Annotation | 3-level hierarchical (functional / large-scale / small-scale). The gold standard for boundary detection benchmark. |
| Audio | Sourced from Internet Archive — distributable subset |
| **Why prioritise** | The most-cited structure analysis benchmark. Multi-annotator agreement makes it the strongest evaluation. |
| Why NOT priority 1 | Genre diversity overlaps with Harmonix/HookTheoryStructure on pop side, and the classical/world tracks are spread thin. Implementation is more expensive than Harmonix because annotations are JAMS multi-level and need flattening. |
| Implementation | ~8 hours. JAMS parsing + 3-level label flattening + standard datamodule. |

#### 9. SongFormDB

The 14k-track training corpus behind SongFormBench. Heterogeneous
supervision (mix of strong/weak/noisy labels). Same team as SongFormer.

**Recommendation:** use as a fine-tuning corpus AFTER establishing a
SongFormBench result. Not a benchmark itself — too heterogeneous. ~10
hours to implement properly.

### Tier 4 — niche / domain-specific

#### 10. Raveform — EDM DJ mixes (TISMIR 2024)

| | |
|---|---|
| Source | [TISMIR](https://transactions.ismir.net/articles/10.5334/tismir.288) |
| Size | 4,902 DJ mixes + 56,873 tracks; 1,423 with detailed structural annotations |
| Annotation | EDM-specific vocabulary (`drop`, `breakdown`, `buildup`, etc.); beat + downbeat + structure |
| Why niche | EDM-specific structural vocabulary doesn't transfer to other genres. |
| When useful | If we want to argue encoders generalise across genre-specific structural conventions. |

#### 11. Annotated Mozart Sonatas (DCML)

| | |
|---|---|
| Domain | Classical (all Mozart piano sonatas) |
| Size | Full Mozart piano sonata corpus |
| Annotation | Roman-numeral harmonic + phrase + cadence (the DCML standard) |
| Why niche | Primarily harmonic analysis, not segmentation. Phrase-level annotations are usable but smaller-grained than functional segments. |
| When useful | As a cross-task probe — can encoders trained on Harmonix structure also do phrase-level Mozart? |

#### 12. YM2413-MDB — FM-synthesis VGM

| | |
|---|---|
| Domain | 8-bit FM-synth video game music |
| Size | 669 songs |
| Annotation | Emotion labels (not structure) |
| Why niche | 8-bit synth domain is very narrow. Emotion not structure. |
| When useful | For a future emotion-recognition task on VGM (parallel to EMO).  |

#### 13. OSSL — Open Screen Sound Library (2025)

| | |
|---|---|
| Domain | Public-domain movie clips + soundtracks |
| Size | 36.5 hours |
| Annotation | Mood (Russell's 4Q taxonomy), not structure |
| Why niche | Mood not structure. Limited to public-domain era films. |
| When useful | Future mood-on-film task; not a structure benchmark. |

#### 14. Mozart Texture Annotations (ISMIR 2022)

Very narrow — 1164 labels of piano texture (melody / accompaniment /
homophonic / etc.) across 9 movements. Interesting *probe* of
texture-specific encoder features but not a standalone structure
benchmark.

#### 15. Isophonics / RWC Popular

Classic benchmarks (Beatles 180 tracks; RWC-P 100 tracks). Well-trodden
in the literature. Mostly historical interest now — Harmonix and SALAMI
are larger and better-annotated. Worth implementing only if writing a
paper that requires comparison to the classic baselines.

---

## Cross-cutting observations

### Symbolic vs audio data path

Several of these datasets are symbolic-only (BPS-Motif, TAVERN,
Annotated Mozart Sonatas, much of SuperMario). For symbolic-only
datasets we have **two implementation paths**:

- **Render to audio** via SGM-Pro 14 (our VGMIDI infrastructure). Lets
  all four audio encoders (CLaMP3, MERT-95M, MuQ, OMARRQ-25hz)
  participate. Slight unrealism (synth audio).
- **Direct symbolic via CLaMP3-symbolic.** Uses the M3 tokeniser. Gives
  CLaMP3-symbolic an advantage but excludes the audio encoders.

For best comparison, do BOTH — render to audio AND emit MIDI alongside.
The renderer's marginal cost is low and it lets us do the same direct
cross-encoder comparison we did for leitmotif (audio encoders vs
CLaMP3-symbolic).

### The "branches / subdatasets" angle

Several datasets have curated subsets or branches worth knowing about:

- **SongFormBench** is itself a curated subset of Harmonix (200 tracks
  manually revised) + Chinese songs.
- **MIREX 2009 / 2010** are derived from Isophonics and RWC subsets
  with standardised evaluation.
- **SLMS (Segmented Lakh MIDI Subset)** is a structural subset of the
  Lakh MIDI Dataset.
- **VGMIDITVar** (already implemented) is a theme/variation subset of
  VGMIDI.

When implementing a new dataset, check for known curated subsets first
— they often come with reproducible splits and pre-flight quality
filtering.

### What's missing from this survey

Datasets we did NOT find good candidates for:
- **Pop song lyrics-aligned structure**: would test text-grounded
  structure detection. Some HookTheory hooks have lyrics but not at
  sufficient scale.
- **Recent ISMIR 2025 papers beyond SongFormer**: limited published-by
  October. Worth re-searching in 6 months.
- **Cross-cultural / non-Western structure annotations**: significant
  gap in the field. No good benchmarks for Indian classical, Arabic
  maqam, etc.
- **Film soundtracks with structural annotations** (vs just mood):
  closest is OSSL but that's mood-only. The leitmotif use case would
  benefit from a real-soundtrack benchmark — see open question below.

### Open question for the leitmotif project specifically

For the user's broader leitmotif research, the **ideal next dataset**
would be a film/game soundtrack corpus with:
- Real (not MIDI-rendered) orchestral audio
- Per-theme occurrence annotations (timestamps + theme labels)
- Multiple appearances of each theme across different scenes

**No public dataset matches this fully.** The closest paths are:
1. Render BPS-Motif (classical motifs, controlled) — does the encoder
   even ID classical motifs?
2. Render TAVERN (theme + variation) — does theme-variation matching
   transfer to classical?
3. Hand-curate a leitmotif benchmark from public soundtracks (~50
   themes from Star Wars / LotR / Skyrim / Halo). This is the
   gold-standard test but it's days of annotation work and small.
   See `docs/leitmotif_findings.md` § Open questions.

---

## How to add a new dataset to MARBLE (process recipe)

Crystallised from HXMSA / VGMIDITVar / SuperMario planning:

1. **Find the canonical source.** GitHub repo, Zenodo record, or
   official dataset page. Check license.
2. **Understand the annotation schema.** Read the paper (especially
   the dataset-stats section); look at one annotation file by hand.
3. **Identify the simplest task framing.** Boundary detection?
   Multi-class section classification? Theme-variation retrieval?
   Match to an existing MARBLE pattern (HookTheoryStructure,
   CoverRetrieval, MTGGenre, etc.) so we can reuse the probe + cache
   integration.
4. **Decide the audio pipeline.** Pre-distributed? YouTube via yt-dlp?
   MIDI render via SGM-Pro? Mel-spectrograms?
5. **Write the build script.** Always include: fail-loud preamble,
   idempotency, `--max-tracks` pilot mode, ffprobe metadata,
   deterministic split. Mirror `scripts/data/build_hxmsa_dataset.py`.
6. **Clone the datamodule + probe** from the closest-matching existing
   task. Patch LABEL2IDX, class names, and JSONL paths only — keep
   cache integration, 4-tuple emit, and defensive getattr verbatim.
7. **Generate configs.** Python loop over the existing template
   configs of the most similar task; swap class paths and out_dim.
8. **Add to `run_all_sweeps.py`**: SweepDef entries + `jsonl_map`
   entry. Verify via `--dry-run`.
9. **Run `audit_cache_integration.py`** before any sweep launch.
10. **Code review pass:** diff the new files against the template they
    were cloned from. All diffs should be intentional.
11. **Write `docs/data/<dataset>_setup.md`** runbook before
    implementation lands.

This recipe takes ~3–10 hours per dataset depending on audio sourcing
complexity.

---

## References

### Datasets cited

- [SongFormer paper](https://arxiv.org/abs/2510.02797) (Hao et al. 2025)
- [SongFormBench HF](https://huggingface.co/datasets/ASLP-lab/SongFormBench)
- [BPS-Motif (ISMIR 2023)](https://archives.ismir.net/ismir2023/paper/000032.pdf)
- [BPSD (TISMIR 2024)](https://transactions.ismir.net/articles/10.5334/tismir.196)
- [JSD (TISMIR 2022)](https://transactions.ismir.net/articles/10.5334/tismir.131)
- [Raveform (TISMIR 2024)](https://transactions.ismir.net/articles/10.5334/tismir.288)
- [NES-VMDB (2024)](https://arxiv.org/html/2404.04420v1)
- [ASAP dataset](https://github.com/fosfrancesco/asap-dataset)
- [Annotated Mozart Sonatas](https://github.com/DCMLab/mozart_piano_sonatas)
- [SALAMI](https://github.com/DDMAL/salami-data-public)
- [VGMIDI](https://github.com/lucasnfe/vgmidi)
- [YM2413-MDB](https://arxiv.org/abs/2211.07131)
- [TAVERN](https://github.com/jcdevaney/TAVERN) (theme + variation Mozart/Beethoven)
- [SuperMario Structure Annotation](https://github.com/ShxLuo-Saxon/supermario-structure-annotation)
- [Harmonix Set](https://github.com/urinieto/harmonixset)

### Review papers

- "Audio-Based Music Structure Analysis: Current Trends, Open
  Challenges, and Applications" — TISMIR survey covering the MSA
  landscape.
- ["Self-Supervised Learning of Multi-Level Audio Representations for
  Music Segmentation"](https://arxiv.org/abs/2303.13518) — relevant
  baseline for MARBLE comparison.

### Companion docs

- [`docs/data/hxmsa_setup.md`](data/hxmsa_setup.md)
- [`docs/data/supermario_setup.md`](data/supermario_setup.md)
- [`docs/data/vgmiditvar_setup.md`](data/vgmiditvar_setup.md)
- [`docs/layer_analysis.md`](layer_analysis.md)
- [`docs/leitmotif_findings.md`](leitmotif_findings.md)

---

## Changelog

- **2026-05-17** — Initial survey. 15 candidate datasets ranked.
  Recommended sequence: HXMSA (done) → SuperMario → SongFormBench →
  BPS-Motif. The first 4 ranks cover the four most distinct axes
  (large pop, VGM, SOTA general, classical motif) in ~14 hours total
  implementation cost. User-suggested datasets from Gemini conversation
  to be added in a later pass if they don't overlap with this list.
