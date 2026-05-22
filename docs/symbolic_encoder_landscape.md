# Symbolic music encoders — landscape report

**Snapshot date:** 22 May 2026. Re-check after ISMIR 2026 acceptances
publish (expected Jul–Aug 2026).

**Purpose:** decision-support reference for which symbolic encoders to
add to MARBLE alongside CLaMP3-symbolic. Scope = leitmotif analysis +
structure analysis on video-game / film music + Beethoven sonata motifs
(BPS-Motif).

**Status:** research-only — none of the encoders below are integrated
yet. CLaMP3-symbolic is the only symbolic encoder in MARBLE today.
Integration cost notes are sketches, not commitments.

---

## TL;DR recommendation

When we add a second/third symbolic encoder, the priority order is:

1. **Aria-medium-embedding** — solo-piano contrastive embeddings; top pick for BPS-Motif and any solo-piano leitmotif work.
2. **MidiBERT-Piano** — the literature baseline for BPS-Motif (Hsiao TISMIR'24); needed for any head-to-head with published numbers.
3. **Moonbeam-839M** — modern multi-instrument alternative to AMT; right slot if we extend beyond solo-piano (VGM/film coverage).

**Skip for now:** MusicBERT-original (fairseq pain), CLaMP 2 (low
diversity vs. CLaMP 3 — same M3 backbone), MuPT (no 2026 maintenance),
MuseTok (2026 newcomer; weights not yet packaged for drop-in use).

Unchanged from late-2025 advice except: AMT demoted, Moonbeam suggested
in its place. No 2026 release has supplanted Aria + MidiBERT.

---

## 1. What's new since Nov 2025

Genuinely new symbolic-music items in the Nov 2025 → 22 May 2026 window:

- **MuseTok** — ICASSP 2026 (Liu et al., arXiv [2510.16273](https://arxiv.org/abs/2510.16273)). RQ-VAE bar-wise tokenizer + transformer encoder–decoder. Tested on melody extraction, chord recognition, emotion. Code: <https://github.com/Yuer867/MuseTok>. Strongest 2026 *new pretrained encoder* signal — but weights are not yet polished for off-the-shelf use; worth watching for late 2026.
- **PianoRoll-Event** — ICASSP 2026 (arXiv [2601.19951](https://arxiv.org/abs/2601.19951)). Event-style encoding *of pianoroll*; 1.36–7.16× compression vs REMI/MIDI-like. A *representation*, not a model — adoption story to track.
- **BACHI** — ICASSP 2026, boundary-aware symbolic chord recognition through masked iterative decoding. Code: <https://github.com/AndyWeasley2004/BACHI_Chord_Recognition>. Task-specific (chord recognition), not a general encoder.
- **MIDI-LLaMA** — Jan 2026 (arXiv [2601.21740](https://arxiv.org/abs/2601.21740)). First MIDI–text instruction-following MLLM. Crucially uses *MusicBERT (OctupleMIDI) as the frozen symbolic encoder*, confirming MusicBERT is still treated as the strongest off-the-shelf bidirectional encoder in early 2026.
- **Probabilistic Multilabel Graphical Modelling of Motif Transformations** — Mar 2026 (arXiv [2603.26478](https://arxiv.org/abs/2603.26478)). Multilabel CRF over motif features applied to Beethoven piano sonatas with typed transformations (transposition, inversion, rhythmic, harmonic). Directly adjacent to the BPS-Motif retrieval probe — worth reading carefully if/when we reconsider the retrieval scoring rule.
- **SAVGM** — 19 Jan 2026 (Luo et al.). 309 post-2000 video-game tracks with structural segmentation (functional / phrase-motive / section). The most directly relevant 2026 *dataset* for the VGM thread. CNN-RNN baseline: P=0.512, R=0.667, F1=0.537 at 3 s tolerance.
- **Hugging Face State of Open Source Spring 2026** ([blog](https://huggingface.co/blog/huggingface/state-of-os-hf-spring-2026)) — surveyed; no new symbolic-music foundation-model checkpoints called out.

## 2. What did NOT appear in the window

Searched for but did not find:

- **No CLaMP 4 release.** CLaMP 3 (ACL 2025) remains current.
- **No Aria v2 / new Aria checkpoint** beyond `loubb/aria-medium-embedding` + `loubb/aria-medium-base`.
- **No 2026 MidiBERT retrain** or new size.
- **No MuPT 2026 update.**
- **No 2026 re-benchmark of BPS-Motif** — Hsiao TISMIR'24 (MidiBERT-based) remains the SOTA reference.
- **ISMIR 2026 acceptances not yet published** (Abu Dhabi, Oct 2026). CFP only: <https://ismir2026.ismir.net/authors/call-for-papers>. Re-check Jul–Aug 2026.
- **No new SongFormBench leaderboard entries** since SongFormer (Oct 2025).

## 3. Candidate list

Downloadable symbolic encoders considered for MARBLE integration, with
their relevance to the leitmotif + VGM/film scope.

| Model | Year | Size | Tokenization | HF / Repo | License |
|---|---|---|---|---|---|
| **Aria-medium-embedding** | ISMIR 2025 (no 2026 v2) | ~1B (LLaMA-3.2-1B) | Aria piano-event | [loubb/aria-medium-embedding](https://huggingface.co/loubb/aria-medium-embedding) | Apache-2.0 |
| **MidiBERT-Piano** | 2021 / JCMS 2024 | ~110M (BERT-12) | CP / REMI | [wazenmai/MIDI-BERT](https://github.com/wazenmai/MIDI-BERT) | MIT |
| **Adversarial-MidiBERT** | 2024 | ~110M | OctupleMIDI | [RS2002/Adversarial-MidiBERT](https://huggingface.co/RS2002/Adversarial-MidiBERT) | MIT |
| **MusicBERT** | MM 2021 (re-validated Jan 2026 by MIDI-LLaMA) | 110M / 340M | OctupleMIDI | `microsoft/musicbert` (fairseq) | MIT |
| **Moonbeam** | May 2025 | 309M / 839M | Multidim. Relative Attention | [code](https://github.com/guozixunnicolas/moonbeam-midi-foundation-model) | MIT |
| **Anticipatory Music Transformer (AMT)** | Stanford CRFM 2023 | 360M / 780M | Arrival-time event | `stanford-crfm/music-medium-800k` | Apache-2.0 |
| **CLaMP 3** | ACL 2025 | ~340M (audio+symbolic+text) | M3 patches | [sander-wood/clamp3](https://huggingface.co/sander-wood/clamp3) | MIT |
| **MuseTok encoder** | ICASSP 2026 | small RQ-VAE | RQ-VAE codes over REMI bars | [Yuer867/MuseTok](https://github.com/Yuer867/MuseTok) | MIT (repo) |
| **NotaGen** | Feb 2025 | 110M – 1.4B | ABC | [ElectricAlexis/NotaGen](https://github.com/ElectricAlexis/NotaGen) | MIT |
| **MuPT** | OpenReview 2024 (no 2026 update) | 190M – 1.97B | SMT-ABC + BPE | `m-a-p/MuPT-*` | Apache-2.0 |
| **CLaMP 2** | 2024 | 12-layer × 768 | M3 patches (same as CLaMP 3) | [sander-wood/clamp2](https://huggingface.co/sander-wood/clamp2) | MIT |
| **MIDI-RWKV** | Jun 2025 | Small RWKV-7 | RWKV linear-attention | [HF paper](https://huggingface.co/papers/2506.13001) | MIT |

## 4. Detailed pick rationales

### Pick A — Aria-medium-embedding

- **Why:** the strongest current "MIDI foundation model" by training scale (~60k h of solo-piano transcribed MIDI, Aria-MIDI dataset = ~1M files), with a *dedicated contrastive-embedding finetune* released as a separate checkpoint. Piano-centric matches the Beethoven sonata target directly.
- **Integration sketch:** needs `aria-utils` tokenizer; LLaMA-3.2-1B class architecture with `output_hidden_states=True`, ~16 layers. 30 s clip ≈ 1500–3000 tokens; fp16 forward ~50–150 ms on 16 GB. Pooling: time-mean or the contrastive head's CLS-equivalent.
- **URLs:** <https://huggingface.co/loubb/aria-medium-embedding>, <https://github.com/EleutherAI/aria>, <https://arxiv.org/abs/2506.23869>.

### Pick B — MidiBERT-Piano

- **Why:** the encoder behind the published BPS-Motif SOTA (Hsiao TISMIR'24: per-note F1=0.721 with pseudo-label boosting). Non-optional if we want direct comparability with that number. Also re-validated by MIDI-LLaMA in Jan 2026 as a frozen MIDI encoder.
- **Integration sketch:** BERT-base 12 layers over CP (Compound Word) tokens; ~110M params; ~1 day of glue (CP preprocessor + custom stacked embedding layer; the repo expects specific dataset prep via `prepare_data/`). 30 s clip ≈ 512 CP tokens, <20 ms forward.
- **URLs:** <https://github.com/wazenmai/MIDI-BERT>, <https://github.com/CUHK-CMD/MIDI-BERT-2> (more actively maintained JCMS-2024 variant), <https://transactions.ismir.net/articles/10.5334/tismir.250> (BPS-Motif follow-up).

### Pick C — Moonbeam (third slot, optional)

- **Why:** modern (May 2025) multi-instrument MIDI foundation model trained on 81.6k h / 18B tokens. Replaces AMT's "third encoder" role with a model that's actually being maintained in the 2026 timeframe. Covers the VGM/film multi-instrument extension that Aria's piano-only training doesn't.
- **Integration sketch:** Multidim. Relative Attention tokens; 309M or 839M variants; HF weights linked from the repo. Standard `AutoModelForCausalLM` load with `output_hidden_states=True`.
- **URLs:** <https://github.com/guozixunnicolas/moonbeam-midi-foundation-model>, <https://arxiv.org/abs/2505.15559>.

## 5. Skip / red flags

- **MusicBERT (original Microsoft)** — fairseq-only, OctupleMIDI pipeline tied to MS's MUMIDI scripts. No HF model card. The community port (`manoskary/musicbert`) uses a different tokenizer (REMI+BPE) so it's effectively a different model. Re-implementation is hours of work; only worth it if we specifically need OctupleMIDI for a paper claim.
- **CLaMP 2** — same M3 patch tokenizer as CLaMP3's symbolic side ⇒ low diversity vs. what we already have. Useful only as an ablation.
- **MuPT** — large, ABC-only, weights release fragmented across versions. No 2026 update — NotaGen (Feb 2025) is the better current ABC model if we need that tokenisation family.
- **MuseTok** — 2026 newcomer with promising design but weights / preprocessing not yet packaged for drop-in HF use. Track for late 2026.
- **Anticipatory Music Transformer (AMT)** — not deprecated but now the oldest of the originally recommended trio (2023). No 2026 maintenance signal. Moonbeam is a stronger current substitute.
- **Adversarial-MidiBERT** — drop-in variant of MidiBERT-Piano with adversarial pre-training; consider as an upgrade *after* MidiBERT-Piano is in, not as an alternative.
- **Pop2Piano / YuE / GPT-Music / MIDI-LLaMA** — not symbolic encoders in the sense MARBLE uses (audio→MIDI generators, audio-domain models, or MLLMs). Out of scope.

## 6. Motif- and structure-tuned models

- **MidiBERT-Piano on BPS-Motif** — the literature baseline. Required for direct head-to-head with Hsiao TISMIR'24.
- **Aria's contrastive head** is the only large-scale, recently-released MIDI embedding model with an explicit similarity objective — natural for motif retrieval probing.
- **No purpose-built leitmotif / VGM symbolic encoder** with a permissive licence and downloadable weights surfaced as of May 2026. BPSD (Beethoven Piano Sonata Dataset) is a dataset, not a model.
- **For leitmotif / VGM specifically:** Aria (piano-leaning) + Moonbeam (LMD-trained, multi-instrument) is the best complementary pair currently.

## 7. New benchmarks/datasets — 2026

Besides encoders, the 2026 datasets relevant to this scope:

- **SAVGM** (Jan 2026) — 309 video-game tracks with structural segmentation. Direct hit for VGM. CNN-RNN baseline F1=0.537.
- **SongFormBench / SongFormDB** (Sep–Oct 2025, carried over) — structure-analysis benchmark of record for general pop. No 2026 successors.
- **BPS-Motif** (ISMIR 2023, integrated 2026-05-22, see [data/bps_motif_setup.md](data/bps_motif_setup.md)) — still the symbolic leitmotif-adjacent benchmark. Nothing new in 2026.
- **ICASSP 2026 Automatic Song Aesthetics Evaluation Challenge** ([page](https://aslp-lab.github.io/Automatic-Song-Aesthetics-Evaluation-Challenge/)) — audio, not symbolic; fresh 2026 MIR benchmark but tangential.
- No new film-music symbolic dataset in 2026.

## Source list

- Aria: <https://arxiv.org/abs/2506.23869> · <https://huggingface.co/loubb/aria-medium-embedding> · <https://huggingface.co/loubb/aria-medium-base> · <https://github.com/EleutherAI/aria> · <https://www.eleuther.ai/papers-blog/aria-midi-a-dataset-of-midi-files-for-symbolic-music-modeling>
- AMT: <https://huggingface.co/stanford-crfm/music-medium-800k> · <https://huggingface.co/stanford-crfm/music-large-800k> · <https://github.com/jthickstun/anticipation> · <https://crfm.stanford.edu/2023/06/16/anticipatory-music-transformer.html> · <https://arxiv.org/html/2306.08620v2>
- MidiBERT-Piano: <https://github.com/wazenmai/MIDI-BERT> · <https://github.com/CUHK-CMD/MIDI-BERT-2> · <https://arxiv.org/abs/2107.05223> · <https://transactions.ismir.net/articles/10.5334/tismir.250>
- BPS-Motif: <https://zenodo.org/records/10265277> · <https://github.com/Wiilly07/Beethoven_motif> · <https://archives.ismir.net/ismir2023/paper/000032.pdf>
- BPSD: <https://transactions.ismir.net/articles/10.5334/tismir.196>
- CLaMP 3: <https://huggingface.co/sander-wood/clamp3> · <https://github.com/sanderwood/clamp3> · <https://arxiv.org/abs/2502.10362>
- CLaMP 2: <https://huggingface.co/sander-wood/clamp2> · <https://arxiv.org/abs/2410.13267>
- MusicBERT (community port): <https://huggingface.co/manoskary/musicbert> · <https://huggingface.co/manoskary/musicbert-large> · <https://malcolmsailor.com/2025/02/24/musicbert-hf.html>
- MusicBERT (Microsoft original): <https://github.com/microsoft/muzic/tree/main/musicbert> · <https://arxiv.org/pdf/2106.05630>
- MuPT: <https://arxiv.org/abs/2404.06393>
- NotaGen: <https://arxiv.org/abs/2502.18008> · <https://github.com/ElectricAlexis/NotaGen>
- Moonbeam: <https://arxiv.org/abs/2505.15559> · <https://github.com/guozixunnicolas/moonbeam-midi-foundation-model>
- MIDI-RWKV: <https://arxiv.org/abs/2506.13001> · <https://huggingface.co/papers/2506.13001>
- MuseTok (ICASSP 2026): <https://arxiv.org/abs/2510.16273> · <https://github.com/Yuer867/MuseTok> · <https://musetok.github.io/>
- PianoRoll-Event (ICASSP 2026): <https://arxiv.org/abs/2601.19951>
- BACHI (ICASSP 2026): <https://github.com/AndyWeasley2004/BACHI_Chord_Recognition>
- MIDI-LLaMA (Jan 2026): <https://arxiv.org/abs/2601.21740>
- Motif-Transformations CRF (Mar 2026): <https://arxiv.org/abs/2603.26478>
- Boundary Regression for Leitmotif Detection (Mar 2025, carried over): <https://arxiv.org/abs/2503.07977>
- Discovering "Words" in Music (Sep 2025, carried over): <https://arxiv.org/abs/2509.24603>
- Barwise Section Boundary (Sep 2025, carried over): <https://arxiv.org/abs/2509.16566>
- SongFormer (Oct 2025): <https://arxiv.org/abs/2510.02797> · [SongFormBench dataset](https://huggingface.co/datasets/ASLP-lab/SongFormBench)
- HF State of Open Source Spring 2026: <https://huggingface.co/blog/huggingface/state-of-os-hf-spring-2026>
- Adversarial-MidiBERT: <https://huggingface.co/RS2002/Adversarial-MidiBERT> · <https://github.com/RS2002/Adversarial-MidiBERT>
- ISMIR 2026 CFP: <https://ismir2026.ismir.net/authors/call-for-papers>
- ICASSP 2026 Song Aesthetics Challenge: <https://aslp-lab.github.io/Automatic-Song-Aesthetics-Evaluation-Challenge/> · <https://arxiv.org/abs/2601.07237>
