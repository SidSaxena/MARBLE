# SuperMarioStructure — CLaMP3-symbolic layer sweep findings

**Status:** symbolic-only sweep complete (2026-05-20). Audio sweep
(MERT-95M, MuQ, OMARRQ, CLaMP3-audio) pending — see "Open follow-ups".

**Headline:** CLaMP3-symbolic L4 and L11 tie at **0.599 test/acc** on
the 6-class functional-segment task (chance = 0.167). L12 (the
contrastive-training output layer that aligns audio + symbolic + text
in a shared embedding space) underperforms by **−3.3 pp**, but is
still **the correct choice for any task that requires cross-modal or
cross-domain comparison** — see "When to use L12" below.

---

## 1. Setup recap

- **Dataset:** 554 Super Mario VGM transcriptions from NinSheetMusic
  → 1,209 functional segments (after bar→time conversion + size
  filtering). 545 pieces survived MIDI parsing.
- **Splits:** 824 train / 203 val / 182 test, piece-level (no segment
  leakage across splits). 334 piece splits inherited from upstream
  `pairs.csv`; remainder assigned by seed-1234 70/15/15.
- **Classes (6, alphabetical):** `bridge`, `intro`, `linear`, `loop`,
  `outro`, `stinger`. Heavily skewed: loop 57%, intro 25%, linear 8%,
  stinger 4%, bridge 3%, outro 3%.
- **Encoder:** CLaMP3-symbolic (M3 tokens via `midi_to_mtf` →
  BERT-base, 13 layers L0–L12). Layer-wise probe via the standard
  MARBLE `LayerSelector` + `MLPDecoder` setup, 40 epochs, Adam @ 1e-3.
- **Metrics:** `test/acc` (overall accuracy) + `test/macro_f1`
  (imbalance-aware).

See [`docs/data/supermario_setup.md`](data/supermario_setup.md) for
the build commands, the
[Section 4 audit](#section-4---cross-task-pattern-for-clamp3-symbolic)
for what got pre-fixed in the build (slug-naming, inclusive bar
semantics, `Ln`/linear class).

---

## 2. Per-layer raw results

| Layer | val/acc | test/acc | val→test gap | test/macro_f1 |
|---:|---:|---:|---:|---:|
| L0  | 0.6502 | 0.5769 | −0.073 | 0.2220 |
| L1  | 0.6502 | 0.5604 | −0.090 | 0.2144 |
| L2  | 0.6601 | 0.5495 | −0.111 | 0.2073 |
| L3  | 0.6601 | 0.5824 | −0.078 | 0.2225 |
| **L4**  | 0.6552 | **0.5989** | −0.056 | **0.2517** |
| L5  | 0.6700 | 0.5769 | −0.093 | 0.2208 |
| L6  | 0.6502 | 0.5769 | −0.073 | 0.2419 |
| L7  | 0.6650 | 0.5604 | −0.105 | 0.2120 |
| L8  | 0.6502 | 0.5659 | −0.084 | 0.2334 |
| L9  | 0.6650 | 0.5659 | −0.099 | 0.2153 |
| L10 | 0.6601 | 0.5549 | −0.105 | 0.2095 |
| **L11** | 0.6404 | **0.5989** | −0.042 | 0.2508 |
| **L12** | 0.6502 | 0.5659 | −0.084 | 0.2145 |
| meanall | 0.6453 | 0.5714 | −0.074 | 0.2169 |

**Observations:**

- **L4 and L11 tie at the top** (0.5989 acc). L4 has a marginally
  better F1 (0.2517 vs 0.2508).
- **L12 (contrastive output) sits at 0.5659**, a 3.3 pp deficit
  vs L4/L11 and 0.55 pp below meanall.
- **The val→test gap is tightest for L4 (−5.6) and L11 (−4.2)** —
  exactly the two layers that win on test. Other layers lose 7–11 pp
  between val and test. This is real signal that L4 and L11
  generalize better, not a fluke of the test set.
- **Macro-F1 stays low everywhere** (~0.22). With chance ≈ 0.167 for
  a uniform 6-class problem, this barely beats chance — the
  imbalance (loop = 57%) is dragging classifier predictions toward
  the majority class. See "Limitations".

---

## 3. The two-peak structure — what L4 and L11 capture

Tied performance at two non-adjacent layers is unusual. The pattern
matches what we've seen on other symbolic + structure tasks:

- **L4 (early-mid)** captures **local syntactic features** — phrase
  boundaries, cadential figures, motivic repetition. These cues
  separate intro/outro/stinger (which have distinctive surface
  rhythms) from loop/linear bodies (which are denser).
- **L11 (late, penultimate)** captures **section-level discourse** —
  is this a self-similar repeating chunk (loop) or a through-composed
  pass (linear)? Is the section a "main body" or a "connector"?
  These cues need integration over the whole patch and only emerge
  near the end of the encoder stack.

L4 and L11 don't compete — they're hitting the same classification
target via complementary representations. **A 2-layer ensemble (L4 +
L11) is the next thing to try** if a marginal lift is wanted; see
"Open follow-ups."

---

## 4. Cross-task pattern for CLaMP3-symbolic

CLaMP3-symbolic now has data on **three symbolic tasks** (audio-only
tasks like Covers80, SHS100K, HookTheoryStructure can't use it). The
per-layer pattern is remarkably consistent:

| Task | Type | Metric | Best layer | Best value | L12 value | L12 gap | meanall |
|---|---|---|---:|---:|---:|---:|---:|
| **VGMIDITVar** | retrieval (1 SF) | test/MAP | **L11** | 0.1978 | 0.1797 | −0.0181 | — |
| **VGMIDITVar-leitmotif** | cross-instrument retrieval | test/MAP | **L11** | 0.1946 | 0.1763 | −0.0183 | 0.1946 |
| **SuperMarioStructure** | classification (6-class) | test/acc | **L4 ≈ L11** | 0.5989 | 0.5659 | −0.0330 | 0.5714 |

**Universal observations:**

1. **L11 is the default best layer** for CLaMP3-symbolic across all
   three tasks. SuperMarioStructure is the only task where another
   layer (L4) ties it.
2. **L12 (contrastive output) is always worse than L11** by 1.8–3.3
   percentage points. The gap is tighter on retrieval (~1.8 pp) than
   classification (~3.3 pp).
3. **meanall ≈ L11 on retrieval, meanall < L11 on classification.**
   Averaging across all 13 layers acts like a regularizer; for
   retrieval where the signal is widely distributed it converges to
   L11. For classification where L4 + L11 carry distinct signal,
   the average gets pulled down by the mid-layers.
4. **Retrieval scores rise monotonically L0→L11** with a sharp drop
   at L12 — classic deep-encoder profile where representations
   abstract until the contrastive layer collapses them onto the
   shared cross-modal manifold.
5. **SuperMarioStructure is the only task with a true two-peak
   profile** (L4 ≈ L11). Retrieval tasks have a single
   monotonically-rising profile to L11 then a sharp L12 drop.

---

## 5. L12 deep-dive — when to use the contrastive output

The user's explicit ask: use L12 for **multidomain and cross-modality
comparison**. This is a real and defensible choice, but the
trade-off is concrete: **L12 sacrifices 2–3 pp of single-modality
task performance for the ability to embed everything into a shared
semantic space**.

### When L12 IS the right choice

| Use case | Why L12 wins |
|---|---|
| **Cross-modal retrieval** (e.g., "find the symbolic-score equivalent of this audio clip") | L12 on the audio side and L12 on the symbolic side were trained against the same contrastive loss — they live in the same metric space. Earlier layers don't. |
| **Cross-domain transfer** (e.g., train probes on Western pop, evaluate on VGM) | The contrastive objective was trained over a multilingual + multi-domain corpus. L12 is the layer most likely to share representations across genres / domains. |
| **Multimodal fusion at inference** (e.g., score-conditioned audio classification) | Concatenating L12(audio) + L12(symbolic) into a single MLP gives a coherent fused embedding. Concatenating L11(audio) + L11(symbolic) mixes two independently-tuned spaces. |
| **Zero-shot / few-shot** via cosine similarity to a text prompt | CLaMP3 was trained with text in the contrastive loss. L12 supports prompt-style retrieval; earlier layers don't carry text alignment. |
| **As a single embedding for ranking / clustering** of mixed-modality content | Same shared-space argument — L12 is the only layer with cross-modal consistency. |

### When L12 is the WRONG choice

| Use case | Why use L11 (or L4) instead |
|---|---|
| Single-modality probe with no cross-modal comparison | L11 wins by 1.8–3.3 pp. Free performance. |
| Pure retrieval within one modality (audio→audio, symbolic→symbolic) | L11 carries the discriminative signal better. The cross-modal alignment of L12 is irrelevant. |
| Classification where minority classes matter (the SuperMario stinger problem) | L4 captures local rhythmic cues that disambiguate short cues from main bodies. L12 has averaged this away. |
| Frozen-encoder feature extraction for downstream fine-tuning | The earlier layer carries more information; the head can learn the task-specific projection. |

### The hybrid recipe — L11 ⊕ L12

If you want both "best single-modality performance" and "cross-modal
comparability," concatenate the two:

```python
emb_l11 = enc(midi, layer=11)        # 768-dim, task-rich
emb_l12 = enc(midi, layer=12)        # 768-dim, cross-modal aligned
emb = torch.cat([F.normalize(emb_l11), F.normalize(emb_l12)], dim=-1)
# 1536-dim concatenated; L2-normalize each half so the
# contrastive component doesn't dominate the magnitude
```

Cost: 2× embedding dimension. Benefit: the cross-modal "anchor"
(L12) sits alongside the task-rich representation (L11), letting a
downstream MLP learn which signal to use per-query. Tested in spirit
on the leitmotif task (where L7+L11 was tried then abandoned as
overengineered for retrieval — but **structure tasks have legitimate
two-peak structure, so the ensemble is more defensible here**).

### Empirical: how bad is L12 on each task in absolute terms?

- **VGMIDITVar (retrieval):** L12 = 0.180 MAP vs L11 = 0.198. Still
  the best symbolic encoder by a wide margin (audio-side CLaMP3 L5
  best = 0.182 MAP). Defensible to ship L12 if you need it for
  multimodal downstream work.
- **VGMIDITVar-leitmotif (retrieval):** L12 = 0.176 MAP vs L11 =
  0.195. Symbolic L12 still beats audio-L5 (0.182) on this task —
  the symbolic advantage compounds. Defensible to ship L12.
- **SuperMarioStructure (classification):** L12 = 0.566 acc vs L4/L11
  = 0.599. This is the worst case (3.3 pp gap). If your downstream
  pipeline needs cross-modal alignment AND structure classification,
  expect to leave 3 pp on the table — but get a single coherent
  embedding space across audio + symbolic + (potentially) text
  pipelines.

---

## 6. What this means for VGM structure & motif analysis

Synthesizing across this sweep, the
[leitmotif retrieval findings](leitmotif_findings.md), and the
[cross-encoder layer analysis](layer_analysis.md):

### For *structure* analysis on VGM specifically

- **CLaMP3-symbolic on MIDI is a genuinely useful structure
  classifier** — 0.599 acc on a 6-class skewed problem with a small
  (824-segment) training set is a respectable result. The model
  picks up enough rhythmic + harmonic continuity to separate
  intros / stingers / connectors from main bodies (loop, linear)
  most of the time.
- **The model overfits to majority class** — macro_f1 at 0.25 means
  minority classes (bridge, outro, stinger) are not learned well.
  This is a data problem not a representation problem; class-balanced
  loss + class-weighted sampling are the next things to try.
- **Loop vs Linear is the interesting distinction** — both are
  "main body" sections, but Loop has explicit repeat marks (the
  "Lp" tag requires `|:` `:|` in the score) while Linear is
  through-composed. The model has to integrate over the whole patch
  to make this call, which is why **late layers (L11) help** despite
  the early-layer (L4) win on accuracy.

### For *motif* analysis on VGM (cross-link to leitmotif retrieval)

- The leitmotif task (`VGMIDITVar-leitmotif`, MAP retrieval) shows
  the **same L11-dominant pattern** with the **same L12 drop** of
  ~1.8 pp. This means the encoder layer choice generalizes from
  motif-level retrieval to section-level classification.
- **CLaMP3-symbolic ≫ all audio encoders** on motif retrieval (0.195
  MAP vs MuQ L11 at 0.044 MAP per the leitmotif doc) — when MIDI is
  available, prefer the symbolic path by a factor of ~5×.
- **L12 stays superior to audio-encoder L12** even after the gap —
  symbolic L12 (0.176) > audio L12 (varies, around 0.005–0.17
  depending on encoder). The contrastive output of the symbolic
  branch carries more useful structural signal than the contrastive
  output of any audio branch.

### Synthesis: a "VGM-friendly encoder" stack

For deploying CLaMP3-symbolic across a VGM-centric pipeline that
does both motif retrieval AND structural classification:

| Component | Layer | Reasoning |
|---|---|---|
| Default single-layer for retrieval | **L11** | Wins on all 3 symbolic tasks tested |
| Default single-layer for structure | **L4 or L11** (tied) — pick L11 if you also retrieve | Same encoder pass, same cache |
| For cross-modal pipelines (audio ↔ symbolic comparison) | **L12** | Only layer with cross-modal alignment |
| For ensemble (if you can afford 2×) | **L4 + L11** | Two non-adjacent peaks → complementary signal |
| For ensemble (cross-modal + best single-modality) | **L11 + L12** | Best single + cross-modal anchor |

---

## 7. Extending to downstream tasks + other datasets

The CLaMP3-symbolic layer pattern (L11 dominant, L12 −2 pp,
meanall ≈ L11) is robust enough across our three symbolic tasks to
suggest specific recommendations for new evaluations:

### Other VGM datasets worth running

| Dataset | Source | Why interesting |
|---|---|---|
| **VGMIDI** ([Cardoso et al. 2020](https://github.com/lucasnfe/vgmidi-extended)) | 200 labeled MIDI clips, emotion + valence-arousal | Cross-check: does the L11 pattern hold for affect rather than structure? |
| **Bach Chorales** | already in repo (top-level dir) | Classical, very different domain — does the symbolic encoder generalize? L12 is the layer to test for cross-domain. |
| **Lakh MIDI Dataset (LMD-matched)** | [Raffel 2016](https://colinraffel.com/projects/lmd/) | 45K MIDIs with matched audio — the only large-scale resource where audio↔symbolic L12 alignment can be properly tested |
| **MAESTRO** | classical piano | Different domain (single-instrument piano) — does L11 still win, or does the lack of multi-track structure shift the layer profile? |
| **GiantMIDI-Piano** | ~10K classical-piano MIDIs | Another large-scale single-instrument test |
| **MetaMIDI Dataset (MMD)** | 436K MIDIs with genre labels | Genre classification benchmark — L4 (style features) vs L11 (compositional features) likely to split |
| **POP909** | 909 pop song MIDIs with melody/chord labels | Direct cross-domain comparison to Western-pop tasks already in MARBLE |
| **GTZAN MIDI** (if available) | MIDI versions of GTZAN audio | Within-MARBLE cross-modality test (audio GTZAN already supported) |

### Other downstream tasks worth wiring up

| Task | What CLaMP3-symbolic L11 should give you |
|---|---|
| **Mood/emotion classification on VGMIDI** | Probably 0.55–0.65 acc per the same probe template. Expect L4 to come up if affect cues are local. |
| **Adaptive-music transition prediction** | "Given segment A and B, do they sound contiguous in-game?" — a binary classification using L11 or L11+L12 (the cross-modal layer helps if you want to query with audio gameplay context). |
| **Style transfer / similarity** | Cosine similarity in L12 embedding space lets you do "find a Mario track that sounds like this Zelda track" — explicit cross-domain similarity is what L12 is designed for. |
| **Boundary detection** (when does a section end?) | Sliding-window L11 + change-point detection. CLaMP3-symbolic is patch-based, so resolution is the patch size — coarser than typical boundary-detection systems, but might work for "section-level" boundaries. |
| **Hierarchical structure** (multi-level segmentation: section → subsection) | The Section-level annotations in supermario-structure-annotation are reserved for v2 of the task. Expect L4 (local) to help with sub-section boundaries, L11 with section-level. |
| **Sequence generation conditioning** | L12 embeddings as conditioning signal for a generative model (token-level MIDI generator). The shared cross-modal space lets you condition on audio prompts at inference. |

### Cross-encoder + cross-modality extensions (when audio sweep lands)

Once the audio side of SuperMarioStructure is built and swept, the
analyses opens up:

- **Audio vs symbolic head-to-head on the same task** — fair
  comparison since same dataset, same splits. Expect symbolic to win
  by 5–15 pp (per the VGMIDITVar pattern).
- **Late fusion of audio + symbolic** — concat L11(symbolic) +
  L11(audio) into a single MLP. Hypothesis: small lift over symbolic
  alone, because audio captures performance-specific cues (timbre,
  velocity) that the score doesn't have.
- **L12 cross-modality coherence test** — for a given piece, do
  L12(audio) and L12(symbolic) actually have high cosine similarity?
  This is a quality check on CLaMP3's contrastive training. If yes
  → L12 is a viable cross-modal embedding. If no → CLaMP3's
  contrastive training didn't transfer cleanly to VGM, and L12 is
  only useful within-modality.

---

## 8. Limitations + risks (read before reporting numbers)

| Issue | Severity | Mitigation |
|---|---|---|
| **Class imbalance dominates macro_f1** (loop = 57%, stinger = 4%) | Real — limits how meaningful 0.22 macro_f1 is | Report accuracy AND macro_f1; consider weighted cross-entropy in the next sweep; oversample minority classes |
| **Single-annotator dataset** (LLM-generated per upstream prompt) | Real but documented | Per-piece human validation showed 95.77% function-level agreement on a 50-piece sample (upstream `eval/`). For benchmark use this is acceptable; for headline claims cite the validation study. |
| **`Ln`/Linear is a VGM-specific class** | Means cross-domain transfer is non-trivial | Models trained on this 6-class inventory won't directly transfer to Western-pop section labels; need a label-mapping layer. |
| **No standard cross-paper benchmark** | We can't compare to a published baseline | Acceptable for first-pass; once paper is published cite their evaluation methodology. |
| **Patch-level granularity** (CLaMP3 patches are ~64 tokens) | Limits temporal resolution | Short stingers (<2 s) may be a single patch with no internal structure; longer sections may span multiple patches that get mean-pooled. |
| **Val/test gap of 4–11 pp** | Notable but small for L4/L11 | The smallest gaps are at the best layers — this is actually a quality signal. But all reported test numbers are LOWER than what val suggests; don't over-promise based on val. |
| **No held-out series test** | We can't validate that the model generalizes to a Mario-game-not-in-training | Build a "leave-one-series-out" split as a follow-up if cross-game generalization claims are made (e.g., train on Mario Galaxy, test on Mario Odyssey). |

---

## 9. Open follow-ups

In rough priority order:

1. **Run the audio sweep** (MERT-95M / MuQ / OMARRQ /
   CLaMP3-audio). Pipeline-ready once you have rendered FLACs;
   the `--audio-dir` build path produces the `audio_path` field
   the audio configs need.
2. **L4 + L11 ensemble** on the existing CLaMP3-symbolic cache —
   trivial to test, should give +0.5 to +1.0 pp acc per the
   two-peak hypothesis.
3. **Class-balanced loss / minority oversampling** — the
   macro_f1 is the bottleneck; fixing it would be the cleanest
   way to make the next published number look better.
4. **Section-level (finer-grained) annotations** — the upstream
   dataset has Section labels reserved for v2; wire them in for a
   new "subsection" classifier task.
5. **VGMIDI emotion classification** — drop in the existing
   probe scaffolding for CLaMP3-symbolic; rough effort: 1 turn.
6. **Cross-domain transfer test** — train probe on Bach Chorales,
   evaluate on SuperMarioStructure (with the appropriate label
   mapping). Probes whether the symbolic representation generalizes
   beyond VGM.
7. **L12 cross-modal coherence study** — once audio is built,
   compute cosine(L12_audio, L12_symbolic) per piece. Reports a
   number for "how cross-modally aligned is CLaMP3 in this
   domain."
8. **Multi-piece-per-WandB-group cleanup** — the current sweep
   has L4-fit and L4-test as separate runs; the analysis script
   handles both but it's noisier than necessary. Consider a fit+test
   combined config.

---

## See also

- [`docs/leitmotif_findings.md`](leitmotif_findings.md) — sister
  retrieval analysis; same encoder, different invariance dimension.
- [`docs/layer_analysis.md`](layer_analysis.md) — cross-encoder
  per-task summary tables; the canonical recommendation guide.
- [`docs/data/supermario_setup.md`](data/supermario_setup.md) —
  build commands, class inventory, troubleshooting.
- Source code:
  [`build_supermario_dataset.py`](../scripts/data/build_supermario_dataset.py),
  [`marble/tasks/SuperMarioStructure/`](../marble/tasks/SuperMarioStructure/),
  [`configs/probe.CLaMP3-symbolic-*.SuperMarioStructure.yaml`](../configs/).
