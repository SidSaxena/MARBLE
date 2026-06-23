# CLaMP3 Contrastive Metric-Learning Plan — Sharpening Symbolic Embeddings for Motif/Leitmotif Discovery

**Status:** PLAN (no training code written). Code-grounded against the MARBLE repo
at `/Users/sid/Developer/Python/marble` (branch `main`).

**Goal.** Learn a small contrastive projection that sharpens CLaMP3-symbolic
embeddings for motif/leitmotif similarity, trained supervised on labeled proxy
corpora (BPS-Motif, MTC-ANN, JKUPDD, full MTC) and deployed UNSUPERVISED on Zelda
game music. The end task has no Zelda labels, so the whole plan is gated on
**cross-domain transfer between proxies** — if a head/adapter doesn't transfer
BPS↔MTC, it will not be trusted toward Zelda.

**Cheapest-first ladder.** (1) Frozen-encoder projection head on the L9 hidden
state → (2) LoRA on the M3 backbone layers feeding L9, only if (1) helps → with a
mandatory cross-domain holdout GATE before either is trusted toward Zelda.

---

## 0. Established facts this plan builds on (given; not re-derived)

- CLaMP3-symbolic encoder = M3 backbone: a HF `BertModel` with
  `PATCH_NUM_LAYERS = 12` transformer layers → **13 hidden states (indices
  0–12)**. `hidden_states[12] == last_hidden_state`; the `symbolic_proj` head
  (L12-projected) is the WORST layer for motif similarity.
- Best layer for cross-piece / cross-melody motif matching is **deep, ~L8–L9**
  (validated JKUPDD + MTC-ANN this campaign). **This plan taps the L9 hidden
  state** (`hidden_states[9]`).
- Input format is **ABC** (score-native interleaved). ABC's advantage scales with
  fragment size: wins at phrase/theme scale, ties MTF at the 3–5-note cell scale.
- **Centering** embeddings before cosine helps consistently; **whitening is a
  wash** (`docs/symbolic_clamp3_methodology_lessons.md` §d). The learned head
  should subsume centering, but we keep centered-cosine eval as the reference.

---

## 1. Where the code already does what we need (grounding)

| Need | Existing code | Notes |
|---|---|---|
| Per-layer L9 hidden state from frozen encoder | `marble/encoders/CLaMP3/model.py` → `CLaMP3_Symbolic_Encoder.forward()` returns **tuple of 13 tensors**, each `(B, 1, H=768)`; the per-layer pooling is `_get_symbolic_layer_embeddings()` (lines ~711–777), which runs the inner `BertModel` with `output_hidden_states=True` and mask-mean-pools each of the 13 `hidden_states` over real patches. | `hidden_states[9]` = element 9 of that tuple. We reuse this verbatim. |
| Selecting one layer | `marble/modules/transforms.py` → `LayerSelector(layers=[9], mode="select")` indexes the tuple: `selected = [hidden_states[i] for i in self.layers]` (line 280). Driven by YAML `init_args.layers: [N]`. | Layer choice is one integer in a config — cheap to change. |
| ABC tokenisation per fragment | `_BPSMotifABCMixin._tokenise_abc` (BPSMotif) and `_MTCANNRetrievalABCDataset._tokenise_abc` (MTCANN) — both call `M3Patchilizer.encode(abc, add_special_patches=True)` then pad to `PATCH_LENGTH`. | Already produces the exact `(P, PATCH_SIZE)` patch tensor the encoder consumes. |
| Labeled occurrences → jsonl | `scripts/data/build_mtc_ann_dataset.py`, `build_bps_motif_dataset.py`/`build_bps_motif_abc.py`, `build_jkupdd_retrieval.py`/`build_jkupdd_abc.py`. | Schema below. |
| Relevance grouping (positive pair definition) | `_work_id(group)` = `sha1(group)[:8]`; MTC-ANN motif `group = "<family>|<motifclass>"`, tunefamily `group = family`; BPS `_encode_work_id(piece_id, motif_letter)`. `_family_id(group)` = `sha1(group.split("|")[0])` for same-family masks. | Same hash that drives eval relevance → train/eval label definitions are identical. |
| Same-family hard-distractor MAP (the GATE metric) | `marble/utils/retrieval_metrics.py` → `compute_masked_map(sim, work_ids, gallery_groups=..., query_subset=...)` (line 824). | Reused as-is for the gate and length-stratified eval. |
| Full retrieval orchestration (cosine, centering, MAP, same-family MAP, length strat) | `marble/tasks/Covers80/probe.py` → `CoverRetrievalTask.on_test_epoch_end` (raw/centered/whitened MAP; same-family + len-stratified block lines 534–635). MTCANN/BPS tasks are thin subclasses. | Our eval re-embeds with the trained head, then calls the SAME metric block. |
| Forward-once / cache-all-layers | `marble/utils/emb_cache.py` → `EmbeddingCacheMixin._cached_forward_layer_tuple` caches the full `(B, L, H)` pooled stack per clip; `cache_embeddings: true` in configs. | The full 13-layer stack is already cached per fragment — exactly the "cache the layer stack so layer choice stays cheap" strategy. |

**Key consequence:** we do NOT need new encoder code. The frozen-head phase reads
the existing per-fragment L9 vector (or the cached 13-layer stack) and trains a
head offline; eval reuses the existing `CoverRetrievalTask` MAP machinery.

---

## 2. Data pipeline — assembling (anchor, positive, negative) batches

### 2.1 Relevance labels already exist in the jsonl

Every retrieval jsonl row carries a `group` string (the relevance key) plus an
`occurrence_id`, the `abc` field (ABC arm) or `midi_path` (MTF arm), and for
MTC-ANN motifs `n_src_notes`. From `build_mtc_ann_dataset.py`:

- **MTC-ANN Motif** (`data/MTC-ANN/MTCANN.Motif.ABC.jsonl`):
  `group = "<family>|<motifclass>"`, plus `n_src_notes`. A positive pair = two
  rows with the **same `group`**; a same-family hard negative = same
  `family` prefix, **different** `motifclass` suffix.
- **MTC-ANN TuneFamily** (`MTCANN.TuneFamily.ABC.jsonl`): `group = family` →
  melody-scale positives (whole-melody fragments). Pool for the large theme-scale
  contrastive signal.
- **BPS-Motif Retrieval** (`data/BPS-Motif/BPSMotifABC.Retrieval.fold{f}.{split}.jsonl`):
  relevance key from `_encode_work_id(piece_id, motif_letter)` (within-movement
  motif identity). 263 motifs / 4,944 occ — the largest positive pool.
- **JKUPDD** (`build_jkupdd_abc.py` output): 23 patterns / 91 occ; small set of
  extra cross-piece positives.
- **Full MTC** (`D:/datasets/MTC` = WSL `/mnt/d/datasets/MTC`, ~19 GB): tens of
  thousands of melodies grouped by tune family. A melody-level theme contrastive
  pool. **Phase-2 only** (needs a build pass to emit ABC + `group=family` jsonl
  in the same schema; reuse `build_mtc_ann_dataset.py`'s `kern_to_abc` /
  `score_to_abc` path with `--task tunefamily` semantics). Do not block Phase 1
  on this.

### 2.2 Class-balanced P-K sampler (the batch construction)

We treat each unique `group` as a **class**. Build a `group → [row indices]` index
from the jsonl at load. Per batch use **P-K sampling**:

- Sample **P classes** (groups with ≥2 occurrences — singletons can never form a
  positive pair and are dropped from the *anchor* pool but kept as in-batch
  negatives), and **K occurrences** per class.
- Batch size `B = P·K`. Every item then has K−1 in-batch positives and
  (P−1)·K in-batch negatives, which is exactly what InfoNCE/NT-Xent over the
  `B×B` similarity matrix consumes (and what the existing `ClipLoss` in
  `clamp3_util.py` already implements as a cross-entropy over a logits matrix —
  we mirror its structure but for same-class supervision).
- Suggested start: **P = 16, K = 4 → B = 64** (single-GPU friendly). For the
  cell-scale MTC-ANN motifs many groups have only 2–3 occ, so cap K at the class
  size and oversample classes to keep B stable.

### 2.3 Same-family hard-negative sampler (MTC-ANN only)

The discriminative signal we ultimately care about is "same motif vs
**same-family-different-motif**," not "same motif vs random other melody" (the
full gallery is ~87% other-family easy negatives per the Covers80 probe note).
So for MTC-ANN Motif we bias the P-class draw:

- Group classes by `family` (the `group` prefix). When sampling the P classes,
  with probability `p_hard` (start **0.5**) draw a **cluster of classes from the
  SAME family** rather than P independent random classes. This packs each batch
  with same-family-different-motif pairs → InfoNCE's denominator is dominated by
  hard negatives, which is the gate's target distribution.
- This is the train-time analogue of `compute_masked_map`'s `gallery_groups`
  mask: train against the hard distractor we evaluate against.
- BPS / JKUPDD / full-MTC have no same-family-different-class structure (BPS
  motif letters are movement-local; full-MTC is family-level), so they fall back
  to plain P-K with random classes.

### 2.4 Mixing datasets

Phase 1 trains a head per **single source domain at a time** (the gate needs
held-out *other* domains — see §5). For a final "all-proxy" head we concatenate
the per-dataset group indices into one namespace (groups are already globally
unique via the sha1 of the full `group` string), then P-K sample across the
union. Keep a `dataset_id` per row so the gate can hold a whole dataset out.

---

## 3. Embedding extraction + caching

### 3.1 Exact tap

Per fragment:

1. Tokenise the `abc` field → `(P, PATCH_SIZE)` via the existing `_tokenise_abc`
   (BPSMotif/MTCANN datamodules).
2. Run `CLaMP3_Symbolic_Encoder.forward(patches)` (frozen, `train_mode="freeze"`)
   → tuple of 13 tensors `(1, 1, 768)`.
3. **L9 vector = element 9 of that tuple**, squeezed to `(768,)`. Equivalent to
   `LayerSelector(layers=[9]) → TimeAvgPool` (TimeAvgPool is a no-op on the
   already-pooled `T=1` cache output).

### 3.2 Cache the full layer stack once (so layer choice stays cheap)

CLaMP3 forwards ONCE and returns all 13 layers (`_get_symbolic_layer_embeddings`
already does `output_hidden_states=True`). `EmbeddingCacheMixin` persists the
pooled `(L=13, H=768)` stack per `clip_id`. **Strategy:**

- Run **one offline extraction pass** per dataset/arm that writes the
  `(13, 768)` pooled stack for every fragment to the emb cache (set
  `cache_embeddings: true`; reuse the existing keying — `make_clip_id` on
  `occurrence_id`).
- The contrastive trainer then loads cached **L9** slices (or any layer) with
  **zero encoder forwards** — training is a tiny-MLP problem over a fixed
  `(N, 768)` matrix. This makes Phase 1 essentially free on GPU and lets us
  re-pick the layer (L8 vs L9 vs mean-of-deep) without re-encoding.
- For LoRA (Phase 2) the cache is bypassed for the adapted layers (we must
  backprop through them) — cache only the FROZEN lower layers' inputs if
  feasible, else forward live (see §4).

Concretely: a one-time `scripts/data/extract_clamp3_layers.py`-style pass (NOT
written here) that iterates each jsonl, tokenises, forwards once, and saves the
`(13,768)` stack keyed by `occurrence_id`. Everything downstream is matrix algebra.

---

## 4. Model + loss

### 4.1 Projection head architecture (Phase 1)

A small MLP on the L9 vector:

```
in_dim   = 768            # H of hidden_states[9]
hidden   = 512            # one hidden layer
out_dim  = 256            # projection dim for cosine retrieval
head     = Linear(768→512) → GELU → LayerNorm(512) → Linear(512→256)
```

- **Depth 2** (one hidden layer). Deeper overfits the few-hundred-class proxy
  sets; a linear-only head underfits the same-family discrimination.
- **out_dim 256**: smaller than 768 to regularise + speed cosine; large enough to
  preserve motif structure. (Ablate 128/256/512 in the smoke test.)
- **Pre-head centering**: subtract the *training-set* mean of L9 (inductive, saved
  with the head) before the MLP, since centering is the one preprocessing known to
  help and we want it applied identically at deploy time on Zelda (no per-corpus
  transductive fit available there).
- Output is **L2-normalised** before cosine (matches the eval's `F.normalize`).

### 4.2 Loss — **InfoNCE / NT-Xent (supervised, multi-positive)** — recommended

Reasoning:

- The data is **multi-positive per class** (K−1 positives per anchor in a P-K
  batch). Supervised-contrastive InfoNCE (SupCon-style) uses *all* same-class
  items as positives in the denominator-normalised softmax — strictly more signal
  per batch than triplet's single (a,p,n).
- It maps **directly onto existing code**: CLaMP3's `ClipLoss`
  (`clamp3_util.py`) is already an InfoNCE over a logits matrix with a
  temperature (`LOGIT_SCALE`). We reuse that *pattern* but build the positive mask
  from `group` equality instead of the diagonal-only cross-modal pairing.
- Triplet/margin loss needs hard-negative mining heuristics and a margin to tune,
  and wastes the multi-positive structure. We keep triplet only as a fallback if
  InfoNCE collapses on the tiny JKUPDD set.

Loss form (supervised NT-Xent), per batch with L2-normalised projections `z`:

```
sim = (z @ z.T) / temperature          # B×B, self-masked to -inf on diagonal
for anchor i:  positives = {j : group_j == group_i, j != i}
loss_i = -mean_{p in positives} log( exp(sim[i,p]) / sum_{k!=i} exp(sim[i,k]) )
loss   = mean_i loss_i
```

- **Temperature**: start **0.07** (standard NT-Xent); ablate 0.05–0.1. (Note the
  pretrained `LOGIT_SCALE = 1.0` is NOT a good init for from-scratch metric
  learning — 0.07 is.)
- **Optimizer**: AdamW, **lr 3e-4** for the head (it's tiny and from-scratch — the
  existing CLaMP3 configs use 1e-3 for the linear probe; 3e-4 is safer for a
  2-layer MLP), weight_decay 1e-4, cosine LR schedule, **20–50 epochs** over the
  cached matrix (each epoch is seconds).
- **Early stopping** on the held-out same-family MAP (§5), NOT on train loss.

### 4.3 Why not just fine-tune the existing `symbolic_proj` head

`symbolic_proj` (L12-projected) is the *worst* layer; retraining it would fight
the cross-modal pretraining and discard the deep-layer advantage. We deliberately
tap L9 *before* that head and learn a fresh projection.

---

## 5. LoRA variant (Phase 2 — only if the head helps)

Trigger: proceed to LoRA **only if** the frozen head clears the gate (§6) with a
clear margin over the centered-cosine L9 baseline. Otherwise LoRA's extra
capacity just over-fits the proxy genre.

- **What to adapt:** LoRA adapters on the **attention q/v projections of M3
  transformer layers that feed L9** — i.e. encoder layers **~6–9** (the blocks
  whose output becomes `hidden_states[9]`). Adapting layers *above* 9 is useless
  (they don't affect `hidden_states[9]`); adapting all of 0–9 is more capacity but
  more genre-overfit risk, so start with the top contributing window (7–9).
- **Rank:** start **r = 8**, alpha 16 (low-rank, cheap, low overfit). Ablate r ∈
  {4, 8, 16}.
- **Frozen:** patch embedding, `text_model`, `audio_model`, both other proj heads,
  and all M3 layers **outside** the adapted window. The Phase-1 MLP head stays on
  top of L9 and is trained jointly.
- **Caching caveat:** the emb cache can no longer serve the adapted layers
  (gradients required). Cache the **input to the first adapted layer** (the frozen
  prefix output) per fragment so the live forward is only the 3–4 adapted blocks +
  head — keeps Phase 2 affordable. If that prefix-cache is too fiddly, forward the
  full frozen encoder live (still feasible at these corpus sizes on the PC GPU).
- Same loss / sampler / gate as Phase 1.

---

## 6. Cross-domain transfer GATE (mandatory)

This is the load-bearing decision step. Zelda has no labels, so the only evidence
we can get is **does the learned metric transfer across proxy domains?**

### 6.1 Splits

Two symmetric holdouts (train on one domain family, test on the disjoint other):

- **Split A:** TRAIN on **BPS-Motif + JKUPDD** (Beethoven piano + classical
  patterns). TEST on **MTC-ANN** (Dutch folk melodies). No tune family, motif
  class, or melody seen in training.
- **Split B (reverse):** TRAIN on **MTC-ANN** (+ full-MTC in Phase 2). TEST on
  **BPS-Motif** held-out folds + **JKUPDD**.

Within each test domain, eval is the existing zero-shot `CoverRetrievalTask` run:
re-embed test fragments with the trained head, then call the standard metric
block. The head must NEVER see test-domain `group`s during training.

### 6.2 Metric (reuse `compute_masked_map`)

Primary number = **same-family MAP** on MTC-ANN Motif
(`test/map_samefamily` / `test/map_samefamily_centered`), i.e.
`compute_masked_map(sim, work_ids, gallery_groups=family_id)` — the confound-free
"is this the same motif, given same tune family?" number. Secondary:

- Length-stratified same-family MAP (`map_samefamily_len_le3` /
  `_len_gt3`) — guard against short-cell collapse (the ABC=MTF tie regime).
- Standard `map_centered` on BPS held-out folds and JKUPDD for Split B.

Baseline to beat = **frozen centered-cosine L9** (no head) on the SAME test
splits — already the campaign's current best symbolic configuration.

### 6.3 GO / NO-GO decision rule

Let `Δ_sf` = (head same-family MAP) − (frozen centered-cosine L9 same-family MAP),
measured on the **held-out** domain, in BOTH directions A and B.

- **GO (proceed toward Zelda / to Phase 2):** `Δ_sf ≥ +0.03` (absolute MAP) in
  **both** Split A and Split B, AND the long-motif bucket (`len_gt3`) is **not
  worse** than baseline (no regression on the regime where ABC is supposed to
  win), AND no collapse on the short-motif bucket beyond baseline. The +0.03 / both
  directions requirement is what distinguishes a *transferable* metric from one
  that memorised proxy-specific structure.
- **WEAK / ITERATE:** improvement in one direction only, or `0 < Δ_sf < 0.03` in
  both. Treat as "head learned something domain-specific." Do NOT advance to LoRA;
  instead revisit sampler (`p_hard`), temperature, out_dim, or layer (try L8 / a
  mean of L8–L9). Re-run the gate.
- **NO-GO (stop the supervised-head direction):** `Δ_sf ≤ 0` on either split, OR a
  bucket regression. The head is overfitting proxy genre and will not help on
  Zelda. Fall back to the frozen centered-cosine L9 deployment and reconsider
  (model/layer fusion, or abandon the contrastive lead per the thesis ordering).

**Phase-2 gate is the SAME rule** applied to the LoRA model, additionally
requiring LoRA to beat the *Phase-1 head* (not just the frozen baseline) by
`≥ +0.02` on the held-out same-family MAP — otherwise the extra capacity isn't
earning its overfit risk.

---

## 7. Compute

All training/eval on the PC GPU: `ssh my-pc` → WSL `/home/sid/developer/marble`
(CLaMP3 weights cached at `~/.cache/clamp3/`, wandb authed). Mac stays the git
home; move artifacts PC→Mac via `cat`/`tar` over SSH
(`docs/symbolic_clamp3_methodology_lessons.md` §e).

| Phase | Work | Rough cost |
|---|---|---|
| Extraction | One forward per fragment, cache `(13,768)` stack. BPS ~5k + MTC ~2k + JKU ~0.1k ≈ 7k fragments; full-MTC adds tens of thousands (Phase 2). | ~minutes for proxies on GPU (encoder is the only cost; sub-ms tokenise). Full-MTC ~1–2 h one-time. |
| Phase 1 head train | 2-layer MLP over cached `(N,768)`; no encoder forwards. 20–50 epochs × seconds/epoch. | **<5 min per run.** Whole sampler/temp/out_dim/layer sweep in well under an hour. |
| Phase 1 gate eval | 2 holdout directions × `CoverRetrievalTask` zero-shot on cached embeddings re-projected through the head. N≈700 sim is sub-second. | **minutes total.** |
| Phase 2 LoRA | Live forward through 3–4 adapted blocks + head (or prefix-cached). ~7k fragments × small epochs. | **~tens of min – ~2 h** per run depending on prefix-cache; modest. |

The frozen-head phase is cheap enough to run many configs; LoRA is the only
non-trivial cost and is gated behind a Phase-1 GO.

---

## 8. Risks + mitigations

1. **Genre gap (THE main risk).** Proxies are Beethoven piano + Dutch folk +
   classical themes; deploy is Zelda orchestral game music. A head can memorise
   proxy-genre surface statistics that don't exist in Zelda.
   - *Mitigation:* the cross-domain gate (§6) is precisely the genre-gap detector —
     BPS↔MTC are already a large genre jump (piano sonata ↔ monophonic folksong).
     Requiring transfer in BOTH directions is the strongest signal we can get
     without Zelda labels. Prefer the **smaller / centering-equivalent** head and
     LoRA window; capacity = overfit here.
2. **Cell-scale collapse.** ABC ties MTF at the 3–5-note cell scale; a head tuned
   on phrase-scale positives could degrade tiny motifs.
   - *Mitigation:* the length-stratified same-family MAP (`len_le3` / `len_gt3`)
     is a gate sub-criterion; a short-bucket regression is a NO-GO.
3. **Notation confound leaking into labels** (`methodology_lessons.md` §b):
   BPS MNID negatives' tuplet-mess. We use the **Retrieval** jsonl (not MNID), so
   the binary-negative notation confound does not apply, but verify the ABC arm's
   rendered notation parity if mixing BPS positives at the cell scale.
4. **Singleton groups / tiny JKUPDD.** Many MTC motif classes have 2–3 occ;
   JKUPDD is 91 occ total. InfoNCE can collapse with too few positives per batch.
   - *Mitigation:* drop singleton groups from the anchor pool (keep as negatives);
     oversample multi-occ classes; cap K at class size; triplet fallback if
     InfoNCE collapses on JKUPDD-only.
5. **Layer assumption (L9).** Best layer is regime-dependent
   (`methodology_lessons.md` §c). L9 is right for cross-piece retrieval but worth
   a cheap check.
   - *Mitigation:* the cached 13-layer stack makes re-picking the tap layer free —
     ablate L8 / L9 / mean(L8–L9) in the smoke test before committing.
6. **Transductive vs inductive centering.** Eval uses transductive centering (fit
   on the test corpus); Zelda deploy has no such fit.
   - *Mitigation:* bake an **inductive** (train-set) center into the head (§4.1) so
     train/deploy preprocessing match; report both numbers.
7. **Distribution of negatives at deploy.** On Zelda, retrieval is unsupervised
   within one game's score — the negative distribution differs from the gate's.
   - *Mitigation:* the same-family hard-negative sampler approximates
     "within-piece, similar-but-different" negatives, the closest proxy to Zelda's
     within-game distractors. Document this as a residual unknown (open question).

---

## 9. Phased checklist

### Build first (Phase 0 — extraction & plumbing)

- [ ] Confirm the four ABC jsonls exist on the PC (`data/BPS-Motif/…ABC…`,
      `data/MTC-ANN/MTCANN.{Motif,TuneFamily}.ABC.jsonl`, JKUPDD ABC). Rebuild via
      the existing `build_*` scripts if missing.
- [ ] One-time **layer-stack extraction** pass: tokenise each fragment, forward
      `CLaMP3_Symbolic_Encoder` once, cache `(13,768)` per `occurrence_id`
      (reuse `cache_embeddings: true` machinery / `make_clip_id`).
- [ ] Build the `group → indices` and `family → groups` indices from each jsonl;
      implement the **P-K + same-family hard-negative sampler** (data-only; no
      model).

### Smoke test (single source, fast)

- [ ] Train the Phase-1 head on **MTC-ANN Motif ABC** only (richest same-family
      structure), L9, InfoNCE, P=16/K=4, temp 0.07. Confirm train loss decreases
      and in-domain (train-family) same-family MAP rises above frozen L9.
- [ ] Ablate tap layer (L8 / L9 / mean) and out_dim (128/256/512) on this single
      run — pick the config for the real gate. (Each run <5 min.)

### Full run (the gate)

- [ ] **Split A:** train on BPS+JKUPDD, eval same-family MAP on MTC-ANN (held-out).
- [ ] **Split B:** train on MTC-ANN(+full-MTC), eval `map_centered` + same-family
      MAP on BPS held-out folds + JKUPDD.
- [ ] Apply the **GO/NO-GO rule** (§6.3). Log baseline (frozen centered-cosine L9)
      on the identical splits for `Δ_sf`.
- [ ] wandb: representation tag `CLaMP3-symbolic-abc-contrastive`, task name clean
      (`MTCANNMotif` / `BPSMotifRetrieval`), `job_type ∈ {fit,test}`
      (`methodology_lessons.md` §f). Commit a campaign-report doc + the gate table.

### Phase 2 (only on a Phase-1 GO)

- [ ] Add LoRA (r=8, layers ~7–9 q/v), prefix-cache the frozen prefix, retrain
      head+adapters jointly, re-run BOTH gate splits with the Phase-2 rule.
- [ ] If GO: produce the deployable head/adapter checkpoint + the inductive
      center; document the exact Zelda deploy recipe (tokenise ABC → frozen
      encoder → L9 → (LoRA) → center → head → L2-norm → cosine).

---

## 10. Open questions for the user (need a decision)

1. **Full-MTC build scope.** The ~19 GB full MTC needs a one-time ABC +
   `group=family` build pass (reuse `build_mtc_ann_dataset.py` tunefamily path).
   Is the **melody/theme-scale** family pool worth building for Phase 1, or hold
   it for Phase 2 (it's family-scale, not motif-cell-scale — helps theme-scale
   transfer more than cell-scale motif discrimination)?
2. **Gate threshold.** Is `Δ_sf ≥ +0.03 in both directions` the right bar, or do
   you want it tighter/looser given how noisy small-corpus MAP is (JKUPDD = 91
   occ)? A bootstrap CI on the held-out MAP could replace the fixed threshold.
3. **Deploy representation on a NO-GO.** If the head doesn't transfer, do we (a)
   fall back to frozen centered-cosine L9 (current best) and drop the contrastive
   lead, or (b) try the model/layer-fusion direction first (the thesis future-work
   ordering puts fusion before the contrastive lead)?
4. **Which negative distribution best mirrors Zelda?** Within-game distractors are
   unlabeled. Is the same-family-different-motif negative an acceptable proxy, or
   should we mine cross-instrument/cross-orchestration negatives (the
   re-orchestration scenario) once a Zelda gallery exists?
5. **ABC vs MTF arm for training.** Plan trains on the **ABC** arm (score-native,
   wins at phrase scale). Confirm we don't also want an MTF-arm head for the
   cell-scale tie regime, or a 2-arm ensemble.
