# MedleyDB Melody — dataset comparison vs HookTheoryMelody, and layer-sweep plan

Status: **planning** (greenfield — no MedleyDB code exists in this fork as of 2026-06-30).
Audience: layer-selection for music-foundation-model probing toward **leitmotif discovery in
the Breath of the Wild OST**.

---

## 0. TL;DR

- **Proxy verdict:** **MedleyDB is the better domain proxy for BotW leitmotifs** (instrumental,
  classical/acoustic/orchestral content; predominant-melody-from-dense-polyphony framing).
  **HookTheoryMelody is the better-powered, lower-variance proxy** (22k clips vs 108 tracks) but
  pop/rock/vocal-skewed. **Recommendation: keep both; weight MedleyDB for the BotW question and
  treat agreement of the two layer rankings as the confidence signal.**
- **Implementation:** clone `marble/tasks/HookTheoryMelody/` → `MedleyDBMelody/`, reuse `probe.py`
  verbatim (frame-level 128-MIDI, RPA/RCA, `MelodyCrossEntropyLoss`), and write a new datamodule
  whose only difference is the **label source**: MedleyDB gives per-frame f0 in **Hz at 5.8 ms hop**
  directly, so we drop HookTheory's beat→time `interp1d` and just bin `Hz → MIDI` per frame.
- **The one big decision:** frame-level tasks are **excluded from the current embedding cache**, so
  HookTheoryMelody re-runs the frozen encoder every epoch × every layer. **MedleyDB is small enough
  (7.3 h) that a frame-level `(L,T,H)` precompute cache becomes feasible (~17–40 GB/encoder) where
  HookTheory's (~50 h → ~400 GB) does not.** Building it = forward-once-all-layers → ~10× sweep
  speedup. Two tracks below: **A) no-cache (start today)**, **B) frame-cache (recommended).**

---

## 1. What each dataset is

### HookTheoryMelody (our current task)
- **Source:** Donahue, Thickstun & Liang, *Melody transcription via generative pre-training*,
  ISMIR 2022 (arXiv 2212.01884) — the "Sheet Sage" dataset; in our fork via `m-a-p/HookTheory`'s
  `Hooktheory.json.gz`, audio downloaded from YouTube.
- **Scale:** ~22k annotated segments / ~13k unique recordings / **~50 h** labeled audio.
- **Content:** crowd-sourced TheoryTab hooks → **pop/rock-skewed** (some EDM/jazz/classical), audio
  is **YouTube full mixes** (variable fidelity), melody is **monophonic** but lives inside polyphony.
- **Annotations:** beat-synchronous melody notes (scale-degree + relative octave originally),
  converted to **absolute MIDI** in our datamodule (`pitch_class + (5 + octave)*12`). Octave is
  partly synthetic given the source's relative-octave ambiguity.
- **Our task framing (verified in code):** frame-level **128-class MIDI** classification,
  `label_freq` = encoder token rate (MERT 75 Hz, MuQ/OMARRQ 25 Hz), silence = `-1` (masked),
  `clip_seconds=15.0`, `MLPDecoderKeepTime`, metrics **RPA** (monitored) + **RCA**,
  `cache_embeddings: false`.

### MedleyDB (proposed task)
- **Source:** Bittner et al., *MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research*,
  ISMIR 2014. Site: medleydb.weebly.com. License **CC BY-NC-SA** (3.0 in paper / 4.0 on site).
- **Scale:** V1 = 122 songs, **108 with melody annotations**, totaling **26,831 s ≈ 7.3 h**.
  105/122 are full-length (mostly 3–5 min). (V2 adds 74 more multitracks if ever needed.)
- **Content:** professional/studio **multitrack** recordings, **WAV 44.1 kHz/16-bit**. Nine genres
  incl. **Classical, Jazz, Fusion, World/Folk, Singer-Songwriter, Rock, Pop, Musical Theatre, Rap**.
  **57 % vocal / 43 % instrumental** among the 108 melodic tracks. For melody extraction we feed the
  **MIX** (not stems/raw). 25 songs have inter-stem bleed (irrelevant once we use the mix).
- **Annotations:** three definitions — **MELODY1** (single-source predominant f0, MIREX def),
  **MELODY2** (predominant f0 allowing the lead source to change — *what MARBLE's paper uses*),
  **MELODY3** (all melodic lines, polyphonic). Format: time-stamped **f0 in Hz, hop 256 samples =
  5.8 ms (~172 fps)**, **unvoiced = 0 Hz**, melodic range ~43–3662 Hz.
- **Reference task framing (MARBLE *paper*, not in our upstream code):** 360 freq bins over
  0–8 kHz + unvoiced, MELODY2, BiLSTM-512 probe, predictions resampled token-rate→label-rate by
  nearest interpolation, metric = mir_eval **Overall Accuracy**, split **67/15/26**.

> **Note on "MERT used MedleyDB":** the MERT *paper* (arXiv 2306.00107) does **not** report a
> MedleyDB / melody task. Its only pitch task is **NSynth pitch** (128-class, monophonic, accuracy;
> MERT-330M ≈ 94.4). The MedleyDB 360-bin melody task belongs to the **MARBLE paper**, which is a
> separate (companion) publication and was **never implemented in the `a43992899/MARBLE` code** we
> forked. So MedleyDB melody is greenfield for us.

---

## 2. Head-to-head

| Dimension | HookTheoryMelody (Donahue 2022) | MedleyDB (Bittner 2014) |
|---|---|---|
| Labeled audio | ~50 h | **7.3 h** |
| Units | ~22k short hooks / ~13k recordings | **108 full songs** (3–5 min) |
| Genre/timbre | pop/rock, **vocal-heavy**, YouTube fidelity | **9 genres incl. classical/jazz/fusion/world**, studio fidelity, 43% instrumental |
| Audio | full mix (polyphonic), monophonic melody target | full mix (polyphonic), monophonic predominant-f0 target |
| Label origin | beat-synced symbolic notes → abs MIDI (octave partly synthetic) | **acoustic f0 in Hz** (pYIN + manual Tony correction) |
| Native label rate | beat grid (16th-note) → resampled to token rate | **5.8 ms (~172 fps)** → resampled to token rate |
| Test-set size | large (~2k+ clips) → **low-variance rankings** | **26 tracks → noisier rankings** |
| Cache feasibility (frame-level) | infeasible (~400 GB/encoder) | **feasible (~17–40 GB/encoder)** |
| Canonical metric | frame RPA/RCA (ours); onset-F (paper) | Overall Accuracy (mir_eval); we can use RPA/RCA for parity |
| License | YouTube-derived (gray) | CC BY-NC-SA, gated Zenodo |

**Core difference in one line:** HookTheory tests *melodic salience in contemporary pop mixes* with
lots of data; MedleyDB tests *predominant-f0 extraction from dense, often acoustic/orchestral
polyphony* across real genres with little data.

---

## 3. Which is the better proxy for BotW leitmotif discovery?

The probe answers "**which frozen-encoder layer best encodes the melodic/thematic line inside a
dense mix?**" — that layer is then what we'd use for leitmotif representation/retrieval on the
orchestral BotW OST. Judging the two proxies on the axes that matter:

1. **Timbre/domain match (favors MedleyDB, strongly).** BotW is orchestral/instrumental game music
   (strings, woodwinds, piano, percussion, choir). MedleyDB contains classical, jazz, fusion,
   world/folk, and 43% instrumental melodies on acoustic instruments — far closer than HookTheory's
   pop/rock/EDM, ~57%-vocal YouTube clips. A layer that wins at picking a violin/oboe line out of an
   acoustic ensemble is more likely the layer that helps with BotW themes.
2. **Task geometry (favors MedleyDB, mildly).** "Find the predominant melodic line in a thick,
   changing-lead polyphonic texture" (MELODY2) is conceptually what a leitmotif detector must do in
   orchestration. HookTheory mixes are typically thinner pop arrangements.
3. **Statistical power / ranking stability (favors HookTheory).** 22k clips vs 26 test tracks —
   HookTheory's layer ranking is far less noisy. MedleyDB's 26-track test set can give jumpy
   per-layer numbers; mitigate with cross-validation or by trusting the broad layer *region* rather
   than the single argmax.
4. **Both share two honest caveats vs the real target:** neither is *game/orchestral VGM*, and both
   are *f0/pitch* tasks, not *theme recognition*. They locate "where melody lives" in the network,
   which is necessary-but-not-sufficient for leitmotif identity (which also needs motif/contour
   memory). Treat them as **layer locators**, not as leitmotif benchmarks.

**Verdict.** For the BotW question specifically, **MedleyDB is the better-aligned proxy** on domain
and task geometry — the two axes that transfer. Its weakness is variance, which HookTheory covers.
**So run both; weight MedleyDB's ranking for BotW, and use HookTheory as the high-N cross-check.**
If the two disagree on the best layer region, prefer MedleyDB's but flag low confidence and confirm
on the actual `LeitmotifDetection` task. (This is consistent with the project's standing rule to not
over-claim and to fact-check transfer — see memory `feedback_verify_novelty_claims`.)

---

## 4. Sourcing MedleyDB

**Long-lead item — start the access request today** (manual approval, can take days):

1. **Audio (gated):** MedleyDB V1 Audio on Zenodo — https://zenodo.org/records/1649325 — log in,
   "Request access." We only need the **`*_MIX.wav`** per track (108 files), not stems/raw, so the
   working set is ~**4–5 GB** even though the full archive is tens of GB.
2. **Annotations (free, no gate):**
   - `marl-internal/medleydb` GitHub repo (annotations + YAML metadata + instrument taxonomy), or
   - **`mirdata`** with the `medleydb_melody` loader (wraps the Zenodo *MedleyDB Melody* record
     https://zenodo.org/records/2628782) — gives MELODY1/2/3 CSVs (time, Hz) per track.
   - `uv pip install mirdata` then `mirdata.initialize('medleydb_melody')`; it validates checksums and
     exposes `track.melody2` arrays.
3. **Place on the PC** under `/mnt/d/datasets/medleydb/` (datasets live on D: per project setup):
   `audio_mix/<track>_MIX.wav` + `annotations/<track>_MELODY2.csv`.
   *Keep the corpus on D:, but keep sweep **output/checkpoints on ext4** — DrvFs breaks checkpoint
   moves (sendfile ENOMEM) and dataloader AF_UNIX (see memory `reference_wsl_drvfs_checkpoint_gotcha`).*

Fallback if Zenodo access lags: the small free **sample** on medleydb.weebly.com/downloads.html lets
us build and smoke-test the pipeline end-to-end before the full grant lands.

---

## 5. Build plan

### 5.1 Task code (`marble/tasks/MedleyDBMelody/`)
- **`probe.py`: copy HookTheoryMelody's verbatim.** Same `ProbeAudioTask`, `MLPDecoderKeepTime`,
  `MelodyCrossEntropyLoss(ignore_index=-1)`, `RawPitchAccuracy`/`RawChromaAccuracy`,
  `_crop_to_min_t(time_dim_mismatch_tol=5)`. No changes needed — it's label-source agnostic.
- **`datamodule.py`: adapt from HookTheoryMelody.** Keep clip-splitting, channel handling, resample,
  `precompute_labels`, the `cache_check_fn` hook, the `label_freq × clip_seconds ∈ ℤ` guard, and the
  4-tuple `__getitem__`. **Replace only `_compute_labels`:**
  - Load the track's MELODY2 CSV → arrays `(t_sec, f0_hz)` at 5.8 ms hop.
  - For a clip `[start_sec, start_sec+clip_seconds)` build `label_len = round(label_freq*clip_seconds)`
    frames; for each frame center time, **nearest-neighbor sample** the f0 series (matches MARBLE's
    "nearest interpolation" token-rate alignment).
  - **`f0==0 → label -1` (unvoiced, masked).** Else `midi = int(round(69 + 12*log2(f0/440)))`,
    clamp to [0,127]. (Optional toggle for the MARBLE-faithful 360-bin/0–8 kHz scheme — see 5.4.)
  - No beat→time `interp1d` (MedleyDB is already in seconds) — simpler and exact.
- **JSONL schema (`scripts/data/build_medleydb_melody_jsonl.py`):** one record per track:
  `{"track_id", "audio_path": ".../<track>_MIX.wav", "melody_csv": ".../<track>_MELODY2.csv",
  "sample_rate", "num_samples"}`. Pre-probe `sample_rate`/`num_samples` (reuse
  `scripts/data/cache_audio_info_in_jsonl.py`) to skip per-file `torchaudio.info`.
- **Split:** reproduce MARBLE's **67/15/26** track partition for comparability; emit
  `data/MedleyDB/MedleyDB.{train,val,test}.jsonl`. (Optionally also a 5-fold variant to tame the
  26-track test variance.)

### 5.2 Configs (`configs/probe.*-layers.MedleyDBMelody.yaml` + `-meanall` siblings)
Clone the five HookTheoryMelody configs (MERT-95M, MERT-330M, MuQ, MusicFM, OMARRQ-25hz) and change
only: task tag/group/tags/name → `MedleyDBMelody`; datamodule class → `MedleyDBMelodyDataModule`;
jsonl + audio paths; W&B `save_dir`. **Keep unchanged:** `out_dim: 128`, `sample_rate: 24000`,
`clip_seconds: 15.0`, `cache_embeddings: false`, `cache_pool_time: false`, and **per-encoder
`label_freq`** (MERT 75, MuQ/OMARRQ/MusicFM 25). W&B naming follows house convention
(`name: <model>-layers`, `job_type: probe`, `project: marble`, `group: "<model> / MedleyDBMelody"`).

### 5.3 Sweep registration & launch
Add a `SweepDef` block in `scripts/sweeps/run_all_sweeps.py` mirroring HookTheoryMelody's priority
order: **MuQ (13) → MERT-v1-95M (13) → OMARRQ-multifeature-25hz (24)**. Launch via the canonical
launcher (never bare `cli.py` — see memory `feedback_marble_sweep_naming`):
```
python scripts/sweeps/run_sweep_local.py \
  --base-config configs/probe.MuQ-layers.MedleyDBMelody.yaml \
  --num-layers 13 --model-tag MuQ --task-tag MedleyDBMelody \
  --accelerator gpu --precision bf16-mixed
```
`run_sweep_local.py` runs the `meanall` sibling first (baseline + cache pre-warm if cached), then
fit+test per layer, skipping completed layers on resume. Calibrate `--concurrency` with
`scripts/sweeps/autotune_concurrency.py`; **smoke first** with `scripts/sweeps/smoke_one_layer.sh`.

### 5.4 Optional MARBLE-faithful variant
If we want leaderboard comparability, add a config flag for 360-bin/0–8 kHz output + OA metric +
BiLSTM head. **Not recommended as primary** — 128-MIDI + RPA/RCA keeps MedleyDB *directly comparable
to our HookTheoryMelody layer curves*, which is the whole point of the cross-check. Keep 360-bin as a
secondary run only if a paper needs it.

---

## 6. Caching strategy (the performance crux)

Frame-level melody is **excluded from the existing `(L,H)` post-pool embedding cache**
(`docs/embedding_cache_plan.md` §"Out of scope"): caching needs the pre-pool `(L,T,H)` tensor, which
for HookTheory (~50 h) is 50–500 GB/encoder. **MedleyDB changes the calculus** — at 7.3 h it's
~7× smaller, so a frame-level cache is genuinely affordable:

| Encoder | (L, T@clip, H) per 15 s clip, fp16 | × ~1.75k clips |
|---|---|---|
| MuQ (13L, 25 Hz, 1024) | 13×375×1024×2 ≈ 10 MB | **~17 GB** |
| OMARRQ (24L, 25 Hz, 1024) | 24×375×1024×2 ≈ 18 MB | **~32 GB** |
| MERT-95M (13L, 75 Hz, 768) | 13×1125×768×2 ≈ 22 MB | **~39 GB** |

All comfortably fit on D:. **Two tracks:**

- **Track A — no cache (ship today).** Identical to HookTheoryMelody. Encoder re-runs every epoch ×
  every layer. Correct and zero new code, just slow. Use this to get first numbers while Track B is
  built / Zenodo access lands. Use `--concurrency 2` if 16 GB VRAM fits two frozen forwards.
- **Track B — frame-level precompute cache (recommended).** Extend `marble/utils/emb_cache.py` (or a
  standalone `scripts/embeddings/extract_frames.py`) to store **per-track `(L, T_full, H)` fp16** to
  `/mnt/d/.../.emb_cache_frames/<encoder>/<track>.pt` in **one forward pass over all layers**, then
  have the datamodule's `cache_check_fn` slice `[start_frame:end_frame]` per clip and the probe skip
  the encoder on hit. Forward-once-all-layers → the 13/24-layer sweep reads tensors from disk → est.
  **~10× faster**. Caveat: validate `.pt` **reads** from DrvFs/D: are fine (the WSL gotcha is about
  checkpoint *moves* via sendfile and AF_UNIX sockets, not plain reads); if flaky, stage the cache on
  ext4 (the 17–39 GB/encoder fits if checkpoints are pruned).

---

## 7. Expected runtime (5060 Ti, 16 GB)

No standalone benchmark yet — **anchor to your live HookTheoryMelody sweep**, which is the same
frame-level/uncached machinery on the same GPU:

- **MedleyDB ≈ 1/7 the per-epoch cost of HookTheoryMelody** (7.3 h vs ~50 h of audio, same encoder,
  same uncached forward). So read your actual HTM per-epoch wall-time off W&B and divide by ~7.
- For scale, `run_all_sweeps.py` annotates the *cached, clip-level* NSynth sweeps at ~25–35 h
  (13 layers) / ~50–70 h (24 layers). MedleyDB is **uncached** but **~12× less audio than capped
  NSynth**, so **Track A** likely lands in the **same order of magnitude per encoder, dominated by
  the per-epoch encoder forward × ~15–25 early-stopped epochs × layers**. Plan ballpark:
  - **Track A (no cache):** ~mid-single-digit to low-double-digit **hours per layer**;
    **~1–2 days per 13-layer encoder**, **~2–4 days for OMARRQ (24L)**. ~**5–9 days** for all three.
  - **Track B (frame cache):** one ~10–15 min forward/encoder to build, then probe-from-disk layers
    in tens of minutes → **~half a day to a day per encoder; ~1.5–2 days total.**
- **First action either way: `smoke_one_layer.sh` on MuQ layer 6** to get a measured epoch time, then
  multiply. Don't trust the estimate above over a real smoke number.
- WSL sees only ~15 GB RAM by default (raise via `.wslconfig memory=`); keep `num_workers` ≤ 8 and
  watch the frame-cache loader's RAM (see memory `reference_pc_system_specs`).

---

## 8. Step order (checklist)

1. [ ] Request Zenodo access to MedleyDB V1 Audio (record 1649325) — **do first, it's the long pole.**
2. [ ] `uv pip install mirdata`; pull MELODY2 CSVs + metadata (works without the gated audio).
3. [ ] Build the 67/15/26 split + `build_medleydb_melody_jsonl.py`; stage `*_MIX.wav` on D:.
4. [ ] Clone `HookTheoryMelody/` → `MedleyDBMelody/`; reuse `probe.py`, swap `_compute_labels`.
5. [ ] Clone the 5 configs; verify the `label_freq × clip_seconds ∈ ℤ` guard for each token rate.
6. [ ] `smoke_one_layer.sh` (MuQ L6) on the free sample → confirm pipeline + measure epoch time.
7. [ ] Register `SweepDef`s; launch **Track A** (MuQ → MERT-95M → OMARRQ) via `run_sweep_local.py`.
8. [ ] (Recommended) build **Track B** frame cache; re-run sweeps from disk; compare wall-times.
9. [ ] Plot MedleyDB vs HookTheoryMelody per-layer RPA/RCA curves; report agreement/divergence of the
       best-layer region; confirm the winning layer against `LeitmotifDetection` on BotW.

---

## 9. Decisions (resolved)

- **Framing:** 128-MIDI + RPA/RCA (comparable to HookTheory). No whitening/centering/ABTT — that
  machinery is **retrieval-only** (transductive cosine-MAP on pooled per-file embeddings) and has no
  coherent insertion point in a trained frame-level probe; melody metrics stay RPA/RCA.
- **Cache:** Track A (no cache) ships first; Track B frame cache deferred (§6).
- **Split / variance:** **artist-conditional 5-fold CV** keyed by the track-name prefix
  (`split('_')[0]`). The `medleydb` package's artist index is NOT used — it doesn't import on modern
  PyYAML and its metadata lumps `MusicDelta_*` inconsistently anyway; the prefix treats those genre
  demos as one leak-safe group (a known fold concentration, mitigated by 5-fold averaging).
  There is **no fetchable MARBLE split** — the paper's 67/15/26 traces to Wang et al. 2022b, whose
  exact list isn't published — so we use a reproducible, leak-free, honestly-labeled artist-conditional
  CV (not "MARBLE-bit-exact"); `--split-json` pins an exact list if one ever surfaces.

---

## 10. Implementation status (built, audited & validated on Mac)

Built, TDD-tested, real-data validated, then **independently code-reviewed** (4 finder agents) and
fixed. **MPS confirmed working end-to-end** (MERT-95M trains on Apple MPS — sanity + Epoch 0 with
loss/RPA/RCA, no unimplemented-op errors). GPU runtime numbers still need the PC.

**New code (branch `feat/bps-within-piece`):**
- `marble/tasks/MedleyDBMelody/melody_labels.py` — pure f0→MIDI + nearest-sample frame labeller +
  `validate_native_grid` (rejects wrong-hop / dropped-row / non-zero-start CSVs at load).
- `marble/tasks/MedleyDBMelody/split.py` — `fold_split` (artist-conditional 5-fold, leak-free,
  empty-split guarded; artist = track-name prefix).
- `marble/tasks/MedleyDBMelody/datamodule.py` — `MedleyDBMelody{DataModule,Train,Val,Test}` on
  `BaseAudioDataset`, BPS fold pattern (`jsonl_template`+`split`+`fold_idx`, exposes `fold_idx` for
  `LogSweepCoordsCallback`); validates each MELODY2 grid; `get_targets` nearest-sample gather.
- `marble/tasks/MedleyDBMelody/probe.py` — re-exports HookTheory's supervised probe (RPA/RCA).
- `scripts/data/build_medleydb_melody_jsonl.py` — recursive-glob discovery (both annotation
  layouts), medleydb-package artist-conditional **5-fold** output
  `MedleyDBMelody.fold{F}.{split}.jsonl` (+ `--split-json` with disjointness/missing checks,
  `--smoke`); `torchaudio.info` + duplicate-basename guards.
- `configs/probe.{MuQ,MERT-v1-95M,OMARRQ-multifeature-25hz}-{layers,meanall}.MedleyDBMelody.yaml`
  — 6 configs; per-encoder `label_freq`, `out_dim 128`, `cache_embeddings:false`,
  `LogSweepCoordsCallback`, fold_idx data, explicit encoder/FE init_args (HookTheory parity).
- `scripts/sweeps/run_all_sweeps.py` — SweepDefs (fold0 only) + `fold0.train` data marker.
- `scripts/sweeps/run_medleydb_melody_folds.sh` — full 5-fold × layer sweep + per-fold meanall
  (mirrors `run_bps_mnid_abc_folds.sh`; `--run-name-suffix foldF --dir-suffix .foldF`,
  `NUM_WORKERS=0` default for the WSL spawn deadlock).
- Tests (**68 pass** incl. HookTheory regression): `test_medleydb_melody_labels.py` (19),
  `test_medleydb_melody_datamodule.py` (6), `test_medleydb_split.py` (7).

**Audit fixes applied** (from the 4-agent review): medleydb-package split fixes MusicDelta lumping;
empty-split guard; CSV grid validation; `--split-json` disjointness/missing checks; corrupt-audio /
duplicate-basename guards; MERT/OMARRQ comment + init_args parity; dead `_crop_to_min_t` re-export
dropped. Label math (`+0.5` frame-centering) was independently **confirmed correct** (same bucket
grid as HookTheory → curves comparable).

**Local correctness smoke (real 2-track sample):** `--smoke` → fold0 jsonls → 31 clips; full path
(decode + resample 44.1k→24k + f0→MIDI + grid validation) verified, voiced frac 0.64, MIDI 47–81.

**Not yet done (needs the PC / full data):** GPU runtime numbers; Track B frame cache (§6).

---

## 11. PC smoke runbook (WSL + CUDA, RTX 5060 Ti)

Goal: a real single-layer GPU run to **measure per-epoch wall-time**, then extrapolate. Two phases.

### A. Sample smoke (fastest — 2 tracks, ~415 MB, free, no Zenodo needed)
```bash
cd ~/developer/marble
# 1. Sample data (free)
curl -L -o /tmp/mdb_sample.tar.gz \
  "https://zenodo.org/records/1438309/files/MedleyDB_Sample.tar.gz?download=1"
mkdir -p /mnt/d/datasets/medleydb_sample && tar xzf /tmp/mdb_sample.tar.gz -C /mnt/d/datasets/medleydb_sample
# 2. Build smoke JSONLs (every track in train+val+test)
uv run python scripts/data/build_medleydb_melody_jsonl.py \
  --audio-root /mnt/d/datasets/medleydb_sample/MedleyDB_sample/Audio \
  --annotation-root /mnt/d/datasets/medleydb_sample/MedleyDB_sample/Annotations \
  --out-dir data/MedleyDB --smoke
# 3. Single-layer fit+test on MuQ layer 6 — TIMING RUN.
#    NUM_WORKERS=0: the box has deadlocked at worker spawn with num_workers>0
#    (see smoke_one_layer.sh). Bump to 4/6 only once confirmed stable.
WANDB_MODE=offline uv run python scripts/sweeps/run_sweep_local.py \
  --base-config configs/probe.MuQ-layers.MedleyDBMelody.yaml \
  --num-layers 13 --model-tag MuQ --task-tag MedleyDBMelody \
  --layers 6 --no-skip --skip-meanall \
  --num-workers 0 --accelerator gpu --precision bf16-mixed
```
Read **seconds/epoch** off the progress bar (or W&B). Let it run ≥2 epochs, then Ctrl-C — early epochs
are enough for the per-epoch number. Sanity: `val/acc_rpa` should climb above the silent-frame
baseline within a few epochs (it'll overfit fast on 2 tracks — that's fine, this is a timing run).

### B. Extrapolate to the full sweep
- Smoke "train" = 31 clips (2 tracks). **Full train ≈ 67 tracks ≈ ~1,090 clips → ×~35.**
- Full per-layer ≈ `(smoke s/epoch) × 35 × (early-stopped epochs, ~15–25)`.
- MuQ/MERT = 13 layers, OMARRQ = 24. Cross-check against your **live HookTheoryMelody** s/epoch:
  MedleyDB should land at **~1/7** of it (7.3 h vs ~50 h audio).
- If per-layer is painful, that's the cue to build the **Track B frame cache** (§6) — feasible here
  (17–40 GB/encoder) precisely because MedleyDB is small.

### C. Full 5-fold run (once you've staged Zenodo V1/V2 audio + GitHub annotations)
```bash
# Artist grouping = track-name prefix (MusicDelta_* → one leak-safe group). No
# medleydb package needed (it doesn't import on modern PyYAML).
uv run python scripts/data/build_medleydb_melody_jsonl.py \
  --audio-root /mnt/d/datasets/medleydb/Audio \
  --annotation-root /mnt/d/datasets/medleydb/Annotations \
  --out-dir data/MedleyDB    # → MedleyDBMelody.fold{0..4}.{train,val,test}.jsonl
# Full 5-fold × layer sweep for all 3 encoders + per-fold meanall:
NUM_WORKERS=0 bash scripts/sweeps/run_medleydb_melody_folds.sh --accelerator gpu
# (fold0-only quick pass instead: run_all_sweeps.py picks up MedleyDBMelody fold0.)
```
Outputs land at `output/probe.MedleyDBMelody.<M>-layers.layer{N}.fold{F}/`; W&B runs are
`layer-<N>-{fit,test}-fold<F>` (job_type clean `fit`/`test`), grouped `<M> / MedleyDBMelody`, with
`sweep/fold` stamped by `LogSweepCoordsCallback` → average per-layer across folds in the dashboard.
Keep `output/` on ext4 (not `/mnt/d`) — DrvFs breaks checkpoint moves + dataloader AF_UNIX.
Annotations: clone `marl/medleydb` and point `--annotation-root` at `medleydb/medleydb/data/Annotations`
(the recursive glob finds `*_MELODY2.csv` under either layout).
