# Symbolic CLaMP3 A/B methodology — hard-won lessons

A running record of the methodology lessons from the score-native **ABC** vs
lossy **MIDI→MTF** campaign for CLaMP3-symbolic motif tasks (BPS-Motif, JKUPDD,
and the MTC-ANN work to come). These are the traps that produced wrong verdicts
before they were caught — written down so the next A/B does not re-pay for them.

Companion result docs:
[`bps_motif_abc_vs_mtf.md`](bps_motif_abc_vs_mtf.md) (BPS-Motif A/B + bar-
granularity ceiling) and [`jkupdd_abc_vs_mtf.md`](jkupdd_abc_vs_mtf.md) (JKUPDD
content-matched A/B).

---

## (a) The content confound — match content, slice at note level

**Trap.** Comparing two encodings of "the same piece" where one arm embeds a
*whole measure* and the other an *isolated motif*. The arms then differ in
**content**, not encoding — the comparison measures "more notes vs fewer," not
"ABC vs MTF." The first JKUPDD ABC result ("ABC loses −0.30") was exactly this:
an artifact of whole-measure ABC vs isolated-motif MTF, **not** a real ABC
weakness.

**Fix.** Both arms must embed the **identical note set**. Slice the motif at the
**note level** — reconstruct the *same notes* the other arm contains, re-zeroed,
and embed those. Once content-matched, ABC *beats* MTF on JKUPDD and the depth
peak moves deep (L9–L11). The corrected JKUPDD build is the one to trust
(`c930a65`); the "−0.30" number is a confound artifact and must **not** be cited.

## (b) The notation confound — match notation STYLE, not just note count

**Trap.** Even with the *same notes*, the two arms can differ in how those notes
are **notated**. BPS-Motif's MNID negatives are sampled with `rng.uniform` window
boundaries; their clipped note durations land on un-notatable fractions that snap
to awkward **tuplets** in ABC (10–65× the tuplet density of the integer-bounded
positives). The class label then leaks through **notation style** ("tuplet-mess ⇒
non-motif"), not musical content — the probe partly cheats. MTF (MIDI ticks)
tolerates any float, so it is *immune*; its number is the believable one.

**Fix.** A content-parity gate that only checks **note count** is insufficient —
it passed here while the confound was wide open. The parity check must inspect the
**rendered ABC** (tuplet density, integer-boundary fraction, note-head count),
not just `n_notes`. When boundaries are sampled, sample them on the **same grid**
as the positives (integer beats), so positives and negatives share a notation
style. (See the BPS-Motif doc's integer-boundary-negative rebuild TODO.)

## (c) Layer choice is regime-dependent — don't assume one peak

The best layer is **not** a fixed property of CLaMP3; it depends on the task
regime:

| regime | example | layer behaviour |
|---|---|---|
| **cross-piece retrieval** | JKUPDD, MTC | **deep** (L9–L11) wins cleanly |
| **within-piece retrieval** | BPS-Motif | **bimodal** — shallow L2 ≈ deep L11; centering tips it shallow (L2 0.546 > L11 0.534) |
| **supervised probe** | BPS MNID | shallow peaks, but **suspect** — a shallow peak can be a confound artifact, not a representational fact |

**Lesson.** A shallow supervised-probe peak is a *red flag*, not a result: check
whether a notation/content confound is what the shallow layer is picking up before
declaring "shallow is best." Cross-piece tasks are the cleaner place to read the
depth signal.

## (d) Centering helps; whitening is a wash

Per-query mean-subtraction (`map_centered`) **helps** consistently — it removes a
per-query offset that depresses raw MAP, and it disproportionately lifts shallow
layers (it is what flips BPS retrieval's winner from deep L11 to shallow L2).
**Whitening**, by contrast, was a **wash** in the broader eval (no reliable gain;
see the prior eval audit). Default to centering; do not reach for whitening
expecting a lift.

## (e) Topology — Mac = git home, PC = compute; always `git cat-file` to verify

**Mac** (`/Users/sid/Developer/Python/marble`) is the **git home** — commits land
here. **PC** (`ssh my-pc` → WSL `/home/sid/developer/marble`) is **compute** —
sweeps, outputs, embeddings live there. Mac sandboxes Python, so run Python on the
PC and do git on the Mac. Move files PC→Mac via `cat`/`tar` over SSH.

**Never trust a reported commit hash — `git cat-file -t <hash>` it on the machine
you expect it on.** A wrong-machine hash check (looking for a Mac commit on the PC)
caused a **false "fabrication" scare** — the commit existed, just on the other
host. The verification command must name the machine: `git -C <mac-path>
cat-file -t <hash>` must print `commit`. A hash that hasn't been cat-file-verified
on the right machine is not "done."

## (f) wandb convention for the symbolic sweeps

- **Representation** carries the ABC marker: `CLaMP3-symbolic-abc`; the **task**
  name stays clean (`BPSMotifRetrieval`, not `BPSMotifRetrievalABC`). In the run
  `repr` field it reads `CLaMP3-symbolic-abc / <Task>` — ABC lives in the *repr*,
  not the task name.
- Exactly **two job_types**: `fit` and `test` (clean stages). The per-fold suffix
  goes on the run **name** (`--run-name-suffix fold3` → `layer-6-test-fold3`),
  never on `job_type` — so runs still group by `sweep/layer` and filter by
  `sweep/stage=test`. `--extra-tag` (which dirties `job_type` to `test-<tag>`)
  stays reserved for backfill marking only.
- `sweep/{layer,fold,stage,repr}` are stamped at **write time** by
  `LogSweepCoordsCallback`, which resolves `fold_idx` **authoritatively** from the
  datamodule (datasets → split-config `init_args.fold_idx`), so even **fit** runs
  — whose names carry no `foldN` token — get a correct `sweep/fold` (fold 0
  honored via `is not None`). Before this fix those runs landed `sweep/fold=None`
  and were unrecoverable. Shared parser: `marble/utils/sweep_coords.py`.

## (g) Parallelize the builds — and smoke-test serial == parallel byte-identical

The ABC/ceiling builds use `ProcessPoolExecutor` (≈25 min serial → ≈1 min on 14
workers). **Always** prove serial and parallel produce **byte-identical** output
on stdout before trusting the parallel run — micro counts are sum-aggregated so
worker order is irrelevant, but that has to be *verified*, not assumed. Both
`build_bps_motif_abc.py` and `ceiling_mnid_bar_granularity.py` send only the
result JSON to stdout (mode + timing to stderr) precisely so the two modes diff
clean.

## (h) Embedding cache — forward CLaMP3 once, reuse across the sweep

A layer sweep does **not** re-run the encoder per layer. CLaMP3 forwards **once**
with `output_hidden_states`; **all** layer embeddings come from the returned
`hidden_states` tuple and are cached/reused across the 13-layer sweep (and the
`meanall` baseline). An in-RAM LRU memoization kills the per-clip `torch.load` in
the forward path. This is what makes a full 13-layer × 5-fold × 2-task sweep cheap
enough to iterate on.

## (i) TISMIR (Wang, Kuo & Su 2025) — their MNID is NOT our task

Their "motif note identification" is **per-note** and serves as a **preprocessing
step** for a MIREX-scored *discovery* pipeline. **Our** MNID is a **window-level**
binary classification (does this window contain a motif). The granularities differ,
so **our numbers are not comparable to theirs** — do not cite one against the
other. The obstacle to building a *per-note* CLaMP3 version is CLaMP3's
**per-patch (≈ per-bar) granularity**: a per-patch head emits one label per bar,
and (see the BPS-Motif doc's ceiling section) the all-voices per-bar ceiling
(0.612) sits **below** the discovery SOTA (0.721) because 78% of bars mix motif +
accompaniment. Voice-aware patching (per-staff ABC) is the prerequisite for a
competitive per-note CLaMP3 MNID.

---

### One-line summary of each lesson

1. **(a)** Match content; slice the motif at note level (the "−0.30" was a confound).
2. **(b)** Match notation *style*; parity must check rendered ABC, not just note count.
3. **(c)** Layer choice is regime-dependent; a shallow supervised peak is a red flag.
4. **(d)** Centering helps; whitening is a wash.
5. **(e)** Mac = git, PC = compute; always `git cat-file -t` on the right machine.
6. **(f)** wandb: ABC in *repr* not task name; two job_types; `sweep/*` stamped at write time.
7. **(g)** Parallelize builds; prove serial == parallel byte-identical.
8. **(h)** Embedding cache: forward once, all layers from `hidden_states`.
9. **(i)** TISMIR MNID (per-note, discovery-prep) ≠ our window-level MNID; not comparable.
