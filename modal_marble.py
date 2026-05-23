"""
modal_marble.py — MARBLE probe & layer-sweep runner on Modal GPU instances.

Quick start
-----------
# One-time: store your HuggingFace token so gated datasets (Chords1217) work
modal secret create hf-secret HF_TOKEN=hf_...

# Download datasets
modal run modal_marble.py::download                 # GTZAN + Chords1217 (+ Covers80)
modal run modal_marble.py::download_gtzan_only      # just the free ones (no token needed)

# Run layer sweeps
modal run modal_marble.py::sweep_omarrq_chords1217
modal run modal_marble.py::sweep_omarrq_gtzan
modal run modal_marble.py::sweep_clamp3_gtzan

# Or run an arbitrary probe (fit + test)
modal run modal_marble.py::run_probe --config configs/probe.MuQ.Chords1217.yaml

# Browse outputs
modal volume ls marble-output
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

# ──────────────────────────────────────────────
# App + image
# ──────────────────────────────────────────────

APP_NAME = "marble-leitmotif"
WORK_DIR = "/root/marble"
HF_CACHE = f"{WORK_DIR}/data/.hf_cache"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["ffmpeg", "libsndfile1", "git", "curl", "wget"])
    .pip_install(
        [
            # Core ML stack — pinned to match pyproject.toml. Torch 2.7 needed for
            # the cu128 wheels used locally (RTX 5060 Ti = sm_120 / Blackwell);
            # Modal still pulls CUDA 12.x at runtime and 2.7 wheels work cleanly.
            "torch==2.7.0",
            "torchaudio==2.7.0",
            "transformers==4.52.3",
            "lightning==2.5.1",
            # MARBLE deps
            "jsonargparse[signatures]>=4.27.7",
            "albumentations==1.4.4",
            "datasets==3.6.0",
            "peft==0.15.2",
            "einops",
            "requests",
            "librosa",
            "omegaconf",
            "wandb",
            "mir_eval",
            "pretty_midi",
            "torchmetrics",
            "soundfile",
            "huggingface_hub>=0.24.0",
            "hf-xet>=1.5.0",  # HF filesystem helpers
            "accelerate",
            # MARBLE deps added since the last Modal-touching commit (5c2ee41):
            "mido",  # CLaMP3 MIDI→MTF tokenisation (SuperMarioStructure path)
            "yt-dlp",  # HookTheoryMelody + HXMSA audio recovery
            "numpy>=1.19",
            "scipy>=1.5",
            "matplotlib",  # scripts/analysis/*
            "seaborn",  # scripts/analysis/*
            "numba",  # GTZANBeatTracking HMM
            "patchright>=1.59.1",  # NinSheetMusic MIDI scrape fallback (SuperMario)
            # OMAR-RQ
            "git+https://github.com/MTG/omar-rq.git",
        ]
    )
    .add_local_dir(
        local_path=".",
        remote_path=WORK_DIR,
        copy=True,  # bake into image layer (not runtime mount)
        ignore=[
            "data/**",
            "output/**",
            ".git/**",
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pth",
            "memory/**",
            ".venv/**",
        ],
    )
    .run_commands([f"pip install -e {WORK_DIR} --no-deps --quiet"])
    .env(
        {
            "PYTHONPATH": WORK_DIR,
            "HF_HOME": HF_CACHE,
            "HF_DATASETS_CACHE": HF_CACHE,
            "WANDB_MODE": "offline",  # logs saved locally; sync with: wandb sync output/
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

# ──────────────────────────────────────────────
# Persistent storage
# ──────────────────────────────────────────────

data_vol = modal.Volume.from_name("marble-data", create_if_missing=True)
output_vol = modal.Volume.from_name("marble-output", create_if_missing=True)

VOL = {
    f"{WORK_DIR}/data": data_vol,
    f"{WORK_DIR}/output": output_vol,
}

# ──────────────────────────────────────────────
# GPU selection
# ──────────────────────────────────────────────
#
# Modal evaluates `@app.function(gpu=...)` at app-definition time, which runs
# locally before submitting to Modal. So an env var read here works as a
# per-invocation knob:
#
#     MARBLE_GPU=A100-40GB modal run modal_marble.py::run_probe --config ...
#     MARBLE_GPU=H100      modal run modal_marble.py::run_sweep  ...
#     MARBLE_GPU=L4        modal run modal_marble.py::run_probe ...   # default per fn
#
# Accepted GPU strings are anything Modal recognises:
#   T4, L4, A10G, A100-40GB, A100-80GB, H100, H200, B200, ...
# See https://modal.com/docs/guide/gpu for the current list + pricing.
#
# Per-function default: short, cheap GPUs (A10G / L4). Override per run via the
# env var. Functions read this once at module load — relaunching the Modal app
# with a different MARBLE_GPU value applies it to all subsequent containers.
PROBE_GPU = os.environ.get("MARBLE_GPU", "A10G")
SWEEP_GPU = os.environ.get("MARBLE_GPU", "A10G")
PARALLEL_GPU = os.environ.get("MARBLE_GPU", "L4")

# ──────────────────────────────────────────────
# Helpers shared across functions
# ──────────────────────────────────────────────


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a subprocess, stream stdout/stderr, raise on error."""
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def _chdir():
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)


def _gen_sweep_configs(base_config: str, num_layers: int, model_tag: str, task_tag: str) -> str:
    """Generate per-layer YAML configs and return the sweep dir path."""
    sweep_dir = f"configs/sweeps/{model_tag}.{task_tag}"
    _run(
        [
            "python",
            "scripts/sweeps/gen_sweep_configs.py",
            "--base-config",
            base_config,
            "--num-layers",
            str(num_layers),
            "--model-tag",
            model_tag,
            "--task-tag",
            task_tag,
            "--out-dir",
            sweep_dir,
        ]
    )
    return sweep_dir


def _has_test_metrics(summary_path: Path) -> bool:
    """True if a WandB summary file contains at least one `test/...` key.
    Mirrors scripts/sweeps/run_sweep_local.py:80 — the only reliable completion
    signal for both supervised and zero-shot sweeps."""
    import json

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(k.startswith("test/") for k in data)


def _layer_done(task_tag: str, model_tag: str, layer: int) -> bool:
    """Port of scripts/sweeps/run_sweep_local.py:97 — checks output/.../wandb/run-*/
    files/wandb-summary.json for `test/*` keys. Used for resume-skip both
    locally and inside Modal containers (volume mounts the same path)."""
    patterns = [
        f"*{model_tag}*{task_tag}*layer{layer}",
        f"*{task_tag}*{model_tag}*layer{layer}",
    ]
    for pat in patterns:
        for d in Path("output").glob(pat):
            if not d.is_dir():
                continue
            for summary in d.glob("wandb/run-*/files/wandb-summary.json"):
                if _has_test_metrics(summary):
                    return True
    return False


# ──────────────────────────────────────────────
# Checkpoint resume + cleanup (ports of run_sweep_local.py helpers)
# ──────────────────────────────────────────────


def _checkpoint_dirpath_from_config(cfg_path: str | Path) -> Path | None:
    """Parse the YAML and return the ModelCheckpoint.dirpath, or None.

    Shared by ``_resume_args_for_config`` (looks for ``last.ckpt`` here)
    and ``_delete_last_ckpt`` (deletes ``last.ckpt`` from here after a
    successful fit). Single source of truth for "where this config's
    checkpoints live."
    """
    import yaml

    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    callbacks = (cfg or {}).get("trainer", {}).get("callbacks", []) or []
    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        if "ModelCheckpoint" not in cb.get("class_path", ""):
            continue
        dirpath = (cb.get("init_args") or {}).get("dirpath")
        if dirpath:
            return Path(dirpath)
    return None


def _resume_args_for_config(cfg_path: str | Path) -> list[str]:
    """Return ``["--ckpt_path", "<path>"]`` if a non-empty ``last.ckpt`` exists
    for this config's ModelCheckpoint dirpath, otherwise an empty list.

    Lightning's ``ModelCheckpoint(save_last=True)`` writes ``last.ckpt``
    after every epoch. Passing ``--ckpt_path <last.ckpt>`` to ``cli.py fit``
    restores epoch + optimizer + LR scheduler + RNG state — equivalent to
    resuming where the previous run was killed (e.g. by a Modal job
    timeout). Critical for sweeps on time-bounded containers.
    """
    dirpath = _checkpoint_dirpath_from_config(cfg_path)
    if dirpath is None:
        return []
    last_ckpt = dirpath / "last.ckpt"
    if last_ckpt.exists() and last_ckpt.stat().st_size > 1000:
        return ["--ckpt_path", str(last_ckpt)]
    return []


def _delete_last_ckpt(cfg_path: str | Path) -> None:
    """Delete ``last.ckpt`` from this config's ModelCheckpoint dirpath.

    Called after a successful fit: ``last.ckpt`` was the resume pointer
    but is now obsolete (the run completed normally; test loads
    ``best.ckpt``). On Modal sweeps each layer's ``last.ckpt`` lives on
    the marble-output volume — without cleanup a 24-layer OMAR-RQ sweep
    leaves ~24 GB of dead weight per run. Idempotent + best-effort.
    """
    dirpath = _checkpoint_dirpath_from_config(cfg_path)
    if dirpath is None:
        return
    last_ckpt = dirpath / "last.ckpt"
    if not last_ckpt.exists():
        return
    try:
        size_mb = last_ckpt.stat().st_size / (1024 * 1024)
        last_ckpt.unlink()
        # Defensive: Lightning may write last-v1.ckpt, last-v2.ckpt if
        # multiple ModelCheckpoint callbacks contend for the same dir.
        for stale in dirpath.glob("last-v*.ckpt"):
            stale.unlink(missing_ok=True)
        print(f"  cleanup: removed {last_ckpt} ({size_mb:.0f} MB)")
    except OSError as e:
        print(f"  cleanup: could not remove {last_ckpt}: {e}", file=sys.stderr)


# ──────────────────────────────────────────────
# Meanall (mean-of-all-layers) baseline (port of run_sweep_local.py helpers)
# ──────────────────────────────────────────────


def _meanall_config_for(base_config: str) -> Path | None:
    """Find the meanall sibling config for a per-layer base config.

    Pattern: ``probe.<encoder>(-layers)?.<task>.yaml`` →
             ``probe.<encoder>-meanall.<task>.yaml``
    Returns the path if it exists on disk, else None.
    """
    p = Path(base_config)
    parts = p.name.split(".")
    # Expected: ['probe', '<encoder>(-layers)?', '<task>', 'yaml']
    if len(parts) != 4 or parts[0] != "probe" or parts[-1] != "yaml":
        return None
    encoder = parts[1].removesuffix("-layers")
    task = parts[2]
    candidate = p.with_name(f"probe.{encoder}-meanall.{task}.yaml")
    return candidate if candidate.exists() else None


def _meanall_done(task_tag: str, model_tag: str) -> bool:
    """Mirror of ``_layer_done`` for the meanall run."""
    patterns = [
        f"*{model_tag}-meanall*{task_tag}*",
        f"*{task_tag}*{model_tag}-meanall*",
        f"*{model_tag}*{task_tag}*-meanall*",
        f"*{task_tag}*{model_tag}*-meanall*",
    ]
    for pat in patterns:
        for d in Path("output").glob(pat):
            if not d.is_dir():
                continue
            for summary in d.glob("wandb/run-*/files/wandb-summary.json"):
                if _has_test_metrics(summary):
                    return True
    return False


def _run_meanall_first(
    base_config: str,
    model_tag: str,
    task_tag: str,
    skip_if_done: bool = True,
    continue_on_failure: bool = False,
) -> bool:
    """Run the mean-of-all-layers baseline before the per-layer sweep.

    Returns True if meanall ran (or was correctly skipped); False if it
    failed and ``continue_on_failure`` is True (caller decides what to do
    with that). Raises subprocess.CalledProcessError if meanall fails and
    continue_on_failure is False — meanall failure usually means every
    per-layer job will hit the same error.
    """
    cfg = _meanall_config_for(base_config)
    if cfg is None:
        print(
            f"  ! No meanall sibling for {base_config} "
            f"(expected probe.<encoder>-meanall.<task>.yaml). Skipping."
        )
        return True
    if skip_if_done and _meanall_done(task_tag, model_tag):
        print("  ✓ meanall already complete — skipping.")
        return True

    print(
        f"\n{'=' * 60}\n meanall (mean-of-all-layers baseline)  "
        f"[{model_tag} | {task_tag}]\n{'=' * 60}",
        flush=True,
    )
    for stage in ("fit", "test"):
        cmd = [
            "python",
            "cli.py",
            stage,
            "-c",
            str(cfg),
            f"--trainer.logger.init_args.name=layer-meanall-{stage}",
            f"--trainer.logger.init_args.job_type={stage}",
        ]
        if stage == "fit":
            cmd += _resume_args_for_config(cfg)
        try:
            _run(cmd)
        except subprocess.CalledProcessError as e:
            if continue_on_failure:
                print(
                    f"  ⚠ meanall {stage} failed (exit {e.returncode}); "
                    f"continue_on_failure=True → proceeding with per-layer.",
                    file=sys.stderr,
                )
                return False
            raise
        if stage == "fit":
            _delete_last_ckpt(cfg)
    return True


# ──────────────────────────────────────────────
# Dataset downloads
# ──────────────────────────────────────────────


@app.function(
    image=image,
    volumes=VOL,
    timeout=3 * 3600,  # 3 h for large downloads
    secrets=[modal.Secret.from_name("huggingface")],
)
def _download_marble_datasets(datasets: list[str]):
    """Download MARBLE datasets from HuggingFace m-a-p/<name>."""
    from huggingface_hub import snapshot_download

    _chdir()
    hf_token = os.environ.get("HF_TOKEN")

    for ds in datasets:
        target = f"{WORK_DIR}/data/{ds}"
        os.makedirs(target, exist_ok=True)
        print(f"\n── Downloading m-a-p/{ds} → {target}")
        snapshot_download(
            repo_id=f"m-a-p/{ds}",
            repo_type="dataset",
            local_dir=target,
            token=hf_token,
            local_dir_use_symlinks=False,
        )
        # HookTheory requires post-download extraction
        if ds == "HookTheory":
            script = Path(target) / "extract.sh"
            if script.exists():
                _run(["bash", str(script)])
        print(f"  ✓ {ds}")

    data_vol.commit()
    print("\nAll downloads committed to volume.")


@app.function(
    image=image,
    volumes=VOL,
    timeout=1800,
)
def _download_covers80():
    """
    Download Covers80 (80 works × 2 versions each) and write JSONL files.
    Audio → /data/Covers80/   JSONL → train / val / test splits.
    """
    import json
    import random
    import tarfile
    import urllib.request

    import torchaudio

    dest = Path(f"{WORK_DIR}/data/Covers80")
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "covers80.tar.bz2"

    url = "http://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz"
    print(f"Downloading {url} …")
    try:
        urllib.request.urlretrieve(url, archive)
    except Exception as e:
        raise RuntimeError(
            f"Covers80 download failed: {e}\n"
            "Place covers80.tar.bz2 manually in /data/Covers80/ and re-run."
        ) from e

    print("Extracting …")
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(dest)

    # Locate list1 / list2 under whatever sub-directory the archive created
    list1 = next(dest.glob("**/list1"), None) or next(dest.glob("**/covers1"), None)
    list2 = next(dest.glob("**/list2"), None) or next(dest.glob("**/covers2"), None)
    if not list1 or not list2:
        raise RuntimeError(f"Unexpected archive layout. Found: {list(dest.iterdir())}")

    def audio_files(d: Path):
        return sorted(list(d.glob("**/*.mp3")) + list(d.glob("**/*.wav")))

    def song_id(p: Path) -> str:
        return p.stem.lower().replace(" ", "_")

    def make_records(files: list[Path]) -> list[dict]:
        recs = []
        for p in files:
            try:
                info = torchaudio.info(str(p))
                recs.append(
                    {
                        "audio_path": str(p),
                        "label": song_id(p),
                        "sample_rate": info.sample_rate,
                        "num_samples": info.num_frames,
                    }
                )
            except Exception as e:
                print(f"  ⚠ skip {p.name}: {e}")
        return recs

    list1_recs = make_records(audio_files(list1))
    list2_recs = make_records(audio_files(list2))
    print(f"list1: {len(list1_recs)} files, list2: {len(list2_recs)} files")

    random.seed(42)
    random.shuffle(list1_recs)
    split_n = max(1, int(len(list1_recs) * 0.2))
    val_recs = list1_recs[:split_n]
    train_recs = list1_recs[split_n:]

    def write_jsonl(recs, path):
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  {Path(path).name}: {len(recs)} records")

    write_jsonl(train_recs, dest / "train.jsonl")
    write_jsonl(val_recs, dest / "val.jsonl")
    write_jsonl(list2_recs, dest / "test.jsonl")

    # Write sorted label list for pasting into YAML configs
    all_labels = sorted({r["label"] for r in list1_recs + list2_recs})
    (dest / "labels.json").write_text(json.dumps(all_labels, indent=2))
    print(f"\n{len(all_labels)} unique works (classes). Labels saved to Covers80/labels.json")

    data_vol.commit()
    return all_labels


# ──────────────────────────────────────────────
# SHS100K JSONL setup (run once before SHS100K sweeps)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    volumes=VOL,
    timeout=2 * 60 * 60,  # 2 h: fast at default settings (~30 s), generous if
    # use_torchaudio=True is opted in (~2 h for 7k files).
)
def setup_shs100k_jsonl(use_torchaudio: bool = False):
    """One-time setup: rewrite SHS100K.test.jsonl on the marble-data volume
    to point at the Modal-mounted audio dir, dropping entries whose audio
    file is missing.

    Default check: existence + size (sub-minute on 7k files).
    Opt-in: use_torchaudio=True also verifies each file decodes (~2 h,
    only worth it if you suspect upload corruption).

    Prerequisite (one-time, from your laptop):
        modal volume put marble-data \\
            data/SHS100K/SHS100K.test.jsonl \\
            SHS100K/SHS100K.test.jsonl

    After this runs, the JSONL on the volume references
    `data/SHS100K/audio/<ytid>.m4a` and only includes entries whose audio
    file exists on the volume.
    """
    _chdir()
    data_vol.reload()

    jsonl = f"{WORK_DIR}/data/SHS100K/SHS100K.test.jsonl"
    audio_dir = f"{WORK_DIR}/data/SHS100K/audio"

    if not os.path.exists(jsonl):
        raise FileNotFoundError(
            f"{jsonl} not found on marble-data volume.\n"
            f"  Upload it first:\n"
            f"    modal volume put marble-data "
            f"data/SHS100K/SHS100K.test.jsonl SHS100K/SHS100K.test.jsonl"
        )

    cmd = [
        "python",
        "scripts/verify/verify_shs100k.py",
        "--jsonl",
        jsonl,
        "--audio-dir",
        audio_dir,
        "--rewrite",
    ]
    if use_torchaudio:
        cmd.append("--torchaudio")
    _run(cmd)
    data_vol.commit()
    print("SHS100K JSONL rebuilt and committed to marble-data:/SHS100K/")


# ──────────────────────────────────────────────
# HookTheory full setup (for HookTheoryMelody)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    volumes=VOL,
    timeout=8 * 60 * 60,  # 8 h — 104 GB download + extract + JSONL build
    secrets=[modal.Secret.from_name("huggingface")],
)
def setup_hooktheory_full():
    """One-time setup for HookTheoryMelody:
      1. Downloads m-a-p/HookTheory complete (clips + 104 GB full audio).
      2. Extracts the audio tars into data/HookTheory/audio/<ytid>.mp3.
      3. Builds data/HookTheory/HookTheory.{train,val,test}.jsonl from the
         raw annotation tree, filtered to entries whose audio exists.

    Idempotent: huggingface_hub.snapshot_download skips already-downloaded
    files; the JSONL build always rewrites the three files.

    Disk needed on marble-data: ~110 GB after extraction.
    """
    _chdir()
    data_vol.reload()
    sys.path.insert(0, WORK_DIR)
    from download import download_dataset

    download_dataset("HookTheory", f"{WORK_DIR}/data", with_full_audio=True)
    data_vol.commit()  # flush the heavy bits before the JSONL build

    # Build the Melody JSONL using the volume's audio dir for the filter.
    audio_dir = f"{WORK_DIR}/data/HookTheory/audio"
    out_dir = f"{WORK_DIR}/data/HookTheory"
    _run(
        [
            "python",
            "scripts/data/build_hooktheory_melody_jsonl.py",
            "--out-dir",
            out_dir,
            "--audio-dir",
            audio_dir,
            "--filter-by-audio",
            # Land the HF cache on the data volume to survive container restarts
            "--hf-cache-dir",
            f"{WORK_DIR}/data/.hf_cache",
        ]
    )
    data_vol.commit()
    print("HookTheoryMelody ready on marble-data:/HookTheory/")


# ──────────────────────────────────────────────
# HXMSA setup (Harmonix Set MSA — yt-dlp + slice)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    volumes=VOL,
    timeout=8 * 60 * 60,  # 8 h — yt-dlp's 2–5 s rate limits dominate
)
def setup_hxmsa(
    max_tracks: int | None = None,
    skip_download: bool = False,
    skip_slice: bool = False,
):
    """Build the HXMSA (Harmonix Set MSA) dataset on the marble-data volume.

    Pipeline (mirrors scripts/data/build_hxmsa_dataset.py):
      1. Clone harmonixset annotation repo into data/HXMSA/harmonixset
      2. yt-dlp full tracks → data/HXMSA/audio_full/<track_id>.flac
      3. ffmpeg slice → data/HXMSA/audio/<track_id>__<seg_id>.flac
      4. Build HXMSA.{train,val,test}.jsonl

    Prerequisite (one-time, upload YouTube cookies to the volume so yt-dlp
    succeeds on age-gated tracks):
        modal volume put marble-data cookies.txt cookies.txt

    Defaults to the full ~912-track build (~5.5 GB, 3–6 h). Pass
    ``max_tracks=N`` for a pilot.
    """
    _chdir()
    data_vol.reload()

    out_dir = f"{WORK_DIR}/data/HXMSA"
    cookies = f"{WORK_DIR}/cookies.txt"

    cmd = [
        "python",
        "scripts/data/build_hxmsa_dataset.py",
        "--out-dir",
        out_dir,
    ]
    if os.path.exists(cookies):
        cmd += ["--cookies-file", cookies]
    else:
        print(
            "  ! cookies.txt not found at marble-data root — yt-dlp will "
            "attempt anonymous download; expect some age-gated failures.\n"
            "  Upload cookies first to recover them:\n"
            "    python scripts/data/export_youtube_cookies.py --browser firefox\n"
            "    modal volume put marble-data cookies.txt cookies.txt",
            file=sys.stderr,
        )
    if max_tracks is not None:
        cmd += ["--max-tracks", str(max_tracks)]
    if skip_download:
        cmd.append("--skip-download")
    if skip_slice:
        cmd.append("--skip-slice")

    _run(cmd)
    data_vol.commit()
    print("HXMSA ready on marble-data:/HXMSA/")


# ──────────────────────────────────────────────
# SuperMarioStructure setup (user-uploaded MIDIs + audio)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    volumes=VOL,
    timeout=2 * 60 * 60,  # 2 h — symbolic-only ~30s; audio slicing dominates
)
def setup_supermario_structure(
    skip_midi_download: bool = True,
    skip_midi_slice: bool = False,
    skip_slice: bool = False,
    max_pieces: int | None = None,
):
    """Build the SuperMarioStructure dataset on the marble-data volume.

    NinSheetMusic blocks scrapers (HTTP 403 regardless of headers), so this
    relies on user-uploaded MIDIs (and optionally user-uploaded audio).

    Prerequisites (one-time uploads from your laptop):
        # NSM MIDIs you obtained manually (or via `ohsheet`):
        modal volume put marble-data path/to/your/midis SuperMarioStructure/midi_source
        # Optional: user-supplied audio aligned with the MIDIs:
        modal volume put marble-data path/to/your/audio SuperMarioStructure/audio_source

    Then run this. Pipeline (mirrors scripts/data/build_supermario_dataset.py):
      1. Clone supermario-structure-annotation repo → data/SuperMarioStructure/annotations
      2. Slice symbolic MIDIs from user uploads → per-segment MIDIs
      3. (Optional) Slice user audio via bar→time mapping → per-segment FLACs
      4. Build SuperMarioStructure.{train,val,test}.jsonl

    Defaults: skip the broken auto-MIDI-download (skip_midi_download=True),
    do MIDI slicing, do audio slicing if audio_source exists.
    """
    _chdir()
    data_vol.reload()

    out_dir = f"{WORK_DIR}/data/SuperMarioStructure"
    midi_source = f"{out_dir}/midi_source"
    audio_source = f"{out_dir}/audio_source"

    if not os.path.exists(midi_source):
        raise FileNotFoundError(
            f"{midi_source} not found on marble-data volume.\n"
            f"  Upload NSM MIDIs first:\n"
            f"    modal volume put marble-data <local_midi_dir> "
            f"SuperMarioStructure/midi_source"
        )

    cmd = [
        "python",
        "scripts/data/build_supermario_dataset.py",
        "--out-dir",
        out_dir,
        "--midi-source-dir",
        midi_source,
    ]
    if os.path.exists(audio_source):
        cmd += ["--audio-dir", audio_source]
    else:
        print(
            f"  ! {audio_source} not found — symbolic-only build (audio "
            f"slicing will be skipped). Upload audio to enable both paths.",
            file=sys.stderr,
        )
    if skip_midi_download:
        cmd.append("--skip-midi-download")
    if skip_midi_slice:
        cmd.append("--skip-midi-slice")
    if skip_slice:
        cmd.append("--skip-slice")
    if max_pieces is not None:
        cmd += ["--max-pieces", str(max_pieces)]

    _run(cmd)
    data_vol.commit()
    print("SuperMarioStructure ready on marble-data:/SuperMarioStructure/")


# ──────────────────────────────────────────────
# BPS-Motif setup (Beethoven sonata leitmotif — symbolic)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    volumes=VOL,
    timeout=15 * 60,  # symbolic-only build is fast (~30 s for 32 movements)
)
def setup_bps_motif(max_movements: int | None = None):
    """Build the BPS-Motif dataset on the marble-data volume.

    Pipeline (mirrors scripts/data/build_bps_motif_dataset.py):
      1. Clone Wiilly07/Beethoven_motif into data/BPS-Motif/_upstream/.
      2. Synthesise full-movement MIDIs at 60 QPM from csv_notes/.
      3. Slice per-occurrence + sampled-negative window MIDIs.
      4. Emit 5-fold CV JSONLs (BPSMotif.{MNID,Retrieval}.fold{0..4}.{train,val,test}.jsonl).

    Symbolic-only — no audio rendering. ~52 MB on disk after build. The
    audio variant (probing MERT/MuQ/OMARRQ on real Beethoven recordings)
    is a separate, deferred follow-up.
    """
    _chdir()
    data_vol.reload()

    out_dir = f"{WORK_DIR}/data/BPS-Motif"
    cmd = ["python", "scripts/data/build_bps_motif_dataset.py", "--out-dir", out_dir]
    if max_movements is not None:
        cmd += ["--max-movements", str(max_movements)]
    _run(cmd)
    data_vol.commit()
    print("BPS-Motif ready on marble-data:/BPS-Motif/")


# ──────────────────────────────────────────────
# Core probe runner
# ──────────────────────────────────────────────


@app.function(
    image=image,
    gpu=PROBE_GPU,
    volumes=VOL,
    timeout=4 * 3600,  # 4 h max per run
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_probe(config: str, skip_if_done: bool = True):
    """
    Fit + test a single MARBLE probe from a YAML config path (relative to WORK_DIR).
    Returns the test output as a string.

    Set WANDB_API_KEY via: modal secret create wandb-secret WANDB_API_KEY=<key>
    """
    _chdir()

    # Optionally wire up WandB online mode
    if os.environ.get("WANDB_API_KEY"):
        os.environ.pop("WANDB_MODE", None)  # unset offline flag

    # Quick skip if checkpoint already exists
    if skip_if_done:
        cfg_tag = Path(config).stem
        ckpt_dir = Path(f"output/{cfg_tag}/checkpoints")
        if (ckpt_dir / "best.ckpt").exists():
            print(f"Checkpoint found at {ckpt_dir}/best.ckpt — skipping fit.")
        else:
            _run(["python", "cli.py", "fit", "-c", config])
    else:
        _run(["python", "cli.py", "fit", "-c", config])

    result = subprocess.run(
        ["python", "cli.py", "test", "-c", config],
        capture_output=True,
        text=True,
        cwd=WORK_DIR,
    )
    output_vol.commit()
    print(result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        result.check_returncode()
    return result.stdout


# ──────────────────────────────────────────────
# Layer sweep runner  (sequential on a single GPU)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    gpu=SWEEP_GPU,
    volumes=VOL,
    timeout=24 * 3600,  # up to 24 h for a full 12-layer sweep
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_sweep(
    base_config: str,
    num_layers: int,
    model_tag: str,
    task_tag: str,
    layers: list[int] | None = None,  # None → all layers
    skip_meanall: bool = False,
    continue_on_meanall_failure: bool = False,
):
    """
    Generate per-layer YAML configs and run fit+test for each layer sequentially.

    Mirrors scripts/sweeps/run_sweep_local.py's flow:
      1. Run the meanall baseline first (if a probe.<enc>-meanall.<task>.yaml
         sibling exists). Aborts the sweep on meanall failure unless
         continue_on_meanall_failure=True — meanall shares the encoder /
         dataloader / GPU init with every per-layer job, so a meanall
         failure typically dooms the whole sweep.
      2. For each layer: resume fit from last.ckpt if present (interrupted
         Modal containers continue from their last epoch); delete last.ckpt
         after fit succeeds so it doesn't accumulate on marble-output.
      3. Commit marble-output after each layer.
    """
    _chdir()
    # Reload so we see prior layers' completion markers + last.ckpts written
    # by sibling containers / PC runs into the same volume.
    output_vol.reload()

    if os.environ.get("WANDB_API_KEY"):
        os.environ.pop("WANDB_MODE", None)

    if not skip_meanall:
        _run_meanall_first(
            base_config,
            model_tag,
            task_tag,
            continue_on_failure=continue_on_meanall_failure,
        )
        output_vol.commit()

    sweep_dir = _gen_sweep_configs(base_config, num_layers, model_tag, task_tag)

    run_layers = layers if layers is not None else list(range(num_layers))
    results = {}

    for layer in run_layers:
        cfg = f"{sweep_dir}/sweep.{model_tag}.{task_tag}.layer{layer}.yaml"
        print(f"\n{'=' * 60}")
        print(f" Layer {layer}/{num_layers - 1}  [{model_tag} | {task_tag}]")
        print(f"{'=' * 60}")

        # WandB naming overrides (same convention as run_sweep_local.py)
        fit_overrides = [
            f"--trainer.logger.init_args.name=layer-{layer}-fit",
            "--trainer.logger.init_args.job_type=fit",
        ]
        test_overrides = [
            f"--trainer.logger.init_args.name=layer-{layer}-test",
            "--trainer.logger.init_args.job_type=test",
        ]

        # fit — resume from last.ckpt if present (e.g. previous container timed out)
        fit_cmd = ["python", "cli.py", "fit", "-c", cfg, *fit_overrides]
        fit_cmd += _resume_args_for_config(cfg)
        _run(fit_cmd)
        # Fit succeeded → drop last.ckpt; best.ckpt remains for the test stage.
        _delete_last_ckpt(cfg)

        # test
        res = subprocess.run(
            ["python", "cli.py", "test", "-c", cfg, *test_overrides],
            capture_output=True,
            text=True,
            cwd=WORK_DIR,
        )
        print(res.stdout)
        if res.returncode != 0:
            print("STDERR:", res.stderr, file=sys.stderr)
        results[layer] = res.stdout

        output_vol.commit()  # persist after each layer so we don't lose work on timeout

    print("\n=== Sweep complete ===")
    for layer, out in results.items():
        acc_line = next((l for l in out.splitlines() if "acc" in l.lower()), "")
        print(f"  layer {layer:2d}: {acc_line.strip()}")

    return results


# ──────────────────────────────────────────────
# Layer sweep runner  (parallel — one container per layer)
# ──────────────────────────────────────────────


@app.function(
    image=image,
    # L4 default (Ada, 24 GB, bf16) — best price/perf vs A10G, ~30 % cheaper
    # at ~1.05× speed. Override via MARBLE_GPU=A100-40GB / H100 / etc.
    gpu=PARALLEL_GPU,
    volumes=VOL,
    timeout=4 * 3600,  # 4 h cap per layer
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb-secret"),
    ],
)
def run_one_layer(
    base_config: str,
    model_tag: str,
    task_tag: str,
    num_layers: int,
    layer: int,
    retest_only: bool = False,
) -> dict:
    """Run fit+test for a single layer of a sweep — designed to be invoked
    via `.starmap()` N times in parallel by `run_parallel_sweep`.

    Resume-safe: checks the wandb-summary marker on `marble-output` volume
    and skips if `test/*` keys are already present.
    """
    _chdir()

    if os.environ.get("WANDB_API_KEY"):
        os.environ.pop("WANDB_MODE", None)

    # Reload volume so we see completion markers committed by parallel siblings
    # or by prior sequential PC runs syncing to the same volume.
    output_vol.reload()

    sweep_dir = _gen_sweep_configs(base_config, num_layers, model_tag, task_tag)
    cfg = f"{sweep_dir}/sweep.{model_tag}.{task_tag}.layer{layer}.yaml"

    if not retest_only and _layer_done(task_tag, model_tag, layer):
        print(f"Layer {layer}: already done — skipping.")
        return {"layer": layer, "status": "skipped"}

    print(f"\n{'=' * 60}\n Layer {layer}/{num_layers - 1}  [{model_tag} | {task_tag}]\n{'=' * 60}")

    # Match the WandB naming convention used by run_sweep_local.py so fit and
    # test land as distinct, clearly-labelled runs (group/name/tags/job_type).
    name_overrides_fit = [
        f"--trainer.logger.init_args.name=layer-{layer}-fit",
        "--trainer.logger.init_args.job_type=fit",
    ]
    name_overrides_test = [
        f"--trainer.logger.init_args.name=layer-{layer}-test",
        "--trainer.logger.init_args.job_type=test",
    ]

    if not retest_only:
        # Resume from last.ckpt if a previous container died mid-fit before
        # writing the wandb-summary completion marker.
        fit_cmd = ["python", "cli.py", "fit", "-c", cfg, *name_overrides_fit]
        fit_cmd += _resume_args_for_config(cfg)
        _run(fit_cmd)
        _delete_last_ckpt(cfg)

    res = subprocess.run(
        ["python", "cli.py", "test", "-c", cfg, *name_overrides_test],
        capture_output=True,
        text=True,
        cwd=WORK_DIR,
    )
    print(res.stdout)
    if res.returncode != 0:
        print("STDERR:", res.stderr, file=sys.stderr)
        output_vol.commit()
        res.check_returncode()

    output_vol.commit()
    return {"layer": layer, "status": "completed", "stdout_tail": res.stdout[-2000:]}


@app.function(
    image=image,
    volumes=VOL,
    timeout=24 * 3600,
)
def run_parallel_sweep(
    base_config: str,
    num_layers: int,
    model_tag: str,
    task_tag: str,
    layers: list[int] | None = None,
    retest_only: bool = False,
) -> list[dict]:
    """Spawn N parallel `run_one_layer` containers via `.starmap()`.

    Same wall-clock as a single layer (≈ 1-2 h for a probe sweep) instead of
    the sequential N × that. Cost is identical — N GPUs × 1 hr = 1 GPU × N hr.
    """
    targets = layers if layers is not None else list(range(num_layers))
    print(f"Spawning {len(targets)} parallel layer jobs for {model_tag} × {task_tag} ...")

    args = [(base_config, model_tag, task_tag, num_layers, lyr, retest_only) for lyr in targets]
    results = list(run_one_layer.starmap(args))

    print("\n=== Parallel sweep complete ===")
    for r in results:
        print(f"  layer {r['layer']:2d}: {r['status']}")
    return results


# ──────────────────────────────────────────────
# Named sweep entry points (convenience)
# ──────────────────────────────────────────────


@app.local_entrypoint()
def download():
    """Download Chords1217 (gated, needs hf-secret) + GTZAN + GS + EMO + Covers80."""
    print("Downloading GTZAN (free) …")
    _download_marble_datasets.remote(datasets=["GTZAN"])

    print("\nDownloading Chords1217 (gated — requires hf-secret with accepted terms) …")
    _download_marble_datasets.remote(datasets=["Chords1217"])

    print("\nDownloading GiantSteps key (GS) and EmoMusic (EMO) …")
    _download_marble_datasets.remote(datasets=["GS", "EMO"])

    print("\nDownloading Covers80 …")
    labels = _download_covers80.remote()
    print(f"Covers80 ready. {len(labels)} works.")


@app.local_entrypoint()
def download_gs_emo():
    """Download only GiantSteps key (GS) and EmoMusic (EMO) — no token needed."""
    _download_marble_datasets.remote(datasets=["GS", "EMO"])


@app.local_entrypoint()
def download_gtzan_only():
    """Download only the free, non-gated datasets (no HF token needed)."""
    _download_marble_datasets.remote(datasets=["GTZAN", "EMO", "MTG", "MTT", "GS"])


@app.local_entrypoint()
def sweep_omarrq_chords1217():
    """OMAR-RQ multifeature-25hz full layer sweep on Chords1217 (24 layers, depth=24)."""
    run_sweep.remote(
        base_config="configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml",
        num_layers=24,
        model_tag="OMARRQ-multifeature25hz",
        task_tag="Chords1217",
    )


@app.local_entrypoint()
def sweep_omarrq_gtzan():
    """OMAR-RQ multifeature-25hz full layer sweep on GTZANGenre (24 layers, depth=24)."""
    run_sweep.remote(
        base_config="configs/probe.OMARRQ-multifeature25hz.GTZANGenre.yaml",
        num_layers=24,
        model_tag="OMARRQ-multifeature25hz",
        task_tag="GTZANGenre",
    )


@app.local_entrypoint()
def sweep_clamp3_gtzan():
    """CLaMP3 BERT audio layer sweep on GTZANGenre (13 layers, 0-12)."""
    run_sweep.remote(
        base_config="configs/probe.CLaMP3-layers.GTZANGenre.yaml",
        num_layers=13,
        model_tag="CLaMP3",
        task_tag="GTZANGenre",
    )


@app.local_entrypoint()
def sweep_omarrq_beat():
    """OMAR-RQ multifeature-25hz full layer sweep on GTZANBeatTracking (24 layers, depth=24)."""
    run_sweep.remote(
        base_config="configs/probe.OMARRQ-multifeature25hz.GTZANBeatTracking.yaml",
        num_layers=24,
        model_tag="OMARRQ-multifeature25hz",
        task_tag="GTZANBeatTracking",
    )


@app.local_entrypoint()
def sweep_omarrq_gs():
    """OMAR-RQ multifeature-25hz full layer sweep on GiantSteps key detection (24 layers)."""
    run_sweep.remote(
        base_config="configs/probe.OMARRQ-multifeature25hz.GS.yaml",
        num_layers=24,
        model_tag="OMARRQ-multifeature25hz",
        task_tag="GS",
    )


@app.local_entrypoint()
def sweep_omarrq_emo():
    """OMAR-RQ multifeature-25hz full layer sweep on EmoMusic valence/arousal (24 layers)."""
    run_sweep.remote(
        base_config="configs/probe.OMARRQ-multifeature25hz.EMO.yaml",
        num_layers=24,
        model_tag="OMARRQ-multifeature25hz",
        task_tag="EMO",
    )


@app.local_entrypoint()
def sweep_clamp3_gs():
    """CLaMP3 BERT audio layer sweep on GiantSteps key detection (13 layers, 0-12)."""
    run_sweep.remote(
        base_config="configs/probe.CLaMP3-layers.GS.yaml",
        num_layers=13,
        model_tag="CLaMP3",
        task_tag="GS",
    )


@app.local_entrypoint()
def sweep_clamp3_emo():
    """CLaMP3 BERT audio layer sweep on EmoMusic valence/arousal (13 layers, 0-12)."""
    run_sweep.remote(
        base_config="configs/probe.CLaMP3-layers.EMO.yaml",
        num_layers=13,
        model_tag="CLaMP3",
        task_tag="EMO",
    )


@app.local_entrypoint()
def run_existing_baselines():
    """
    Run the standard MERT-95M and MuQ probes on Chords1217 and GTZANGenre
    as baselines (no layer sweep — uses MLPReduce over all layers).
    """
    for cfg in [
        "configs/probe.MERT-v1-95M.Chords1217.yaml",
        "configs/probe.MuQ.Chords1217.yaml",
        "configs/probe.MERT-v1-95M.GTZANGenre.yaml",
        "configs/probe.MuQ.GTZANGenre.yaml",
    ]:
        print(f"\n── {cfg}")
        run_probe.remote(config=cfg)


@app.local_entrypoint()
def run_gs_emo_baselines():
    """
    Run the standard MERT-95M and MuQ probes on GiantSteps key + EmoMusic
    as baselines (no layer sweep).
    """
    for cfg in [
        "configs/probe.MERT-v1-95M.GS.yaml",
        "configs/probe.MuQ.GS.yaml",
        "configs/probe.MERT-v1-95M.EMO.yaml",
        "configs/probe.MuQ.EMO.yaml",
    ]:
        print(f"\n── {cfg}")
        run_probe.remote(config=cfg)


@app.local_entrypoint()
def run_probe_cli(config: str = ""):
    """Run an arbitrary probe by config path. Usage: modal run modal_marble.py::run_probe_cli --config <path>"""
    if not config:
        print("Provide --config <path>")
        return
    run_probe.remote(config=config)
