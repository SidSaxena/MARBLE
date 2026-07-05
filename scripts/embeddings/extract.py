#!/usr/bin/env python3
"""scripts/embeddings/extract.py
─────────────────────────────────
Pre-warm the per-clip embedding cache for a probe config WITHOUT running
a Lightning fit/test (and so without creating a duplicate WandB run).

Use this when you want to populate the cache offline before launching a
sweep, e.g. overnight so the next morning's sweep is all cache hits.
For the meanall-first sweep flow you don't actually need this script —
``run_sweep_local.py`` already pre-warms via the meanall job — but it's
handy for:

  * Resuming a partially-completed sweep without an extra WandB run
  * Pre-extracting multiple (encoder, task) caches in parallel offline
  * Populating the cache from a different machine than the one running
    the sweep (rsync the cache dir afterward)

What it does
------------
1. Parses the YAML config (any probe config with `cache_embeddings: true`
   in `model.init_args`).
2. Instantiates the encoder + datamodule the same way LightningCLI would.
3. Builds the EmbeddingCache with the exact same key the runtime would
   produce (so cache hits during the sweep land on these files).
4. Iterates the chosen split's DataLoader, runs `task.forward(x,
   clip_ids=...)`, which populates the cache on miss.
5. Exits. No WandB, no Lightning Trainer.

Usage
-----
::

    # Pre-warm SHS100K's cache for OMARRQ — uses the meanall sibling
    # config (any config with the right encoder + datamodule works, but
    # meanall configs typically have max_epochs=0 so they're guaranteed
    # not to require trainable state).
    uv run python scripts/embeddings/extract.py \\
        --config configs/probe.OMARRQ-multifeature-25hz-meanall.SHS100K.yaml

    # Different split (default: test)
    uv run python scripts/embeddings/extract.py \\
        --config <config> --split train

    # Force re-extraction (re-write every cache file)
    uv run python scripts/embeddings/extract.py --config <config> --no-skip

Reads the same fields the runtime cache derivation reads
(``marble/tasks/Covers80/probe.py:_ensure_cache``), so the cache key
matches deterministically.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from marble.core.utils import instantiate_from_config, instantiate_recursive


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a probe config YAML (model.init_args must accept "
        "cache_embeddings; usually a meanall sibling config).",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which dataloader split to walk. Default: test.",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the encoder on (default: auto cuda/cpu).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the datamodule's batch_size. Default: use config value.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the datamodule's num_workers. Default: use config value.",
    )
    ap.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-extract even when every clip is already cached.",
    )
    ap.add_argument(
        "--precision",
        default="bf16",
        choices=["fp32", "tf32", "bf16"],
        help="Encoder forward precision. bf16 (default) = TF32 matmuls + "
        "bf16 autocast — ~2-4x on tensor-core GPUs; matches the bf16-mixed "
        "precision the live Lightning runs already use, so cached embeddings "
        "are no further from the live path than fp32 extraction was. "
        "tf32 = TF32 matmuls only (~1.5-2x, smaller numeric delta). "
        "fp32 = the historical exact-fp32 behaviour.",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Print progress every N batches. Default: 20.",
    )
    return ap.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(args.config.read_text())

    # ── Apply overrides to the data config before instantiation
    if args.batch_size is not None:
        cfg.setdefault("data", {}).setdefault("init_args", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("data", {}).setdefault("init_args", {})["num_workers"] = args.num_workers

    # ── Instantiate the datamodule + the chosen dataloader
    print(f"Instantiating datamodule for split={args.split} ...")
    dm = instantiate_from_config(cfg["data"])
    dm.setup(stage=args.split)
    loader = {
        "train": dm.train_dataloader,
        "val": dm.val_dataloader,
        "test": dm.test_dataloader,
    }[args.split]()
    n_batches = len(loader)
    print(f"  loader has {n_batches} batch(es)")

    # ── Instantiate the task (loads the encoder). Recursive: BaseTask expects
    #    instantiated submodules (encoder/decoders/metrics); non-recursive
    #    instantiate_from_config would pass raw dicts → "dict is not a Module
    #    subclass". (The datamodule above stays non-recursive on purpose — it
    #    stores split configs raw until setup().)
    print(f"\nInstantiating task + encoder on {args.device} ...")
    model_cfg = cfg["model"]
    model_cfg.setdefault("init_args", {})["cache_embeddings"] = True
    task = instantiate_recursive(model_cfg)
    task.eval().to(args.device)

    # ── Build the cache via the task's OWN _ensure_cache, so the cache key
    #    (encoder model_id, sample_rate, clip_seconds, pipeline signature,
    #    pool_time) is byte-identical to what the training runtime derives — no
    #    reimplementation drift. _ensure_cache only reads the datamodule + the
    #    wandb group off the trainer, so a minimal shim is enough.
    group = None
    try:
        group = cfg["trainer"]["logger"]["init_args"]["group"]
    except (KeyError, TypeError):
        group = None
    task.trainer = types.SimpleNamespace(
        datamodule=dm,
        logger=types.SimpleNamespace(_wandb_init={"group": group} if group else {}),
    )
    task._cache = None
    task._cache_init_attempted = False
    task._ensure_cache()
    cache = task._cache
    if cache is None:
        sys.exit("No cache was built — is cache_embeddings supported for this task?")
    print(f"Cache directory: {cache.dir}  (pool_time={cache.pool_time})")

    # ── Fast-path: if every clip is already cached, we're done.
    if not args.no_skip:
        print("\nChecking whether cache is already fully populated ...")
        seen_clip_ids: set[str] = set()
        all_cached = True
        for batch in loader:
            if len(batch) < 4:
                print(
                    "  ! batch has < 4 elements (no clip_id field). The target "
                    "datamodule isn't cache-aware (needs (waveform, label, path, clip_id)).",
                    file=sys.stderr,
                )
                sys.exit(3)
            for cid in batch[3]:
                seen_clip_ids.add(cid)
                if not cache.has(cid):
                    all_cached = False
        if all_cached:
            print(
                f"  ✓ all {len(seen_clip_ids)} clips already cached. "
                f"Nothing to do (pass --no-skip to force re-extract)."
            )
            return
        print(f"  proceeding: {len(seen_clip_ids)} unique clips, some missing.")

    # ── Walk the loader, forward each batch through the cache-aware
    #    forward(). Misses populate the cache; hits are no-ops.
    #
    # Precision: extraction historically ran plain fp32 with matmul precision
    # "highest" — measured 98-100% SM at the power cap with the tensor cores
    # idle. TF32 ("high") + bf16 autocast engages them (~2-4x). Reductions
    # under autocast stay fp32-accumulated; the fp16 store cast in
    # EmbeddingCache.put() is the final precision floor either way.
    use_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if args.precision != "fp32" and use_cuda:
        torch.set_float32_matmul_precision("high")
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.precision == "bf16" and use_cuda
        else nullcontext()
    )
    print(f"\nExtracting embeddings → {cache.dir}  (precision={args.precision})")
    t_start = time.time()
    n_clips = 0
    with torch.inference_mode(), autocast_ctx:
        for i, batch in enumerate(loader):
            x = batch[0].to(args.device)
            clip_ids = batch[3]
            task(x, clip_ids=list(clip_ids))
            n_clips += len(clip_ids)
            if (i + 1) % args.log_every == 0 or (i + 1) == n_batches:
                elapsed = time.time() - t_start
                rate = n_clips / max(elapsed, 1e-6)
                eta_s = (n_batches - (i + 1)) * elapsed / max(i + 1, 1)
                print(
                    f"  batch {i + 1:>4} / {n_batches}  "
                    f"{n_clips:>6,} clips  "
                    f"{rate:.1f} clips/s  ETA {eta_s / 60:.1f} min"
                )

    total = time.time() - t_start
    # Measure disk usage of the cache dir for the user's records
    total_bytes = sum(p.stat().st_size for p in cache.dir.iterdir() if p.is_file())
    print(
        f"\n✓ done in {total / 60:.1f} min. "
        f"{n_clips:,} clips → {total_bytes / 1e6:.1f} MB at {cache.dir}"
    )


if __name__ == "__main__":
    main()
