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
from pathlib import Path

import torch
import yaml

from marble.core.utils import instantiate_from_config
from marble.utils.emb_cache import EmbeddingCache, compute_config_hash


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
        "--log-every",
        type=int,
        default=20,
        help="Print progress every N batches. Default: 20.",
    )
    return ap.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Cache-key derivation from a raw YAML config
# ──────────────────────────────────────────────────────────────────────────


def _dig(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def derive_cache_from_config(cfg: dict) -> tuple[EmbeddingCache, dict]:
    """Mirror of CoverRetrievalTask._ensure_cache, driven by raw YAML
    dict instead of trainer/datamodule attributes. Returns the built
    cache + a dict of the derivation inputs (for logging)."""

    # 1. encoder_slug + task_name from the WandB group, with fallbacks
    #    that match the runtime's _derive_cache_slugs logic.
    group = _dig(cfg, "trainer", "logger", "init_args", "group")
    if isinstance(group, str) and " / " in group:
        encoder_slug, task_name = (s.strip() for s in group.split(" / ", 1))
    else:
        # Fallback: class names taken from the config (no class-instantiation
        # needed for this step — `class_path` ends with the class name).
        enc_cp = _dig(cfg, "model", "init_args", "encoder", "class_path", default="")
        encoder_slug = enc_cp.rsplit(".", 1)[-1] if enc_cp else "encoder"
        task_cp = _dig(cfg, "model", "class_path", default="")
        task_name = task_cp.rsplit(".", 1)[-1] if task_cp else "task"

    # 2. encoder_model_id from model.init_args.encoder.init_args.model_id
    #    when present (OMARRQ), else fall back to the slug — same as the
    #    runtime's `getattr(self.encoder, "HUGGINGFACE_MODEL_NAME", ...)`.
    model_id = _dig(cfg, "model", "init_args", "encoder", "init_args", "model_id")
    if not model_id:
        model_id = encoder_slug

    # 3. sample_rate from model.init_args.sample_rate (encoder isn't yet
    #    instantiated here, so we trust the config's declared sample_rate).
    sample_rate = _dig(cfg, "model", "init_args", "sample_rate", default=0)

    # 4. clip_seconds from the test dataset's init_args (where the encoder
    #    actually consumes audio — matches the runtime's _derive_clip_seconds).
    clip_seconds = _dig(
        cfg,
        "data",
        "init_args",
        "test",
        "init_args",
        "clip_seconds",
        default=0.0,
    )

    # 5. pipeline signature from the test audio_transforms class_paths.
    transforms = _dig(cfg, "data", "init_args", "audio_transforms", "test", default=[]) or []
    sig_parts = [t.get("class_path", repr(t)) for t in transforms]
    pipeline_signature = "|".join(sig_parts)

    config_hash = compute_config_hash(
        encoder_model_id=model_id,
        sample_rate=sample_rate,
        clip_seconds=clip_seconds,
        pipeline_signature=pipeline_signature,
    )

    cache = EmbeddingCache(
        encoder_slug=encoder_slug,
        task_name=task_name,
        config_hash=config_hash,
        metadata={
            "encoder_model_id": str(model_id),
            "sample_rate": sample_rate,
            "clip_seconds": float(clip_seconds),
            "pipeline_signature": pipeline_signature,
            "extracted_via": "scripts/embeddings/extract.py",
        },
    )

    return cache, {
        "encoder_slug": encoder_slug,
        "task_name": task_name,
        "config_hash": config_hash,
        "model_id": model_id,
        "sample_rate": sample_rate,
        "clip_seconds": clip_seconds,
        "pipeline_signature": pipeline_signature,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(2)

    cfg = yaml.safe_load(args.config.read_text())

    # ── Build the cache up front so we can do an early "everything's
    #    already cached, exit fast" check when --no-skip isn't set.
    cache, derived = derive_cache_from_config(cfg)
    print(f"Cache directory: {cache.dir}")
    print("Derived cache key:")
    for k, v in derived.items():
        print(f"  {k}: {v}")

    # ── Apply overrides to the data config before instantiation
    if args.batch_size is not None:
        cfg.setdefault("data", {}).setdefault("init_args", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.setdefault("data", {}).setdefault("init_args", {})["num_workers"] = args.num_workers

    # ── Instantiate the datamodule + the chosen dataloader
    print(f"\nInstantiating datamodule for split={args.split} ...")
    dm = instantiate_from_config(cfg["data"])
    dm.setup(stage=args.split)
    loader = {
        "train": dm.train_dataloader,
        "val": dm.val_dataloader,
        "test": dm.test_dataloader,
    }[args.split]()

    n_batches = len(loader)
    print(f"  loader has {n_batches} batch(es)")

    # ── Fast-path: scan the loader once and see if all clip_ids are cached.
    #    Saves spinning up the encoder model just to find a fully-warm cache.
    if not args.no_skip:
        print("\nChecking whether cache is already fully populated ...")
        seen_clip_ids: set[str] = set()
        all_cached = True
        for batch in loader:
            if len(batch) < 4:
                # Datamodule doesn't emit clip_id — can't pre-warm via
                # the cache layer in this script. Bail with a clear msg.
                print(
                    "  ! batch has < 4 elements (no clip_id field). The "
                    "target datamodule isn't cache-aware. Update its "
                    "__getitem__ to return (waveform, label, path, clip_id).",
                    file=sys.stderr,
                )
                sys.exit(3)
            clip_ids = batch[3]
            for cid in clip_ids:
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

    # ── Instantiate the task (needs the encoder to be loaded). We force
    #    cache_embeddings=True here so the task's forward() writes via
    #    the cache infrastructure we just constructed.
    print(f"\nInstantiating task + encoder on {args.device} ...")
    model_cfg = cfg["model"]
    model_cfg.setdefault("init_args", {})["cache_embeddings"] = True
    task = instantiate_from_config(model_cfg)
    task.eval().to(args.device)

    # Attach the pre-built cache and short-circuit lazy init. This bypasses
    # the runtime _ensure_cache, which needs a trainer we don't have.
    task._cache = cache
    task._cache_init_attempted = True

    # ── Walk the loader, forward each batch through the cache-aware
    #    forward(). Misses populate the cache; hits are no-ops.
    print(f"\nExtracting embeddings → {cache.dir}")
    t_start = time.time()
    n_clips = 0
    with torch.no_grad():
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
