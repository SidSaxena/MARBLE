#!/usr/bin/env python3
"""
scripts/analysis/vgmiditvar_leitmotif_breakdown.py
──────────────────────────────────────────────────
Compute per-instrument-pair retrieval MAP for the VGMIDITVar-leitmotif
test set, using cached encoder embeddings produced by a layer sweep.

Aggregate MAP is misleading on this dataset — idx=0 is always piano, so
the easy "piano theme → piano theme of a different work" pairs dominate.
The real story is in per-pair slices:

  * same-instrument MAP   (piano→piano, strings→strings, …)
                          — ceiling, weakest test of invariance
  * cross-instrument MAP  (piano→strings, horn→flute, …)
                          — the actual leitmotif challenge
  * per-pair grid         — every (query_program, target_program) cell

The script reads:
  1. The JSONL at  data/VGMIDITVar-leitmotif/VGMIDITVar.jsonl
     (must have a `gm_program` field per record, written by the renderer
     when --instrument-map is passed).
  2. Cached embeddings under  output/.emb_cache/<encoder>/
     VGMIDITVar-leitmotif__<config_hash>/  — written by the cache during
     the layer sweep.

Per-clip embeddings are aggregated per audio path via mean+L2-norm, then
the cosine similarity matrix is sliced by gm_program to compute MAP per
(query_program, target_program) cell.

Usage:
    uv run python scripts/analysis/vgmiditvar_leitmotif_breakdown.py \\
        --encoder MuQ --layer 11

The default --cache-root is output/.emb_cache (the MARBLE convention).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _clip_id(audio_path: str, slice_idx: int = 0) -> str:
    """Replicate ``marble.utils.emb_cache.make_clip_id``.

    Format: ``<stem>__<sha1(posix-path)[:8]>__c<slice_idx>``.

    IMPORTANT: must normalize via ``as_posix()`` BEFORE hashing —
    otherwise on Windows the backslash vs forward-slash difference
    yields a different hash than what the cache used at write time
    and every lookup misses. This precisely matches the upstream
    implementation in marble/utils/emb_cache.py:make_clip_id.

    Inline to avoid pulling in the full marble import chain (heavy)
    for a small analysis script.
    """
    stem = Path(audio_path).stem
    norm = Path(audio_path).as_posix()
    h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
    return f"{stem}__{h}__c{int(slice_idx)}"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _find_cache_dir(cache_root: Path, encoder: str) -> Path:
    """Locate the VGMIDITVar-leitmotif cache subdir for the given encoder.

    The structure is ``cache_root / <encoder> / VGMIDITVar-leitmotif__<hash>``.
    Picks the most recently modified match if multiple hashes exist
    (e.g. after a clip_seconds change).
    """
    enc_dir = cache_root / encoder
    if not enc_dir.exists():
        raise SystemExit(f"Encoder dir not found in cache root: {enc_dir}")
    cands = sorted(
        enc_dir.glob("VGMIDITVar-leitmotif__*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise SystemExit(f"No VGMIDITVar-leitmotif cache dir under {enc_dir}")
    if len(cands) > 1:
        log.warning(
            "Multiple cache dirs found for %s/VGMIDITVar-leitmotif; using most-recent: %s",
            encoder,
            cands[0].name,
        )
    chosen = cands[0]
    n_pt = sum(1 for _ in chosen.glob("*.pt"))
    if n_pt == 0:
        raise SystemExit(
            f"\nCache directory {chosen} contains 0 .pt files.\n"
            f"\n  This typically means the encoder's datamodule does not emit "
            f"clip_ids and therefore bypasses the cache entirely.\n"
            f"  CLaMP3-symbolic is one such case — VGMIDITVarSymbolicBase."
            f"__getitem__ returns a 3-tuple (patches, work_id, midi_path) with "
            f"no clip_id, so the cache mechanism never fires for it.\n"
            f"\n  Workaround: use the WandB aggregate test/map for symbolic and "
            f"only run this breakdown script on audio encoders (CLaMP3, "
            f"MERT-v1-95M, MuQ, OMARRQ-multifeature-25hz)."
        )
    log.info("Cache contains %d .pt files", n_pt)
    return chosen


def _load_clip_embedding(cache_dir: Path, clip_id: str, layer: int) -> torch.Tensor | None:
    """Load the (H,) embedding for ``layer`` from the cached ``(L, H)`` tensor.

    The MARBLE cache stores ``{"embedding": tensor}`` (dict, not raw tensor) —
    see ``EmbeddingCache.put`` which calls ``torch.save({"embedding": emb}, ...)``.
    We extract the inner tensor here.
    """
    path = cache_dir / f"{clip_id}.pt"
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=True)
    # Dict (current MARBLE convention) or bare tensor (defensive fallback).
    if isinstance(obj, dict):
        tensor = obj.get("embedding")
        if tensor is None:
            log.warning("Cache file %s missing 'embedding' key (keys=%s)", path, list(obj))
            return None
    else:
        tensor = obj
    if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
        log.warning(
            "Unexpected cache content at %s: type=%s shape=%s",
            path,
            type(tensor).__name__,
            getattr(tensor, "shape", None),
        )
        return None
    if layer < 0 or layer >= tensor.size(0):
        log.warning(
            "Layer %d out of range for shape %s at %s; skipping",
            layer,
            tuple(tensor.shape),
            path,
        )
        return None
    return tensor[layer]


def _aggregate_per_path(
    records: list[dict], cache_dir: Path, layer: int
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Mean-pool clip embeddings per audio_path, return (embs (N, H), work_ids, gm_programs)."""
    path2embs: dict[str, list[torch.Tensor]] = defaultdict(list)
    path2work: dict[str, int] = {}
    path2program: dict[str, int] = {}
    missing = 0

    for rec in records:
        audio_path = rec["audio_path"]
        cid = _clip_id(audio_path, 0)
        emb = _load_clip_embedding(cache_dir, cid, layer)
        if emb is None:
            missing += 1
            continue
        path2embs[audio_path].append(emb)
        path2work[audio_path] = int(rec["work_id"])
        # gm_program may be absent on older JSONLs — fall back to 0 (piano)
        # and warn the user.
        if "gm_program" not in rec:
            log.warning(
                "Record missing gm_program: %s — defaulting to 0 (piano). "
                "Re-run renderer with --instrument-map to populate this field.",
                audio_path,
            )
        path2program[audio_path] = int(rec.get("gm_program", 0))

    if missing:
        log.warning(
            "Missing cache embeddings for %d/%d records; result reflects only "
            "the %d that were cached.",
            missing,
            len(records),
            len(records) - missing,
        )

    paths = sorted(path2embs)
    embs: list[torch.Tensor] = []
    work_ids: list[int] = []
    programs: list[int] = []
    for p in paths:
        stacked = torch.stack(path2embs[p]).mean(0)
        embs.append(F.normalize(stacked, dim=-1))
        work_ids.append(path2work[p])
        programs.append(path2program[p])

    return torch.stack(embs), work_ids, programs


def _map_for_subset(
    sim: torch.Tensor,
    work_ids: list[int],
    programs: list[int],
    query_program: int | None,
    target_program: int | None,
) -> tuple[float, int]:
    """Compute MAP for queries with `query_program` against candidates with
    `target_program`. None means "any program". Returns (MAP, N_queries)."""
    n = sim.size(0)
    wids = torch.tensor(work_ids)
    prog_t = torch.tensor(programs)

    if query_program is None:
        query_mask = torch.ones(n, dtype=torch.bool)
    else:
        query_mask = prog_t == query_program

    if target_program is None:
        target_mask = torch.ones(n, dtype=torch.bool)
    else:
        target_mask = prog_t == target_program

    aps: list[float] = []
    for i in range(n):
        if not query_mask[i]:
            continue
        # Allowed candidates: target_mask, excluding self
        allowed = target_mask.clone()
        allowed[i] = False
        if allowed.sum() == 0:
            continue
        sims_i = sim[i].clone()
        sims_i[~allowed] = -2.0
        order = sims_i.argsort(descending=True)
        order = order[: int(allowed.sum())]
        is_rel = (wids[order] == wids[i]) & allowed[order]
        n_rel = int(is_rel.sum().item())
        if n_rel == 0:
            continue
        hits = 0
        ap = 0.0
        for rank, rel in enumerate(is_rel.tolist(), start=1):
            if rel:
                hits += 1
                ap += hits / rank
        ap /= n_rel
        aps.append(ap)
    return (float(torch.tensor(aps).mean().item()) if aps else 0.0, len(aps))


# GM program → human label, just for the table headers
GM_LABELS = {
    0: "Piano",
    48: "Strings",
    56: "Trumpet",
    60: "Horn",
    73: "Flute",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--cache-root",
        default="output/.emb_cache",
        help="Cache root. Default matches MARBLE's DEFAULT_CACHE_ROOT.",
    )
    ap.add_argument(
        "--encoder",
        required=True,
        help="Encoder slug as it appears in the cache root, e.g. 'MuQ'.",
    )
    ap.add_argument(
        "--jsonl",
        default="data/VGMIDITVar-leitmotif/VGMIDITVar.jsonl",
    )
    ap.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer index to slice from each cached (L, H) tensor.",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=("train", "test", "all"),
        help="Restrict analysis to one split (matches the `split` JSONL field).",
    )
    args = ap.parse_args()

    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        sys.exit(f"JSONL not found: {jsonl}")
    records = _load_jsonl(jsonl)
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]
    if not records:
        sys.exit(f"No records after filtering by split={args.split}")

    cache_dir = _find_cache_dir(Path(args.cache_root), args.encoder)
    log.info("Using cache: %s", cache_dir)
    log.info("Loaded %d records from %s (split=%s)", len(records), jsonl, args.split)

    embs, work_ids, programs = _aggregate_per_path(records, cache_dir, args.layer)
    n = len(work_ids)
    if n == 0:
        sys.exit("No usable records after embedding lookup; aborting.")

    log.info("Computing similarity over %d files", n)
    sim = embs @ embs.T

    progs_sorted = sorted({p for p in programs if p is not None})

    # ── Headline: aggregate MAP ──────────────────────────────────────────
    agg_map, agg_n = _map_for_subset(sim, work_ids, programs, None, None)
    print(f"\n## VGMIDITVar-leitmotif breakdown — {args.encoder} L{args.layer}")
    print(f"\nFiles: {n}  ({len(set(work_ids))} works, split={args.split})")
    print(f"\n**Aggregate MAP:** {agg_map:.4f}  (N queries = {agg_n})")
    print("Aggregate is misleading on this dataset — see per-pair table.\n")

    # ── Same-instrument vs cross-instrument summary ──────────────────────
    same_aps: list[float] = []
    cross_aps: list[float] = []
    for q in progs_sorted:
        for t in progs_sorted:
            m, nq = _map_for_subset(sim, work_ids, programs, q, t)
            if nq == 0:
                continue
            (same_aps if q == t else cross_aps).append(m)
    print(
        f"**Same-instrument MAP** (mean across cells): "
        f"{(sum(same_aps) / len(same_aps)) if same_aps else 0:.4f}"
    )
    print(
        f"**Cross-instrument MAP** (mean across cells): "
        f"{(sum(cross_aps) / len(cross_aps)) if cross_aps else 0:.4f}"
    )
    print()

    # ── Per-pair table ───────────────────────────────────────────────────
    print("### Per (query → target) instrument pair\n")
    header = (
        "| query \\ target | " + " | ".join(f"{GM_LABELS.get(p, p)}" for p in progs_sorted) + " |"
    )
    sep = "|" + "|".join(["---"] * (len(progs_sorted) + 1)) + "|"
    print(header)
    print(sep)
    for q in progs_sorted:
        cells: list[str] = []
        for t in progs_sorted:
            m, nq = _map_for_subset(sim, work_ids, programs, q, t)
            if nq == 0:
                cells.append("—")
            else:
                cells.append(f"{m:.3f} (N={nq})")
        row = f"| **{GM_LABELS.get(q, q)}** | " + " | ".join(cells) + " |"
        print(row)

    print(
        "\n_Each cell is MAP for queries of `query` program retrieving "
        "candidates of `target` program. Diagonal = same-instrument (easy). "
        "Off-diagonal = cross-instrument (real leitmotif test)._"
    )


if __name__ == "__main__":
    main()
