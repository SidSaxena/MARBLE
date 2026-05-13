#!/usr/bin/env python3
"""
scripts/sweeps/gen_sweep_configs.py
─────────────────────────────
Generate one YAML config per transformer layer for a MARBLE layer sweep.

For each layer N it:
  1. Copies the base config verbatim
  2. Patches  `layers: [N]`  inside the LayerSelector init_args block
  3. Patches checkpoint dirpath and WandB save_dir to include ".layerN"
  4. Sets WandB run name → "layer-N"
  5. Injects WandB group / tags / job_type for organised W&B workspace:
       group    : "{model_tag} / {task_tag}"   (all layers of one sweep)
       tags     : ["{model_tag}", "{task_tag}", "layer-sweep"]
       job_type : "probe"

Usage
-----
python scripts/sweeps/gen_sweep_configs.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml \\
    --num-layers  12 \\
    --model-tag   OMARRQ-multifeature25hz \\
    --task-tag    Chords1217 \\
    --out-dir     configs/sweeps/OMARRQ-multifeature25hz.Chords1217
"""

import argparse
import re
import sys
from pathlib import Path


def patch_layers(text: str, layer: int) -> str:
    """Replace bare `layers: [<anything>]` with `layers: [<layer>]`.
    Uses a negative lookbehind to avoid matching `hidden_layers`, `num_layers`, etc."""
    return re.sub(r'(?<!\w)layers:\s*\[.*?\]', f'layers: [{layer}]', text)


def patch_wandb_name(text: str, layer: int) -> str:
    """Set WandB run name to "layer-N" (short; group carries model×task context)."""
    return re.sub(
        r'(?<!\w)(name:\s*")([^"\n]+)(")',
        lambda m: f'{m.group(1)}layer-{layer}{m.group(3)}',
        text,
    )


def append_layer_tag(text: str, layer: int) -> str:
    """Append ``layer-N`` to the existing WandB tags array if not present.

    Runs unconditionally (independent of inject_wandb_group_tags) so the
    per-layer tag is always added even when the base config already had
    group/tags injected by an earlier migration.
    """
    layer_tag = f'"layer-{layer}"'

    def replacer(m: re.Match) -> str:
        existing = m.group(2).strip()
        if layer_tag in existing:
            return m.group(0)
        new_inner = f"{existing}, {layer_tag}" if existing else layer_tag
        return f"{m.group(1)}[{new_inner}]"

    return re.sub(r'(tags:\s*)\[([^\]\n]*)\]', replacer, text, count=1)


def inject_wandb_group_tags(text: str, group: str, tags: list[str]) -> str:
    """
    Insert group, tags, and job_type lines immediately after the
    ``project: "marble"`` line in the WandB logger init_args block.

    Indentation is inferred from the project: line so the YAML stays valid
    regardless of how deeply nested the logger block is.
    Idempotent: skips injection if ``group:`` is already present.
    """
    if 'group:' in text:
        return text  # already injected (e.g. on re-run)

    tags_yaml = "[" + ", ".join(f'"{t}"' for t in tags) + "]"

    def replacer(m: re.Match) -> str:
        indent = m.group(1)   # whitespace that precedes "project:"
        return (
            f'{m.group(0)}\n'
            f'{indent}group: "{group}"\n'
            f'{indent}tags: {tags_yaml}\n'
            f'{indent}job_type: "probe"'
        )

    return re.sub(r'([ \t]+)project:\s*"[^"]*"', replacer, text)


def main():
    parser = argparse.ArgumentParser(description="Generate per-layer MARBLE sweep configs.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--num-layers",  type=int, required=True)
    parser.add_argument("--model-tag",   required=True)
    parser.add_argument("--task-tag",    required=True)
    parser.add_argument("--out-dir",     required=True)
    parser.add_argument("--layers",      type=int, nargs="*",
                        help="Subset of layers to generate (default: all 0..num_layers-1)")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    if not base_path.exists():
        print(f"Error: base config not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_text = base_path.read_text()
    layers = args.layers if args.layers is not None else list(range(args.num_layers))

    # Strip the "-layers" suffix so the WandB tags / group name unify
    # layer-sweep runs with any single-best-layer probe of the same model.
    # e.g.  "CLaMP3-layers"      →  "CLaMP3"
    #       "MERT-v1-95M-layers" →  "MERT-v1-95M"
    model_base = args.model_tag.removesuffix("-layers")

    # WandB group: groups all layers of one model×task sweep together
    group     = f"{model_base} / {args.task_tag}"
    base_tags = [model_base, args.task_tag, "layer-sweep", "probe"]

    for layer in layers:
        text = base_text

        # 1. LayerSelector index
        text = patch_layers(text, layer)

        # 2. Checkpoint dirpath  (output/<base>  →  output/<base>.layerN)
        text = re.sub(
            r'(dirpath:\s*"?\.?/?)output/([^/\n"\']+)',
            lambda m: f'{m.group(1)}output/{m.group(2)}.layer{layer}',
            text,
        )

        # 3. WandB save_dir  (same pattern)
        text = re.sub(
            r'(save_dir:\s*"?\.?/?)output/([^/\n"\']+)',
            lambda m: f'{m.group(1)}output/{m.group(2)}.layer{layer}',
            text,
        )

        # 4. WandB run name → "layer-N"  (fit/test suffix is added at runtime
        #    via --trainer.logger.init_args.name override in run_sweep_local)
        text = patch_wandb_name(text, layer)

        # 5. Inject group / tags / job_type into WandB init_args.
        #    inject_wandb_group_tags is idempotent — it no-ops if the base
        #    config has already been migrated.  For per-layer enrichment we
        #    always append the layer-specific tag separately.
        text = inject_wandb_group_tags(text, group, base_tags)
        text = append_layer_tag(text, layer)

        out_path = out_dir / f"sweep.{args.model_tag}.{args.task_tag}.layer{layer}.yaml"
        out_path.write_text(text)
        print(f"  layer {layer:2d} → {out_path}")

    print(f"\nGenerated {len(layers)} configs in {out_dir}/")


if __name__ == "__main__":
    main()
