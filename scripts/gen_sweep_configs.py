#!/usr/bin/env python3
"""
scripts/gen_sweep_configs.py
─────────────────────────────
Generate one YAML config per transformer layer for a MARBLE layer sweep.

For each layer N it:
  1. Copies the base config verbatim
  2. Patches  `layers: [N]`  inside the LayerSelector init_args block
  3. Patches checkpoint dirpath and WandB name / save_dir to include ".layerN"

Usage
-----
python scripts/gen_sweep_configs.py \\
    --base-config configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml \\
    --num-layers  12 \\
    --model-tag   OMARRQ-multifeature25hz \\
    --task-tag    Chords1217 \\
    --out-dir     configs/sweeps/OMARRQ-multifeature25hz.Chords1217
"""

import argparse
import os
import re
import sys
from pathlib import Path


def patch_layers(text: str, layer: int) -> str:
    """Replace bare `layers: [<anything>]` with `layers: [<layer>]`.
    Uses a negative lookbehind to avoid matching `hidden_layers`, `num_layers`, etc."""
    return re.sub(r'(?<!\w)layers:\s*\[.*?\]', f'layers: [{layer}]', text)


def patch_dirpath(text: str, original_tag: str, new_tag: str) -> str:
    """Patch ModelCheckpoint dirpath."""
    return re.sub(
        r'(dirpath:\s*["\']?)(.*?)(' + re.escape(original_tag) + r')(.*?["\']?)',
        lambda m: m.group(1) + m.group(2) + new_tag + m.group(4),
        text,
    )


def patch_wandb(text: str, original_tag: str, new_tag: str) -> str:
    """Patch WandB name and save_dir."""
    return re.sub(
        r'((?:name|save_dir):\s*["\']?)(.*?)(' + re.escape(original_tag) + r')(/?["\']?)',
        lambda m: m.group(1) + m.group(2) + new_tag + m.group(4),
        text,
    )


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

    # Derive the "base tag" that appears in dirpath / wandb name from the base config stem
    # e.g.  probe.OMARRQ-multifeature25hz.Chords1217  →  base_tag
    base_tag = base_path.stem  # e.g. "probe.OMARRQ-multifeature25hz.Chords1217"

    for layer in layers:
        layer_tag = f"{base_tag}.layer{layer}"
        text = base_text

        # 1. Patch LayerSelector
        text = patch_layers(text, layer)

        # 2. Patch checkpoint dirpath:
        #    Match the FIRST path segment after "output/" (no slashes allowed),
        #    append ".layerN", and leave the rest of the path unchanged.
        text = re.sub(
            r'(dirpath:\s*"?\.?/?)output/([^/\n"\']+)',
            lambda m: f'{m.group(1)}output/{m.group(2)}.layer{layer}',
            text,
        )

        # 3. Patch WandB logger name and save_dir
        #    Use negative lookbehind to avoid matching "filename: ..." (where "name" is a suffix).
        text = re.sub(
            r'(?<!\w)(name:\s*")([^"\n]+)(")',
            lambda m: f'{m.group(1)}{m.group(2)}.layer{layer}{m.group(3)}',
            text,
        )
        text = re.sub(
            r'(save_dir:\s*"?\.?/?)output/([^/\n"\']+)',
            lambda m: f'{m.group(1)}output/{m.group(2)}.layer{layer}',
            text,
        )

        out_path = out_dir / f"sweep.{args.model_tag}.{args.task_tag}.layer{layer}.yaml"
        out_path.write_text(text)
        print(f"  layer {layer:2d} → {out_path}")

    print(f"\nGenerated {len(layers)} configs in {out_dir}/")


if __name__ == "__main__":
    main()
