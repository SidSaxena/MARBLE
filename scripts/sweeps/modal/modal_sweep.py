"""scripts/sweeps/modal/modal_sweep.py — submit a MARBLE layer sweep to Modal in parallel.

Mirrors `scripts/sweeps/run_sweep_local.py`'s CLI surface but submits one Modal
container per layer (via `modal_marble.run_parallel_sweep`).

Same total compute cost as the sequential `run_sweep`, but wall-clock ≈
single-layer time instead of N × single-layer.

ALWAYS pass `--detach`. Without it `modal run` starts an *ephemeral* app whose
lifetime is bound to this CLI process — `main()` blocks on `.starmap()` for the
whole sweep, so if the client disconnects (session closed, laptop sleeps,
network drop) Modal tears the app down and kills every layer container
mid-training. `--detach` decouples the run from the client: it keeps going on
Modal's side regardless. Monitor with `modal app logs <app-id>` (or the
dashboard); stop with `modal app stop <app-id> --yes`. (Detach is a `modal run`
CLI flag — it can't be defaulted from inside the script, so it must be on every
invocation below.)

Usage
-----
    modal run --detach scripts/sweeps/modal/modal_sweep.py \\
        --base-config configs/probe.OMARRQ-multifeature-25hz.GS.yaml \\
        --num-layers 24 \\
        --model-tag OMARRQ-multifeature-25hz \\
        --task-tag GS

Subset of layers (comma-separated):
    modal run --detach scripts/sweeps/modal/modal_sweep.py \\
        --base-config configs/probe.CLaMP3-layers.GS.yaml \\
        --num-layers 13 --model-tag CLaMP3 --task-tag GS \\
        --layers 0,1,2,3

Re-run only the test stage (skip fit for completed layers):
    modal run --detach scripts/sweeps/modal/modal_sweep.py --retest-only ... [other flags]

Pass Lightning-CLI overrides through to each layer's fit+test (e.g. bump the
dataloader worker count — Modal A10G/A100 containers expose ~17 cores, so 16
keeps the GPU fed where the config's local-safe default of 8 starves it):
    modal run --detach scripts/sweeps/modal/modal_sweep.py ... \\
        --cli-overrides "--data.init_args.num_workers=16"
Multiple overrides go in one quoted string (shlex-split), e.g.
    --cli-overrides "--data.init_args.num_workers=16 --trainer.max_epochs=50"
This mirrors the sequential entrypoints' `_HOOKTHEORY_CLI_OVERRIDES`.
"""

import shlex

from modal_marble import app, run_parallel_sweep


@app.local_entrypoint()
def main(
    base_config: str,
    num_layers: int,
    model_tag: str,
    task_tag: str,
    layers: str = "",
    retest_only: bool = False,
    warmup_audio_dir: str = "",
    cli_overrides: str = "",
):
    layer_list: list[int] | None = None
    if layers:
        layer_list = [int(x) for x in layers.split(",") if x.strip()]

    # shlex.split so quoted values survive and multiple "--flag=value" tokens
    # in one string become a clean list — the shape run_one_layer expects.
    override_list: list[str] | None = shlex.split(cli_overrides) or None

    results = run_parallel_sweep.remote(
        base_config=base_config,
        num_layers=num_layers,
        model_tag=model_tag,
        task_tag=task_tag,
        layers=layer_list,
        retest_only=retest_only,
        # Inline volume warmup inside each layer container (network-backed
        # Modal volume: first open of each file is slow). Pass the relative
        # dir — run_one_layer chdirs to WORK_DIR before warming.
        warmup_audio_dir=warmup_audio_dir or None,
        cli_overrides=override_list,
    )

    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDone: {completed} completed, {skipped} skipped (of {len(results)} layers)")
