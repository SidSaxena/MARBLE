"""scripts/sweeps/modal/modal_sweep.py — submit a MARBLE layer sweep to Modal in parallel.

Mirrors `scripts/sweeps/run_sweep_local.py`'s CLI surface but submits one Modal
container per layer (via `modal_marble.run_parallel_sweep`).

Same total compute cost as the sequential `run_sweep`, but wall-clock ≈
single-layer time instead of N × single-layer.

Usage
-----
    modal run scripts/sweeps/modal/modal_sweep.py \\
        --base-config configs/probe.OMARRQ-multifeature25hz.GS.yaml \\
        --num-layers 24 \\
        --model-tag OMARRQ-multifeature25hz \\
        --task-tag GS

Subset of layers (comma-separated):
    modal run scripts/sweeps/modal/modal_sweep.py \\
        --base-config configs/probe.CLaMP3-layers.GS.yaml \\
        --num-layers 13 --model-tag CLaMP3 --task-tag GS \\
        --layers 0,1,2,3

Re-run only the test stage (skip fit for completed layers):
    modal run scripts/sweeps/modal/modal_sweep.py --retest-only ... [other flags]
"""

from modal_marble import app, run_parallel_sweep


@app.local_entrypoint()
def main(
    base_config: str,
    num_layers: int,
    model_tag: str,
    task_tag: str,
    layers: str = "",
    retest_only: bool = False,
):
    layer_list: list[int] | None = None
    if layers:
        layer_list = [int(x) for x in layers.split(",") if x.strip()]

    results = run_parallel_sweep.remote(
        base_config=base_config,
        num_layers=num_layers,
        model_tag=model_tag,
        task_tag=task_tag,
        layers=layer_list,
        retest_only=retest_only,
    )

    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(f"\nDone: {completed} completed, {skipped} skipped (of {len(results)} layers)")
