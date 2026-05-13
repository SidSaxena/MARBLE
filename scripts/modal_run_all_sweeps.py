"""scripts/modal_run_all_sweeps.py — orchestrate the Modal-migrated sweeps.

Submits the 5 long sweeps in sequence (each one parallelizes internally over
its N layers). Run this on its own, or alongside `run_all_sweeps.py` on the
PC for everything else.

Usage
-----
    modal run scripts/modal_run_all_sweeps.py
    modal run scripts/modal_run_all_sweeps.py --only OMARRQ-NSynth
    modal run scripts/modal_run_all_sweeps.py --skip MERT-NSynth,MERT-Chords1217

Each sweep spawns its layers in parallel (one Modal container per layer).
"""

from modal_marble import app, run_parallel_sweep


SWEEPS: list[dict] = [
    {
        "tag": "OMARRQ-NSynth",
        "base_config": "configs/probe.OMARRQ-multifeature25hz.NSynth.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "NSynth",
    },
    {
        "tag": "OMARRQ-Chords1217",
        "base_config": "configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "Chords1217",
    },
    {
        "tag": "OMARRQ-GTZANBeatTracking",
        "base_config": "configs/probe.OMARRQ-multifeature25hz.GTZANBeatTracking.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "GTZANBeatTracking",
    },
    {
        "tag": "MERT-NSynth",
        "base_config": "configs/probe.MERT-v1-95M-layers.NSynth.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "NSynth",
    },
    {
        "tag": "MERT-Chords1217",
        "base_config": "configs/probe.MERT-v1-95M-layers.Chords1217.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "Chords1217",
    },
]


@app.local_entrypoint()
def main(only: str = "", skip: str = ""):
    only_set = {x.strip() for x in only.split(",") if x.strip()}
    skip_set = {x.strip() for x in skip.split(",") if x.strip()}

    selected = [s for s in SWEEPS
                if (not only_set or s["tag"] in only_set)
                and s["tag"] not in skip_set]

    print(f"Running {len(selected)} sweep(s) sequentially "
          f"(layers parallelize within each sweep):")
    for s in selected:
        print(f"  - {s['tag']}  ({s['num_layers']} layers)")

    for s in selected:
        print(f"\n{'='*64}\n  {s['tag']}  ({s['num_layers']} layers)\n{'='*64}")
        results = run_parallel_sweep.remote(
            base_config=s["base_config"],
            num_layers=s["num_layers"],
            model_tag=s["model_tag"],
            task_tag=s["task_tag"],
        )
        completed = sum(1 for r in results if r["status"] == "completed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        print(f"  {s['tag']}: {completed} completed, {skipped} skipped")
