"""scripts/modal_run_all_sweeps.py — orchestrate the Modal-migrated sweeps.

Per-sweep semantics: each sweep parallelizes across its N layers via
`modal_marble.run_parallel_sweep` (one L4 container per layer). This script
just iterates over sweeps sequentially.

Priority order (user-set):
  Tier 1  MERT × ... (most complete coverage)
  Tier 2  CLaMP3 × ... (no Chords1217, no GTZANBeatTracking, no HookTheoryMelody)
  Tier 3  OMARRQ × ... (deferred — 24-layer sweeps cost ~2× MERT/CLaMP3)

GS is skipped on Modal (already done locally).

Usage
-----
    modal run scripts/modal_run_all_sweeps.py                  # all selected
    modal run scripts/modal_run_all_sweeps.py --only MERT-NSynth
    modal run scripts/modal_run_all_sweeps.py --skip MERT-NSynth,CLaMP3-NSynth
    modal run scripts/modal_run_all_sweeps.py --tier 1         # just MERT
    modal run scripts/modal_run_all_sweeps.py --tier 2         # just CLaMP3
    modal run scripts/modal_run_all_sweeps.py --tier 3         # just OMARRQ
"""

from modal_marble import app, run_parallel_sweep


# Tier 1 — MERT (~$160 total at L4 spot, all 13-layer)
TIER1_MERT: list[dict] = [
    {
        "tag": "MERT-SHS100K",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.SHS100K.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "SHS100K",
        "note": "Zero-shot retrieval — fast smoke test",
    },
    {
        "tag": "MERT-HookTheoryMelody",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.HookTheoryMelody.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "HookTheoryMelody",
        "note": "Requires setup_hooktheory_full first",
    },
    {
        "tag": "MERT-HookTheoryStructure",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.HookTheoryStructure.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "HookTheoryStructure",
    },
    {
        "tag": "MERT-HookTheoryKey",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.HookTheoryKey.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "HookTheoryKey",
    },
    {
        "tag": "MERT-GTZANBeatTracking",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.GTZANBeatTracking.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "GTZANBeatTracking",
    },
    {
        "tag": "MERT-Chords1217",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.Chords1217.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "Chords1217",
    },
    {
        "tag": "MERT-NSynth",
        "tier": 1,
        "base_config": "configs/probe.MERT-v1-95M-layers.NSynth.yaml",
        "num_layers": 13,
        "model_tag": "MERT-v1-95M",
        "task_tag": "NSynth",
    },
]


# Tier 2 — CLaMP3 (fewer sweeps available — no Chords1217, no BeatTracking, no Melody)
TIER2_CLAMP3: list[dict] = [
    {
        "tag": "CLaMP3-SHS100K",
        "tier": 2,
        "base_config": "configs/probe.CLaMP3-layers.SHS100K.yaml",
        "num_layers": 13,
        "model_tag": "CLaMP3",
        "task_tag": "SHS100K",
        "note": "Zero-shot retrieval — fast",
    },
    {
        "tag": "CLaMP3-HookTheoryKey",
        "tier": 2,
        "base_config": "configs/probe.CLaMP3-layers.HookTheoryKey.yaml",
        "num_layers": 13,
        "model_tag": "CLaMP3",
        "task_tag": "HookTheoryKey",
    },
    {
        "tag": "CLaMP3-HookTheoryStructure",
        "tier": 2,
        "base_config": "configs/probe.CLaMP3-layers.HookTheoryStructure.yaml",
        "num_layers": 13,
        "model_tag": "CLaMP3",
        "task_tag": "HookTheoryStructure",
    },
    {
        "tag": "CLaMP3-NSynth",
        "tier": 2,
        "base_config": "configs/probe.CLaMP3-layers.NSynth.yaml",
        "num_layers": 13,
        "model_tag": "CLaMP3",
        "task_tag": "NSynth",
    },
]


# Tier 3 — OMARRQ (deferred; 24-layer sweeps, ~2× cost of MERT/CLaMP3)
TIER3_OMARRQ: list[dict] = [
    {
        "tag": "OMARRQ-HookTheoryMelody",
        "tier": 3,
        "base_config": "configs/probe.OMARRQ-multifeature25hz.HookTheoryMelody.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "HookTheoryMelody",
    },
    {
        "tag": "OMARRQ-HookTheoryStructure",
        "tier": 3,
        "base_config": "configs/probe.OMARRQ-multifeature25hz.HookTheoryStructure.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "HookTheoryStructure",
    },
    {
        "tag": "OMARRQ-GTZANBeatTracking",
        "tier": 3,
        "base_config": "configs/probe.OMARRQ-multifeature25hz.GTZANBeatTracking.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "GTZANBeatTracking",
    },
    {
        "tag": "OMARRQ-Chords1217",
        "tier": 3,
        "base_config": "configs/probe.OMARRQ-multifeature25hz.Chords1217.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "Chords1217",
    },
    {
        "tag": "OMARRQ-NSynth",
        "tier": 3,
        "base_config": "configs/probe.OMARRQ-multifeature25hz.NSynth.yaml",
        "num_layers": 24,
        "model_tag": "OMARRQ-multifeature25hz",
        "task_tag": "NSynth",
    },
]


SWEEPS = TIER1_MERT + TIER2_CLAMP3 + TIER3_OMARRQ


@app.local_entrypoint()
def main(only: str = "", skip: str = "", tier: int = 0):
    only_set = {x.strip() for x in only.split(",") if x.strip()}
    skip_set = {x.strip() for x in skip.split(",") if x.strip()}

    selected = []
    for s in SWEEPS:
        if tier > 0 and s["tier"] != tier:
            continue
        if only_set and s["tag"] not in only_set:
            continue
        if s["tag"] in skip_set:
            continue
        selected.append(s)

    print(f"Running {len(selected)} sweep(s) sequentially "
          f"(layers parallelize within each sweep):")
    for s in selected:
        note = f"  ({s['note']})" if s.get("note") else ""
        print(f"  - tier {s['tier']}  {s['tag']:<32}  "
              f"{s['num_layers']} layers{note}")

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
