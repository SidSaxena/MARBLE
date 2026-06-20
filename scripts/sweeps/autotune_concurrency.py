#!/usr/bin/env python3
"""Autotune sweep concurrency + embedding-cache RAM mode to the live machine.

Recommends (concurrency, MARBLE_EMB_RAM_CACHE_BYTES) per (encoder, task) that
maximizes throughput WITHOUT OOMing, from live RAM/GPU + MEASURED per-process
cost (2026-06-19, RTX 5060 Ti 16GB / WSL). Verdict baked in: uncached (RAM
cache off) at concurrency ~3 is the proven sweet spot for VGMLoopStructure on
24GB WSL -- C5 OOMs, the RAM cache's ~1.7x is within noise and not worth the
extra RAM, so we prefer uncached unless cached wins throughput by >20%.
"""
import argparse, glob, os, re, subprocess

# --- MEASURED per-process cost (PSS = true physical, shared counted once) ---
# Uncached PSS (model + 8 workers, RAM cache off) and cached PSS (full working
# set held in RAM). GPU = encoder resident + batch. Speedup = cached epoch /
# uncached epoch, measured (89s -> 52s solo). Unmeasured encoders use a
# conservative estimate (flagged in output).
RSS_UNCACHED = {"MuQ": 5.2, "MusicFM": 4.8, "CLaMP3": 6.4, "OMARRQ-multifeature-25hz": 6.5}
RSS_CACHED   = {"MuQ": 9.1, "MusicFM": 9.3, "CLaMP3": 8.8, "OMARRQ-multifeature-25hz": 14.0}
GPU_PROC     = {"MuQ": 1.5, "MusicFM": 1.5, "CLaMP3": 2.5, "OMARRQ-multifeature-25hz": 2.5}
MEASURED     = {"MuQ", "CLaMP3"}  # directly measured; others estimated
CACHE_SPEEDUP = 1.7
RAM_SAFETY = 0.80      # conservative post-OOM (5x5.2=26GB OOM'd at 24GB)
GPU_SAFETY = 0.85
CACHED_WIN_MARGIN = 1.20  # cached must beat uncached by >20% to override the
                          # simplicity preference for uncached
DEFAULT_RSS_UNC, DEFAULT_RSS_CAC, DEFAULT_GPU = 6.0, 11.0, 2.5

def mem_avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 1024 / 1024
    return 0.0

def gpu_free_gb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode()
    return int(out.splitlines()[0]) / 1024

def working_set_gb(cache_dir):
    return sum(os.path.getsize(p) for p in glob.glob(f"{cache_dir}/*.pt")) / 1024**3 * 0.90

def fd(a, b):
    return int(a // b) if b > 0 else 0

def recommend(enc, ram_b, gpu_b, max_layers):
    rss_u = RSS_UNCACHED.get(enc, DEFAULT_RSS_UNC)
    rss_c = RSS_CACHED.get(enc, DEFAULT_RSS_CAC)
    gpu = GPU_PROC.get(enc, DEFAULT_GPU)
    c_gpu = fd(gpu_b, gpu)
    c_unc = max(0, min(c_gpu, fd(ram_b, rss_u), max_layers))
    c_cac = max(0, min(c_gpu, fd(ram_b, rss_c), max_layers))
    t_unc, t_cac = c_unc * 1.0, c_cac * CACHE_SPEEDUP
    if c_cac >= 1 and t_cac > t_unc * CACHED_WIN_MARGIN:
        return "cached", c_cac, rss_c, t_cac, gpu
    return "uncached", c_unc, 0.0, t_unc, gpu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--layers", type=int, default=999)
    a = ap.parse_args()
    ram, gpu = mem_avail_gb(), gpu_free_gb()
    ram_b, gpu_b = ram * RAM_SAFETY, gpu * GPU_SAFETY
    print(f"MACHINE: RAM avail {ram:.1f}GB (budget {ram_b:.1f}) | GPU free {gpu:.1f}GB "
          f"(budget {gpu_b:.1f}) | RAM_SAFETY={RAM_SAFETY}")
    print(f"{'encoder':26}{'wset':>6}{'mode':>10}{'CONC':>5}{'cap':>9}{'gpu/proc':>9}{'meas?':>7}")
    for cdir in sorted(glob.glob(f"output/.emb_cache/*/{a.task}__*")):
        enc = os.path.basename(os.path.dirname(cdir))
        ws = working_set_gb(cdir)
        mode, c, cap, t, g = recommend(enc, ram_b, gpu_b, a.layers)
        meas = "yes" if enc in MEASURED else "est"
        cap_s = f"{cap:.1f}GB" if cap else "0(off)"
        gpu_tot = c * g
        warn = "  <- GPU-bound" if gpu_tot > gpu_b else ""
        print(f"{enc:26}{ws:5.1f}G{mode:>10}{c:5d}{cap_s:>9}{g:8.1f}G{meas:>7}{warn}")

if __name__ == "__main__":
    main()
