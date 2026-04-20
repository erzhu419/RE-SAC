#!/usr/bin/env python3
"""Multi-GPU smart scheduler for RE-SAC experiments.

Extends smart_scheduler with per-GPU resource tracking:
- Probes each visible GPU's free VRAM every cycle
- Assigns each new job to the GPU with the most free memory
- Caps total parallelism by min(per-GPU cap sum, RAM cap, CPU-based cap)
- Launches child processes with CUDA_VISIBLE_DEVICES set to a single GPU

Designed for shared servers where other users may grab VRAM unpredictably.

Usage:
    python jax_experiments/multi_gpu_scheduler.py
    python jax_experiments/multi_gpu_scheduler.py --dry-run
    python jax_experiments/multi_gpu_scheduler.py --per-job-vram 1500
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from jax_experiments.smart_scheduler import Job, build_job_queue, is_job_done


# ============================================================
# Per-GPU hardware detection
# ============================================================

def get_per_gpu_info() -> List[Dict]:
    """Return [{index, total_mib, used_mib, free_mib, name}, ...] for each GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,name",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return []

    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({
            "index": int(parts[0]),
            "total_mib": int(parts[1]),
            "used_mib": int(parts[2]),
            "free_mib": int(parts[3]),
            "name": parts[4],
        })
    return gpus


def get_ram_info() -> Dict:
    try:
        with open("/proc/meminfo") as f:
            info = f.read()
        total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1)) // 1024
        avail = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) // 1024
        return {"total_mib": total, "available_mib": avail}
    except Exception:
        return {"total_mib": 0, "available_mib": 0}


def get_cpu_count() -> int:
    return os.cpu_count() or 4


# ============================================================
# Scheduler
# ============================================================

class MultiGPUScheduler:
    """Tracks per-GPU running-job counts and launches new jobs on the GPU
    with the largest free-memory headroom."""

    def __init__(
        self,
        per_job_vram_mib: int,
        per_job_ram_mib: int,
        gpu_reserve_mib: int,
        ram_reserve_mib: int,
        cpu_reserve: int,
        log_dir: str = "jax_experiments/logs",
    ):
        self.per_job_vram = per_job_vram_mib
        self.per_job_ram = per_job_ram_mib
        self.gpu_reserve = gpu_reserve_mib
        self.ram_reserve = ram_reserve_mib
        self.cpu_reserve = cpu_reserve
        self.log_dir = log_dir

        self.running: List[Job] = []
        self.gpu_counts: Dict[int, int] = {}  # gpu_idx -> #running jobs
        self.completed = 0
        self.failed = 0
        self._shutdown = False

        os.makedirs(log_dir, exist_ok=True)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    # ---- signals ----
    def _handle_signal(self, signum, frame):
        print(f"\n[!] Signal {signum} received. Killing {len(self.running)} running jobs...")
        self._shutdown = True
        for job in self.running:
            if job.pid:
                try:
                    os.kill(job.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        sys.exit(1)

    # ---- capacity planning ----
    def _cpu_cap(self) -> int:
        return max(1, get_cpu_count() - self.cpu_reserve)

    def _ram_cap(self) -> int:
        ram = get_ram_info()
        usable = max(0, ram["available_mib"] - self.ram_reserve)
        return max(0, usable // self.per_job_ram)

    def _pick_gpu(self) -> Optional[int]:
        """Return the GPU index that can fit one more job.

        JAX lazy-allocates: a just-launched job won't show in nvidia-smi for
        5-15s (Python import + JIT compile). If we only trust nvidia-smi, we'll
        happily launch 8 jobs into an empty-looking card and then OOM.

        Fix: take the pessimistic min of (reported_free) and (total minus
        budget-based projection of already-running-here). Once jobs finish
        allocating, both numbers agree."""
        gpus = get_per_gpu_info()
        if not gpus:
            return None
        candidates = []
        for g in gpus:
            running_here = self.gpu_counts.get(g["index"], 0)
            reported_free = g["free_mib"]
            projected_free = g["total_mib"] - self.per_job_vram * running_here
            effective_free = min(reported_free, projected_free)
            needed = self.per_job_vram + self.gpu_reserve
            if effective_free >= needed:
                candidates.append((effective_free, g["index"]))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (-t[0], self.gpu_counts.get(t[1], 0)))
        return candidates[0][1]

    def _launch(self, job: Job, gpu_idx: int):
        job.log_file = os.path.join(self.log_dir, f"{job.name}.log")
        cmd = f"python -u -m jax_experiments.train {job.args} --run_name {job.name}"
        log_f = open(job.log_file, "w")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        # Cap each child's XLA pool to ~per_job_vram / gpu_total.
        # On a 12 GB card with per_job_vram=1500 -> ~0.13 fraction.
        gpus = get_per_gpu_info()
        total_mib = next((g["total_mib"] for g in gpus if g["index"] == gpu_idx), 12000)
        frac = max(0.08, min(0.5, self.per_job_vram / max(total_mib, 1)))
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{frac:.2f}"

        proc = subprocess.Popen(
            cmd.split(), stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd()
        )
        job.pid = proc.pid
        job._proc = proc
        job._log_f = log_f
        job._gpu = gpu_idx
        self.running.append(job)
        self.gpu_counts[gpu_idx] = self.gpu_counts.get(gpu_idx, 0) + 1

    def _poll(self):
        still = []
        for job in self.running:
            ret = job._proc.poll()
            if ret is None:
                still.append(job)
            else:
                job._log_f.close()
                job.done = True
                self.gpu_counts[job._gpu] = max(0, self.gpu_counts.get(job._gpu, 1) - 1)
                if ret == 0:
                    self.completed += 1
                    print(f"  [ok] {job.name} done (gpu{job._gpu})")
                else:
                    self.failed += 1
                    print(f"  [fail] {job.name} exit={ret} (gpu{job._gpu}, log={job.log_file})")
        self.running = still

    # ---- main loop ----
    def run(self, jobs: List[Job]):
        total = len(jobs)
        queue = list(jobs)

        gpus = get_per_gpu_info()
        ram = get_ram_info()
        print(f"\n{'='*62}")
        print(f"  Multi-GPU Smart Scheduler")
        print(f"  Total jobs: {total}")
        print(f"  CPU cores: {get_cpu_count()} (cap={self._cpu_cap()})")
        print(f"  RAM: {ram['total_mib']} MiB total, {ram['available_mib']} MiB avail")
        for g in gpus:
            print(f"  GPU{g['index']}: {g['name']} — {g['total_mib']} MiB total, {g['free_mib']} MiB free")
        print(f"  Per-job budget: {self.per_job_vram} MiB VRAM, {self.per_job_ram} MiB RAM")
        print(f"{'='*62}\n")

        start = time.time()

        while queue or self.running:
            if self._shutdown:
                break

            self._poll()

            # Launch while we have global capacity and some GPU has room.
            while queue:
                # Global caps
                if len(self.running) >= self._cpu_cap():
                    break
                if len(self.running) >= self._ram_cap() + len(self.running) and self._ram_cap() <= 0:
                    break
                if len(self.running) - self.completed - self.failed >= self._ram_cap():
                    # conservative RAM check: current running can't exceed ram cap
                    if self._ram_cap() > 0 and len(self.running) >= self._ram_cap():
                        break

                gpu_idx = self._pick_gpu()
                if gpu_idx is None:
                    break  # no GPU has headroom right now; wait a cycle

                job = queue.pop(0)
                if is_job_done(job.name):
                    self.completed += 1
                    print(f"  [skip] {job.name} already done")
                    continue
                self._launch(job, gpu_idx)
                time.sleep(4)  # let JAX allocate before probing again

            done_total = self.completed + self.failed
            elapsed_h = (time.time() - start) / 3600
            per_gpu_str = ", ".join(
                f"gpu{i}:{self.gpu_counts.get(i, 0)}" for i in sorted({g['index'] for g in gpus})
            )
            print(
                f"  [{time.strftime('%H:%M:%S')}] running={len(self.running)} "
                f"[{per_gpu_str}] done={done_total}/{total} elapsed={elapsed_h:.2f}h"
            )

            time.sleep(45)

        elapsed_h = (time.time() - start) / 3600
        print(f"\n{'='*62}")
        print(f"  Finished in {elapsed_h:.2f}h — completed={self.completed} failed={self.failed}")
        print(f"{'='*62}")


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Multi-GPU scheduler for RE-SAC experiments")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--per-job-vram", type=int, default=1500,
                   help="Budget per job in MiB (default 1500).")
    p.add_argument("--per-job-ram", type=int, default=2500,
                   help="Host RAM budget per job in MiB (default 2500).")
    p.add_argument("--gpu-reserve", type=int, default=800,
                   help="Keep this much VRAM free as safety margin (default 800).")
    p.add_argument("--ram-reserve", type=int, default=4000,
                   help="Keep this much RAM free for OS (default 4000).")
    p.add_argument("--cpu-reserve", type=int, default=2,
                   help="Leave this many CPU cores unused (default 2).")
    args = p.parse_args()

    gpus = get_per_gpu_info()
    ram = get_ram_info()
    print("Hardware detected:")
    for g in gpus:
        print(f"  GPU{g['index']}: {g['name']} — {g['total_mib']} MiB total, {g['free_mib']} MiB free")
    print(f"  RAM: {ram['total_mib']} MiB total, {ram['available_mib']} MiB available")
    print(f"  CPU: {get_cpu_count()} cores")

    jobs = build_job_queue(args.device)
    if not jobs:
        print("All experiments already completed!")
        return

    p0 = [j for j in jobs if j.priority == 0]
    p1 = [j for j in jobs if j.priority == 1]
    print(f"\nJob queue ({len(jobs)} remaining): P2a={len(p0)}, P2b={len(p1)}")

    if args.dry_run:
        print("\n--- DRY RUN: Job list ---")
        for i, j in enumerate(jobs):
            print(f"  [{i+1:3d}] P{j.priority} {j.name}")
        # Rough estimate: sum of per-GPU capacities
        gpu_caps = [
            max(0, (g["free_mib"] - args.gpu_reserve) // args.per_job_vram) for g in gpus
        ]
        total_cap = sum(gpu_caps) if gpu_caps else 1
        total_cap = max(1, min(total_cap, get_cpu_count() - args.cpu_reserve))
        avg_h = 5.0
        print(f"\nEstimated wall time: ~{len(jobs) * avg_h / total_cap:.1f}h "
              f"({len(jobs)} jobs / {total_cap} parallel slots)")
        return

    sched = MultiGPUScheduler(
        per_job_vram_mib=args.per_job_vram,
        per_job_ram_mib=args.per_job_ram,
        gpu_reserve_mib=args.gpu_reserve,
        ram_reserve_mib=args.ram_reserve,
        cpu_reserve=args.cpu_reserve,
    )
    sched.run(jobs)


if __name__ == "__main__":
    main()
