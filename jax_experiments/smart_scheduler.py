#!/usr/bin/env python3
"""Smart experiment scheduler: auto-detects hardware and maximizes parallelism.

Probes GPU VRAM, system RAM, and CPU cores to determine safe concurrency.
Maintains a job queue and launches new jobs as slots free up.

Usage:
    python jax_experiments/smart_scheduler.py [--device gpu] [--dry-run]
"""
import subprocess
import os
import sys
import time
import json
import re
import signal
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ============================================================
# Hardware detection
# ============================================================

def get_gpu_info() -> Dict:
    """Detect GPU total VRAM and current free VRAM (MiB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,name",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL
        ).strip()
        parts = [x.strip() for x in out.split(",")]
        return {
            "total_mib": int(parts[0]),
            "used_mib": int(parts[1]),
            "free_mib": int(parts[2]),
            "name": parts[3],
        }
    except Exception:
        return {"total_mib": 0, "used_mib": 0, "free_mib": 0, "name": "none"}


def get_ram_info() -> Dict:
    """Detect total and available system RAM (MiB)."""
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


def estimate_max_parallel(gpu: Dict, ram: Dict) -> int:
    """Estimate safe number of parallel training jobs.

    Based on empirical measurements:
    - Each job uses ~700-800 MiB GPU VRAM
    - Each job uses ~2000-2500 MiB system RAM (replay buffer + JIT cache)
    - Reserve 1500 MiB GPU for system/overhead
    - Reserve 4000 MiB RAM for OS/other processes
    """
    GPU_PER_JOB = 800   # MiB per job
    GPU_RESERVE = 1500   # MiB reserved
    RAM_PER_JOB = 2500   # MiB per job
    RAM_RESERVE = 4000   # MiB reserved for OS

    gpu_slots = max(1, (gpu["total_mib"] - GPU_RESERVE) // GPU_PER_JOB)
    ram_slots = max(1, (ram["total_mib"] - RAM_RESERVE) // RAM_PER_JOB)

    # Conservative: take the minimum, cap at 8 (diminishing returns beyond)
    max_jobs = min(gpu_slots, ram_slots, 8)
    return max(1, max_jobs)


# ============================================================
# Job definition
# ============================================================

@dataclass
class Job:
    name: str
    args: str  # full command-line args for train.py
    priority: int = 0  # lower = higher priority
    done: bool = False
    pid: Optional[int] = None
    log_file: str = ""


def is_job_done(name: str, save_root: str = "jax_experiments/results") -> bool:
    """Check if a job already has completed results."""
    result_dir = os.path.join(save_root, name, "logs", "eval_reward.npy")
    return os.path.exists(result_dir)


# ============================================================
# Job queue builder
# ============================================================

def build_job_queue(device: str) -> List[Job]:
    """Build the full job queue for P2a (sensitivity) + P2b (non-stationary)."""
    jobs = []

    SEED = 8
    MAX_ITERS = 2000
    SAVE_ROOT = "jax_experiments/results"
    BACKEND = "spring"

    COMMON = (f"--seed {SEED} --max_iters {MAX_ITERS} "
              f"--save_root {SAVE_ROOT} --backend {BACKEND} --device {device} "
              f"--stationary --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 "
              f"--beta_start -2.0 --beta_warmup 0.2")

    # Per-env base configs
    env_configs = {
        "HalfCheetah-v2": {
            "base": f"--algo resac --env HalfCheetah-v2 {COMMON} --adaptive_beta",
            "defaults": "--ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0",
        },
        "Ant-v2": {
            "base": f"--algo resac --env Ant-v2 {COMMON} --adaptive_beta",
            "defaults": "--ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5",
        },
    }

    # ── P2a: Sensitivity sweeps (priority=0, run first) ──
    sweeps = {
        "anchor": {
            "values": [0, 0.001, 0.01, 0.1, 0.5, 1.0],
            "arg_template": "--ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda {val} --independent_ratio {ratio}",
        },
        "betaend": {
            "values": [-2.0, -1.5, -1.0, -0.5, 0.0],
            "arg_template": "--ensemble_size 10 --beta_end {val} --ema_tau 0.005 --anchor_lambda {anchor} --independent_ratio {ratio}",
        },
        "ratio": {
            "values": [0.0, 0.25, 0.5, 0.75, 1.0],
            "arg_template": "--ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda {anchor} --independent_ratio {val}",
        },
        "K": {
            "values": [2, 5, 10, 20],
            "arg_template": "--ensemble_size {val} --beta_end -1.0 --ema_tau 0.005 --anchor_lambda {anchor} --independent_ratio {ratio}",
        },
        "ema": {
            "values": [0.0, 0.001, 0.005, 0.01, 0.05],
            "arg_template": "--ensemble_size 10 --beta_end -1.0 --ema_tau {val} --anchor_lambda {anchor} --independent_ratio {ratio}",
        },
    }

    env_specific = {
        "HalfCheetah-v2": {"anchor": 0.1, "ratio": 1.0},
        "Ant-v2": {"anchor": 0.01, "ratio": 0.5},
    }

    for sweep_name, sweep_cfg in sweeps.items():
        for val in sweep_cfg["values"]:
            for env in ["HalfCheetah-v2", "Ant-v2"]:
                name = f"sens_{sweep_name}_{val}_{env}_{SEED}"
                if is_job_done(name):
                    continue

                esp = env_specific[env]
                tmpl = sweep_cfg["arg_template"]

                # For the swept param, use val; for others use defaults
                if sweep_name == "anchor":
                    extra = tmpl.format(val=val, ratio=esp["ratio"])
                elif sweep_name == "betaend":
                    extra = tmpl.format(val=val, anchor=esp["anchor"], ratio=esp["ratio"])
                elif sweep_name == "ratio":
                    extra = tmpl.format(val=val, anchor=esp["anchor"])
                elif sweep_name == "K":
                    extra = tmpl.format(val=int(val), anchor=esp["anchor"], ratio=esp["ratio"])
                elif sweep_name == "ema":
                    extra = tmpl.format(val=val, anchor=esp["anchor"], ratio=esp["ratio"])

                args = f"{env_configs[env]['base']} {extra}"
                jobs.append(Job(name=name, args=args, priority=0))

    # ── P2b: Non-stationary (priority=1, run after sensitivity) ──
    NS_ARGS = "--varying_params gravity --task_num 40 --test_task_num 40 --log_scale_limit 3.0"
    NS_ENVS = ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]

    ns_algos = {
        "ns_sac": {"algo": "sac", "K": 2, "extra": ""},
        "ns_dsac": {"algo": "dsac", "K": 10, "extra": ""},
    }

    for prefix, cfg in ns_algos.items():
        for env in NS_ENVS:
            name = f"{prefix}_{env}_{SEED}"
            if is_job_done(name):
                continue
            args = (f"--algo {cfg['algo']} --env {env} --seed {SEED} --max_iters {MAX_ITERS} "
                    f"--save_root {SAVE_ROOT} --backend {BACKEND} --device {device} "
                    f"--ensemble_size {cfg['K']} {NS_ARGS}")
            jobs.append(Job(name=name, args=args, priority=1))

    # NS RE-SAC (per-env config)
    ns_resac_args = {
        "Hopper-v2": "--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0",
        "Walker2d-v2": "--adaptive_beta --beta_end -0.5 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 1.0",
        "HalfCheetah-v2": "--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0",
        "Ant-v2": "--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5",
    }
    for env in NS_ENVS:
        name = f"ns_resac_{env}_{SEED}"
        if is_job_done(name):
            continue
        args = (f"--algo resac --env {env} --seed {SEED} --max_iters {MAX_ITERS} "
                f"--save_root {SAVE_ROOT} --backend {BACKEND} --device {device} "
                f"--ensemble_size 10 "
                f"--beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 "
                f"--beta_start -2.0 --beta_warmup 0.2 "
                f"{ns_resac_args[env]} {NS_ARGS}")
        jobs.append(Job(name=name, args=args, priority=1))

    # Sort by priority
    jobs.sort(key=lambda j: j.priority)
    return jobs


# ============================================================
# Scheduler
# ============================================================

class Scheduler:
    def __init__(self, max_parallel: int, device: str, log_dir: str = "jax_experiments/logs"):
        self.max_parallel = max_parallel
        self.device = device
        self.log_dir = log_dir
        self.running: List[Job] = []
        self.completed = 0
        self.failed = 0
        self._shutdown = False

        os.makedirs(log_dir, exist_ok=True)
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n⚠ Signal {signum} received. Killing {len(self.running)} running jobs...")
        self._shutdown = True
        for job in self.running:
            if job.pid:
                try:
                    os.kill(job.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        sys.exit(1)

    def _get_current_load(self) -> Tuple[int, int]:
        """Get current GPU used MiB and RAM used MiB by our jobs."""
        gpu = get_gpu_info()
        ram = get_ram_info()
        return gpu["used_mib"], ram["total_mib"] - ram["available_mib"]

    def _can_launch(self) -> bool:
        """Check if we can launch another job based on current resource usage."""
        if len(self.running) >= self.max_parallel:
            return False

        gpu = get_gpu_info()
        ram = get_ram_info()

        # Check headroom: need at least 800 MiB GPU and 2500 MiB RAM free
        gpu_ok = gpu["free_mib"] > 1000
        ram_ok = ram["available_mib"] > 3000

        return gpu_ok and ram_ok

    def _launch(self, job: Job):
        """Launch a single training job."""
        job.log_file = os.path.join(self.log_dir, f"{job.name}.log")

        cmd = f"python -u -m jax_experiments.train {job.args} --run_name {job.name}"
        log_f = open(job.log_file, "w")

        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"

        proc = subprocess.Popen(
            cmd.split(), stdout=log_f, stderr=subprocess.STDOUT,
            env=env, cwd=os.getcwd()
        )
        job.pid = proc.pid
        job._proc = proc
        job._log_f = log_f
        self.running.append(job)

    def _poll(self):
        """Check running jobs, remove finished ones."""
        still_running = []
        for job in self.running:
            ret = job._proc.poll()
            if ret is None:
                still_running.append(job)
            else:
                job._log_f.close()
                job.done = True
                if ret == 0:
                    self.completed += 1
                    print(f"  ✓ {job.name} done")
                else:
                    self.failed += 1
                    print(f"  ✗ {job.name} FAILED (exit {ret})")
        self.running = still_running

    def run(self, jobs: List[Job]):
        """Run all jobs with adaptive parallelism."""
        total = len(jobs)
        queue = list(jobs)  # copy

        print(f"\n{'='*60}")
        print(f"  Smart Scheduler")
        print(f"  Total jobs: {total}")
        print(f"  Max parallel: {self.max_parallel}")
        print(f"  Device: {self.device}")

        gpu = get_gpu_info()
        ram = get_ram_info()
        print(f"  GPU: {gpu['name']} ({gpu['total_mib']} MiB)")
        print(f"  RAM: {ram['total_mib']} MiB total, {ram['available_mib']} MiB available")

        # Time estimate
        avg_hours_per_job = 5.0  # conservative average
        est_hours = (total * avg_hours_per_job) / self.max_parallel
        print(f"  Estimated wall time: ~{est_hours:.0f}h ({total} jobs / {self.max_parallel} parallel)")
        print(f"{'='*60}\n")

        start_time = time.time()

        while queue or self.running:
            if self._shutdown:
                break

            self._poll()

            # Launch new jobs if we have capacity
            launched = 0
            while queue and self._can_launch():
                job = queue.pop(0)
                # Double-check not already done (in case of restart)
                if is_job_done(job.name):
                    self.completed += 1
                    print(f"  ⊘ {job.name} already done, skipping")
                    continue
                self._launch(job)
                launched += 1
                time.sleep(3)  # small delay between launches for GPU alloc

            if launched > 0:
                done_total = self.completed + self.failed
                elapsed = (time.time() - start_time) / 3600
                print(f"  [{time.strftime('%H:%M')}] Running: {len(self.running)} | "
                      f"Done: {done_total}/{total} | "
                      f"Elapsed: {elapsed:.1f}h")

            time.sleep(30)  # poll every 30s

        elapsed = (time.time() - start_time) / 3600
        print(f"\n{'='*60}")
        print(f"  Scheduler finished in {elapsed:.1f}h")
        print(f"  Completed: {self.completed}  Failed: {self.failed}")
        print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Smart experiment scheduler")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--max-parallel", type=int, default=0,
                        help="Override auto-detected parallelism (0=auto)")
    args = parser.parse_args()

    gpu = get_gpu_info()
    ram = get_ram_info()
    cpus = get_cpu_count()

    if args.max_parallel > 0:
        max_p = args.max_parallel
    else:
        max_p = estimate_max_parallel(gpu, ram)

    print(f"Hardware detected:")
    print(f"  GPU: {gpu['name']} — {gpu['total_mib']} MiB total, {gpu['free_mib']} MiB free")
    print(f"  RAM: {ram['total_mib']} MiB total, {ram['available_mib']} MiB available")
    print(f"  CPU: {cpus} cores")
    print(f"  → Max parallel jobs: {max_p}")
    print()

    jobs = build_job_queue(args.device)

    if not jobs:
        print("All experiments already completed!")
        return

    # Group by priority for display
    p0_jobs = [j for j in jobs if j.priority == 0]
    p1_jobs = [j for j in jobs if j.priority == 1]

    print(f"Job queue ({len(jobs)} remaining):")
    if p0_jobs:
        print(f"  P2a Sensitivity: {len(p0_jobs)} jobs")
    if p1_jobs:
        print(f"  P2b Non-stationary: {len(p1_jobs)} jobs")

    if args.dry_run:
        print("\n--- DRY RUN: Job list ---")
        for i, j in enumerate(jobs):
            print(f"  [{i+1:3d}] P{j.priority} {j.name}")
        avg_h = 5.0
        print(f"\nEstimated wall time: ~{len(jobs) * avg_h / max_p:.0f}h "
              f"({len(jobs)} × {avg_h:.0f}h / {max_p} parallel)")
        return

    # Setup conda env
    print("Activating conda environment...")

    # Set up NVIDIA libs for GPU
    if args.device == "gpu":
        try:
            import nvidia
            nvidia_base = os.path.dirname(nvidia.__file__)
            lib_dirs = []
            for subdir in os.listdir(nvidia_base):
                lib_path = os.path.join(nvidia_base, subdir, "lib")
                if os.path.isdir(lib_path):
                    lib_dirs.append(lib_path)
            if lib_dirs:
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        except ImportError:
            pass

    scheduler = Scheduler(max_parallel=max_p, device=args.device)
    scheduler.run(jobs)


if __name__ == "__main__":
    main()
