# RE-SAC server deployment

## TL;DR (one-liner)

```bash
bash jax_experiments/run_server.sh
```

That's it. The script will:

1. Create/reuse conda env `resac-jax` (Python 3.10 + pinned JAX CUDA 12).
2. Probe every visible GPU each cycle; assign each new job to the GPU with
   the most free VRAM via `CUDA_VISIBLE_DEVICES`.
3. Auto-cap parallelism by **CPU cores, host RAM, and per-GPU free VRAM**.
4. Write per-job logs to `jax_experiments/logs/<job>.log`, final metrics to
   `jax_experiments/results/<job>/logs/eval_reward.npy`.
5. Safely resume if killed — finished jobs are auto-detected and skipped.

## Preview without launching

```bash
bash jax_experiments/run_server.sh --dry-run
```

## Background run

```bash
nohup bash jax_experiments/run_server.sh > run.out 2>&1 &
tail -f jax_experiments/logs/scheduler.log
```

## Tuning (rarely needed)

| Flag | Default | Meaning |
|------|---------|---------|
| `--per-job-vram` | 1500 | MiB VRAM reserved per job. Lower if jobs OOM rarely; higher if they OOM often. |
| `--per-job-ram`  | 2500 | MiB host RAM per job. |
| `--gpu-reserve`  | 800  | MiB VRAM kept free on each GPU for other users. |
| `--ram-reserve`  | 4000 | MiB host RAM kept free for the OS. |
| `--cpu-reserve`  | 2    | CPU cores never used. |

Example: on a heavily-shared 12 GB GPU, raise the reserve:

```bash
bash jax_experiments/run_server.sh --gpu-reserve 3000
```

## What gets run

62 jobs (if no partial results are present):

- **P2a sensitivity sweeps** (50 jobs): anchor / beta_end / independent_ratio /
  ensemble_size / ema_tau on HalfCheetah-v2 + Ant-v2.
- **P2b non-stationary** (12 jobs): RE-SAC / SAC / DSAC on
  Hopper / Walker2d / HalfCheetah / Ant.

Full list: `bash jax_experiments/run_server.sh --dry-run`.

Each job is ~5h wall-clock on a 4060 laptop; expect ~2–3h on a proper server
GPU. With 2 GPUs × ~6 jobs each = ~12 parallel, full run is **≈30 h**.

## Pulling results back

After it finishes, everything needed for paper figures lives under:

```
jax_experiments/results/
jax_experiments/logs/
```

Rsync these directories back to the laptop:

```bash
rsync -a USER@SERVER:PATH/RE-SAC/jax_experiments/results/   ./jax_experiments/results/
rsync -a USER@SERVER:PATH/RE-SAC/jax_experiments/logs/      ./jax_experiments/logs/
```

## Files this bundle adds / needs

- `jax_experiments/multi_gpu_scheduler.py` — the new multi-GPU scheduler
  (built on top of the existing single-GPU `smart_scheduler.py`).
- `jax_experiments/server_setup.sh` — idempotent conda-env bootstrap.
- `jax_experiments/run_server.sh` — one-liner entry point.
- `jax_experiments/requirements_server.txt` — pinned deps.
