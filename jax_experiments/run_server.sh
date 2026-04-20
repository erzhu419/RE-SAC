#!/usr/bin/env bash
# One-liner entry point for the RE-SAC experiment bundle on the server.
#
#   Usage (from the project root, i.e. the parent of jax_experiments/):
#     bash jax_experiments/run_server.sh
#     bash jax_experiments/run_server.sh --dry-run           # show queue and exit
#     bash jax_experiments/run_server.sh --per-job-vram 2000 # override budget
#
# What it does:
#   1) Sets up (or reuses) conda env `resac-jax` with pinned deps.
#   2) Launches the multi-GPU scheduler:
#        - probes every visible GPU each cycle
#        - assigns each new job to the GPU with the most free VRAM
#        - caps parallelism by CPU + RAM + per-GPU free memory
#   3) Writes per-job logs to jax_experiments/logs/<job>.log
#   4) Writes results to jax_experiments/results/<job>/...
#      (already-completed jobs are detected and skipped, so re-running is safe)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
cd "${ROOT}"

ENV_NAME="${RESAC_ENV:-resac-jax}"

# 1) bootstrap
bash "${HERE}/server_setup.sh"

# 2) activate env
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -x "/opt/conda/bin/conda" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

# 3) ensure jax_experiments package is importable from project root
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# 4) run the scheduler (tee'd so nohup+tail both work)
mkdir -p jax_experiments/logs
LOG="jax_experiments/logs/scheduler.log"
echo "[*] Launching multi-GPU scheduler. Log: ${LOG}"
echo "[*] Tip: re-running is safe — finished jobs are auto-skipped."
python -u -m jax_experiments.multi_gpu_scheduler "$@" 2>&1 | tee -a "${LOG}"
