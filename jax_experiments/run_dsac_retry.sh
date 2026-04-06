#!/bin/bash
# Usage: bash jax_experiments/run_dsac_retry.sh [cpu|gpu]
DEVICE=${1:-gpu}
eval "$(conda shell.bash hook)"
conda activate jax-rl

# For GPU: setup CUDA libs
if [ "$DEVICE" = "gpu" ]; then
    NVIDIA_BASE=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null || true)
    if [ -n "$NVIDIA_BASE" ]; then
        for d in "$NVIDIA_BASE"/*/lib; do
            [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH}"
        done
    fi
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
fi
for env in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2; do
    nohup python -u -m jax_experiments.train --algo dsac --env $env --seed 8 --max_iters 2000 --save_root jax_experiments/results --run_name dsac_${env}_8 --backend spring --device $DEVICE --ensemble_size 2 > jax_experiments/logs/dsac_${env}_8.log 2>&1 &
done
