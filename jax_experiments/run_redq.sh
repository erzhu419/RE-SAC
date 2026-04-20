#!/bin/bash
# REDQ baseline: 4 envs, seed=8
# Usage: bash jax_experiments/run_redq.sh [gpu|cpu]

DEVICE=${1:-gpu}
SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
LOG_DIR="jax_experiments/logs"
mkdir -p "$LOG_DIR"

eval "$(conda shell.bash hook)"
conda activate jax-rl

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

ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")

echo "============================================================"
echo "  REDQ baseline (N=10, M=2)"
echo "============================================================"

PIDS=()
for env in "${ENVS[@]}"; do
    RUN_NAME="redq_${env}_${SEED}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    echo "Starting: $env -> $RUN_NAME"
    python -u -m jax_experiments.train \
        --algo redq --env "$env" --seed $SEED --max_iters $MAX_ITERS \
        --save_root "$SAVE_ROOT" --run_name "$RUN_NAME" \
        --backend "$BACKEND" --device "$DEVICE" \
        --ensemble_size 10 --stationary \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo "4 jobs launched. PIDs: ${PIDS[*]}"
echo "Monitor: tail -f jax_experiments/logs/redq_*_${SEED}.log"

FAILED=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED + 1))
done

[ $FAILED -eq 0 ] && echo "All done!" || echo "$FAILED failed."
