#!/bin/bash
# Run REDQ, SAC-N, TQC sequentially (4 envs parallel per algo)
# Usage: nohup bash jax_experiments/run_baselines_serial.sh gpu &

DEVICE=${1:-gpu}
cd /home/erzhu419/mine_code/RE-SAC

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
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
fi

ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")
SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"

run_algo_batch() {
    local ALGO=$1
    local K=$2
    echo ""
    echo "============================================================"
    echo "  $(date): Starting $ALGO (K=$K)"
    echo "============================================================"
    PIDS=()
    for ENV in "${ENVS[@]}"; do
        NAME="${ALGO}_${ENV}_${SEED}"
        LOG="${LOG_DIR}/${NAME}.log"
        echo "  $ENV -> $NAME"
        python -u -m jax_experiments.train \
            --algo "$ALGO" --env "$ENV" --seed $SEED --max_iters $MAX_ITERS \
            --save_root "$SAVE_ROOT" --run_name "$NAME" \
            --backend "$BACKEND" --device "$DEVICE" \
            --ensemble_size $K --stationary \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
    echo "  4 jobs launched for $ALGO. Waiting..."
    FAILED=0
    for pid in "${PIDS[@]}"; do wait $pid || FAILED=$((FAILED+1)); done
    [ $FAILED -eq 0 ] && echo "$(date): $ALGO done!" || echo "$(date): $ALGO: $FAILED failed"
}

run_algo_batch redq 10
run_algo_batch sacn 10
run_algo_batch tqc 5

echo ""
echo "$(date): All baselines complete!"
