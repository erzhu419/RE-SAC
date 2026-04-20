#!/bin/bash
# ============================================================
# New baselines: REDQ + SAC-N + TQC on 4 envs
# Usage: bash jax_experiments/run_new_baselines.sh <algo> [gpu|cpu]
#   algo: redq | sacn | tqc | all
# ============================================================

ALGO=${1:?Usage: $0 <redq|sacn|tqc|all> [gpu|cpu]}
DEVICE=${2:-gpu}
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

run_algo() {
    local A=$1
    local K=10
    [ "$A" = "tqc" ] && K=5  # TQC default N=5

    echo "--- $A (K=$K) ---"
    for env in "${ENVS[@]}"; do
        local NAME="${A}_${env}_${SEED}"
        local LOG="${LOG_DIR}/${NAME}.log"
        echo "  $env -> $NAME"
        python -u -m jax_experiments.train \
            --algo "$A" --env "$env" --seed $SEED --max_iters $MAX_ITERS \
            --save_root "$SAVE_ROOT" --run_name "$NAME" \
            --backend "$BACKEND" --device "$DEVICE" \
            --ensemble_size $K --stationary \
            > "$LOG" 2>&1 &
        PIDS+=($!)
    done
}

PIDS=()

if [ "$ALGO" = "all" ]; then
    run_algo redq
    run_algo sacn
    run_algo tqc
else
    run_algo "$ALGO"
fi

echo ""
echo "${#PIDS[@]} jobs launched."
echo "Monitor: tail -f ${LOG_DIR}/${ALGO}_*_${SEED}.log"

FAILED=0
for pid in "${PIDS[@]}"; do wait $pid || FAILED=$((FAILED+1)); done
[ $FAILED -eq 0 ] && echo "All done!" || echo "$FAILED failed."
