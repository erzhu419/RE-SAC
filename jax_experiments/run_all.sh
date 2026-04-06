#!/bin/bash
# JAX Experiments: RE-SAC + SAC/TD3/DSAC baselines
# ALL experiments use STATIONARY (classic MuJoCo) environments.
#
# Phase 1: SAC + TD3 + DSAC → 3 algos × 4 envs = 12, max 9 concurrent
# Phase 2: RE-SAC (ensemble_size=10) → 4 envs, sequential
#
# Usage: bash jax_experiments/run_all.sh [cpu|gpu]
# Monitor: tail -f jax_experiments/logs/*.log
# Kill all: pkill -f "jax_experiments.train"

# NOTE: no 'set -e' — individual failures should not kill the suite

# Device: gpu (default) or cpu
DEVICE=${1:-gpu}

SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"

ENVS=("Hopper-v2" "HalfCheetah-v2" "Walker2d-v2" "Ant-v2")
BASELINES=("sac" "td3" "dsac")

LOG_DIR="jax_experiments/logs"
mkdir -p "$LOG_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate jax-rl

# For GPU: setup CUDA libs (LD_LIBRARY_PATH must be set before process starts)
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

declare -a ACTIVE_PIDS=()

reap_pids() {
    local alive=()
    for pid in "${ACTIVE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done
    ACTIVE_PIDS=("${alive[@]}")
}

wait_for_slot() {
    local max=$1
    while true; do
        reap_pids
        if [ "${#ACTIVE_PIDS[@]}" -lt "$max" ]; then
            break
        fi
        sleep 3
    done
}

wait_all() {
    while true; do
        reap_pids
        if [ "${#ACTIVE_PIDS[@]}" -eq 0 ]; then
            break
        fi
        sleep 5
    done
}

BASELINE_TOTAL=$(( ${#BASELINES[@]} * ${#ENVS[@]} ))
RESAC_TOTAL=${#ENVS[@]}
TOTAL=$(( BASELINE_TOTAL + RESAC_TOTAL ))
COUNT=0

echo "=============================================="
echo "  JAX Experiments: ${TOTAL} total  [STATIONARY]"
echo "  Phase 1: SAC/TD3/DSAC baselines (${BASELINE_TOTAL}, max 9 concurrent)"
echo "  Phase 2: RE-SAC ensemble_size=10 (${RESAC_TOTAL}, sequential)"
echo "  Seed: $SEED, Max iters: $MAX_ITERS, Backend: $BACKEND"
echo "  Device: $DEVICE"
echo "=============================================="

# ========== Phase 1: Baselines (SAC + TD3 + DSAC) ==========
echo ""
echo ">>> Phase 1: SAC/TD3/DSAC baselines (max 9 concurrent)..."

for algo in "${BASELINES[@]}"; do
    for env in "${ENVS[@]}"; do
        run_name="${algo}_${env}_${SEED}"
        log_file="${LOG_DIR}/${run_name}.log"
        COUNT=$((COUNT + 1))

        wait_for_slot 9
        echo "[${COUNT}/${TOTAL}] ${algo} ${env} (active: ${#ACTIVE_PIDS[@]}/9) — $(date '+%H:%M:%S')"

        python -u -m jax_experiments.train \
            --algo "$algo" \
            --env "$env" \
            --seed $SEED \
            --max_iters $MAX_ITERS \
            --save_root "$SAVE_ROOT" \
            --run_name "$run_name" \
            --backend "$BACKEND" \
            --device "$DEVICE" \
            --ensemble_size 2 \
            --stationary \
            > "$log_file" 2>&1 &
        ACTIVE_PIDS+=($!)
    done
done

echo ""
echo ">>> Waiting for all baselines to complete..."
wait_all
echo ">>> All baselines finished! — $(date '+%H:%M:%S')"

# ========== Phase 2: RE-SAC (GPU-heavy) ==========
echo ""
echo ">>> Phase 2: RE-SAC (ensemble_size=10, sequential, STATIONARY)..."

for env in "${ENVS[@]}"; do
    run_name="resac_${env}_${SEED}"
    log_file="${LOG_DIR}/${run_name}.log"
    COUNT=$((COUNT + 1))

    echo "[${COUNT}/${TOTAL}] resac ${env} — $(date '+%H:%M:%S')"

    python -u -m jax_experiments.train \
        --algo resac \
        --env "$env" \
        --seed $SEED \
        --max_iters $MAX_ITERS \
        --save_root "$SAVE_ROOT" \
        --run_name "$run_name" \
        --backend "$BACKEND" \
        --device "$DEVICE" \
        --ensemble_size 10 \
        --stationary \
        > "$log_file" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "  ⚠️  FAILED on $env (exit $EXIT_CODE)"
    else
        echo "  ✅ $env — $(date '+%H:%M:%S')"
    fi
done

echo "=============================================="
echo "  All $TOTAL experiments finished!"
echo "  Results: $SAVE_ROOT/"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
