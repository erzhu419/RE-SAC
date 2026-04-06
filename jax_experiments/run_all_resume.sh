#!/bin/bash
# JAX Experiments: RE-SAC + SAC/TD3/DSAC baselines — RESUMABLE VERSION
# ALL experiments use STATIONARY (classic MuJoCo) environments.
#
# This script adds --resume to every training command:
#   - If a checkpoint exists and is complete → automatically skipped
#   - If a checkpoint exists but is incomplete → resumes from last checkpoint
#   - If no checkpoint exists → trains from scratch
#
# Phase 1: SAC + TD3 + DSAC → 3 algos × 4 envs = 12, max 9 concurrent
# Phase 2: RE-SAC (ensemble_size=10) → 4 envs, sequential
#
# Usage: bash jax_experiments/run_all_resume.sh [cpu|gpu]
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

# --- Check what's already done ---
echo ""
echo "Scanning existing progress..."
ALREADY_DONE=0
NEEDS_RESUME=0
NEEDS_FRESH=0

check_status() {
    local run_name=$1
    local ckpt_dir="${SAVE_ROOT}/${run_name}/checkpoints"
    local log_dir="${SAVE_ROOT}/${run_name}/logs"
    
    if [ -f "${ckpt_dir}/train_state.pkl" ]; then
        local ckpt_iter=$(python3 -c "
import pickle
with open('${ckpt_dir}/train_state.pkl', 'rb') as f:
    s = pickle.load(f)
print(s['iteration'])
" 2>/dev/null || echo "-1")
        if [ "$ckpt_iter" -ge $((MAX_ITERS - 1)) ]; then
            echo "  ✅ $run_name: DONE (iter $ckpt_iter)"
            ALREADY_DONE=$((ALREADY_DONE + 1))
            return 0  # done
        else
            echo "  🔄 $run_name: RESUME from iter $ckpt_iter"
            NEEDS_RESUME=$((NEEDS_RESUME + 1))
            return 1  # needs resume
        fi
    elif [ -f "${log_dir}/iteration.npy" ]; then
        local log_iter=$(python3 -c "
import numpy as np
a = np.load('${log_dir}/iteration.npy')
print(int(a[-1]) if len(a)>0 else -1)
" 2>/dev/null || echo "-1")
        if [ "$log_iter" -ge $((MAX_ITERS - 1)) ]; then
            echo "  ✅ $run_name: DONE (logs show iter $log_iter)"
            ALREADY_DONE=$((ALREADY_DONE + 1))
            return 0  # done
        else
            echo "  🆕 $run_name: from scratch (logs had iter $log_iter but no checkpoint)"
            NEEDS_FRESH=$((NEEDS_FRESH + 1))
            return 1  # needs training
        fi
    else
        echo "  🆕 $run_name: from scratch"
        NEEDS_FRESH=$((NEEDS_FRESH + 1))
        return 1  # needs training
    fi
}

# Pre-scan all experiments
for algo in "${BASELINES[@]}"; do
    for env in "${ENVS[@]}"; do
        check_status "${algo}_${env}_${SEED}"
    done
done
for env in "${ENVS[@]}"; do
    check_status "resac_${env}_${SEED}"
done

TOTAL_REMAINING=$((NEEDS_RESUME + NEEDS_FRESH))
TOTAL=$((ALREADY_DONE + TOTAL_REMAINING))

echo ""
echo "=============================================="
echo "  JAX Experiments: ${TOTAL} total  [STATIONARY]"
echo "  Already done:  $ALREADY_DONE"
echo "  Need resume:   $NEEDS_RESUME"
echo "  Need fresh:    $NEEDS_FRESH"
echo "  Remaining:     $TOTAL_REMAINING"
echo "  Device: $DEVICE"
echo "=============================================="

if [ $TOTAL_REMAINING -eq 0 ]; then
    echo ""
    echo "  All experiments already completed! Nothing to do."
    exit 0
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

COUNT=0

# ========== Phase 1: Baselines (SAC + TD3 + DSAC) ==========
echo ""
echo ">>> Phase 1: SAC/TD3/DSAC baselines (max 9 concurrent)..."

for algo in "${BASELINES[@]}"; do
    for env in "${ENVS[@]}"; do
        run_name="${algo}_${env}_${SEED}"
        log_file="${LOG_DIR}/${run_name}.log"
        COUNT=$((COUNT + 1))

        wait_for_slot 9
        echo "[${COUNT}/${TOTAL}] ${algo} ${env} --resume (active: ${#ACTIVE_PIDS[@]}/9) — $(date '+%H:%M:%S')"

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
            --resume \
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

    echo "[${COUNT}/${TOTAL}] resac ${env} --resume — $(date '+%H:%M:%S')"

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
        --resume \
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
