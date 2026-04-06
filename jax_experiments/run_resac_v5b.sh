#!/bin/bash
# ============================================================
# RE-SAC v5b: Walker2d + Ant with relaxed anchoring
# ============================================================
# Changes from v5:
#   - anchor_lambda=0.01 (was 0.1, too restrictive for Walker2d/Ant)
#   - beta_end=-0.5 for Walker2d (needs more exploration)
#   - beta_end=-1.0 for Ant (keep conservative, Q-std already high)
#
# Usage: bash jax_experiments/run_resac_v5b.sh [cpu|gpu]
# ============================================================

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

# Shared params
BETA=-2.0
BETA_OOD=0.001
WEIGHT_REG=0.001
BETA_BC=0.0001
BETA_START=-2.0
BETA_WARMUP=0.2
EMA_TAU=0.005
ANCHOR_LAMBDA=0.01  # relaxed from 0.1

echo "============================================================"
echo "  RE-SAC v5b (Walker2d + Ant, relaxed anchor)"
echo "  anchor_lambda=$ANCHOR_LAMBDA  ema_tau=$EMA_TAU"
echo "  Device: $DEVICE  Backend: $BACKEND  Seed: $SEED"
echo "============================================================"

PIDS=()

# Walker2d: beta_end=-0.5 (needs exploration room)
env="Walker2d-v2"
BETA_END=-0.5
RUN_NAME="resac_v5b_${env}_${SEED}"
LOG_FILE="${LOG_DIR}/resac_v5b_${env}_${SEED}.log"
echo "Starting: $env -> $RUN_NAME (beta_end=$BETA_END, anchor=$ANCHOR_LAMBDA)"
python -u -m jax_experiments.train \
    --algo resac --env "$env" --seed $SEED --max_iters $MAX_ITERS \
    --save_root "$SAVE_ROOT" --run_name "$RUN_NAME" \
    --backend "$BACKEND" --device "$DEVICE" --ensemble_size 10 --stationary \
    --beta $BETA --beta_ood $BETA_OOD --weight_reg $WEIGHT_REG --beta_bc $BETA_BC \
    --adaptive_beta --beta_start $BETA_START --beta_end $BETA_END --beta_warmup $BETA_WARMUP \
    --ema_tau $EMA_TAU --anchor_lambda $ANCHOR_LAMBDA \
    > "$LOG_FILE" 2>&1 &
PIDS+=($!)

# Ant: beta_end=-1.0 (Q-std already high, keep conservative)
env="Ant-v2"
BETA_END=-1.0
RUN_NAME="resac_v5b_${env}_${SEED}"
LOG_FILE="${LOG_DIR}/resac_v5b_${env}_${SEED}.log"
echo "Starting: $env -> $RUN_NAME (beta_end=$BETA_END, anchor=$ANCHOR_LAMBDA)"
python -u -m jax_experiments.train \
    --algo resac --env "$env" --seed $SEED --max_iters $MAX_ITERS \
    --save_root "$SAVE_ROOT" --run_name "$RUN_NAME" \
    --backend "$BACKEND" --device "$DEVICE" --ensemble_size 10 --stationary \
    --beta $BETA --beta_ood $BETA_OOD --weight_reg $WEIGHT_REG --beta_bc $BETA_BC \
    --adaptive_beta --beta_start $BETA_START --beta_end $BETA_END --beta_warmup $BETA_WARMUP \
    --ema_tau $EMA_TAU --anchor_lambda $ANCHOR_LAMBDA \
    > "$LOG_FILE" 2>&1 &
PIDS+=($!)

echo ""
echo "2 jobs launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f jax_experiments/logs/resac_v5b_*_8.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "Job $i completed successfully."
    else
        echo "Job $i FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "All jobs completed successfully!"
else
    echo "$FAILED job(s) failed."
fi
