#!/bin/bash
# ============================================================
# RE-SAC v5: Independent targets + EMA policy + Anchoring +
#            Performance-aware beta + beta_end=-1.0
# ============================================================
# Key changes from v4:
#   - EMA policy for stable evaluation
#   - Policy anchoring (L2 penalty toward best policy)
#   - Performance-aware beta (tightens if eval drops)
#   - beta_end=-1.0 (was -0.5, less aggressive in late training)
#
# Results saved to resac_v5_{env}_8/
#
# Usage: bash jax_experiments/run_resac_v5.sh [cpu|gpu]
# ============================================================

DEVICE=${1:-gpu}
SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"

ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")

LOG_DIR="jax_experiments/logs"
mkdir -p "$LOG_DIR"

# Activate conda env
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

# v5 hyperparameters
BETA=-2.0
BETA_OOD=0.001
WEIGHT_REG=0.001
BETA_BC=0.0001
# Adaptive beta: more conservative end
BETA_START=-2.0
BETA_END=-1.0
BETA_WARMUP=0.2
# Stabilization
EMA_TAU=0.005
ANCHOR_LAMBDA=0.1

echo "============================================================"
echo "  RE-SAC v5 (EMA + anchoring + perf-aware beta)"
echo "  beta: ${BETA_START} -> ${BETA_END}  warmup=${BETA_WARMUP}"
echo "  ema_tau=$EMA_TAU  anchor_lambda=$ANCHOR_LAMBDA"
echo "  Device: $DEVICE  Backend: $BACKEND  Seed: $SEED"
echo "============================================================"

PIDS=()
for env in "${ENVS[@]}"; do
    RUN_NAME="resac_v5_${env}_${SEED}"
    LOG_FILE="${LOG_DIR}/resac_v5_${env}_${SEED}.log"

    echo "Starting: $env -> $RUN_NAME (log: $LOG_FILE)"

    python -u -m jax_experiments.train \
        --algo resac \
        --env "$env" \
        --seed $SEED \
        --max_iters $MAX_ITERS \
        --save_root "$SAVE_ROOT" \
        --run_name "$RUN_NAME" \
        --backend "$BACKEND" \
        --device "$DEVICE" \
        --ensemble_size 10 \
        --stationary \
        --beta $BETA \
        --beta_ood $BETA_OOD \
        --weight_reg $WEIGHT_REG \
        --beta_bc $BETA_BC \
        --adaptive_beta \
        --beta_start $BETA_START \
        --beta_end $BETA_END \
        --beta_warmup $BETA_WARMUP \
        --ema_tau $EMA_TAU \
        --anchor_lambda $ANCHOR_LAMBDA \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All 4 jobs launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f jax_experiments/logs/resac_v5_*_8.log"
echo ""

# Wait for all jobs
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "${ENVS[$i]} completed successfully."
    else
        echo "${ENVS[$i]} FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "All 4 environments completed successfully!"
else
    echo ""
    echo "$FAILED environment(s) failed."
fi
