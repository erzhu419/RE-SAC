#!/bin/bash
# ============================================================
# P0: Formal Ablation Study
# ============================================================
# Base = best config per env. Remove one component at a time.
#
# Ablations:
#   abl_noind   : independent_ratio=0.0 (revert to shared min target)
#   abl_noema   : ema_tau=0 (no EMA policy, eval with live policy)
#   abl_noanc   : anchor_lambda=0 (no policy anchoring)
#   abl_noadapt : fixed beta=-2.0 (no adaptive beta)
#   abl_noall   : all removed (= RE-SAC v1 baseline with independent target only)
#
# Usage: bash jax_experiments/run_ablation.sh [cpu|gpu]
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

# ============================================================
# Per-env best config (base for ablation)
# ============================================================
# Format: ENV BETA_END ANCHOR_LAMBDA INDEPENDENT_RATIO
declare -A ENV_BETA_END=(
    ["Hopper-v2"]="-1.0"
    ["Walker2d-v2"]="-0.5"
    ["HalfCheetah-v2"]="-1.0"
    ["Ant-v2"]="-1.0"
)
declare -A ENV_ANCHOR=(
    ["Hopper-v2"]="0.1"
    ["Walker2d-v2"]="0.01"
    ["HalfCheetah-v2"]="0.1"
    ["Ant-v2"]="0.01"
)
declare -A ENV_RATIO=(
    ["Hopper-v2"]="1.0"
    ["Walker2d-v2"]="1.0"
    ["HalfCheetah-v2"]="1.0"
    ["Ant-v2"]="0.5"
)

ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")

# Shared base params
COMMON="--algo resac --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --ensemble_size 10 --stationary \
  --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --beta_start -2.0 --beta_warmup 0.2"

run_ablation() {
    local ABL_NAME=$1  # e.g. "abl_noind"
    local ENV=$2
    local EXTRA_ARGS=$3

    local BETA_END=${ENV_BETA_END[$ENV]}
    local ANCHOR=${ENV_ANCHOR[$ENV]}
    local RATIO=${ENV_RATIO[$ENV]}

    # Apply per-env defaults, then override with ablation-specific args
    local ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BETA_END \
      --ema_tau 0.005 --anchor_lambda $ANCHOR --independent_ratio $RATIO"

    # Apply ablation overrides
    case $ABL_NAME in
        abl_noind)
            ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BETA_END \
              --ema_tau 0.005 --anchor_lambda $ANCHOR --independent_ratio 0.0"
            ;;
        abl_noema)
            # ema_tau=0 means EMA = live policy (no smoothing)
            ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BETA_END \
              --ema_tau 0.0 --anchor_lambda $ANCHOR --independent_ratio $RATIO"
            ;;
        abl_noanc)
            ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BETA_END \
              --ema_tau 0.005 --anchor_lambda 0.0 --independent_ratio $RATIO"
            ;;
        abl_noadapt)
            # No adaptive beta: fixed beta=-2.0 (don't pass --adaptive_beta)
            ARGS="$COMMON --env $ENV \
              --ema_tau 0.005 --anchor_lambda $ANCHOR --independent_ratio $RATIO"
            ;;
        abl_noall)
            # Remove all: min target, no EMA, no anchor, fixed beta
            ARGS="$COMMON --env $ENV --independent_ratio 0.0 \
              --ema_tau 0.0 --anchor_lambda 0.0"
            ;;
    esac

    local RUN_NAME="${ABL_NAME}_${ENV}_${SEED}"
    local LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

    echo "  $ABL_NAME $ENV -> $RUN_NAME"
    python -u -m jax_experiments.train $ARGS \
        --run_name "$RUN_NAME" \
        > "$LOG_FILE" 2>&1 &
}

echo "============================================================"
echo "  RE-SAC Ablation Study"
echo "  5 ablations x 4 envs = 20 runs"
echo "============================================================"

PIDS=()

for ABL in abl_noind abl_noema abl_noanc abl_noadapt abl_noall; do
    echo ""
    echo "--- $ABL ---"
    for ENV in "${ENVS[@]}"; do
        run_ablation "$ABL" "$ENV"
        PIDS+=($!)
    done
done

echo ""
echo "All ${#PIDS[@]} jobs launched."
echo "Monitor: tail -f jax_experiments/logs/abl_*_8.log"
echo ""

FAILED=0
for pid in "${PIDS[@]}"; do
    if wait $pid; then
        true
    else
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "All ablation experiments completed successfully!"
else
    echo "$FAILED experiment(s) failed."
fi
