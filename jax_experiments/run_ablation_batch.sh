#!/bin/bash
# ============================================================
# Ablation: run one ablation at a time (4 envs parallel)
# Usage: bash jax_experiments/run_ablation_batch.sh <abl_name> [gpu|cpu]
#   abl_name: abl_noind | abl_noema | abl_noanc | abl_noadapt | abl_noall
# ============================================================

ABL_NAME=${1:?Usage: $0 <abl_name> [gpu|cpu]}
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

declare -A ENV_BETA_END=(["Hopper-v2"]="-1.0" ["Walker2d-v2"]="-0.5" ["HalfCheetah-v2"]="-1.0" ["Ant-v2"]="-1.0")
declare -A ENV_ANCHOR=(["Hopper-v2"]="0.1" ["Walker2d-v2"]="0.01" ["HalfCheetah-v2"]="0.1" ["Ant-v2"]="0.01")
declare -A ENV_RATIO=(["Hopper-v2"]="1.0" ["Walker2d-v2"]="1.0" ["HalfCheetah-v2"]="1.0" ["Ant-v2"]="0.5")

ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")

COMMON="--algo resac --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --ensemble_size 10 --stationary \
  --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --beta_start -2.0 --beta_warmup 0.2"

echo "============================================================"
echo "  Ablation: $ABL_NAME (4 envs)"
echo "============================================================"

PIDS=()
for ENV in "${ENVS[@]}"; do
    BE=${ENV_BETA_END[$ENV]}
    AN=${ENV_ANCHOR[$ENV]}
    RA=${ENV_RATIO[$ENV]}

    case $ABL_NAME in
        abl_noind)
            ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BE --ema_tau 0.005 --anchor_lambda $AN --independent_ratio 0.0" ;;
        abl_noema)
            ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BE --ema_tau 0.0 --anchor_lambda $AN --independent_ratio $RA" ;;
        abl_noanc)
            ARGS="$COMMON --env $ENV --adaptive_beta --beta_end $BE --ema_tau 0.005 --anchor_lambda 0.0 --independent_ratio $RA" ;;
        abl_noadapt)
            ARGS="$COMMON --env $ENV --ema_tau 0.005 --anchor_lambda $AN --independent_ratio $RA" ;;
        abl_noall)
            ARGS="$COMMON --env $ENV --independent_ratio 0.0 --ema_tau 0.0 --anchor_lambda 0.0" ;;
        *) echo "Unknown ablation: $ABL_NAME"; exit 1 ;;
    esac

    RUN_NAME="${ABL_NAME}_${ENV}_${SEED}"
    LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
    echo "  $ENV -> $RUN_NAME"
    python -u -m jax_experiments.train $ARGS --run_name "$RUN_NAME" > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "4 jobs launched. PIDs: ${PIDS[*]}"
echo "Monitor: tail -f jax_experiments/logs/${ABL_NAME}_*_${SEED}.log"

FAILED=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAILED=$((FAILED + 1))
done

[ $FAILED -eq 0 ] && echo "All done!" || echo "$FAILED failed."
