#!/bin/bash
# ============================================================
# P2: Hyperparameter Sensitivity Sweep
# ============================================================
# Sweep one param at a time on HalfCheetah + Ant (most informative envs)
# Other params at best-config default.
#
# Usage: bash jax_experiments/run_sensitivity.sh <sweep_name> [gpu|cpu]
#   sweep_name: anchor | beta_end | ratio | ensemble | ema
# ============================================================

SWEEP=${1:?Usage: $0 <anchor|beta_end|ratio|ensemble|ema> [gpu|cpu]}
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

# Base config (HalfCheetah=v5, Ant=v6b)
BASE_HC="--algo resac --env HalfCheetah-v2 --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --stationary --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --adaptive_beta --beta_start -2.0 --beta_warmup 0.2"

BASE_ANT="--algo resac --env Ant-v2 --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --stationary --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --adaptive_beta --beta_start -2.0 --beta_warmup 0.2"

# Default per-env params
HC_DEFAULTS="--ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
ANT_DEFAULTS="--ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"

launch() {
    local NAME=$1; local ARGS=$2
    local LOG="${LOG_DIR}/${NAME}.log"
    echo "  $NAME"
    python -u -m jax_experiments.train $ARGS --run_name "$NAME" > "$LOG" 2>&1 &
    PIDS+=($!)
}

echo "============================================================"
echo "  Sensitivity Sweep: $SWEEP"
echo "============================================================"

PIDS=()

case $SWEEP in
    anchor)
        for val in 0 0.001 0.01 0.1 0.5 1.0; do
            launch "sens_anchor_${val}_HalfCheetah-v2_$SEED" \
                "$BASE_HC --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda $val --independent_ratio 1.0"
            launch "sens_anchor_${val}_Ant-v2_$SEED" \
                "$BASE_ANT --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda $val --independent_ratio 0.5"
        done ;;
    beta_end)
        for val in -2.0 -1.5 -1.0 -0.5 0.0; do
            launch "sens_betaend_${val}_HalfCheetah-v2_$SEED" \
                "$BASE_HC --ensemble_size 10 --beta_end $val --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
            launch "sens_betaend_${val}_Ant-v2_$SEED" \
                "$BASE_ANT --ensemble_size 10 --beta_end $val --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"
        done ;;
    ratio)
        for val in 0.0 0.25 0.5 0.75 1.0; do
            launch "sens_ratio_${val}_HalfCheetah-v2_$SEED" \
                "$BASE_HC --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio $val"
            launch "sens_ratio_${val}_Ant-v2_$SEED" \
                "$BASE_ANT --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio $val"
        done ;;
    ensemble)
        for val in 2 5 10 20; do
            launch "sens_K_${val}_HalfCheetah-v2_$SEED" \
                "$BASE_HC --ensemble_size $val --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
            launch "sens_K_${val}_Ant-v2_$SEED" \
                "$BASE_ANT --ensemble_size $val --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"
        done ;;
    ema)
        for val in 0.0 0.001 0.005 0.01 0.05; do
            launch "sens_ema_${val}_HalfCheetah-v2_$SEED" \
                "$BASE_HC --ensemble_size 10 --beta_end -1.0 --ema_tau $val --anchor_lambda 0.1 --independent_ratio 1.0"
            launch "sens_ema_${val}_Ant-v2_$SEED" \
                "$BASE_ANT --ensemble_size 10 --beta_end -1.0 --ema_tau $val --anchor_lambda 0.01 --independent_ratio 0.5"
        done ;;
    *) echo "Unknown sweep: $SWEEP"; exit 1 ;;
esac

echo "${#PIDS[@]} jobs launched."
echo "Monitor: tail -f ${LOG_DIR}/sens_${SWEEP}_*.log"

FAILED=0
for pid in "${PIDS[@]}"; do wait $pid || FAILED=$((FAILED+1)); done
[ $FAILED -eq 0 ] && echo "All done!" || echo "$FAILED failed."
