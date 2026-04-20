#!/bin/bash
# ============================================================
# P2: Non-stationary Environment Experiments
# ============================================================
# Gravity varies every 20k steps (40 tasks, log_scale_limit=3.0)
# Compare RE-SAC best vs baselines under regime changes.
#
# Usage: bash jax_experiments/run_nonstationary.sh [gpu|cpu]
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

# Non-stationary params
NS_ARGS="--varying_params gravity --task_num 40 --test_task_num 40 \
  --log_scale_limit 3.0"

ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")

# Algorithms to compare
# SAC, DSAC, REDQ, RE-SAC v5 best config per env
declare -A ENV_RESAC_ARGS
ENV_RESAC_ARGS["Hopper-v2"]="--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
ENV_RESAC_ARGS["Walker2d-v2"]="--adaptive_beta --beta_end -0.5 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 1.0"
ENV_RESAC_ARGS["HalfCheetah-v2"]="--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
ENV_RESAC_ARGS["Ant-v2"]="--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"

echo "============================================================"
echo "  Non-stationary experiments (gravity perturbation)"
echo "============================================================"

PIDS=()

for ENV in "${ENVS[@]}"; do
    echo "--- $ENV ---"

    # SAC baseline
    NAME="ns_sac_${ENV}_${SEED}"
    python -u -m jax_experiments.train --algo sac --env "$ENV" --seed $SEED \
        --max_iters $MAX_ITERS --save_root "$SAVE_ROOT" --run_name "$NAME" \
        --backend "$BACKEND" --device "$DEVICE" --ensemble_size 2 \
        $NS_ARGS > "${LOG_DIR}/${NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "  $NAME"

    # DSAC baseline
    NAME="ns_dsac_${ENV}_${SEED}"
    python -u -m jax_experiments.train --algo dsac --env "$ENV" --seed $SEED \
        --max_iters $MAX_ITERS --save_root "$SAVE_ROOT" --run_name "$NAME" \
        --backend "$BACKEND" --device "$DEVICE" --ensemble_size 10 \
        $NS_ARGS > "${LOG_DIR}/${NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "  $NAME"

    # RE-SAC v5 best config
    NAME="ns_resac_v5_${ENV}_${SEED}"
    python -u -m jax_experiments.train --algo resac --env "$ENV" --seed $SEED \
        --max_iters $MAX_ITERS --save_root "$SAVE_ROOT" --run_name "$NAME" \
        --backend "$BACKEND" --device "$DEVICE" --ensemble_size 10 \
        --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
        --beta_start -2.0 --beta_warmup 0.2 \
        ${ENV_RESAC_ARGS[$ENV]} \
        $NS_ARGS > "${LOG_DIR}/${NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "  $NAME"
done

echo ""
echo "${#PIDS[@]} jobs launched."
echo "Monitor: tail -f ${LOG_DIR}/ns_*.log"

FAILED=0
for pid in "${PIDS[@]}"; do wait $pid || FAILED=$((FAILED+1)); done
[ $FAILED -eq 0 ] && echo "All done!" || echo "$FAILED failed."
