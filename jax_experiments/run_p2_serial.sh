#!/bin/bash
# P2: Sensitivity sweeps + Non-stationary, fully sequential
# Waits for P1 (baselines) to finish first, then auto-starts.
# Each "batch" = 2 jobs (HalfCheetah + Ant), to stay within 8GB GPU limit.
#
# Usage: nohup bash jax_experiments/run_p2_serial.sh gpu &

DEVICE=${1:-gpu}
cd /home/erzhu419/mine_code/RE-SAC

LOG_DIR="jax_experiments/logs"
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
SEED=8
MAX_ITERS=2000

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

# ============================================================
# Wait for P1 (baselines) to finish
# ============================================================
echo "$(date): Waiting for P1 baselines to finish..."
while pgrep -f "jax_experiments.train" > /dev/null 2>&1; do
    sleep 120
done
echo "$(date): P1 done. Starting P2."

# ============================================================
# Helper: run 2 jobs (HC + Ant) and wait
# ============================================================
BASE_HC="--algo resac --env HalfCheetah-v2 --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --stationary --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --adaptive_beta --beta_start -2.0 --beta_warmup 0.2"

BASE_ANT="--algo resac --env Ant-v2 --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --stationary --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --adaptive_beta --beta_start -2.0 --beta_warmup 0.2"

run_pair() {
    local NAME_HC=$1; local ARGS_HC=$2
    local NAME_ANT=$3; local ARGS_ANT=$4
    echo "  $NAME_HC  |  $NAME_ANT"
    python -u -m jax_experiments.train $ARGS_HC --run_name "$NAME_HC" \
        > "${LOG_DIR}/${NAME_HC}.log" 2>&1 &
    PID1=$!
    python -u -m jax_experiments.train $ARGS_ANT --run_name "$NAME_ANT" \
        > "${LOG_DIR}/${NAME_ANT}.log" 2>&1 &
    PID2=$!
    wait $PID1 $PID2
}

# ============================================================
# P2a: Sensitivity sweeps (one value pair at a time)
# ============================================================
echo ""
echo "============================================================"
echo "  $(date): P2a Sensitivity Sweeps"
echo "============================================================"

# --- anchor ---
echo "--- sweep: anchor_lambda ---"
for val in 0 0.001 0.01 0.1 0.5 1.0; do
    run_pair \
        "sens_anchor_${val}_HalfCheetah-v2_${SEED}" \
        "$BASE_HC --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda $val --independent_ratio 1.0" \
        "sens_anchor_${val}_Ant-v2_${SEED}" \
        "$BASE_ANT --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda $val --independent_ratio 0.5"
done

# --- beta_end ---
echo "--- sweep: beta_end ---"
for val in -2.0 -1.5 -1.0 -0.5 0.0; do
    run_pair \
        "sens_betaend_${val}_HalfCheetah-v2_${SEED}" \
        "$BASE_HC --ensemble_size 10 --beta_end $val --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0" \
        "sens_betaend_${val}_Ant-v2_${SEED}" \
        "$BASE_ANT --ensemble_size 10 --beta_end $val --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"
done

# --- independent_ratio ---
echo "--- sweep: independent_ratio ---"
for val in 0.0 0.25 0.5 0.75 1.0; do
    run_pair \
        "sens_ratio_${val}_HalfCheetah-v2_${SEED}" \
        "$BASE_HC --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio $val" \
        "sens_ratio_${val}_Ant-v2_${SEED}" \
        "$BASE_ANT --ensemble_size 10 --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio $val"
done

# --- ensemble size ---
echo "--- sweep: ensemble_size ---"
for val in 2 5 10 20; do
    run_pair \
        "sens_K_${val}_HalfCheetah-v2_${SEED}" \
        "$BASE_HC --ensemble_size $val --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0" \
        "sens_K_${val}_Ant-v2_${SEED}" \
        "$BASE_ANT --ensemble_size $val --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"
done

# --- ema_tau ---
echo "--- sweep: ema_tau ---"
for val in 0.0 0.001 0.005 0.01 0.05; do
    run_pair \
        "sens_ema_${val}_HalfCheetah-v2_${SEED}" \
        "$BASE_HC --ensemble_size 10 --beta_end -1.0 --ema_tau $val --anchor_lambda 0.1 --independent_ratio 1.0" \
        "sens_ema_${val}_Ant-v2_${SEED}" \
        "$BASE_ANT --ensemble_size 10 --beta_end -1.0 --ema_tau $val --anchor_lambda 0.01 --independent_ratio 0.5"
done

echo "$(date): P2a sensitivity sweeps done."

# ============================================================
# P2b: Non-stationary (3 algos × 4 envs, 4 envs parallel per algo)
# ============================================================
echo ""
echo "============================================================"
echo "  $(date): P2b Non-stationary"
echo "============================================================"

NS_ARGS="--varying_params gravity --task_num 40 --test_task_num 40 --log_scale_limit 3.0"
ENVS=("Hopper-v2" "Walker2d-v2" "HalfCheetah-v2" "Ant-v2")

declare -A ENV_RESAC_ARGS
ENV_RESAC_ARGS["Hopper-v2"]="--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
ENV_RESAC_ARGS["Walker2d-v2"]="--adaptive_beta --beta_end -0.5 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 1.0"
ENV_RESAC_ARGS["HalfCheetah-v2"]="--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.1 --independent_ratio 1.0"
ENV_RESAC_ARGS["Ant-v2"]="--adaptive_beta --beta_end -1.0 --ema_tau 0.005 --anchor_lambda 0.01 --independent_ratio 0.5"

run_4envs() {
    local PREFIX=$1; local ALGO=$2; local EXTRA=$3; local K=$4
    echo "  $PREFIX ($ALGO)"
    PIDS=()
    for ENV in "${ENVS[@]}"; do
        NAME="${PREFIX}_${ENV}_${SEED}"
        python -u -m jax_experiments.train --algo "$ALGO" --env "$ENV" \
            --seed $SEED --max_iters $MAX_ITERS \
            --save_root "$SAVE_ROOT" --run_name "$NAME" \
            --backend "$BACKEND" --device "$DEVICE" --ensemble_size $K \
            $NS_ARGS $EXTRA \
            > "${LOG_DIR}/${NAME}.log" 2>&1 &
        PIDS+=($!)
    done
    FAILED=0
    for pid in "${PIDS[@]}"; do wait $pid || FAILED=$((FAILED+1)); done
    echo "$(date): $PREFIX done ($FAILED failed)"
}

run_4envs "ns_sac"   "sac"   ""   2
run_4envs "ns_dsac"  "dsac"  ""   10
for ENV in "${ENVS[@]}"; do
    # RE-SAC per-env args need to be run individually in pairs for memory safety
    NAME="ns_resac_${ENV}_${SEED}"
    python -u -m jax_experiments.train --algo resac --env "$ENV" \
        --seed $SEED --max_iters $MAX_ITERS \
        --save_root "$SAVE_ROOT" --run_name "$NAME" \
        --backend "$BACKEND" --device "$DEVICE" --ensemble_size 10 \
        --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
        --beta_start -2.0 --beta_warmup 0.2 \
        ${ENV_RESAC_ARGS[$ENV]} \
        $NS_ARGS \
        > "${LOG_DIR}/${NAME}.log" 2>&1
    echo "$(date): ns_resac $ENV done"
done

echo ""
echo "$(date): All P2 experiments complete!"
