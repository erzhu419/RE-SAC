#!/bin/bash
# ============================================================
# RE-SAC v6: Ant stability ablation (4 fixes)
# ============================================================
# v6a: lcb_normalize + q_std_clip=0.5
# v6b: independent_ratio=0.5 (half independent, half min)
# v6c: ensemble_size=5 (smaller ensemble)
# v6d: all fixes combined
#
# Usage: bash jax_experiments/run_resac_v6_ant.sh [cpu|gpu]
# ============================================================

DEVICE=${1:-gpu}
SEED=8
MAX_ITERS=2000
SAVE_ROOT="jax_experiments/results"
BACKEND="spring"
ENV="Ant-v2"

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

# Shared base params (from v5b best)
BASE="--algo resac --env $ENV --seed $SEED --max_iters $MAX_ITERS \
  --save_root $SAVE_ROOT --backend $BACKEND --device $DEVICE \
  --stationary --beta -2.0 --beta_ood 0.001 --weight_reg 0.001 --beta_bc 0.0001 \
  --adaptive_beta --beta_start -2.0 --beta_end -1.0 --beta_warmup 0.2 \
  --ema_tau 0.005 --anchor_lambda 0.01"

echo "============================================================"
echo "  RE-SAC v6: Ant stability ablation"
echo "============================================================"

PIDS=()
NAMES=()

# v6a: normalized LCB + Q-std clipping (fixes #1 + #2)
NAME="resac_v6a_${ENV}_${SEED}"
LOG="${LOG_DIR}/${NAME}.log"
echo "v6a: lcb_normalize + q_std_clip=0.5"
python -u -m jax_experiments.train $BASE \
    --ensemble_size 10 --run_name "$NAME" \
    --lcb_normalize --q_std_clip 0.5 \
    > "$LOG" 2>&1 &
PIDS+=($!); NAMES+=("v6a")

# v6b: partial independent target (fix #4)
NAME="resac_v6b_${ENV}_${SEED}"
LOG="${LOG_DIR}/${NAME}.log"
echo "v6b: independent_ratio=0.5"
python -u -m jax_experiments.train $BASE \
    --ensemble_size 10 --run_name "$NAME" \
    --independent_ratio 0.5 \
    > "$LOG" 2>&1 &
PIDS+=($!); NAMES+=("v6b")

# v6c: smaller ensemble (fix #3)
NAME="resac_v6c_${ENV}_${SEED}"
LOG="${LOG_DIR}/${NAME}.log"
echo "v6c: ensemble_size=5"
python -u -m jax_experiments.train $BASE \
    --ensemble_size 5 --run_name "$NAME" \
    > "$LOG" 2>&1 &
PIDS+=($!); NAMES+=("v6c")

# v6d: all fixes combined
NAME="resac_v6d_${ENV}_${SEED}"
LOG="${LOG_DIR}/${NAME}.log"
echo "v6d: all fixes (normalize + clip=0.5 + ratio=0.5 + K=5)"
python -u -m jax_experiments.train $BASE \
    --ensemble_size 5 --run_name "$NAME" \
    --lcb_normalize --q_std_clip 0.5 --independent_ratio 0.5 \
    > "$LOG" 2>&1 &
PIDS+=($!); NAMES+=("v6d")

echo ""
echo "4 jobs launched. PIDs: ${PIDS[*]}"
echo "Monitor: tail -f jax_experiments/logs/resac_v6*_Ant*_8.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait ${PIDS[$i]}; then
        echo "${NAMES[$i]} completed successfully."
    else
        echo "${NAMES[$i]} FAILED (exit code $?)."
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "All 4 Ant ablations completed!"
else
    echo "$FAILED job(s) failed."
fi
