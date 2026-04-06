#!/bin/bash
# Benchmark: 9 concurrent training processes, CPU vs GPU
# Simulates the actual run_all.sh Phase 1 workload
#
# Usage:
#   bash jax_experiments/bench_concurrent.sh cpu
#   bash jax_experiments/bench_concurrent.sh gpu

DEVICE=${1:-cpu}
ITERS=5  # short run: warmup + a few steady-state iters
ENVS=("Hopper-v2" "HalfCheetah-v2" "Walker2d-v2")
ALGOS=("sac" "td3" "dsac")
# 3 algos × 3 envs = 9 concurrent processes

eval "$(conda shell.bash hook)"
conda activate jax-rl

# For GPU: setup CUDA libs at shell level (LD_LIBRARY_PATH must be set before process starts)
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

TMPDIR=$(mktemp -d -p /home/erzhu419/mine_code/RE-SAC/jax_experiments "bench_XXXXXX")
echo "=============================================="
echo "  Concurrent benchmark: 9 processes, device=$DEVICE"
echo "  Each process: $ITERS iters (~${ITERS}×4000 = $((ITERS*4000)) steps)"
echo "  Temp dir: $TMPDIR"
echo "=============================================="

START=$(date +%s%N)

PIDS=()
for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        python -u -m jax_experiments.train \
            --algo "$algo" \
            --env "$env" \
            --seed 8 \
            --max_iters $ITERS \
            --save_root "$TMPDIR" \
            --run_name "${algo}_${env}" \
            --backend spring \
            --device "$DEVICE" \
            --ensemble_size 2 \
            --stationary \
            > "$TMPDIR/${algo}_${env}.log" 2>&1 &
        PIDS+=($!)
    done
done

echo "Launched ${#PIDS[@]} processes, waiting..."

# Wait for all
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))
ELAPSED_S=$(echo "scale=1; $ELAPSED_MS / 1000" | bc)

echo ""
echo "=============================================="
echo "  RESULT: device=$DEVICE, 9 concurrent"
echo "  Wall-clock time: ${ELAPSED_S}s"
echo "  Failed: $FAILED / ${#PIDS[@]}"

# Show per-process timing from logs
echo ""
echo "  Per-process timings:"
for algo in "${ALGOS[@]}"; do
    for env in "${ENVS[@]}"; do
        LOG="$TMPDIR/${algo}_${env}.log"
        if [ -f "$LOG" ]; then
            TOTAL_TIME=$(grep "Total time:" "$LOG" | grep -oP '\d+s' | head -1)
            LAST_ITER=$(grep "^Iter" "$LOG" | tail -1 | grep -oP '\d+\.\d+s/iter' | head -1)
            echo "    ${algo} ${env}: total=${TOTAL_TIME:-FAIL}, last_iter=${LAST_ITER:-N/A}"
        else
            echo "    ${algo} ${env}: NO LOG"
        fi
    done
done
echo "=============================================="

# Cleanup only on full success
if [ $FAILED -eq 0 ]; then
    rm -rf "$TMPDIR"
else
    echo "  Logs preserved at: $TMPDIR"
fi
