#!/bin/bash
# Run remaining ablation batches sequentially (4 envs parallel per batch)
# Usage: nohup bash jax_experiments/run_ablation_serial.sh gpu &

DEVICE=${1:-gpu}

cd /home/erzhu419/mine_code/RE-SAC

# Wait for any running abl_ jobs to finish
while pgrep -f "jax_experiments.train.*abl_" > /dev/null 2>&1; do
    echo "$(date): Waiting for current batch to finish..."
    sleep 120
done

for ABL in abl_noanc abl_noadapt abl_noall; do
    echo ""
    echo "============================================================"
    echo "  $(date): Starting $ABL"
    echo "============================================================"
    bash jax_experiments/run_ablation_batch.sh "$ABL" "$DEVICE"
    echo "$(date): $ABL completed"
done

echo ""
echo "$(date): All remaining ablations done!"
