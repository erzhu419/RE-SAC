#!/bin/bash

# Run experiments for all Ensemble Sizes in parallel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Ensemble Size Experiments..."

for size in 2 5 10 20 40; do
    echo "Launching Ensemble Size = $size"
    bash "$SCRIPT_DIR/run_ensemble.sh" $size &
done

# Ablation experiments
echo "Launching Aleatoric Only Experiment"
bash "$SCRIPT_DIR/run_aleatoric_only.sh" &

echo "Launching Epistemic Only Experiment"
bash "$SCRIPT_DIR/run_epistemic_only.sh" &

echo "All experiments launched! Waiting for completion..."
wait
echo "All experiments finished."
