#!/bin/bash

# Run the ensemble training with the best parameters
# Default ensemble size is 10, can be overridden by first argument
ENSEMBLE_SIZE=${1:-10}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

python "$PROJECT_DIR/sac_ensemble_original_logging.py" \
    --weight_reg 0.01 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root "$PROJECT_DIR/results/ensemble_${ENSEMBLE_SIZE}" \
    --run_name . \
    --ensemble_size ${ENSEMBLE_SIZE}
