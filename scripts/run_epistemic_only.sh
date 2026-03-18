#!/bin/bash

# Experiment B: Epistemic Only
# Settings: weight_reg=0 (Disabled), beta=-2, beta_bc=0, beta_ood=0.01

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export CUDA_VISIBLE_DEVICES=0

python "$PROJECT_DIR/sac_ensemble_original_logging.py" \
    --weight_reg 0 \
    --beta -2 \
    --beta_bc 0 \
    --beta_ood 0.01 \
    --maximum_alpha 0.6 \
    --critic_actor_ratio 2 \
    --replay_buffer_size 1000000 \
    --hidden_dim 64 \
    --max_episodes 500 \
    --save_root "$PROJECT_DIR/results/epistemic_only" \
    --run_name .
