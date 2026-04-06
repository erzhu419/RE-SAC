#!/bin/bash
# ============================================================
# Master Script: All Supplementary Experiments for RE-SAC Paper
# ============================================================
#
# Run all 10 experiments in sequence. Each experiment generates
# its own results and plots.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh
#
# Or run individual experiments:
#   ./run_all_experiments.sh --exp 1    # Q-estimation only
#   ./run_all_experiments.sh --exp 6    # β ablation only
# ============================================================

set -e
CONDA_ENV="jax-rl"
BASE_CMD="conda run -n ${CONDA_ENV} python"

echo "============================================================"
echo "  RE-SAC Supplementary Experiments"
echo "  $(date)"
echo "============================================================"

# Parse which experiments to run
EXPERIMENTS="${1:-all}"
if [ "$1" == "--exp" ]; then
    EXPERIMENTS="$2"
fi

run_exp() {
    local exp_num=$1
    echo ""
    echo "============================================================"
    echo "  EXPERIMENT $exp_num"
    echo "============================================================"
}

# ------------------------------------------------------------------
# Phase 1: Training runs (expensive, GPU-bound)
# ------------------------------------------------------------------

# Exp 6: β Ablation — generate and run training scripts
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "6" ]; then
    run_exp "6: Adaptive β Ablation"
    echo "Generating β ablation run scripts..."
    ${BASE_CMD} -m jax_experiments.experiments.exp6_beta_ablation \
        --mode gen_script --all
    echo "Run the generated scripts manually for training:"
    echo "  jax_experiments/experiments/results/exp6_beta_ablation/run_beta_ablation.sh"
fi

# Exp 9: Ensemble Size Ablation
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "9" ]; then
    run_exp "9: Ensemble Size Ablation"
    echo "Generating ensemble ablation run scripts..."
    ${BASE_CMD} -m jax_experiments.experiments.exp9_ensemble_ablation \
        --mode gen_script --all
    echo "Run the generated scripts manually:"
    echo "  jax_experiments/experiments/results/exp9_ensemble/run_ensemble_ablation.sh"
fi

# Exp 5: Multi-seed training
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "5" ]; then
    run_exp "5: Q-Value Stability (Multi-seed)"
    echo "Generating multi-seed run scripts..."
    ${BASE_CMD} -m jax_experiments.experiments.exp5_q_stability \
        --mode gen_script --all --seeds 1 2 3 4 5 8
    echo "Run the generated scripts manually for multi-seed training."
fi

# Exp 7: Environment expansion
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "7" ]; then
    run_exp "7: Environment Expansion"
    echo "Generating environment expansion run scripts..."
    ${BASE_CMD} -m jax_experiments.experiments.exp7_env_expansion \
        --mode gen_script
fi

# Exp 10: BAC comparison training
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "10" ]; then
    run_exp "10: RE-SAC vs BAC"
    echo "Generating BAC comparison scripts..."
    ${BASE_CMD} -m jax_experiments.experiments.exp10_bac_comparison \
        --mode gen_script --all
fi

# ------------------------------------------------------------------
# Phase 2: Analysis experiments (can run from existing checkpoints)
# ------------------------------------------------------------------

# Exp 1: Q-estimation accuracy
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "1" ]; then
    run_exp "1: Q-Estimation Accuracy"
    for ENV in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2; do
        echo "  Processing ${ENV}..."
        ${BASE_CMD} -m jax_experiments.experiments.exp1_q_estimation \
            --mode both --env ${ENV} --n_episodes 20 2>&1 | tail -5
    done
fi

# Exp 2: Δ(μ,π) analysis
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "2" ]; then
    run_exp "2: Δ(μ,π) Analysis"
    for ENV in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2; do
        echo "  Processing ${ENV}..."
        ${BASE_CMD} -m jax_experiments.experiments.exp2_delta_analysis \
            --mode both --env ${ENV} --n_episodes 10 2>&1 | tail -5
    done
fi

# Exp 3: Uncertainty decomposition
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "3" ]; then
    run_exp "3: Uncertainty Decomposition"
    for ENV in Hopper-v2 HalfCheetah-v2 Walker2d-v2 Ant-v2; do
        echo "  Processing ${ENV}..."
        ${BASE_CMD} -m jax_experiments.experiments.exp3_uncertainty_decomp \
            --mode both --env ${ENV} --n_episodes 20 2>&1 | tail -5
    done
fi

# Exp 4: Serendipity exploitation (requires GPU training)
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "4" ]; then
    run_exp "4: Serendipity Exploitation"
    for ENV in Hopper-v2 HalfCheetah-v2; do
        echo "  Training SAC and RE-SAC with expert injection on ${ENV}..."
        ${BASE_CMD} -m jax_experiments.experiments.exp4_serendipity \
            --mode both --env ${ENV} \
            --algos sac resac \
            --inject_at 500 --n_expert 5000 --max_iters 1000 2>&1 | tail -5
    done
fi

# Exp 8: Efficiency benchmark (requires GPU)
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "8" ]; then
    run_exp "8: Computational Efficiency"
    ${BASE_CMD} -m jax_experiments.experiments.exp8_efficiency \
        --mode both --env Hopper-v2 --n_warmup 5 --n_measure 20 2>&1 | tail -10
fi

# ------------------------------------------------------------------
# Phase 3: Post-training analysis (after training runs complete)
# ------------------------------------------------------------------

# These only work after the training from Phase 1 is complete
if [ "$EXPERIMENTS" == "analyze" ]; then
    echo ""
    echo "Running all analysis experiments on existing data..."

    # Exp 5 analysis
    ${BASE_CMD} -m jax_experiments.experiments.exp5_q_stability \
        --mode both --all

    # Exp 6 analysis
    ${BASE_CMD} -m jax_experiments.experiments.exp6_beta_ablation \
        --mode both --all

    # Exp 7 analysis
    ${BASE_CMD} -m jax_experiments.experiments.exp7_env_expansion \
        --mode both

    # Exp 9 analysis
    ${BASE_CMD} -m jax_experiments.experiments.exp9_ensemble_ablation \
        --mode both --all

    # Exp 10 analysis
    ${BASE_CMD} -m jax_experiments.experiments.exp10_bac_comparison \
        --mode both --all
fi

# ------------------------------------------------------------------
# Final: Combined paper figures
# ------------------------------------------------------------------
if [ "$EXPERIMENTS" == "all" ] || [ "$EXPERIMENTS" == "paper" ]; then
    run_exp "Paper Figures"
    ${BASE_CMD} -m jax_experiments.experiments.plot_paper_supplement \
        2>&1 || echo "Paper figure generation skipped (may need all data first)"
fi

echo ""
echo "============================================================"
echo "  All experiments complete!"
echo "  Results: jax_experiments/experiments/results/"
echo "============================================================"
