"""Unified paper figure generation from all supplementary experiments.

Creates publication-quality figures combining results from Exp 1-10.
All figures use consistent styling (serif fonts, 300 DPI, color scheme).

Usage:
    python -m jax_experiments.experiments.plot_paper_supplement
"""
import os
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Publication styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

RESULTS_BASE = "jax_experiments/experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/paper_figures"
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
ENV_SHORT = {"Hopper-v2": "Hopper", "HalfCheetah-v2": "HalfCheetah",
             "Walker2d-v2": "Walker2d", "Ant-v2": "Ant"}

# Consistent color scheme across all figures
ALGO_COLORS = {
    'sac': '#ff7f0e', 'resac': '#1f77b4', 'resac_v2': '#9467bd',
    'td3': '#2ca02c', 'dsac': '#d62728', 'bac': '#e377c2',
}
ALGO_LABELS = {
    'sac': 'SAC', 'resac': 'RE-SAC', 'resac_v2': 'RE-SAC v2',
    'td3': 'TD3', 'dsac': 'DSAC', 'bac': 'BAC',
}
ALGO_MARKERS = {
    'sac': 's', 'resac': 'o', 'resac_v2': 'P',
    'td3': '^', 'dsac': 'D', 'bac': 'v',
}


def smooth(y, window=10):
    """Moving average smoothing."""
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')


def load_pkl(subdir, filename):
    """Load a pickle file from experiment results."""
    path = os.path.join(RESULTS_BASE, subdir, filename)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ====================================================================
# Figure 1: Q-Estimation Accuracy (Exp 1) — Main paper contribution
# ====================================================================
def figure_q_estimation():
    """2×4 grid: Q-error distribution + bar chart for each env."""
    fig, axes = plt.subplots(2, len(ENVS), figsize=(16, 7))
    fig.suptitle('Q-Value Estimation Accuracy Across Environments', fontsize=14, y=0.98)

    for col, env_name in enumerate(ENVS):
        data = load_pkl("exp1_q_estimation", f"q_estimation_{env_name}.pkl")
        if data is None:
            axes[0, col].text(0.5, 0.5, 'No data', ha='center', transform=axes[0, col].transAxes)
            axes[1, col].text(0.5, 0.5, 'No data', ha='center', transform=axes[1, col].transAxes)
            continue

        results = data['results']

        # Top row: distribution
        ax = axes[0, col]
        for algo in ['sac', 'resac', 'td3', 'dsac']:
            if algo not in results:
                continue
            err = np.clip(results[algo]['normalized_error'], -2, 2)
            ax.hist(err, bins=40, alpha=0.3, label=ALGO_LABELS[algo],
                    color=ALGO_COLORS[algo], density=True)
        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(ENV_SHORT[env_name])
        if col == 0:
            ax.set_ylabel('Density')
        ax.set_xlabel('Norm. Q-Error')
        if col == 0:
            ax.legend(fontsize=7, loc='upper left')

        # Bottom row: bar chart
        ax = axes[1, col]
        algos_present = [a for a in ['sac', 'resac', 'td3', 'dsac'] if a in results]
        means = [results[a]['mean_normalized_error'] for a in algos_present]
        x = np.arange(len(algos_present))
        colors = [ALGO_COLORS[a] for a in algos_present]
        bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='black', width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in algos_present], fontsize=8, rotation=15)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        if col == 0:
            ax.set_ylabel('Mean Norm. Q-Error')
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig1_q_estimation.pdf")
    save_fig(fig, "fig1_q_estimation.png")


# ====================================================================
# Figure 2: Uncertainty Decomposition (Exp 3)
# ====================================================================
def figure_uncertainty():
    """2×2 grid: epistemic over training + aleatoric per head."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Uncertainty Decomposition: Aleatoric vs Epistemic', fontsize=14)

    for idx, env_name in enumerate(ENVS[:4]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        data = load_pkl("exp3_uncertainty", f"uncertainty_{env_name}.pkl")
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            continue

        results = data['results']
        for algo in ['sac', 'resac']:
            if algo not in results:
                continue
            curves = results[algo].get('training_curves', {})
            if 'q_std_mean' in curves:
                q_std = curves['q_std_mean']
                smoothed = smooth(q_std, max(1, len(q_std)//30))
                ax.plot(np.arange(len(smoothed)), smoothed,
                        label=f"{ALGO_LABELS[algo]} (K={results[algo]['ensemble_size']})",
                        color=ALGO_COLORS[algo], linewidth=2)

        ax.set_title(ENV_SHORT[env_name])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Q-Std (Epistemic)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig, "fig2_uncertainty.pdf")
    save_fig(fig, "fig2_uncertainty.png")


# ====================================================================
# Figure 3: Δ(μ,π) Analysis (Exp 2)
# ====================================================================
def figure_delta():
    """1×4 bar chart of Δ(μ,π) across environments."""
    fig, axes = plt.subplots(1, len(ENVS), figsize=(16, 4))
    fig.suptitle('Buffer-Policy Q-Value Gap Δ(μ,π)', fontsize=14)

    for col, env_name in enumerate(ENVS):
        ax = axes[col]
        data = load_pkl("exp2_delta_analysis", f"delta_analysis_{env_name}.pkl")
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            continue

        results = data['results']
        algos_present = [a for a in ['sac', 'resac', 'td3', 'dsac'] if a in results]
        means = [results[a]['mean_delta'] for a in algos_present]
        stds = [results[a]['std_delta'] for a in algos_present]
        x = np.arange(len(algos_present))
        colors = [ALGO_COLORS[a] for a in algos_present]

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                      edgecolor='black', capsize=3, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_LABELS[a] for a in algos_present], fontsize=8)
        ax.set_title(ENV_SHORT[env_name])
        if col == 0:
            ax.set_ylabel('Mean Δ(μ,π)')

    plt.tight_layout()
    save_fig(fig, "fig3_delta.pdf")
    save_fig(fig, "fig3_delta.png")


# ====================================================================
# Figure 4: β Ablation (Exp 6)
# ====================================================================
def figure_beta_ablation():
    """2×2 learning curves for each env with different β configs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Adaptive β_lcb Ablation', fontsize=14)

    for idx, env_name in enumerate(ENVS[:4]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        data = load_pkl("exp6_beta_ablation", f"beta_ablation_{env_name}.pkl")
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            continue

        results = data['results']
        for name, d in results.items():
            cfg = d['config']
            mean = d['eval_mean']
            w = max(1, len(mean) // 20)
            smoothed = smooth(mean, w)
            ax.plot(np.arange(len(smoothed)), smoothed,
                    label=cfg.label, color=cfg.color, linewidth=1.5)

        ax.set_title(ENV_SHORT[env_name])
        ax.set_xlabel('Eval Point')
        ax.set_ylabel('Eval Reward')
        ax.legend(fontsize=6, loc='lower right')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig, "fig4_beta_ablation.pdf")
    save_fig(fig, "fig4_beta_ablation.png")


# ====================================================================
# Figure 5: Ensemble Size Ablation (Exp 9)
# ====================================================================
def figure_ensemble_ablation():
    """2×2 learning curves with K=2,5,10,20."""
    K_COLORS = {2: '#ff7f0e', 5: '#2ca02c', 10: '#1f77b4', 20: '#d62728'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Ensemble Size Ablation', fontsize=14)

    for idx, env_name in enumerate(ENVS[:4]):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        data = load_pkl("exp9_ensemble", f"ensemble_ablation_{env_name}.pkl")
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            continue

        results = data['results']
        for K in [2, 5, 10, 20]:
            if K not in results:
                continue
            d = results[K]
            mean = d['eval_mean']
            w = max(1, len(mean) // 20)
            smoothed = smooth(mean, w)
            ax.plot(np.arange(len(smoothed)), smoothed,
                    label=f'K={K}', color=K_COLORS[K], linewidth=2)

        ax.set_title(ENV_SHORT[env_name])
        ax.set_xlabel('Eval Point')
        ax.set_ylabel('Eval Reward')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig, "fig5_ensemble.pdf")
    save_fig(fig, "fig5_ensemble.png")


# ====================================================================
# Figure 6: Multi-Environment Comparison (Exp 7)
# ====================================================================
def figure_env_comparison():
    """Learning curves across all environments."""
    data = load_pkl("exp7_env_expansion", "env_expansion_results.pkl")
    if data is None:
        print("  No environment comparison data")
        return

    envs_with_data = [e for e in ENVS if e in data and data[e]]
    if not envs_with_data:
        print("  No environment data available")
        return

    n = len(envs_with_data)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle('Multi-Environment Performance Comparison', fontsize=14)

    for ax, env_name in zip(axes, envs_with_data):
        for algo in ['sac', 'resac', 'td3', 'dsac']:
            if algo not in data[env_name]:
                continue
            d = data[env_name][algo]
            mean = d['eval_rewards'].mean(axis=0)
            w = max(1, len(mean) // 20)
            smoothed = smooth(mean, w)
            ax.plot(np.arange(len(smoothed)), smoothed,
                    label=ALGO_LABELS[algo], color=ALGO_COLORS[algo], linewidth=2)
        ax.set_title(ENV_SHORT.get(env_name, env_name))
        ax.set_xlabel('Eval Point')
        ax.set_ylabel('Eval Reward')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    save_fig(fig, "fig6_env_comparison.pdf")
    save_fig(fig, "fig6_env_comparison.png")


# ====================================================================
# Figure 7: Efficiency Comparison (Exp 8)
# ====================================================================
def figure_efficiency():
    """Bar charts of computational cost."""
    data = load_pkl("exp8_efficiency", "efficiency_Hopper-v2.pkl")
    if data is None:
        print("  No efficiency data")
        return

    results = data['results']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Computational Efficiency', fontsize=14)

    algos = [a for a in ['sac', 'resac', 'td3', 'dsac'] if a in results]
    x = np.arange(len(algos))

    # Time breakdown
    ax = axes[0]
    collect = [results[a]['collect_mean']*1000 for a in algos]
    train = [results[a]['train_mean']*1000 for a in algos]
    ax.bar(x, collect, 0.6, label='Collect', color='#66c2a5')
    ax.bar(x, train, 0.6, bottom=collect, label='Train', color='#8da0cb')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos])
    ax.set_ylabel('ms/iter')
    ax.set_title('Per-Iteration Time')
    ax.legend()

    # Per gradient step
    ax = axes[1]
    grad_ms = [results[a]['time_per_grad_step']*1000 for a in algos]
    bars = ax.bar(x, grad_ms, 0.6, color=[ALGO_COLORS[a] for a in algos],
                  edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_LABELS[a] for a in algos])
    ax.set_ylabel('ms/grad step')
    ax.set_title('Per Gradient Step')
    for bar, val in zip(bars, grad_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_fig(fig, "fig7_efficiency.pdf")
    save_fig(fig, "fig7_efficiency.png")


# ====================================================================
# Summary Table (LaTeX)
# ====================================================================
def generate_latex_table():
    """Generate LaTeX table of final performance across all experiments."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Final evaluation reward across environments (mean $\pm$ std).}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Hopper & HalfCheetah & Walker2d & Ant \\")
    lines.append(r"\midrule")

    # Load environment comparison data
    data = load_pkl("exp7_env_expansion", "env_expansion_results.pkl")
    if data is not None:
        for algo in ['sac', 'resac', 'td3', 'dsac']:
            row = f"{ALGO_LABELS.get(algo, algo)}"
            for env_name in ENVS:
                if env_name in data and algo in data[env_name]:
                    d = data[env_name][algo]
                    row += f" & ${d['final_mean']:.1f} \\pm {d['final_std']:.1f}$"
                else:
                    row += " & ---"
            row += r" \\"
            lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_str = '\n'.join(lines)
    table_path = os.path.join(OUTPUT_DIR, "performance_table.tex")
    with open(table_path, 'w') as f:
        f.write(table_str)
    print(f"Saved LaTeX table: {table_path}")
    print(table_str)


def save_fig(fig, filename):
    """Save figure to output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating paper figures...")
    print("=" * 50)

    # Generate each figure (silently skip if data not available)
    for name, func in [
        ("Fig 1: Q-Estimation", figure_q_estimation),
        ("Fig 2: Uncertainty", figure_uncertainty),
        ("Fig 3: Δ(μ,π)", figure_delta),
        ("Fig 4: β Ablation", figure_beta_ablation),
        ("Fig 5: Ensemble Size", figure_ensemble_ablation),
        ("Fig 6: Environment Comparison", figure_env_comparison),
        ("Fig 7: Efficiency", figure_efficiency),
    ]:
        print(f"\n{name}...")
        try:
            func()
        except Exception as e:
            print(f"  Skipped: {e}")

    # LaTeX table
    print("\nLaTeX Table...")
    try:
        generate_latex_table()
    except Exception as e:
        print(f"  Skipped: {e}")

    print(f"\n✅ All paper figures saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
