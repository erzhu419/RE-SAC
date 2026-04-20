"""Generate publication-quality figures for the paper.

1. Combined 2x4 Oracle Q-Error figure (4 envs, each with 2 panels)
2. Combined 2x2 Training Curves figure (eval reward for all algos)

Usage:
    cd RE-SAC
    python -m jax_experiments.q_analysis.plot_paper_figures
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

DATA_DIR = "jax_experiments/q_analysis/results"
RESULTS_DIR = "jax_experiments/results"
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
ENV_SHORT = {"Hopper-v2": "Hopper", "HalfCheetah-v2": "HalfCheetah",
             "Walker2d-v2": "Walker2d", "Ant-v2": "Ant"}

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Algorithm display order and style (for paper)
ALGO_ORDER = ['resac', 'resac_v2', 'sac', 'dsac', 'td3',
              'resac_v5', 'resac_v5b', 'resac_v6b']

# Per-env best new version (for focused comparison in paper)
BEST_NEW = {
    "Hopper-v2": "resac_v5",
    "Walker2d-v2": "resac_v5b",
    "HalfCheetah-v2": "resac_v5",
    "Ant-v2": "resac_v6b",
}

PAPER_LABELS = {
    'resac': 'RE-SAC v1',
    'resac_v2': 'RE-SAC v2',
    'resac_v5': 'RE-SAC v5',
    'resac_v5b': 'RE-SAC v5b',
    'resac_v6b': 'RE-SAC v6b',
    'sac': 'SAC',
    'dsac': 'DSAC',
    'td3': 'TD3',
}
PAPER_COLORS = {
    'resac': '#2166AC',     # dark blue
    'resac_v2': '#762A83',  # purple
    'resac_v5': '#D6604D',  # coral red
    'resac_v5b': '#D6604D', # coral red (same family)
    'resac_v6b': '#D6604D', # coral red (same family)
    'sac': '#E66100',       # orange
    'dsac': '#B2182B',      # red
    'td3': '#1B7837',       # green
}
PAPER_MARKERS = {
    'resac': 'o',
    'resac_v2': 'P',
    'resac_v5': '*',
    'resac_v5b': '*',
    'resac_v6b': '*',
    'sac': 's',
    'dsac': 'D',
    'td3': '^',
}
PAPER_LINESTYLES = {
    'resac': '-',
    'resac_v2': '--',
    'resac_v5': '-',
    'resac_v5b': '-',
    'resac_v6b': '-',
    'sac': '-.',
    'dsac': ':',
    'td3': '-',
}


def plot_oracle_q_error_combined():
    """Create a combined figure: 4 envs, each showing Oracle MAE by rareness."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for idx, env_name in enumerate(ENVS):
        ax = axes[idx]

        stats_path = os.path.join(DATA_DIR, f"bin_stats_{env_name}.pkl")
        if not os.path.exists(stats_path):
            ax.text(0.5, 0.5, f"No data: {env_name}", ha='center', va='center',
                    transform=ax.transAxes)
            continue

        with open(stats_path, 'rb') as f:
            data = pickle.load(f)

        meta = data['meta']
        stats = data['stats']
        bin_centers = meta['bin_centers']

        for algo in ALGO_ORDER:
            if algo not in stats:
                continue
            s = stats[algo]
            mae = s['mae']
            valid = ~np.isnan(mae)
            if not valid.any():
                continue

            is_best_new = (algo == BEST_NEW.get(env_name))
            ax.plot(bin_centers[valid], mae[valid],
                    color=PAPER_COLORS[algo],
                    linewidth=2.5 if is_best_new else 1.8,
                    linestyle=PAPER_LINESTYLES[algo],
                    marker=PAPER_MARKERS[algo],
                    markersize=6 if is_best_new else 4,
                    markevery=3,
                    label=f"{PAPER_LABELS[algo]} ({s['overall_mae']:.1f})",
                    zorder=6 if is_best_new else (5 if algo == 'resac' else 3))

        ax.set_title(ENV_SHORT[env_name], fontweight='bold', fontsize=11)
        ax.set_ylabel('Oracle MAE')
        ax.set_xlabel('Mahalanobis Rareness')
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(loc='upper left', framealpha=0.8, edgecolor='#ccc')

    plt.tight_layout(pad=1.0)
    out_path = os.path.join(DATA_DIR, "paper_oracle_q_error_combined.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def plot_oracle_q_error_with_predictions():
    """Create a combined figure: 4 envs × 2 rows (Q predictions + Oracle MAE)."""
    fig = plt.figure(figsize=(12, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.40, wspace=0.28,
                  left=0.07, right=0.95, top=0.96, bottom=0.04)

    for idx, env_name in enumerate(ENVS):
        stats_path = os.path.join(DATA_DIR, f"bin_stats_{env_name}.pkl")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path, 'rb') as f:
            data = pickle.load(f)

        meta = data['meta']
        stats = data['stats']
        bin_centers = meta['bin_centers']

        # Top panel: Q-Value Predictions vs Ground Truth
        ax_top = fig.add_subplot(gs[idx, 0])
        # Bottom panel: Oracle MAE
        ax_bot = fig.add_subplot(gs[idx, 1])

        for algo in ALGO_ORDER:
            if algo not in stats:
                continue
            s = stats[algo]

            # Q predictions
            q_pred = s['q_pred']
            valid_q = ~np.isnan(q_pred)
            if valid_q.any():
                ax_top.plot(bin_centers[valid_q], q_pred[valid_q],
                           color=PAPER_COLORS[algo],
                           linewidth=1.5,
                           linestyle=PAPER_LINESTYLES[algo],
                           marker=PAPER_MARKERS[algo],
                           markersize=3.5, markevery=3,
                           label=PAPER_LABELS[algo])

            # Ground truth (dashed, same color)
            q_real = s['q_real']
            valid_gt = ~np.isnan(q_real)
            if valid_gt.any():
                ax_top.plot(bin_centers[valid_gt], q_real[valid_gt],
                           color=PAPER_COLORS[algo],
                           linestyle='--', linewidth=0.8, alpha=0.5)

            # Oracle MAE
            mae = s['mae']
            valid_m = ~np.isnan(mae)
            if valid_m.any():
                ax_bot.plot(bin_centers[valid_m], mae[valid_m],
                           color=PAPER_COLORS[algo],
                           linewidth=1.5,
                           linestyle=PAPER_LINESTYLES[algo],
                           marker=PAPER_MARKERS[algo],
                           markersize=3.5, markevery=3,
                           label=f"{PAPER_LABELS[algo]} ({s['overall_mae']:.1f})")

        # Add ground truth legend entry
        ax_top.plot([], [], 'k--', linewidth=0.8, alpha=0.5, label='MC Return')

        ax_top.set_title(f'{ENV_SHORT[env_name]} — Q Predictions', fontweight='bold')
        ax_top.set_ylabel('Q-Value')
        ax_top.set_xlabel('Mahalanobis Rareness')
        ax_top.grid(alpha=0.2)
        ax_top.legend(loc='best', fontsize=6.5, framealpha=0.8)

        ax_bot.set_title(f'{ENV_SHORT[env_name]} — Oracle MAE', fontweight='bold')
        ax_bot.set_ylabel('MAE')
        ax_bot.set_xlabel('Mahalanobis Rareness')
        ax_bot.grid(alpha=0.2)
        ax_bot.legend(loc='upper left', fontsize=6.5, framealpha=0.8)

    out_path = os.path.join(DATA_DIR, "paper_oracle_q_full.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def smooth(x, window=20):
    """Exponential moving average."""
    if len(x) < 2:
        return x
    alpha = 2.0 / (window + 1)
    result = np.empty_like(x, dtype=float)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
    return result


def plot_training_curves_combined():
    """Create a 2x2 grid of training curves (eval reward) for all algos."""
    algos_train = ['sac', 'td3', 'dsac', 'resac', 'resac_v2',
                    'resac_v5', 'resac_v5b', 'resac_v6b']
    seed = 8

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for idx, env_name in enumerate(ENVS):
        ax = axes[idx]

        for algo in algos_train:
            run_dir = os.path.join(RESULTS_DIR, f"{algo}_{env_name}_{seed}", "logs")
            eval_path = os.path.join(run_dir, "eval_reward.npy")
            iter_path = os.path.join(run_dir, "iteration.npy")

            if not os.path.exists(eval_path):
                continue

            eval_rewards = np.load(eval_path)
            # Eval happens every 5 iters, compute x-axis in env steps
            eval_interval = 5
            samples_per_iter = 4000
            x_steps = np.arange(len(eval_rewards)) * eval_interval * samples_per_iter

            smoothed = smooth(eval_rewards, window=10)

            is_best = (algo == BEST_NEW.get(env_name))
            ax.plot(x_steps / 1e6, smoothed,
                    color=PAPER_COLORS[algo],
                    linewidth=2.5 if is_best else 1.5,
                    linestyle=PAPER_LINESTYLES[algo],
                    label=PAPER_LABELS[algo],
                    alpha=0.95 if is_best else 0.9,
                    zorder=6 if is_best else 3)
            # Light fill for raw data variance
            ax.fill_between(x_steps / 1e6, eval_rewards, smoothed,
                           color=PAPER_COLORS[algo], alpha=0.06)

        ax.set_title(ENV_SHORT[env_name], fontweight='bold', fontsize=11)
        ax.set_xlabel('Environment Steps (M)')
        ax.set_ylabel('Eval Return')
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(loc='best', framealpha=0.8, edgecolor='#ccc')

    plt.tight_layout(pad=1.0)
    out_path = os.path.join(DATA_DIR, "paper_training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


def print_summary_table():
    """Print a LaTeX-ready summary table of Oracle MAE."""
    print("\n=== Oracle MAE Summary Table ===")
    print(f"{'Algorithm':<15}", end='')
    for env in ENVS:
        print(f" & {ENV_SHORT[env]:<14}", end='')
    print(" \\\\")
    print("\\midrule")

    for algo in ALGO_ORDER:
        print(f"{PAPER_LABELS[algo]:<15}", end='')
        for env in ENVS:
            stats_path = os.path.join(DATA_DIR, f"bin_stats_{env}.pkl")
            if not os.path.exists(stats_path):
                print(f" & {'--':<14}", end='')
                continue
            with open(stats_path, 'rb') as f:
                data = pickle.load(f)
            stats = data['stats']
            if algo in stats:
                mae = stats[algo]['overall_mae']
                # Bold the best (lowest)
                all_maes = [stats[a]['overall_mae'] for a in ALGO_ORDER if a in stats]
                if mae == min(all_maes):
                    print(f" & \\textbf{{{mae:<.1f}}}", end='')
                else:
                    print(f" & {mae:<13.1f}", end='')
            else:
                print(f" & {'--':<14}", end='')
        print(" \\\\")

    print("\n=== Final Eval Reward Table ===")
    seed = 8
    print(f"{'Algorithm':<15}", end='')
    for env in ENVS:
        print(f" & {ENV_SHORT[env]:<14}", end='')
    print(" \\\\")
    print("\\midrule")

    for algo in ALGO_ORDER:
        print(f"{PAPER_LABELS[algo]:<15}", end='')
        for env in ENVS:
            eval_path = os.path.join(RESULTS_DIR, f"{algo}_{env}_{seed}", "logs", "eval_reward.npy")
            if os.path.exists(eval_path):
                rewards = np.load(eval_path)
                final = rewards[-1]
                best = rewards.max()
                # Bold the best
                all_finals = []
                for a in ALGO_ORDER:
                    p = os.path.join(RESULTS_DIR, f"{a}_{env}_{seed}", "logs", "eval_reward.npy")
                    if os.path.exists(p):
                        all_finals.append(np.load(p)[-1])
                if final == max(all_finals):
                    print(f" & \\textbf{{{final:<.0f}}}", end='')
                else:
                    print(f" & {final:<13.0f}", end='')
            else:
                print(f" & {'--':<14}", end='')
        print(" \\\\")


if __name__ == "__main__":
    print("Generating paper figures...")
    plot_oracle_q_error_combined()
    plot_oracle_q_error_with_predictions()
    plot_training_curves_combined()
    print_summary_table()
    print("\nDone!")
