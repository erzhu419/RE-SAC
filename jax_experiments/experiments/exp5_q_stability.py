"""Experiment 5: Q-Value Stability — Cross-Seed Variance Analysis.

Replicates BAC paper's Fig 13 methodology:
  - Train each algorithm with multiple seeds
  - Compare Q-value estimation variance across seeds
  - RE-SAC's ensemble should provide more stable Q estimates

Usage:
    # Analyze existing multi-seed runs:
    python -m jax_experiments.experiments.exp5_q_stability \
        --env Hopper-v2 --seeds 1 2 3 4 5

    # Launch multi-seed training:
    python -m jax_experiments.experiments.exp5_q_stability \
        --mode train --env Hopper-v2 --seeds 1 2 3 4 5 --algo resac
"""
import os
import sys
import argparse
import pickle
import numpy as np

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config

ALGOS = ["sac", "resac", "resac_v2", "td3", "dsac"]
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/exp5_q_stability"

DEFAULT_SEEDS = [1, 2, 3, 4, 5, 8, 42]


def load_training_curves(algo, env_name, seed):
    """Load training metrics from a run directory."""
    run_name = f"{algo}_{env_name}_{seed}"
    log_dir = os.path.join(RESULTS_ROOT, run_name, "logs")

    if not os.path.isdir(log_dir):
        return None

    curves = {}
    for metric in ["eval_reward", "q_mean", "q_std_mean", "critic_loss",
                    "policy_loss", "alpha", "log_prob", "iteration",
                    "total_steps", "train_reward_mean"]:
        path = os.path.join(log_dir, f"{metric}.npy")
        if os.path.exists(path):
            curves[metric] = np.load(path)

    if not curves or 'eval_reward' not in curves:
        return None
    return curves


def analyze_cross_seed(env_name, seeds=None):
    """Analyze Q-value and reward stability across seeds."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seeds = seeds or DEFAULT_SEEDS

    results = {}

    for algo in ALGOS:
        print(f"\n  {algo.upper()} on {env_name}:")
        seed_data = {}

        for seed in seeds:
            curves = load_training_curves(algo, env_name, seed)
            if curves is not None:
                seed_data[seed] = curves
                n_iters = len(curves.get('eval_reward', []))
                final_reward = curves['eval_reward'][-1] if n_iters > 0 else 0
                print(f"    Seed {seed}: {n_iters} eval points, "
                      f"final reward = {final_reward:.1f}")

        if not seed_data:
            print(f"    No data found for any seed")
            continue

        # Align curves to same length (min across seeds)
        common_len = min(len(d['eval_reward']) for d in seed_data.values())

        # Stack aligned curves
        eval_rewards = np.stack([
            d['eval_reward'][:common_len] for d in seed_data.values()
        ])  # [n_seeds, T]

        q_means = None
        if all('q_mean' in d for d in seed_data.values()):
            q_common = min(len(d['q_mean']) for d in seed_data.values())
            q_means = np.stack([
                d['q_mean'][:q_common] for d in seed_data.values()
            ])

        q_stds = None
        if all('q_std_mean' in d for d in seed_data.values()):
            qs_common = min(len(d['q_std_mean']) for d in seed_data.values())
            q_stds = np.stack([
                d['q_std_mean'][:qs_common] for d in seed_data.values()
            ])

        results[algo] = {
            'seeds': list(seed_data.keys()),
            'n_seeds': len(seed_data),
            'eval_rewards': eval_rewards,  # [n_seeds, T]
            'eval_reward_mean': eval_rewards.mean(axis=0),
            'eval_reward_std': eval_rewards.std(axis=0),
            'q_means': q_means,
            'q_stds': q_stds,
            'final_rewards': eval_rewards[:, -1],
            'final_reward_mean': float(eval_rewards[:, -1].mean()),
            'final_reward_std': float(eval_rewards[:, -1].std()),
            # Cross-seed Q-value variance (mean over time)
            'q_cross_seed_var': float(q_means.var(axis=0).mean()) if q_means is not None else None,
        }

        print(f"    Final reward: {results[algo]['final_reward_mean']:.1f} "
              f"± {results[algo]['final_reward_std']:.1f} "
              f"({len(seed_data)} seeds)")
        if results[algo]['q_cross_seed_var'] is not None:
            print(f"    Q cross-seed variance: {results[algo]['q_cross_seed_var']:.4f}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"q_stability_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'seeds': seeds, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'Algo':<15} {'Seeds':>6} {'Final Reward':>15} {'Q-Var(cross)':>14}")
    print("-" * 55)
    for algo, d in results.items():
        q_var = f"{d['q_cross_seed_var']:.4f}" if d['q_cross_seed_var'] is not None else "N/A"
        print(f"{algo:<15} {d['n_seeds']:>6} "
              f"{d['final_reward_mean']:>7.1f} ± {d['final_reward_std']:<6.1f} "
              f"{q_var:>14}")

    return results


def plot_q_stability(env_name, output_dir=None):
    """Plot Q-value stability comparison (BAC Fig 13 style)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"q_stability_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    output_dir = output_dir or OUTPUT_DIR

    colors = {
        'sac': '#ff7f0e', 'resac': '#4488FF', 'resac_v2': '#9944FF',
        'td3': '#2ca02c', 'dsac': '#d62728'
    }
    labels = {
        'sac': 'SAC', 'resac': 'RE-SAC', 'resac_v2': 'RE-SAC v2',
        'td3': 'TD3', 'dsac': 'DSAC'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Q-Value Stability Analysis — {env_name}', fontsize=14)

    # Panel 1: Eval reward mean ± std band
    ax = axes[0, 0]
    for algo, d in results.items():
        if d['n_seeds'] < 2:
            continue
        mean = d['eval_reward_mean']
        std = d['eval_reward_std']
        x = np.arange(len(mean))
        # Smooth
        w = max(1, len(mean) // 30)
        mean_s = np.convolve(mean, np.ones(w)/w, mode='valid')
        std_s = np.convolve(std, np.ones(w)/w, mode='valid')
        x_s = x[:len(mean_s)]
        ax.plot(x_s, mean_s, label=labels.get(algo, algo),
                color=colors.get(algo, 'gray'), linewidth=2)
        ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                        alpha=0.15, color=colors.get(algo, 'gray'))
    ax.set_xlabel('Eval Point')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Reward Stability Across Seeds (mean ± std)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Q-mean cross-seed band
    ax = axes[0, 1]
    for algo, d in results.items():
        if d['q_means'] is None or d['n_seeds'] < 2:
            continue
        mean = d['q_means'].mean(axis=0)
        std = d['q_means'].std(axis=0)
        x = np.arange(len(mean))
        w = max(1, len(mean) // 30)
        mean_s = np.convolve(mean, np.ones(w)/w, mode='valid')
        std_s = np.convolve(std, np.ones(w)/w, mode='valid')
        x_s = x[:len(mean_s)]
        ax.plot(x_s, mean_s, label=labels.get(algo, algo),
                color=colors.get(algo, 'gray'), linewidth=2)
        ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                        alpha=0.15, color=colors.get(algo, 'gray'))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Q-Mean')
    ax.set_title('Q-Value Stability Across Seeds')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Final reward box plot
    ax = axes[1, 0]
    algo_order = [a for a in ALGOS if a in results and results[a]['n_seeds'] >= 2]
    box_data = [results[a]['final_rewards'] for a in algo_order]
    if box_data:
        bp = ax.boxplot(box_data, labels=[labels.get(a, a) for a in algo_order],
                        patch_artist=True)
        for patch, algo in zip(bp['boxes'], algo_order):
            patch.set_facecolor(colors.get(algo, 'gray'))
            patch.set_alpha(0.6)
    ax.set_ylabel('Final Eval Reward')
    ax.set_title('Final Performance Distribution')
    ax.grid(alpha=0.3)

    # Panel 4: Cross-seed variance bar chart
    ax = axes[1, 1]
    algo_names = [a for a in algo_order if results[a]['q_cross_seed_var'] is not None]
    if algo_names:
        variances = [results[a]['q_cross_seed_var'] for a in algo_names]
        x = np.arange(len(algo_names))
        bars = ax.bar(x, variances,
                      color=[colors.get(a, 'gray') for a in algo_names],
                      alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(a, a) for a in algo_names])
        ax.set_ylabel('Mean Q-Value Variance (cross-seed)')
        ax.set_title('Q-Estimation Consistency (lower = more stable)')
        for bar, val in zip(bars, variances):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"q_stability_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def generate_multi_seed_script(env_name, seeds, algos=None):
    """Generate a bash script to launch multi-seed training runs."""
    algos = algos or ["sac", "resac"]
    lines = [
        "#!/bin/bash",
        f"# Multi-seed training for Q-stability experiment — {env_name}",
        f"# Generated for seeds: {seeds}",
        "",
    ]
    for algo in algos:
        for seed in seeds:
            run_name = f"{algo}_{env_name}_{seed}"
            cmd = (f"conda run -n jax-rl python -m jax_experiments.train "
                   f"--algo {algo} --env {env_name} --seed {seed} "
                   f"--run_name {run_name} --stationary --resume")
            if algo == "sac":
                cmd += " --ensemble_size 2"
            lines.append(f"echo '=== {algo.upper()} seed={seed} ==='")
            lines.append(cmd)
            lines.append("")

    script_path = os.path.join(OUTPUT_DIR, f"run_multiseed_{env_name}.sh")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(script_path, 0o755)
    print(f"Generated script: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description="Exp 5: Q-Value Stability")
    parser.add_argument("--mode", choices=["analyze", "plot", "both", "gen_script"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--algos", nargs="+", default=None,
                        help="Algos for script generation")
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    for env_name in envs:
        if args.mode == "gen_script":
            generate_multi_seed_script(env_name, args.seeds, args.algos)
        elif args.mode in ("analyze", "both"):
            analyze_cross_seed(env_name, args.seeds)
        if args.mode in ("plot", "both"):
            plot_q_stability(env_name)

    print("\n✅ Experiment 5 complete!")


if __name__ == "__main__":
    main()
