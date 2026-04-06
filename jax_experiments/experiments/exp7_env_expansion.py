"""Experiment 7: Environment Coverage Expansion.

Adds DMControl-style environments via Brax for broader evaluation.
Brax provides direct equivalents for many DMControl tasks:
  - quadruped (run/walk)
  - humanoid (stand/walk/run)
  - reacher

Also ensures the standard MuJoCo suite is fully covered.

Usage:
    # Generate expanded run scripts:
    python -m jax_experiments.experiments.exp7_env_expansion --mode gen_script

    # Analyze results:
    python -m jax_experiments.experiments.exp7_env_expansion --mode both
"""
import os
import argparse
import pickle
import numpy as np

RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/exp7_env_expansion"

# Standard MuJoCo (already supported via Brax spring backend)
MUJOCO_ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]

# Extended Brax environments (native Brax, no -v2 suffix)
BRAX_ENVS = ["humanoid", "reacher"]

# Full test suite
ALL_ENVS = MUJOCO_ENVS + BRAX_ENVS

ALGOS = ["sac", "resac", "td3", "dsac"]


def generate_run_scripts(envs=None, algos=None, seeds=None, max_iters=2000):
    """Generate bash scripts for expanded environment coverage."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    envs = envs or ALL_ENVS
    algos = algos or ALGOS
    seeds = seeds or [8]

    all_lines = [
        "#!/bin/bash",
        "# Experiment 7: Expanded Environment Coverage",
        f"# Envs: {envs}",
        f"# Algos: {algos}",
        "",
    ]

    for env_name in envs:
        all_lines.append(f"echo '====== {env_name} ======'")
        for algo in algos:
            for seed in seeds:
                run_name = f"{algo}_{env_name}_{seed}"

                # Base command
                cmd = (
                    f"conda run -n jax-rl python -m jax_experiments.train "
                    f"--algo {algo} --env {env_name} --seed {seed} "
                    f"--run_name {run_name} --stationary --resume "
                    f"--max_iters {max_iters}"
                )

                # SAC uses ensemble_size=2
                if algo == "sac":
                    cmd += " --ensemble_size 2"

                all_lines.append(f"echo '--- {algo.upper()} {env_name} seed={seed} ---'")
                all_lines.append(cmd)
                all_lines.append("")

    script_path = os.path.join(OUTPUT_DIR, "run_env_expansion.sh")
    with open(script_path, 'w') as f:
        f.write('\n'.join(all_lines))
    os.chmod(script_path, 0o755)
    print(f"Generated script: {script_path}")


def load_results(envs=None, algos=None, seeds=None):
    """Load training results for all env×algo combinations."""
    envs = envs or ALL_ENVS
    algos = algos or ALGOS
    seeds = seeds or [8]

    results = {}

    for env_name in envs:
        results[env_name] = {}
        for algo in algos:
            seed_rewards = []
            for seed in seeds:
                run_name = f"{algo}_{env_name}_{seed}"
                log_dir = os.path.join(RESULTS_ROOT, run_name, "logs")
                eval_path = os.path.join(log_dir, "eval_reward.npy")

                if os.path.exists(eval_path):
                    rewards = np.load(eval_path)
                    if len(rewards) > 0:
                        seed_rewards.append(rewards)

            if seed_rewards:
                min_len = min(len(r) for r in seed_rewards)
                stacked = np.stack([r[:min_len] for r in seed_rewards])
                results[env_name][algo] = {
                    'eval_rewards': stacked,
                    'final_mean': float(stacked[:, -1].mean()),
                    'final_std': float(stacked[:, -1].std()),
                    'peak_mean': float(stacked.mean(axis=0).max()),
                    'n_seeds': len(seed_rewards),
                }

    return results


def analyze_all(envs=None, seeds=None):
    """Print comprehensive results table."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = load_results(envs, seeds=seeds)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "env_expansion_results.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved to {out_path}")

    # Print table
    print(f"\n{'Environment':<18}", end="")
    for algo in ALGOS:
        print(f"  {algo.upper():>12}", end="")
    print()
    print("-" * (18 + 14 * len(ALGOS)))

    for env_name in results:
        print(f"{env_name:<18}", end="")
        for algo in ALGOS:
            if algo in results[env_name]:
                d = results[env_name][algo]
                print(f"  {d['final_mean']:>6.1f}±{d['final_std']:<4.1f}", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        print()

    return results


def plot_env_comparison(output_dir=None):
    """Plot multi-environment comparison grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, "env_expansion_results.pkl")
    if not os.path.exists(data_path):
        print("No data found")
        return

    with open(data_path, 'rb') as f:
        results = pickle.load(f)

    output_dir = output_dir or OUTPUT_DIR
    envs_with_data = [e for e in ALL_ENVS if e in results and results[e]]

    if not envs_with_data:
        print("No environment data available")
        return

    colors = {
        'sac': '#ff7f0e', 'resac': '#4488FF', 'resac_v2': '#9944FF',
        'td3': '#2ca02c', 'dsac': '#d62728'
    }
    labels = {
        'sac': 'SAC', 'resac': 'RE-SAC', 'resac_v2': 'RE-SAC v2',
        'td3': 'TD3', 'dsac': 'DSAC'
    }

    n_envs = len(envs_with_data)
    n_cols = min(3, n_envs)
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    fig.suptitle('Multi-Environment Performance Comparison', fontsize=14)

    for idx, env_name in enumerate(envs_with_data):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        for algo in ALGOS:
            if algo not in results[env_name]:
                continue
            d = results[env_name][algo]
            mean = d['eval_rewards'].mean(axis=0)
            x = np.arange(len(mean))
            w = max(1, len(mean) // 30)
            smoothed = np.convolve(mean, np.ones(w)/w, mode='valid')
            ax.plot(x[:len(smoothed)], smoothed,
                    label=labels.get(algo, algo),
                    color=colors.get(algo, 'gray'), linewidth=2)
            if d['n_seeds'] > 1:
                std = d['eval_rewards'].std(axis=0)
                std_s = np.convolve(std, np.ones(w)/w, mode='valid')
                ax.fill_between(x[:len(smoothed)],
                                smoothed - std_s, smoothed + std_s,
                                alpha=0.1, color=colors.get(algo, 'gray'))

        ax.set_title(env_name, fontsize=12)
        ax.set_xlabel('Eval Point')
        ax.set_ylabel('Eval Reward')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(len(envs_with_data), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, "env_comparison.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 7: Environment Coverage Expansion")
    parser.add_argument("--mode", choices=["gen_script", "analyze", "plot", "both"],
                        default="both")
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[8])
    args = parser.parse_args()

    if args.mode == "gen_script":
        generate_run_scripts(args.envs, seeds=args.seeds)
    elif args.mode in ("analyze", "both"):
        analyze_all(args.envs, args.seeds)
    if args.mode in ("plot", "both"):
        plot_env_comparison()

    print("\n✅ Experiment 7 complete!")


if __name__ == "__main__":
    main()
