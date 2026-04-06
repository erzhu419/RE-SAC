"""Experiment 6: Adaptive β Ablation — Systematic β_lcb Schedule Comparison.

Tests multiple β_lcb configurations to answer:
  1. Does removing weight_reg from actor loss improve performance?
  2. What is the optimal adaptive β schedule?
  3. Is fixed vs adaptive β better?

Configurations:
  - RE-SAC-old:  β=-2.0 (fixed), weight_reg=0.01 (original triple conservatism)
  - RE-SAC-noreg: β=-2.0 (fixed), weight_reg removed
  - RE-SAC-mild:  β=-1.0 (fixed)
  - RE-SAC-adapt-v1: β: -2.0 → -0.5, warmup=20%
  - RE-SAC-adapt-v2: β: -1.0 → -0.3, warmup=10%
  - RE-SAC-optimistic: β: -0.5 → 0.0, warmup=20%

Usage:
    # Generate run scripts:
    python -m jax_experiments.experiments.exp6_beta_ablation --mode gen_script

    # Analyze results after training:
    python -m jax_experiments.experiments.exp6_beta_ablation --mode both --env Hopper-v2
"""
import os
import sys
import argparse
import pickle
import numpy as np
from dataclasses import dataclass

ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/exp6_beta_ablation"


@dataclass
class BetaConfig:
    """One β ablation configuration."""
    name: str
    label: str
    beta: float
    adaptive_beta: bool = False
    beta_start: float = -2.0
    beta_end: float = -0.5
    beta_warmup: float = 0.2
    color: str = 'gray'


CONFIGS = [
    BetaConfig("resac_fixed_b2", "RE-SAC (β=-2.0, fixed)",
               beta=-2.0, color='#1f77b4'),
    BetaConfig("resac_fixed_b1", "RE-SAC (β=-1.0, fixed)",
               beta=-1.0, color='#ff7f0e'),
    BetaConfig("resac_fixed_b05", "RE-SAC (β=-0.5, fixed)",
               beta=-0.5, color='#2ca02c'),
    BetaConfig("resac_adapt_v1", "RE-SAC (β: -2→-0.5, adapt)",
               beta=-2.0, adaptive_beta=True,
               beta_start=-2.0, beta_end=-0.5, beta_warmup=0.2,
               color='#d62728'),
    BetaConfig("resac_adapt_v2", "RE-SAC (β: -1→-0.3, adapt)",
               beta=-1.0, adaptive_beta=True,
               beta_start=-1.0, beta_end=-0.3, beta_warmup=0.1,
               color='#9467bd'),
    BetaConfig("resac_optimistic", "RE-SAC (β: -0.5→0.0, adapt)",
               beta=-0.5, adaptive_beta=True,
               beta_start=-0.5, beta_end=0.0, beta_warmup=0.2,
               color='#8c564b'),
]


def generate_run_scripts(envs=None, seeds=None, max_iters=2000):
    """Generate bash scripts for all ablation configurations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    envs = envs or ENVS
    seeds = seeds or [8]

    all_lines = [
        "#!/bin/bash",
        "# Experiment 6: Adaptive β Ablation",
        f"# Envs: {envs}",
        f"# Seeds: {seeds}",
        "",
    ]

    for env_name in envs:
        all_lines.append(f"echo '====== {env_name} ======'")
        for cfg in CONFIGS:
            for seed in seeds:
                run_name = f"{cfg.name}_{env_name}_{seed}"
                cmd = (
                    f"conda run -n jax-rl python -m jax_experiments.train "
                    f"--algo resac --env {env_name} --seed {seed} "
                    f"--run_name {run_name} --stationary --resume "
                    f"--max_iters {max_iters} "
                    f"--beta {cfg.beta}"
                )
                if cfg.adaptive_beta:
                    cmd += (f" --adaptive_beta "
                            f"--beta_start {cfg.beta_start} "
                            f"--beta_end {cfg.beta_end} "
                            f"--beta_warmup {cfg.beta_warmup}")

                all_lines.append(f"echo '--- {cfg.label} seed={seed} ---'")
                all_lines.append(cmd)
                all_lines.append("")

    script_path = os.path.join(OUTPUT_DIR, "run_beta_ablation.sh")
    with open(script_path, 'w') as f:
        f.write('\n'.join(all_lines))
    os.chmod(script_path, 0o755)
    print(f"Generated script: {script_path}")

    # Also generate per-env scripts for parallel execution
    for env_name in envs:
        env_lines = ["#!/bin/bash", f"# Beta ablation for {env_name}", ""]
        for cfg in CONFIGS:
            for seed in seeds:
                run_name = f"{cfg.name}_{env_name}_{seed}"
                cmd = (
                    f"conda run -n jax-rl python -m jax_experiments.train "
                    f"--algo resac --env {env_name} --seed {seed} "
                    f"--run_name {run_name} --stationary --resume "
                    f"--max_iters {max_iters} "
                    f"--beta {cfg.beta}"
                )
                if cfg.adaptive_beta:
                    cmd += (f" --adaptive_beta "
                            f"--beta_start {cfg.beta_start} "
                            f"--beta_end {cfg.beta_end} "
                            f"--beta_warmup {cfg.beta_warmup}")
                env_lines.append(cmd)
                env_lines.append("")

        env_script = os.path.join(OUTPUT_DIR, f"run_beta_{env_name}.sh")
        with open(env_script, 'w') as f:
            f.write('\n'.join(env_lines))
        os.chmod(env_script, 0o755)
        print(f"  {env_script}")


def load_ablation_curves(env_name, seeds=None):
    """Load training curves for all β ablation runs."""
    seeds = seeds or [8]
    results = {}

    for cfg in CONFIGS:
        seed_curves = []
        for seed in seeds:
            run_name = f"{cfg.name}_{env_name}_{seed}"
            log_dir = os.path.join(RESULTS_ROOT, run_name, "logs")

            if not os.path.isdir(log_dir):
                continue

            curves = {}
            for metric in ["eval_reward", "q_mean", "q_std_mean",
                            "critic_loss", "policy_loss", "alpha",
                            "iteration", "beta_lcb"]:
                path = os.path.join(log_dir, f"{metric}.npy")
                if os.path.exists(path):
                    curves[metric] = np.load(path)

            if 'eval_reward' in curves:
                seed_curves.append(curves)

        if seed_curves:
            # Align lengths
            min_len = min(len(c['eval_reward']) for c in seed_curves)
            eval_stacked = np.stack([
                c['eval_reward'][:min_len] for c in seed_curves
            ])
            results[cfg.name] = {
                'config': cfg,
                'n_seeds': len(seed_curves),
                'eval_rewards': eval_stacked,
                'eval_mean': eval_stacked.mean(axis=0),
                'eval_std': eval_stacked.std(axis=0),
                'final_mean': float(eval_stacked[:, -1].mean()),
                'final_std': float(eval_stacked[:, -1].std()),
                'curves': seed_curves,
            }
            print(f"  {cfg.label}: {len(seed_curves)} seeds, "
                  f"final = {results[cfg.name]['final_mean']:.1f} "
                  f"± {results[cfg.name]['final_std']:.1f}")
        else:
            print(f"  {cfg.label}: no data found")

    return results


def analyze_ablation(env_name, seeds=None):
    """Run analysis on β ablation results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  β Ablation Analysis: {env_name}")
    print(f"{'='*60}")

    results = load_ablation_curves(env_name, seeds)

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"beta_ablation_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'Config':<35} {'Seeds':>5} {'Final Reward':>15}")
    print("-" * 58)
    for name, d in sorted(results.items(), key=lambda x: -x[1]['final_mean']):
        cfg = d['config']
        print(f"{cfg.label:<35} {d['n_seeds']:>5} "
              f"{d['final_mean']:>7.1f} ± {d['final_std']:<6.1f}")

    return results


def plot_beta_ablation(env_name, output_dir=None):
    """Plot β ablation comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"beta_ablation_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    output_dir = output_dir or OUTPUT_DIR

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'β_lcb Ablation — {env_name}', fontsize=14)

    # Panel 1: Learning curves
    ax = axes[0]
    for name, d in results.items():
        cfg = d['config']
        mean = d['eval_mean']
        std = d['eval_std']
        x = np.arange(len(mean))
        # Smooth
        w = max(1, len(mean) // 30)
        mean_s = np.convolve(mean, np.ones(w)/w, mode='valid')
        std_s = np.convolve(std, np.ones(w)/w, mode='valid')
        x_s = x[:len(mean_s)]
        ax.plot(x_s, mean_s, label=cfg.label, color=cfg.color, linewidth=2)
        if d['n_seeds'] > 1:
            ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                            alpha=0.1, color=cfg.color)
    ax.set_xlabel('Eval Point')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Learning Curves')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # Panel 2: Final performance bar chart
    ax = axes[1]
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['final_mean'])
    names = [r[1]['config'].label for r in sorted_results]
    means = [r[1]['final_mean'] for r in sorted_results]
    stds = [r[1]['final_std'] for r in sorted_results]
    bar_colors = [r[1]['config'].color for r in sorted_results]
    x = np.arange(len(names))
    bars = ax.barh(x, means, xerr=stds, color=bar_colors, alpha=0.8,
                   edgecolor='black', capsize=3)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Final Eval Reward')
    ax.set_title('Final Performance Comparison')
    ax.invert_yaxis()

    # Panel 3: β schedule visualization
    ax = axes[2]
    max_iters = 2000
    x_iter = np.linspace(0, max_iters, 200)
    for name, d in results.items():
        cfg = d['config']
        if cfg.adaptive_beta:
            warmup_iters = int(cfg.beta_warmup * max_iters)
            beta_vals = []
            for it in x_iter:
                if it < warmup_iters:
                    beta_vals.append(cfg.beta_start)
                else:
                    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
                    progress = min(1.0, progress)
                    beta_vals.append(cfg.beta_start + (cfg.beta_end - cfg.beta_start) * progress)
            ax.plot(x_iter, beta_vals, label=cfg.label, color=cfg.color,
                    linewidth=2, linestyle='-')
        else:
            ax.axhline(cfg.beta, label=cfg.label, color=cfg.color,
                       linewidth=2, linestyle='--')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('β_lcb')
    ax.set_title('β Schedule Over Training')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"beta_ablation_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Exp 6: β Ablation")
    parser.add_argument("--mode", choices=["gen_script", "analyze", "plot", "both"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[8])
    parser.add_argument("--max_iters", type=int, default=2000)
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    if args.mode == "gen_script":
        generate_run_scripts(envs, args.seeds, args.max_iters)
    else:
        for env_name in envs:
            if args.mode in ("analyze", "both"):
                analyze_ablation(env_name, args.seeds)
            if args.mode in ("plot", "both"):
                plot_beta_ablation(env_name)

    print("\n✅ Experiment 6 complete!")


if __name__ == "__main__":
    main()
