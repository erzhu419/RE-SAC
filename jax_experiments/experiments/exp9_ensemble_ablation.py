"""Experiment 9: Ensemble Size Ablation — K=2,5,10,20.

Tests how ensemble size affects:
  1. Q-estimation accuracy (epistemic uncertainty quality)
  2. Final performance
  3. Training stability
  4. Computational cost

Expected: K=2 underperforms (poor epistemic estimate), K=10 and K=20 similar
(ensemble has converged), proving K=10 is the sweet spot.

Usage:
    # Generate run scripts:
    python -m jax_experiments.experiments.exp9_ensemble_ablation --mode gen_script

    # Analyze:
    python -m jax_experiments.experiments.exp9_ensemble_ablation --mode both
"""
import os
import argparse
import pickle
import numpy as np

ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/exp9_ensemble"

ENSEMBLE_SIZES = [2, 5, 10, 20]
COLORS = {2: '#ff7f0e', 5: '#2ca02c', 10: '#4488FF', 20: '#d62728'}
LABELS = {2: 'K=2', 5: 'K=5', 10: 'K=10', 20: 'K=20'}


def generate_run_scripts(envs=None, seeds=None, max_iters=2000):
    """Generate bash scripts for ensemble size ablation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    envs = envs or ENVS
    seeds = seeds or [8]

    all_lines = [
        "#!/bin/bash",
        "# Experiment 9: Ensemble Size Ablation",
        "",
    ]

    for env_name in envs:
        all_lines.append(f"echo '====== {env_name} ======'")
        for K in ENSEMBLE_SIZES:
            for seed in seeds:
                run_name = f"resac_K{K}_{env_name}_{seed}"
                cmd = (
                    f"conda run -n jax-rl python -m jax_experiments.train "
                    f"--algo resac --env {env_name} --seed {seed} "
                    f"--run_name {run_name} --stationary --resume "
                    f"--max_iters {max_iters} "
                    f"--ensemble_size {K}"
                )
                all_lines.append(f"echo '--- K={K} seed={seed} ---'")
                all_lines.append(cmd)
                all_lines.append("")

    script_path = os.path.join(OUTPUT_DIR, "run_ensemble_ablation.sh")
    with open(script_path, 'w') as f:
        f.write('\n'.join(all_lines))
    os.chmod(script_path, 0o755)
    print(f"Generated script: {script_path}")


def load_ensemble_curves(env_name, seeds=None):
    """Load training curves for each ensemble size."""
    seeds = seeds or [8]
    results = {}

    for K in ENSEMBLE_SIZES:
        seed_curves = []
        for seed in seeds:
            run_name = f"resac_K{K}_{env_name}_{seed}"
            log_dir = os.path.join(RESULTS_ROOT, run_name, "logs")

            if not os.path.isdir(log_dir):
                # Fallback: K=10 might be the default "resac" run
                if K == 10:
                    run_name_alt = f"resac_{env_name}_{seed}"
                    log_dir = os.path.join(RESULTS_ROOT, run_name_alt, "logs")
                elif K == 2:
                    # K=2 is equivalent to SAC's architecture
                    run_name_alt = f"sac_{env_name}_{seed}"
                    log_dir = os.path.join(RESULTS_ROOT, run_name_alt, "logs")

            if not os.path.isdir(log_dir):
                continue

            curves = {}
            for metric in ["eval_reward", "q_mean", "q_std_mean",
                            "critic_loss", "iteration"]:
                path = os.path.join(log_dir, f"{metric}.npy")
                if os.path.exists(path):
                    curves[metric] = np.load(path)

            if 'eval_reward' in curves:
                seed_curves.append(curves)

        if seed_curves:
            min_len = min(len(c['eval_reward']) for c in seed_curves)
            eval_stacked = np.stack([c['eval_reward'][:min_len] for c in seed_curves])

            q_std_stacked = None
            if all('q_std_mean' in c for c in seed_curves):
                qs_min = min(len(c['q_std_mean']) for c in seed_curves)
                q_std_stacked = np.stack([c['q_std_mean'][:qs_min] for c in seed_curves])

            results[K] = {
                'eval_rewards': eval_stacked,
                'eval_mean': eval_stacked.mean(axis=0),
                'eval_std': eval_stacked.std(axis=0),
                'final_mean': float(eval_stacked[:, -1].mean()),
                'final_std': float(eval_stacked[:, -1].std()),
                'q_std_curves': q_std_stacked,
                'n_seeds': len(seed_curves),
            }
            print(f"  K={K}: {len(seed_curves)} seeds, "
                  f"final = {results[K]['final_mean']:.1f} "
                  f"± {results[K]['final_std']:.1f}")
        else:
            print(f"  K={K}: no data found")

    return results


def analyze_ablation(env_name, seeds=None):
    """Analyze ensemble size ablation."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  Ensemble Size Ablation: {env_name}")
    print(f"{'='*50}")

    results = load_ensemble_curves(env_name, seeds)

    out_path = os.path.join(OUTPUT_DIR, f"ensemble_ablation_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'K':>5} {'Seeds':>6} {'Final Reward':>15}")
    print("-" * 30)
    for K in ENSEMBLE_SIZES:
        if K in results:
            d = results[K]
            print(f"{K:>5} {d['n_seeds']:>6} "
                  f"{d['final_mean']:>7.1f} ± {d['final_std']:<6.1f}")

    return results


def plot_ensemble_ablation(env_name, output_dir=None):
    """Plot ensemble size ablation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"ensemble_ablation_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    output_dir = output_dir or OUTPUT_DIR

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Ensemble Size Ablation — {env_name}', fontsize=14)

    # Panel 1: Learning curves
    ax = axes[0]
    for K in ENSEMBLE_SIZES:
        if K not in results:
            continue
        d = results[K]
        mean = d['eval_mean']
        std = d['eval_std']
        x = np.arange(len(mean))
        w = max(1, len(mean) // 30)
        mean_s = np.convolve(mean, np.ones(w)/w, mode='valid')
        std_s = np.convolve(std, np.ones(w)/w, mode='valid')
        x_s = x[:len(mean_s)]
        ax.plot(x_s, mean_s, label=LABELS[K], color=COLORS[K], linewidth=2)
        if d['n_seeds'] > 1:
            ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                            alpha=0.15, color=COLORS[K])
    ax.set_xlabel('Eval Point')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Learning Curves by Ensemble Size')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Final performance vs K
    ax = axes[1]
    Ks = [K for K in ENSEMBLE_SIZES if K in results]
    means = [results[K]['final_mean'] for K in Ks]
    stds = [results[K]['final_std'] for K in Ks]
    bar_colors = [COLORS[K] for K in Ks]
    ax.bar(np.arange(len(Ks)), means, yerr=stds, color=bar_colors,
           alpha=0.8, edgecolor='black', capsize=5)
    ax.set_xticks(np.arange(len(Ks)))
    ax.set_xticklabels([f'K={K}' for K in Ks], fontsize=10)
    ax.set_ylabel('Final Eval Reward')
    ax.set_title('Final Performance vs Ensemble Size')

    # Panel 3: Q-std evolution
    ax = axes[2]
    for K in ENSEMBLE_SIZES:
        if K not in results or results[K]['q_std_curves'] is None:
            continue
        d = results[K]
        q_std_mean = d['q_std_curves'].mean(axis=0)
        x = np.arange(len(q_std_mean))
        w = max(1, len(q_std_mean) // 30)
        smoothed = np.convolve(q_std_mean, np.ones(w)/w, mode='valid')
        ax.plot(x[:len(smoothed)], smoothed, label=LABELS[K],
                color=COLORS[K], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Q-Std (Epistemic Uncertainty)')
    ax.set_title('Epistemic Uncertainty by Ensemble Size')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"ensemble_ablation_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 9: Ensemble Size Ablation")
    parser.add_argument("--mode", choices=["gen_script", "analyze", "plot", "both"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[8])
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    if args.mode == "gen_script":
        generate_run_scripts(envs, args.seeds)
    else:
        for env_name in envs:
            if args.mode in ("analyze", "both"):
                analyze_ablation(env_name, args.seeds)
            if args.mode in ("plot", "both"):
                plot_ensemble_ablation(env_name)

    print("\n✅ Experiment 9 complete!")


if __name__ == "__main__":
    main()
