"""Experiment 3: Uncertainty Decomposition — Aleatoric vs Epistemic Over Training.

RE-SAC's core contribution: disentangle aleatoric (L1 weight norm) and
epistemic (ensemble Q-variance) uncertainty. This experiment logs both
components during training to show:

1. Epistemic uncertainty decreases as the critic learns (more data in-distribution)
2. Aleatoric uncertainty stays relatively constant (inherent env stochasticity)
3. Adaptive β_lcb responds to epistemic uncertainty changes

Implementation:
  - Modifies the scan body to output reg_norm (aleatoric) and q_std (epistemic)
  - These are already partially logged (q_std_mean), but reg_norm is not
  - Also adds per-iteration uncertainty decomposition on a held-out batch

Usage:
    python -m jax_experiments.experiments.exp3_uncertainty_decomp \
        --env Hopper-v2 --mode analyze

    # Or train with extended logging (adds reg_norm logging to train loop):
    python -m jax_experiments.experiments.exp3_uncertainty_decomp \
        --env Hopper-v2 --mode train --algo resac
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
from jax_experiments.experiments.exp1_q_estimation import (
    ALGOS, ENVS, RESULTS_ROOT, load_checkpoint_params
)

OUTPUT_DIR = "jax_experiments/experiments/results/exp3_uncertainty"


def compute_uncertainty_decomposition(agent, env, algo, config, n_episodes=20):
    """Compute aleatoric and epistemic uncertainty for a trained agent.

    Aleatoric proxy: L1 weight norm of each critic head (measures sensitivity)
    Epistemic proxy: Std of Q-values across ensemble heads

    Returns dict with per-step uncertainty values.
    """
    if algo not in ("sac", "resac", "resac_v2"):
        print(f"  Skipping {algo} — uncertainty decomposition requires ensemble critic")
        return None

    rng_key = jax.random.PRNGKey(42)

    # --- Aleatoric: L1 weight norm per head ---
    reg_norm = np.array(agent.critic.compute_reg_norm())  # [K]
    aleatoric_per_head = reg_norm

    # --- Epistemic: collect per-state Q-std ---
    all_q_std = []
    all_q_mean = []
    all_obs = []
    ep_returns = []

    for ep in range(n_episodes):
        obs = env.reset()
        episode_rew = []
        done = False
        step = 0

        while not done and step < config.max_episode_steps:
            obs_jax = jnp.array(obs)[None]
            rng_key, act_key = jax.random.split(rng_key)

            # Get action from policy
            action, _ = agent.policy.sample(obs_jax, act_key)
            action_np = np.array(action[0])

            # Get Q-values from all ensemble heads
            act_jax = jnp.array(action_np)[None]
            q_all = agent.critic(obs_jax, act_jax)  # [K, 1]
            q_vals = np.array(q_all[:, 0])  # [K]

            all_q_std.append(float(q_vals.std()))
            all_q_mean.append(float(q_vals.mean()))
            all_obs.append(obs.copy())

            next_obs, reward, terminated, info = env.step(action_np)
            episode_rew.append(float(reward))
            obs = next_obs
            done = terminated
            step += 1

        ep_returns.append(sum(episode_rew))

    q_std_arr = np.array(all_q_std)
    q_mean_arr = np.array(all_q_mean)

    # Normalized epistemic: q_std / |q_mean| (scale-invariant)
    q_mean_safe = np.where(np.abs(q_mean_arr) < 1.0, 1.0, np.abs(q_mean_arr))
    relative_epistemic = q_std_arr / q_mean_safe

    return {
        'aleatoric_per_head': aleatoric_per_head,  # [K]
        'aleatoric_total': float(aleatoric_per_head.mean()),
        'epistemic_mean': float(q_std_arr.mean()),
        'epistemic_std': float(q_std_arr.std()),
        'epistemic_median': float(np.median(q_std_arr)),
        'relative_epistemic': float(relative_epistemic.mean()),
        'q_std_per_step': q_std_arr,
        'q_mean_per_step': q_mean_arr,
        'mean_return': float(np.mean(ep_returns)),
        'ensemble_size': len(aleatoric_per_head),
        'n_samples': len(q_std_arr),
    }


def load_training_curves(algo, env_name, seed=8):
    """Load training metrics logged during training (q_std_mean, etc)."""
    run_name = f"{algo}_{env_name}_{seed}"
    log_dir = os.path.join(RESULTS_ROOT, run_name, "logs")

    curves = {}
    for metric in ["q_std_mean", "q_mean", "critic_loss", "policy_loss",
                    "alpha", "log_prob", "eval_reward", "iteration",
                    "total_steps"]:
        path = os.path.join(log_dir, f"{metric}.npy")
        if os.path.exists(path):
            curves[metric] = np.load(path)

    return curves


def analyze_all(env_name, seed=8, n_episodes=20):
    """Analyze uncertainty decomposition for all ensemble-based algos."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    # Post-hoc analysis on final checkpoints
    for algo in ["sac", "resac", "resac_v2"]:
        print(f"\n{'='*50}")
        print(f"  {algo.upper()} on {env_name} — Uncertainty Decomposition")
        print(f"{'='*50}")
        try:
            agent, env, config = load_checkpoint_params(algo, env_name, seed)
            data = compute_uncertainty_decomposition(
                agent, env, algo, config, n_episodes)
            if data is not None:
                # Also load training curves for temporal analysis
                curves = load_training_curves(algo, env_name, seed)
                data['training_curves'] = curves
                results[algo] = data

                print(f"  Aleatoric (L1 norm mean): {data['aleatoric_total']:.4f}")
                print(f"  Aleatoric per head: {data['aleatoric_per_head']}")
                print(f"  Epistemic (Q-std mean): {data['epistemic_mean']:.4f}")
                print(f"  Relative Epistemic: {data['relative_epistemic']:.4f}")
                print(f"  Mean return: {data['mean_return']:.1f}")
            env.close()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"uncertainty_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    return results


def plot_uncertainty(env_name, output_dir=None):
    """Plot uncertainty decomposition: dual-axis aleatoric vs epistemic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"uncertainty_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    output_dir = output_dir or OUTPUT_DIR

    colors = {'sac': '#ff7f0e', 'resac': '#4488FF', 'resac_v2': '#9944FF'}
    labels = {'sac': 'SAC (K=2)', 'resac': 'RE-SAC (K=10)',
              'resac_v2': 'RE-SAC v2 (K=10)'}

    # Figure 1: Training curves of Q-std (epistemic over time)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Uncertainty Decomposition — {env_name}', fontsize=14)

    # Panel 1: Q-std over training
    ax = axes[0, 0]
    for algo in ["sac", "resac", "resac_v2"]:
        if algo not in results:
            continue
        curves = results[algo].get('training_curves', {})
        if 'q_std_mean' in curves:
            q_std = curves['q_std_mean']
            iters = curves.get('iteration', np.arange(len(q_std)))
            # Smooth
            window = max(1, len(q_std) // 50)
            smoothed = np.convolve(q_std, np.ones(window)/window, mode='valid')
            ax.plot(iters[:len(smoothed)], smoothed, label=labels[algo],
                    color=colors[algo], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Q-Std (Epistemic)')
    ax.set_title('Epistemic Uncertainty Over Training')
    ax.legend(fontsize=9)

    # Panel 2: Aleatoric per head
    ax = axes[0, 1]
    for algo in ["sac", "resac", "resac_v2"]:
        if algo not in results:
            continue
        d = results[algo]
        heads = d['aleatoric_per_head']
        ax.bar(np.arange(len(heads)) + (["sac", "resac", "resac_v2"].index(algo)) * 0.25,
               heads, width=0.2, label=labels[algo], color=colors[algo], alpha=0.8)
    ax.set_xlabel('Critic Head Index')
    ax.set_ylabel('L1 Weight Norm')
    ax.set_title('Aleatoric Proxy Per Critic Head (Final Checkpoint)')
    ax.legend(fontsize=9)

    # Panel 3: Epistemic distribution (histogram of per-step Q-std)
    ax = axes[1, 0]
    for algo in ["sac", "resac", "resac_v2"]:
        if algo not in results:
            continue
        q_std = results[algo]['q_std_per_step']
        ax.hist(q_std, bins=50, alpha=0.35, label=labels[algo],
                color=colors[algo], density=True)
        ax.axvline(q_std.mean(), color=colors[algo], linestyle='--', linewidth=2)
    ax.set_xlabel('Per-Step Q-Std')
    ax.set_ylabel('Density')
    ax.set_title('Epistemic Uncertainty Distribution (Final Policy)')
    ax.legend(fontsize=9)

    # Panel 4: Summary table as text
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for algo in ["sac", "resac", "resac_v2"]:
        if algo not in results:
            continue
        d = results[algo]
        table_data.append([
            labels[algo],
            f"{d['aleatoric_total']:.3f}",
            f"{d['epistemic_mean']:.3f}",
            f"{d['relative_epistemic']:.4f}",
            f"{d['ensemble_size']}",
            f"{d['mean_return']:.0f}",
        ])
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Algorithm', 'Aleatoric', 'Epistemic', 'Rel. Epist.',
                       'K', 'Return'],
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
    ax.set_title('Summary', fontsize=12, pad=20)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"uncertainty_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Uncertainty Decomposition")
    parser.add_argument("--mode", choices=["analyze", "plot", "both"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    for env_name in envs:
        if args.mode in ("analyze", "both"):
            analyze_all(env_name, args.seed, args.n_episodes)
        if args.mode in ("plot", "both"):
            plot_uncertainty(env_name)

    print("\n✅ Experiment 3 complete!")


if __name__ == "__main__":
    main()
