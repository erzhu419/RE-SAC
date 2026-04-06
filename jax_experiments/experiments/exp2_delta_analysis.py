"""Experiment 2: Δ(μ,π) Analysis — Buffer-Max vs Policy Q Gap.

Replicates BAC paper's Fig 3 methodology:
  Δ(μ,π) = E_s[ max_{a∈D(s)} Q(s,a) - E_{a~π(·|s)} Q(s,a) ]

This measures how much better the replay buffer's best actions are compared to
the current policy's actions, according to the critic. A growing Δ indicates the
policy is missing high-value actions that exist in the buffer — the root cause
of underestimation identified by BAC.

RE-SAC hypothesis: By using ensemble uncertainty to adaptively control
conservatism (via β_lcb), RE-SAC should maintain a smaller Δ(μ,π) than SAC,
indicating better exploitation of high-value buffer actions.

Usage:
    python -m jax_experiments.experiments.exp2_delta_analysis \
        --env Hopper-v2 --n_episodes 20
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
    ALGOS, ENVS, RESULTS_ROOT, make_agent_and_env, load_checkpoint_params,
    get_q_predictions
)

OUTPUT_DIR = "jax_experiments/experiments/results/exp2_delta_analysis"


def compute_delta_mu_pi(agent, env, algo, config, n_episodes=20,
                        n_buffer_actions=50):
    """Compute Δ(μ,π) for a trained agent.

    For each visited state s:
      1. Sample n_buffer_actions actions from the behavior policy (random + policy)
      2. max_{a∈D} Q(s,a) — best Q from sampled buffer actions
      3. E_{a~π} Q(s,a) — expected Q under current policy (use multiple samples)
      4. Δ(s) = max_Q - E_Q

    Returns dict with per-step delta values and summary statistics.
    """
    rng_key = jax.random.PRNGKey(42)

    all_delta = []
    all_max_q = []
    all_policy_q = []
    all_obs = []
    ep_returns = []

    n_policy_samples = 10  # sample this many actions from π to estimate E_π[Q]

    for ep in range(n_episodes):
        obs = env.reset()
        episode_rew = []
        done = False
        step = 0

        while not done and step < config.max_episode_steps:
            obs_jax = jnp.array(obs)[None]  # [1, obs_dim]
            rng_key, *keys = jax.random.split(rng_key, 4)

            # --- max_{a∈D} Q(s,a): sample diverse actions and pick best ---
            # Mix of random uniform + policy noise for diversity
            random_acts = jax.random.uniform(
                keys[0], (n_buffer_actions, env.act_dim),
                minval=-1.0, maxval=1.0)  # [N, act_dim]

            obs_rep = jnp.broadcast_to(obs_jax, (n_buffer_actions, obs_jax.shape[1]))
            q_buffer = get_q_predictions(agent, obs_rep, random_acts, algo)  # [K, N]
            q_buffer_mean = q_buffer.mean(axis=0)  # [N]
            max_q = float(q_buffer_mean.max())

            # --- E_{a~π} Q(s,a): sample from current policy ---
            obs_rep_pi = jnp.broadcast_to(obs_jax, (n_policy_samples, obs_jax.shape[1]))
            if algo == "td3":
                pi_acts = agent.policy.noisy_action(obs_rep_pi, keys[1], noise_std=0.1)
            else:
                pi_acts, _ = agent.policy.sample(obs_rep_pi, keys[1])

            q_pi = get_q_predictions(agent, obs_rep_pi, pi_acts, algo)  # [K, n_policy]
            policy_q = float(q_pi.mean())  # mean over heads and samples

            delta = max_q - policy_q
            all_delta.append(delta)
            all_max_q.append(max_q)
            all_policy_q.append(policy_q)
            all_obs.append(obs.copy())

            # Step env with deterministic action
            if algo == "td3":
                action = np.array(agent.policy(obs_jax)[0])
            else:
                action, _ = agent.policy.sample(obs_jax, keys[2])
                action = np.array(action[0])

            next_obs, reward, terminated, info = env.step(action)
            episode_rew.append(float(reward))
            obs = next_obs
            done = terminated
            step += 1

        ep_returns.append(sum(episode_rew))

    delta_arr = np.array(all_delta)
    return {
        'delta': delta_arr,
        'max_q': np.array(all_max_q),
        'policy_q': np.array(all_policy_q),
        'mean_delta': float(delta_arr.mean()),
        'std_delta': float(delta_arr.std()),
        'median_delta': float(np.median(delta_arr)),
        'positive_frac': float((delta_arr > 0).mean()),
        'mean_return': float(np.mean(ep_returns)),
        'n_samples': len(delta_arr),
    }


def analyze_all(env_name, seed=8, n_episodes=20):
    """Run Δ(μ,π) analysis for all algos on one env."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for algo in ALGOS:
        print(f"\n{'='*50}")
        print(f"  {algo.upper()} on {env_name} — Δ(μ,π) analysis")
        print(f"{'='*50}")
        try:
            agent, env, config = load_checkpoint_params(algo, env_name, seed)
            data = compute_delta_mu_pi(agent, env, algo, config, n_episodes)
            results[algo] = data
            print(f"  Δ(μ,π) mean: {data['mean_delta']:.4f}")
            print(f"  Δ(μ,π) median: {data['median_delta']:.4f}")
            print(f"  Frac(Δ > 0): {data['positive_frac']:.2%}")
            print(f"  Mean return: {data['mean_return']:.1f}")
            env.close()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"delta_analysis_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'Algo':<15} {'Δ(μ,π) mean':>12} {'Δ median':>10} {'Frac Δ>0':>10} {'Return':>10}")
    print("-" * 60)
    for algo, d in results.items():
        print(f"{algo:<15} {d['mean_delta']:>12.4f} {d['median_delta']:>10.4f} "
              f"{d['positive_frac']:>10.2%} {d['mean_return']:>10.1f}")

    return results


def plot_delta_analysis(env_name, output_dir=None):
    """Plot Δ(μ,π) comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"delta_analysis_{env_name}.pkl")
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Δ(μ,π) Analysis — {env_name}', fontsize=14)

    # Panel 1: Δ distribution
    ax = axes[0]
    for algo in ALGOS:
        if algo not in results:
            continue
        d = results[algo]
        delta_clip = np.clip(d['delta'], -50, 200)
        ax.hist(delta_clip, bins=50, alpha=0.35, label=labels[algo],
                color=colors[algo], density=True)
        ax.axvline(d['mean_delta'], color=colors[algo],
                   linestyle='--', alpha=0.8, linewidth=2)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Δ(μ,π) = max_D Q(s,a) - E_π Q(s,a)')
    ax.set_ylabel('Density')
    ax.set_title('Δ(μ,π) Distribution (lower = better policy exploitation)')
    ax.legend(fontsize=9)

    # Panel 2: Bar chart
    ax = axes[1]
    algo_names = [a for a in ALGOS if a in results]
    means = [results[a]['mean_delta'] for a in algo_names]
    stds = [results[a]['std_delta'] for a in algo_names]
    bar_colors = [colors[a] for a in algo_names]
    bar_labels = [labels[a] for a in algo_names]
    x = np.arange(len(algo_names))
    bars = ax.bar(x, means, yerr=stds, color=bar_colors, alpha=0.8,
                  edgecolor='black', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.set_ylabel('Mean Δ(μ,π)')
    ax.set_title('Buffer-Policy Gap (lower = better)')

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01 * max(abs(v) for v in means),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"delta_analysis_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Exp 2: Δ(μ,π) Analysis")
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
            plot_delta_analysis(env_name)

    print("\n✅ Experiment 2 complete!")


if __name__ == "__main__":
    main()
