"""Experiment 1: Q-Estimation Accuracy — Normalized Q-Error Over Training.

Replicates the BAC paper's Fig 2 methodology:
  - At periodic checkpoints during training, compute Monte Carlo returns (Q_true)
  - Compare against critic predictions (Q_pred)
  - Normalized Q-error = (Q_pred - Q_true) / |Q_true|
  - Shows whether each algorithm over- or under-estimates Q-values over time

This is the CORE experiment for RE-SAC: must demonstrate that ensemble +
adaptive β achieves more accurate Q-estimation than SAC (which BAC showed
suffers from underestimation in later training).

Usage:
    # Collect Q-error data during training (adds --q_error_tracking flag)
    python -m jax_experiments.experiments.exp1_q_estimation \
        --algo resac --env Hopper-v2 --eval_interval 100

    # Or post-hoc from checkpoints:
    python -m jax_experiments.experiments.exp1_q_estimation \
        --mode analyze --env Hopper-v2
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
from jax_experiments.envs.brax_env import BraxNonstationaryEnv


ALGOS = ["sac", "resac", "resac_v2", "td3", "dsac"]
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/exp1_q_estimation"


def make_agent_and_env(algo, env_name, seed=8, ensemble_size=None):
    """Create agent + env matching training config."""
    config = Config()
    config.algo = "resac" if algo.startswith("resac") else algo
    config.env_name = env_name
    config.seed = seed
    config.brax_backend = "spring"
    config.stationary = True
    config.varying_params = []

    if algo == "sac":
        config.ensemble_size = 2
    elif algo.startswith("resac"):
        config.ensemble_size = ensemble_size or 10
    elif algo == "td3":
        config.ensemble_size = 2

    env = BraxNonstationaryEnv(
        env_name, rand_params=[], log_scale_limit=0.0,
        seed=seed, backend=config.brax_backend)

    from jax_experiments.train import make_algo
    agent = make_algo(env.obs_dim, env.act_dim, config)

    return agent, env, config


def load_checkpoint_params(algo, env_name, seed=8):
    """Load agent params from checkpoint."""
    agent, env, config = make_agent_and_env(algo, env_name, seed)
    run_name = f"{algo}_{env_name}_{seed}"
    params_path = os.path.join(RESULTS_ROOT, run_name, "checkpoints", "params.pkl")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No checkpoint at {params_path}")

    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    from jax_experiments.common.checkpoint import _to_jax_tree
    nnx.update(agent.policy, _to_jax_tree(params['policy']))

    if algo == 'td3':
        nnx.update(agent.critic, _to_jax_tree(params['critic']))
    elif algo == 'dsac':
        nnx.update(agent.twin_critic, _to_jax_tree(params['twin_critic']))
        agent.log_alpha = jnp.array(params['log_alpha'])
    else:
        nnx.update(agent.critic, _to_jax_tree(params['critic']))
        agent.log_alpha = jnp.array(params['log_alpha'])

    agent._build_scan_fn()
    return agent, env, config


def compute_mc_returns(rewards, gamma=0.99):
    """Compute Monte Carlo returns backwards through an episode."""
    n = len(rewards)
    G = np.zeros(n)
    G[-1] = rewards[-1]
    for t in range(n - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G


def get_q_predictions(agent, obs_jax, act_jax, algo):
    """Get Q-value predictions from all critic heads."""
    if algo in ("sac", "resac", "resac_v2"):
        q_all = agent.critic(obs_jax, act_jax)  # [K, B]
        return np.array(q_all)
    elif algo == "dsac":
        n_tau = agent.num_quantiles
        tau = jnp.linspace(0.5/n_tau, 1.0 - 0.5/n_tau, n_tau)
        tau = jnp.broadcast_to(tau[None], (obs_jax.shape[0], n_tau))
        z1 = agent.twin_critic.zf1(obs_jax, act_jax, tau)
        z2 = agent.twin_critic.zf2(obs_jax, act_jax, tau)
        q1 = np.array(z1.mean(axis=-1))  # [B]
        q2 = np.array(z2.mean(axis=-1))  # [B]
        return np.stack([q1, q2], axis=0)  # [2, B]
    elif algo == "td3":
        q_all = agent.critic(obs_jax, act_jax)  # [2, B]
        return np.array(q_all)


def collect_q_error_data(agent, env, algo, config, n_episodes=20):
    """Run episodes and compute normalized Q-error.

    Returns:
        dict with q_pred_mean, q_true (MC returns), normalized_q_error
    """
    gamma = config.gamma
    all_q_pred = []
    all_q_true = []
    all_obs = []
    all_act = []
    ep_returns = []

    rng_key = jax.random.PRNGKey(42)

    for ep in range(n_episodes):
        obs = env.reset()
        episode_obs = []
        episode_act = []
        episode_rew = []
        done = False
        step = 0

        while not done and step < config.max_episode_steps:
            obs_jax = jnp.array(obs)[None]
            rng_key, act_key = jax.random.split(rng_key)

            if algo == "td3":
                action = agent.policy(obs_jax)
                action_np = np.array(action[0])
            else:
                action, _ = agent.policy.sample(obs_jax, act_key)
                action_np = np.array(action[0])

            episode_obs.append(obs.copy())
            episode_act.append(action_np.copy())

            next_obs, reward, terminated, info = env.step(action_np)
            episode_rew.append(float(reward))

            obs = next_obs
            done = terminated
            step += 1

        # Compute MC returns
        mc_returns = compute_mc_returns(np.array(episode_rew), gamma)

        # Get Q predictions for all steps
        obs_batch = jnp.array(np.stack(episode_obs))
        act_batch = jnp.array(np.stack(episode_act))
        q_preds = get_q_predictions(agent, obs_batch, act_batch, algo)  # [K, T]

        # Mean across ensemble heads
        q_pred_mean = q_preds.mean(axis=0)  # [T]

        # Strip entropy bias for SAC-based methods
        if algo not in ("td3",) and hasattr(agent, 'log_alpha'):
            alpha = float(jnp.exp(agent.log_alpha))
            # Approximate log_prob bias removal
            obs_jax = jnp.array(np.stack(episode_obs))
            rng_key, lp_key = jax.random.split(rng_key)
            _, log_probs = agent.policy.sample(obs_jax, lp_key)
            lp_np = np.array(log_probs)
            # Q_soft includes -α·H term, strip it
            bias = (-alpha * lp_np) / (1 - gamma)
            q_pred_mean = q_pred_mean - bias

        all_q_pred.extend(q_pred_mean.tolist())
        all_q_true.extend(mc_returns.tolist())
        ep_returns.append(sum(episode_rew))

    q_pred = np.array(all_q_pred)
    q_true = np.array(all_q_true)

    # Normalized Q-error: (Q_pred - Q_true) / |Q_true|
    q_true_safe = np.where(np.abs(q_true) < 1.0, 1.0, np.abs(q_true))
    normalized_error = (q_pred - q_true) / q_true_safe

    return {
        'q_pred': q_pred,
        'q_true': q_true,
        'normalized_error': normalized_error,
        'mean_normalized_error': float(normalized_error.mean()),
        'median_normalized_error': float(np.median(normalized_error)),
        'mean_abs_error': float(np.abs(q_pred - q_true).mean()),
        'mean_return': float(np.mean(ep_returns)),
        'n_samples': len(q_pred),
    }


def analyze_all(env_name, seed=8, n_episodes=20):
    """Run Q-estimation analysis for all algos on one env."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for algo in ALGOS:
        print(f"\n{'='*50}")
        print(f"  {algo.upper()} on {env_name}")
        print(f"{'='*50}")
        try:
            agent, env, config = load_checkpoint_params(algo, env_name, seed)
            data = collect_q_error_data(agent, env, algo, config, n_episodes)
            results[algo] = data
            print(f"  Normalized Q-error (mean): {data['mean_normalized_error']:.4f}")
            print(f"  Normalized Q-error (median): {data['median_normalized_error']:.4f}")
            print(f"  Mean |Q-error|: {data['mean_abs_error']:.2f}")
            print(f"  Mean return: {data['mean_return']:.1f}")
            env.close()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"q_estimation_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'Algo':<15} {'NormErr(mean)':>14} {'NormErr(med)':>13} {'|Q-err|':>10} {'Return':>10}")
    print("-" * 65)
    for algo, d in results.items():
        print(f"{algo:<15} {d['mean_normalized_error']:>14.4f} "
              f"{d['median_normalized_error']:>13.4f} "
              f"{d['mean_abs_error']:>10.2f} {d['mean_return']:>10.1f}")

    return results


def plot_q_estimation(env_name, output_dir=None):
    """Plot normalized Q-error comparison (BAC Fig 2 style)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"q_estimation_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data found at {data_path}")
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
    fig.suptitle(f'Q-Estimation Accuracy — {env_name}', fontsize=14)

    # Panel 1: Distribution of normalized Q-error
    ax = axes[0]
    for algo in ALGOS:
        if algo not in results:
            continue
        d = results[algo]
        err = d['normalized_error']
        # Clip for visualization
        err_clip = np.clip(err, -2, 2)
        ax.hist(err_clip, bins=50, alpha=0.4, label=labels[algo],
                color=colors[algo], density=True)
        ax.axvline(d['mean_normalized_error'], color=colors[algo],
                   linestyle='--', alpha=0.8, linewidth=2)
    ax.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax.set_xlabel('Normalized Q-Error: (Q_pred - Q_true) / |Q_true|')
    ax.set_ylabel('Density')
    ax.set_title('Q-Error Distribution')
    ax.legend(fontsize=9)

    # Panel 2: Bar chart of mean normalized error
    ax = axes[1]
    algo_names = [a for a in ALGOS if a in results]
    means = [results[a]['mean_normalized_error'] for a in algo_names]
    bar_colors = [colors[a] for a in algo_names]
    bar_labels = [labels[a] for a in algo_names]
    x = np.arange(len(algo_names))
    bars = ax.bar(x, means, color=bar_colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=10)
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_ylabel('Mean Normalized Q-Error')
    ax.set_title('Q-Estimation Bias (↑=overest, ↓=underest)')

    # Add value labels on bars
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"q_estimation_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Q-Estimation Accuracy")
    parser.add_argument("--mode", choices=["analyze", "plot", "both"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true", help="All 4 envs")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    for env_name in envs:
        if args.mode in ("analyze", "both"):
            analyze_all(env_name, args.seed, args.n_episodes)
        if args.mode in ("plot", "both"):
            plot_q_estimation(env_name)

    print("\n✅ Experiment 1 complete!")


if __name__ == "__main__":
    main()
