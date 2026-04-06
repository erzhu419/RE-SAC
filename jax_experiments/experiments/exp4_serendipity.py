"""Experiment 4: Serendipity Exploitation — Expert Injection Recovery.

Replicates BAC paper's Fig 12: inject a small number of expert demonstrations
into the replay buffer partway through training, and measure how quickly each
algorithm exploits these high-quality transitions.

RE-SAC hypothesis: Ensemble uncertainty should be LOW for states near expert
demonstrations (they're well-covered), so adaptive β automatically becomes less
conservative, enabling faster exploitation.

Implementation:
  1. Train normally for warmup_iters
  2. Inject N_expert expert transitions (from a pre-trained policy)
  3. Continue training and measure reward recovery speed

Usage:
    python -m jax_experiments.experiments.exp4_serendipity \
        --env Hopper-v2 --algo resac --inject_at 500 --n_expert 5000
"""
import os
import sys
import argparse
import pickle
import time
import numpy as np

if "JAX_PLATFORMS" not in os.environ:
    _device = "gpu"
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            _device = sys.argv[i + 1].lower()
            break
    os.environ["JAX_PLATFORMS"] = "cuda" if _device == "gpu" else _device

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.common.logging import Logger
from jax_experiments.envs.brax_env import BraxNonstationaryEnv
from jax_experiments.train import make_algo, evaluate, collect_samples

ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
OUTPUT_DIR = "jax_experiments/experiments/results/exp4_serendipity"


def generate_expert_data(env_name, n_transitions=5000, seed=42):
    """Generate expert transitions by running a pre-trained agent.

    If no pre-trained expert exists, generate near-expert data by running
    many random episodes and keeping only high-return episodes.
    """
    # Try to load a pre-trained SAC agent as expert
    from jax_experiments.experiments.exp1_q_estimation import load_checkpoint_params

    expert_data = {
        'obs': [], 'act': [], 'rew': [], 'next_obs': [], 'done': []
    }

    # Try loading a trained agent as expert
    for algo in ["resac", "sac"]:
        try:
            agent, env, config = load_checkpoint_params(algo, env_name, seed=8)
            print(f"  Using trained {algo.upper()} as expert source")

            rng_key = jax.random.PRNGKey(seed)
            collected = 0

            while collected < n_transitions:
                obs = env.reset()
                done = False
                step = 0
                while not done and step < config.max_episode_steps and collected < n_transitions:
                    obs_jax = jnp.array(obs)[None]
                    rng_key, act_key = jax.random.split(rng_key)

                    if algo == "td3":
                        action = np.array(agent.policy(obs_jax)[0])
                    else:
                        action, _ = agent.policy.sample(obs_jax, act_key)
                        action = np.array(action[0])

                    next_obs, reward, terminated, _ = env.step(action)

                    expert_data['obs'].append(obs.copy())
                    expert_data['act'].append(action.copy())
                    expert_data['rew'].append(float(reward))
                    expert_data['next_obs'].append(next_obs.copy())
                    expert_data['done'].append(float(terminated))

                    obs = next_obs
                    done = terminated
                    step += 1
                    collected += 1

            env.close()
            break  # Success, no need to try other algos
        except Exception as e:
            print(f"  Could not load {algo}: {e}")
            continue

    if not expert_data['obs']:
        raise RuntimeError("Could not generate expert data — no trained agents found")

    return {k: np.array(v) for k, v in expert_data.items()}


def run_serendipity_experiment(algo, env_name, inject_at=500, n_expert=5000,
                               max_iters=1000, seed=8):
    """Run training with expert injection at inject_at iteration.

    Returns training curves with injection marker.
    """
    config = Config()
    config.algo = "resac" if algo.startswith("resac") else algo
    config.env_name = env_name
    config.seed = seed
    config.max_iters = max_iters
    config.brax_backend = "spring"
    config.stationary = True
    config.varying_params = []

    if algo == "sac":
        config.ensemble_size = 2
    elif algo.startswith("resac"):
        config.ensemble_size = 10
    elif algo == "td3":
        config.ensemble_size = 2

    # Create env + agent
    env = BraxNonstationaryEnv(
        env_name, rand_params=[], log_scale_limit=0.0,
        seed=seed, backend=config.brax_backend)

    agent = make_algo(env.obs_dim, env.act_dim, config)
    policy_graphdef = nnx.graphdef(agent.policy)
    env.build_rollout_fn(policy_graphdef, context_graphdef=None)

    replay_buffer = ReplayBuffer(env.obs_dim, env.act_dim, config.replay_size)

    eval_env = BraxNonstationaryEnv(
        env_name, rand_params=[], log_scale_limit=0.0,
        seed=seed + 1000, backend=config.brax_backend)
    eval_env.build_rollout_fn(policy_graphdef, context_graphdef=None)

    # Generate expert data ahead of time
    print(f"  Generating {n_expert} expert transitions...")
    expert_data = generate_expert_data(env_name, n_expert, seed=42)
    expert_return = expert_data['rew'].sum() / max(1, expert_data['done'].sum())
    print(f"  Expert avg episode return: ~{expert_return:.1f}")

    # Training loop
    curves = {
        'iteration': [], 'eval_reward': [], 'q_std_mean': [],
        'critic_loss': [], 'inject_at': inject_at
    }
    total_steps = 0

    print(f"\n  Training {algo.upper()} on {env_name} "
          f"(inject expert at iter {inject_at})...")

    for iteration in range(max_iters):
        # Collect
        ep_rewards = collect_samples(
            agent, env, replay_buffer, config, config.samples_per_iter)
        total_steps += config.samples_per_iter

        # INJECT expert data at specified iteration
        if iteration == inject_at:
            print(f"\n  *** INJECTING {n_expert} expert transitions at iter {iteration} ***\n")
            replay_buffer.push_batch(
                expert_data['obs'], expert_data['act'],
                expert_data['rew'].reshape(-1, 1),
                expert_data['next_obs'],
                expert_data['done'].reshape(-1, 1))

        # Train
        metrics = {}
        if replay_buffer.size >= config.start_train_steps:
            sample_key = jax.random.PRNGKey(seed + iteration + total_steps)
            stacked = replay_buffer.sample_stacked(
                config.updates_per_iter, config.batch_size, rng_key=sample_key)

            beta_override = None
            if config.algo == "resac" and hasattr(config, 'adaptive_beta') and config.adaptive_beta:
                beta_override = agent.get_adaptive_beta(iteration, max_iters)

            metrics = agent.multi_update(stacked, **(
                {"beta_override": beta_override} if beta_override is not None else {}
            ))

        # Eval every 5 iters
        if iteration % 5 == 0:
            eval_mean, eval_std = evaluate(agent, eval_env, config, None)
            curves['iteration'].append(iteration)
            curves['eval_reward'].append(eval_mean)
            curves['q_std_mean'].append(metrics.get('q_std_mean', 0))
            curves['critic_loss'].append(metrics.get('critic_loss', 0))

            if iteration % 50 == 0:
                print(f"  Iter {iteration:4d} | Eval: {eval_mean:8.1f} | "
                      f"Q-std: {metrics.get('q_std_mean', 0):.3f}")

    env.close()
    eval_env.close()

    return {k: np.array(v) if isinstance(v, list) else v for k, v in curves.items()}


def run_experiment(env_name, inject_at=500, n_expert=5000, max_iters=1000,
                   seed=8, algos=None):
    """Run serendipity experiment for multiple algorithms."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    algos = algos or ["sac", "resac"]
    results = {}

    for algo in algos:
        print(f"\n{'='*60}")
        print(f"  Serendipity Experiment: {algo.upper()} on {env_name}")
        print(f"  Inject {n_expert} expert transitions at iter {inject_at}")
        print(f"{'='*60}")
        try:
            curves = run_serendipity_experiment(
                algo, env_name, inject_at, n_expert, max_iters, seed)
            results[algo] = curves
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"serendipity_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({
            'env': env_name, 'inject_at': inject_at,
            'n_expert': n_expert, 'results': results
        }, f)
    print(f"\nSaved to {out_path}")
    return results


def plot_serendipity(env_name, output_dir=None):
    """Plot serendipity exploitation comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"serendipity_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    inject_at = data['inject_at']
    output_dir = output_dir or OUTPUT_DIR

    colors = {
        'sac': '#ff7f0e', 'resac': '#4488FF', 'resac_v2': '#9944FF',
        'td3': '#2ca02c', 'dsac': '#d62728'
    }
    labels = {
        'sac': 'SAC', 'resac': 'RE-SAC', 'resac_v2': 'RE-SAC v2',
        'td3': 'TD3', 'dsac': 'DSAC'
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Serendipity Exploitation — {env_name}', fontsize=14)

    for algo, curves in results.items():
        iters = curves['iteration']
        rewards = curves['eval_reward']
        # Smooth
        window = max(1, len(rewards) // 30)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(iters[:len(smoothed)], smoothed, label=labels.get(algo, algo),
                color=colors.get(algo, 'gray'), linewidth=2)

    # Mark injection point
    ax.axvline(inject_at, color='red', linestyle='--', alpha=0.7,
               linewidth=2, label=f'Expert injection (iter {inject_at})')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Eval Reward')
    ax.set_title(f'Recovery After Expert Data Injection')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"serendipity_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Exp 4: Serendipity Exploitation")
    parser.add_argument("--mode", choices=["run", "plot", "both"], default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--algos", nargs="+", default=["sac", "resac"])
    parser.add_argument("--inject_at", type=int, default=500,
                        help="Iteration to inject expert data")
    parser.add_argument("--n_expert", type=int, default=5000,
                        help="Number of expert transitions to inject")
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    if args.mode in ("run", "both"):
        run_experiment(args.env, args.inject_at, args.n_expert,
                      args.max_iters, args.seed, args.algos)
    if args.mode in ("plot", "both"):
        plot_serendipity(args.env)

    print("\n✅ Experiment 4 complete!")


if __name__ == "__main__":
    main()
