"""Experiment 8: Wall-Clock Efficiency Benchmark.

Compares computation costs:
  - Time per iteration (collection + training)
  - Time per gradient step
  - Total training time to reach target reward
  - Breakdown: collection vs training vs eval

Shows that VectorizedCritic + scan fusion makes RE-SAC's ensemble overhead
manageable despite K=10 critics vs K=2 for SAC.

Usage:
    python -m jax_experiments.experiments.exp8_efficiency \
        --env Hopper-v2 --n_warmup 5 --n_measure 20
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
from jax_experiments.envs.brax_env import BraxNonstationaryEnv
from jax_experiments.train import make_algo, collect_samples

ALGOS = ["sac", "resac", "td3", "dsac"]
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v4"]
OUTPUT_DIR = "jax_experiments/experiments/results/exp8_efficiency"


def benchmark_algo(algo, env_name, n_warmup=5, n_measure=20, seed=8):
    """Benchmark wall-clock time for one algorithm.

    Returns dict with timing breakdowns.
    """
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
        config.ensemble_size = 10
    elif algo == "td3":
        config.ensemble_size = 2

    env = BraxNonstationaryEnv(
        env_name, rand_params=[], log_scale_limit=0.0,
        seed=seed, backend=config.brax_backend)

    agent = make_algo(env.obs_dim, env.act_dim, config)
    policy_graphdef = nnx.graphdef(agent.policy)
    env.build_rollout_fn(policy_graphdef, context_graphdef=None)

    replay_buffer = ReplayBuffer(env.obs_dim, env.act_dim, config.replay_size)

    # Pre-fill buffer with random data
    print(f"  Pre-filling replay buffer...")
    obs = env.reset()
    for _ in range(config.start_train_steps + 1000):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()

    # Warmup JIT compilation
    print(f"  JIT warmup ({n_warmup} iters)...")
    for i in range(n_warmup):
        ep_rew = collect_samples(agent, env, replay_buffer, config,
                                 config.samples_per_iter)
        sample_key = jax.random.PRNGKey(seed + i)
        stacked = replay_buffer.sample_stacked(
            config.updates_per_iter, config.batch_size, rng_key=sample_key)
        metrics = agent.multi_update(stacked)
        # Force synchronization
        jax.block_until_ready(metrics)

    # Measure
    print(f"  Measuring ({n_measure} iters)...")
    collect_times = []
    train_times = []
    total_times = []
    sample_times = []

    for i in range(n_measure):
        # --- Collection ---
        t0 = time.perf_counter()
        ep_rew = collect_samples(agent, env, replay_buffer, config,
                                 config.samples_per_iter)
        jax.block_until_ready(replay_buffer.obs)  # sync
        t_collect = time.perf_counter() - t0
        collect_times.append(t_collect)

        # --- Sampling ---
        t0 = time.perf_counter()
        sample_key = jax.random.PRNGKey(seed + n_warmup + i)
        stacked = replay_buffer.sample_stacked(
            config.updates_per_iter, config.batch_size, rng_key=sample_key)
        jax.block_until_ready(stacked['obs'])
        t_sample = time.perf_counter() - t0
        sample_times.append(t_sample)

        # --- Training ---
        t0 = time.perf_counter()
        metrics = agent.multi_update(stacked)
        # Force synchronization to get accurate timing
        for v in metrics.values():
            if hasattr(v, 'block_until_ready'):
                v.block_until_ready()
        t_train = time.perf_counter() - t0
        train_times.append(t_train)

        total_times.append(t_collect + t_sample + t_train)

    env.close()

    collect_arr = np.array(collect_times)
    sample_arr = np.array(sample_times)
    train_arr = np.array(train_times)
    total_arr = np.array(total_times)

    result = {
        'algo': algo,
        'env': env_name,
        'ensemble_size': config.ensemble_size,
        'updates_per_iter': config.updates_per_iter,
        'samples_per_iter': config.samples_per_iter,
        'batch_size': config.batch_size,
        'n_measure': n_measure,
        # Per-iteration times (seconds)
        'collect_mean': float(collect_arr.mean()),
        'collect_std': float(collect_arr.std()),
        'sample_mean': float(sample_arr.mean()),
        'sample_std': float(sample_arr.std()),
        'train_mean': float(train_arr.mean()),
        'train_std': float(train_arr.std()),
        'total_mean': float(total_arr.mean()),
        'total_std': float(total_arr.std()),
        # Per gradient step
        'time_per_grad_step': float(train_arr.mean() / config.updates_per_iter),
        # Estimated total training time (2000 iters)
        'estimated_total_hours': float(total_arr.mean() * 2000 / 3600),
    }

    return result


def benchmark_all(env_name, n_warmup=5, n_measure=20, seed=8):
    """Benchmark all algorithms on one environment."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for algo in ALGOS:
        print(f"\n{'='*50}")
        print(f"  Benchmarking {algo.upper()} on {env_name}")
        print(f"{'='*50}")
        try:
            result = benchmark_algo(algo, env_name, n_warmup, n_measure, seed)
            results[algo] = result

            print(f"  Collect:  {result['collect_mean']*1000:.1f} ± "
                  f"{result['collect_std']*1000:.1f} ms/iter")
            print(f"  Sample:   {result['sample_mean']*1000:.1f} ± "
                  f"{result['sample_std']*1000:.1f} ms/iter")
            print(f"  Train:    {result['train_mean']*1000:.1f} ± "
                  f"{result['train_std']*1000:.1f} ms/iter")
            print(f"  Total:    {result['total_mean']*1000:.1f} ± "
                  f"{result['total_std']*1000:.1f} ms/iter")
            print(f"  Per grad step: {result['time_per_grad_step']*1000:.3f} ms")
            print(f"  Est. 2000-iter: {result['estimated_total_hours']:.2f} hours")

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"efficiency_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'Algo':<10} {'K':>3} {'Collect':>10} {'Train':>10} "
          f"{'Total':>10} {'/GradStep':>10} {'Est.Total':>10}")
    print("-" * 68)
    for algo, r in results.items():
        print(f"{algo:<10} {r['ensemble_size']:>3} "
              f"{r['collect_mean']*1000:>8.1f}ms "
              f"{r['train_mean']*1000:>8.1f}ms "
              f"{r['total_mean']*1000:>8.1f}ms "
              f"{r['time_per_grad_step']*1000:>8.3f}ms "
              f"{r['estimated_total_hours']:>8.2f}h")

    return results


def plot_efficiency(env_name, output_dir=None):
    """Plot efficiency comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"efficiency_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    output_dir = output_dir or OUTPUT_DIR

    colors = {
        'sac': '#ff7f0e', 'resac': '#4488FF',
        'td3': '#2ca02c', 'dsac': '#d62728'
    }
    labels = {
        'sac': 'SAC (K=2)', 'resac': 'RE-SAC (K=10)',
        'td3': 'TD3 (K=2)', 'dsac': 'DSAC (K=2)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Computational Efficiency — {env_name}', fontsize=14)

    algo_order = [a for a in ALGOS if a in results]

    # Panel 1: Stacked bar — time breakdown
    ax = axes[0]
    x = np.arange(len(algo_order))
    width = 0.6
    collect = [results[a]['collect_mean']*1000 for a in algo_order]
    sample = [results[a]['sample_mean']*1000 for a in algo_order]
    train = [results[a]['train_mean']*1000 for a in algo_order]

    ax.bar(x, collect, width, label='Collect', color='#66c2a5')
    ax.bar(x, sample, width, bottom=collect, label='Sample', color='#fc8d62')
    ax.bar(x, train, width,
           bottom=[c+s for c, s in zip(collect, sample)],
           label='Train', color='#8da0cb')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(a, a) for a in algo_order], fontsize=9)
    ax.set_ylabel('Time (ms/iter)')
    ax.set_title('Per-Iteration Time Breakdown')
    ax.legend(fontsize=9)

    # Panel 2: Per gradient step
    ax = axes[1]
    grad_times = [results[a]['time_per_grad_step']*1000 for a in algo_order]
    bar_colors = [colors.get(a, 'gray') for a in algo_order]
    bars = ax.bar(x, grad_times, width, color=bar_colors, alpha=0.8,
                  edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(a, a) for a in algo_order], fontsize=9)
    ax.set_ylabel('Time (ms/grad step)')
    ax.set_title('Per Gradient Step Cost')
    for bar, val in zip(bars, grad_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Panel 3: Estimated total training time
    ax = axes[2]
    total_hours = [results[a]['estimated_total_hours'] for a in algo_order]
    bars = ax.bar(x, total_hours, width, color=bar_colors, alpha=0.8,
                  edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(a, a) for a in algo_order], fontsize=9)
    ax.set_ylabel('Estimated Training Time (hours)')
    ax.set_title('Total Training Time (2000 iters)')
    for bar, val in zip(bars, total_hours):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}h', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"efficiency_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(description="Exp 8: Efficiency Benchmark")
    parser.add_argument("--mode", choices=["benchmark", "plot", "both"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2")
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_measure", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()

    if args.mode in ("benchmark", "both"):
        benchmark_all(args.env, args.n_warmup, args.n_measure, args.seed)
    if args.mode in ("plot", "both"):
        plot_efficiency(args.env)

    print("\n✅ Experiment 8 complete!")


if __name__ == "__main__":
    main()
