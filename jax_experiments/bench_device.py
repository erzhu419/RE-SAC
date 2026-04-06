"""Benchmark CPU vs GPU for Brax env step + SAC update throughput.

Usage (GPU):
  LD_LIBRARY_PATH=... conda run -n jax-rl python -m jax_experiments.bench_device

Usage (CPU forced):
  JAX_PLATFORMS=cpu conda run -n jax-rl python -m jax_experiments.bench_device
"""
import os
import sys
import time
import numpy as np

# ── Must set CUDA lib path before JAX import ──
NVIDIA_LIB = None
for p in sys.path:
    candidate = os.path.join(p, "nvidia")
    if os.path.isdir(candidate):
        NVIDIA_LIB = candidate
        break
if NVIDIA_LIB is None:
    import site
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "nvidia")
        if os.path.isdir(candidate):
            NVIDIA_LIB = candidate
            break
if NVIDIA_LIB is not None:
    lib_dirs = []
    for subdir in os.listdir(NVIDIA_LIB):
        lib_path = os.path.join(NVIDIA_LIB, subdir, "lib")
        if os.path.isdir(lib_path):
            lib_dirs.append(lib_path)
    if lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv


def bench_env_steps(env_name="Hopper-v2", n_steps=2000, warmup=500):
    """Benchmark raw env stepping throughput."""
    env = NonstationaryEnv(env_name, rand_params=["gravity"], seed=0, backend="spring")
    obs = env.reset()

    # Warmup (JIT compilation)
    print(f"  Warming up ({warmup} steps)...")
    for _ in range(warmup):
        action = env.action_space.sample()
        obs, r, d, _ = env.step(action)
        if d:
            obs = env.reset()

    # Timed run
    print(f"  Benchmarking ({n_steps} steps)...")
    t0 = time.perf_counter()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, r, d, _ = env.step(action)
        if d:
            obs = env.reset()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    sps = n_steps / elapsed
    env.close()
    return sps, elapsed


def bench_scan_rollout(env_name="Hopper-v2", n_steps=4000, warmup_rollouts=2):
    """Benchmark scan-fused rollout throughput (the actual training path)."""
    config = Config()
    config.env_name = env_name
    config.ensemble_size = 10
    config.hidden_dim = 256

    env = NonstationaryEnv(env_name, rand_params=["gravity"], seed=0, backend="spring")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Build a real SAC agent for realistic benchmark
    from jax_experiments.algos.sac_base import SACBase
    config.ensemble_size = 2  # SAC uses twin critics
    agent = SACBase(obs_dim, act_dim, config, seed=0)

    policy_graphdef = nnx.graphdef(agent.policy)
    env.build_rollout_fn(policy_graphdef, context_graphdef=None)

    # Warmup (JIT compilation of scan rollout)
    print(f"  Warming up scan rollout ({warmup_rollouts} rollouts of {n_steps} steps)...")
    for i in range(warmup_rollouts):
        rng = jax.random.PRNGKey(i)
        policy_params = nnx.state(agent.policy, nnx.Param)
        _, _ = env.rollout(policy_params, n_steps, rng, context_params=None)

    # Timed run (multiple rollouts)
    n_rollouts = 5
    print(f"  Benchmarking ({n_rollouts} rollouts × {n_steps} steps = {n_rollouts * n_steps} total)...")
    t0 = time.perf_counter()
    for i in range(n_rollouts):
        rng = jax.random.PRNGKey(100 + i)
        policy_params = nnx.state(agent.policy, nnx.Param)
        _, _ = env.rollout(policy_params, n_steps, rng, context_params=None)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total = n_rollouts * n_steps
    sps = total / elapsed
    env.close()
    return sps, elapsed, total


def bench_training_update(env_name="Hopper-v2", n_updates=200, warmup=50):
    """Benchmark gradient update throughput (the other half of training)."""
    config = Config()
    config.env_name = env_name
    config.ensemble_size = 2
    config.hidden_dim = 256
    config.batch_size = 256

    env = NonstationaryEnv(env_name, rand_params=["gravity"], seed=0, backend="spring")
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    from jax_experiments.algos.sac_base import SACBase
    agent = SACBase(obs_dim, act_dim, config, seed=0)

    # Fill replay buffer with random data
    buf = ReplayBuffer(obs_dim, act_dim, capacity=50000)
    for _ in range(10000):
        o = np.random.randn(obs_dim).astype(np.float32)
        a = np.random.randn(act_dim).astype(np.float32)
        r = np.random.randn()
        no = np.random.randn(obs_dim).astype(np.float32)
        d = float(np.random.rand() < 0.01)
        buf.push(o, a, r, no, d, 0)

    # Warmup
    print(f"  Warming up updates ({warmup} batches)...")
    stacked = buf.sample_stacked(warmup, config.batch_size)
    _ = agent.multi_update(stacked)

    # Timed run
    print(f"  Benchmarking ({n_updates} updates)...")
    stacked = buf.sample_stacked(n_updates, config.batch_size)
    t0 = time.perf_counter()
    _ = agent.multi_update(stacked)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ups = n_updates / elapsed
    env.close()
    return ups, elapsed


def main():
    backend = jax.default_backend()
    devices = jax.devices()
    print("=" * 60)
    print(f"  JAX backend: {backend}")
    print(f"  JAX devices: {devices}")
    print(f"  JAX version: {jax.__version__}")
    print("=" * 60)

    env_name = "Hopper-v2"

    # 1. Sequential env step benchmark
    print(f"\n[1/3] Sequential env.step() — {env_name}")
    sps_seq, t_seq = bench_env_steps(env_name, n_steps=2000, warmup=500)
    print(f"  ✓ {sps_seq:.0f} steps/sec  ({t_seq:.2f}s)")

    # 2. Scan-fused rollout benchmark (the real training path)
    print(f"\n[2/3] Scan-fused rollout — {env_name}")
    sps_scan, t_scan, total_scan = bench_scan_rollout(env_name, n_steps=4000, warmup_rollouts=2)
    print(f"  ✓ {sps_scan:.0f} steps/sec  ({t_scan:.2f}s for {total_scan} steps)")

    # 3. Training update benchmark
    print(f"\n[3/3] Training updates (SAC, batch=256) — {env_name}")
    ups, t_upd = bench_training_update(env_name, n_updates=200, warmup=50)
    print(f"  ✓ {ups:.0f} updates/sec  ({t_upd:.2f}s)")

    # Summary
    print("\n" + "=" * 60)
    print(f"  SUMMARY  ({backend.upper()})")
    print(f"  Sequential step:   {sps_seq:>8.0f} steps/sec")
    print(f"  Scan rollout:      {sps_scan:>8.0f} steps/sec")
    print(f"  Training updates:  {ups:>8.0f} updates/sec")
    print("=" * 60)


if __name__ == "__main__":
    main()
