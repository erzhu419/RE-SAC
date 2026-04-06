"""Collect Q-value data for Oracle accuracy analysis.

For each (algo, env), loads the final checkpoint, runs N evaluation episodes,
and records per-step: obs, action, all Q-head predictions, log_prob, reward, done.
Then computes Monte Carlo returns (real Q = G_t) backwards through each episode.

Usage:
    cd RE-SAC
    conda run -n jax-rl python -m jax_experiments.q_analysis.collect_q_data \
        --env Hopper-v2 --n_episodes 50

    # Or collect all envs:
    conda run -n jax-rl python -m jax_experiments.q_analysis.collect_q_data --all
"""
import os
import sys
import argparse
import pickle
import numpy as np

# ── Device setup (before JAX import) ──
# Default to CPU for inference-only collection (avoids CUDA lib path issues)
if "JAX_PLATFORMS" not in os.environ:
    _device = "cpu"
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            _device = sys.argv[i + 1].lower()
            break
    if _device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
        for p in sys.path:
            candidate = os.path.join(p, "nvidia")
            if os.path.isdir(candidate):
                for subdir in os.listdir(candidate):
                    lib_path = os.path.join(candidate, subdir, "lib")
                    if os.path.isdir(lib_path):
                        os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
                break
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.envs.brax_env import BraxNonstationaryEnv


# Algo keys → (algorithm class, run_name prefix, config overrides)
ALGO_CONFIGS = {
    "sac":       {"algo": "sac",   "ensemble_size": 2},
    "resac":     {"algo": "resac", "ensemble_size": 10},
    "resac_v2":  {"algo": "resac", "ensemble_size": 10},
    "dsac":      {"algo": "dsac",  "ensemble_size": 10},
    "td3":       {"algo": "td3",   "ensemble_size": 2},
    # New versions with independent targets + EMA + anchoring
    "resac_v4":  {"algo": "resac", "ensemble_size": 10},
    "resac_v5":  {"algo": "resac", "ensemble_size": 10,
                  "ema_tau": 0.005, "anchor_lambda": 0.1},
    "resac_v5b": {"algo": "resac", "ensemble_size": 10,
                  "ema_tau": 0.005, "anchor_lambda": 0.01},
    "resac_v6b": {"algo": "resac", "ensemble_size": 10,
                  "ema_tau": 0.005, "anchor_lambda": 0.01,
                  "independent_ratio": 0.5},
}
ALGOS = list(ALGO_CONFIGS.keys())
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
SEED = 8
RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/q_analysis/results"


def make_agent_and_env(algo, env_name, seed=SEED):
    """Create agent + env with correct config, matching the training setup."""
    config = Config()
    acfg = ALGO_CONFIGS[algo]
    config.algo = acfg["algo"]
    config.env_name = env_name
    config.seed = seed
    config.brax_backend = "spring"
    config.stationary = True
    config.varying_params = []
    config.ensemble_size = acfg.get("ensemble_size", 10)
    # Apply extra config overrides (ema_tau, anchor_lambda, independent_ratio, etc.)
    for k, v in acfg.items():
        if k not in ("algo", "ensemble_size") and hasattr(config, k):
            setattr(config, k, v)

    # Create env
    env = BraxNonstationaryEnv(
        env_name, rand_params=[], log_scale_limit=0.0,
        seed=seed, backend=config.brax_backend)

    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Create agent
    from jax_experiments.train import make_algo
    agent = make_algo(obs_dim, act_dim, config)

    return agent, env, config


def load_trained_agent(algo, env_name, seed=SEED):
    """Load agent from checkpoint (params only, skip replay buffer)."""
    agent, env, config = make_agent_and_env(algo, env_name, seed)
    run_name = f"{algo}_{env_name}_{seed}"
    ckpt_dir = os.path.join(RESULTS_ROOT, run_name, "checkpoints")
    params_path = os.path.join(ckpt_dir, "params.pkl")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No checkpoint found at {params_path}")

    # Load params directly (skip replay buffer — we only need model weights)
    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    from jax_experiments.common.checkpoint import _to_jax_tree

    # Restore policy
    nnx.update(agent.policy, _to_jax_tree(params['policy']))

    if algo == 'td3':
        nnx.update(agent.critic, _to_jax_tree(params['critic']))
        nnx.update(agent.target_critic, _to_jax_tree(params['target_critic']))
        nnx.update(agent.target_policy, _to_jax_tree(params['target_policy']))
    elif algo == 'dsac':
        nnx.update(agent.twin_critic, _to_jax_tree(params['twin_critic']))
        nnx.update(agent.target_twin_critic, _to_jax_tree(params['target_twin_critic']))
        agent.log_alpha = jnp.array(params['log_alpha'])
    else:
        # SAC / RESAC
        nnx.update(agent.critic, _to_jax_tree(params['critic']))
        nnx.update(agent.target_critic, _to_jax_tree(params['target_critic']))
        agent.log_alpha = jnp.array(params['log_alpha'])

    agent.update_count = params.get('update_count', 0)

    # For v5+ with EMA policy: copy policy → ema_policy as well
    if hasattr(agent, 'ema_policy'):
        nnx.update(agent.ema_policy, nnx.state(agent.policy, nnx.Param))

    # Rebuild scan fn after loading params
    agent._build_scan_fn()

    print(f"  Loaded params from {params_path}")
    return agent, env, config


def collect_episode_data(agent, env, algo, config, rng_key):
    """Run one episode, collecting per-step Q predictions.

    Returns list of step dicts with: obs, action, reward, done, q_vals, log_prob
    """
    gamma = config.gamma
    steps = []

    obs = env.reset()
    done = False
    step_count = 0

    while not done and step_count < config.max_episode_steps:
        obs_jax = jnp.array(obs)[None]  # (1, obs_dim)
        rng_key, act_key = jax.random.split(rng_key)

        # Get deterministic action
        algo_class = ALGO_CONFIGS[algo]["algo"]
        if algo_class == "td3":
            action = agent.policy(obs_jax)
            action_np = np.array(action[0])
            log_prob = None
        else:
            # SAC / RE-SAC / DSAC all use GaussianPolicy
            action, log_prob_val = agent.policy.sample(obs_jax, act_key)
            action_np = np.array(action[0])
            log_prob = float(log_prob_val[0])

        # Get Q-values from all heads
        action_jax = jnp.array(action_np)[None]  # (1, act_dim)

        algo_class = ALGO_CONFIGS[algo]["algo"]
        if algo_class in ("sac", "resac"):
            # EnsembleCritic: returns [ensemble_size, batch]
            q_all = agent.critic(obs_jax, action_jax)  # [N, 1]
            q_vals = np.array(q_all[:, 0])  # [N]
        elif algo_class == "dsac":
            # TwinQuantileCritic: compute expected Q from quantiles
            # Use uniform tau for evaluation
            n_tau = agent.num_quantiles
            tau_uniform = jnp.linspace(0.5/n_tau, 1.0 - 0.5/n_tau, n_tau)[None]  # (1, T)
            z1 = agent.twin_critic.zf1(obs_jax, action_jax, tau_uniform)  # (1, T)
            z2 = agent.twin_critic.zf2(obs_jax, action_jax, tau_uniform)  # (1, T)
            q1 = float(z1.mean())
            q2 = float(z2.mean())
            q_vals = np.array([q1, q2])
        elif algo_class == "td3":
            q_all = agent.critic(obs_jax, action_jax)  # [2, 1]
            q_vals = np.array(q_all[:, 0])  # [2]

        # Step environment
        next_obs, reward, terminated, info = env.step(action_np)

        steps.append({
            'obs': obs.copy(),
            'action': action_np.copy(),
            'reward': float(reward),
            'done': bool(terminated),
            'q_vals': q_vals,
            'log_prob': log_prob,
        })

        obs = next_obs
        done = terminated
        step_count += 1

    # Compute Monte Carlo returns (real Q) backwards
    G = 0.0
    for i in range(len(steps) - 1, -1, -1):
        G = steps[i]['reward'] + gamma * G
        steps[i]['q_real'] = G
        # Reset G on episode boundary (not needed for single ep, but safe)

    return steps


def compute_pure_q(steps, algo, config):
    """Strip entropy/regularization biases from predicted Q to get Pure Q.

    For SAC/RE-SAC/DSAC: Q_pure = Q_pred - alpha * log_prob / (1 - gamma)
    For TD3: Q_pure = Q_pred (no entropy term)
    """
    gamma = config.gamma
    alpha = float(jnp.exp(jnp.array(jnp.log(config.alpha))))

    # Try to get current alpha from agent (auto-tuned)
    # We pass alpha as parameter instead

    for step in steps:
        q_vals = step['q_vals']
        if algo == "td3" or step['log_prob'] is None:
            step['q_pure'] = q_vals.copy()
        else:
            lp = step['log_prob']
            bias = (-alpha * lp) / (1 - gamma)
            step['q_pure'] = q_vals - bias

    return steps


def collect_for_algo_env(algo, env_name, n_episodes=50, seed=SEED):
    """Collect Q-value comparison data for one (algo, env) pair."""
    print(f"\n{'='*60}")
    print(f"  Collecting: {algo.upper()} on {env_name}")
    print(f"  Episodes: {n_episodes}")
    print(f"{'='*60}")

    agent, env, config = load_trained_agent(algo, env_name, seed)

    # Get actual alpha value (may have been auto-tuned)
    if hasattr(agent, 'log_alpha'):
        actual_alpha = float(jnp.exp(agent.log_alpha))
        print(f"  Learned alpha: {actual_alpha:.4f}")
    else:
        actual_alpha = config.alpha

    all_steps = []
    rng_key = jax.random.PRNGKey(42)

    for ep in range(n_episodes):
        rng_key, ep_key = jax.random.split(rng_key)
        ep_steps = collect_episode_data(agent, env, algo, config, ep_key)

        # Strip entropy bias using actual learned alpha
        algo_class = ALGO_CONFIGS[algo]["algo"]
        gamma = config.gamma
        for step in ep_steps:
            q_vals = step['q_vals']
            if algo_class == "td3" or step['log_prob'] is None:
                step['q_pure'] = q_vals.copy()
            else:
                lp = step['log_prob']
                bias = (-actual_alpha * lp) / (1 - gamma)
                step['q_pure'] = q_vals - bias

        all_steps.extend(ep_steps)
        ep_return = sum(s['reward'] for s in ep_steps)
        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"len={len(ep_steps)}, return={ep_return:.1f}")

    print(f"  Total steps collected: {len(all_steps)}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"q_data_{algo}_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({
            'algo': algo,
            'env_name': env_name,
            'n_episodes': n_episodes,
            'gamma': config.gamma,
            'alpha': actual_alpha,
            'ensemble_size': config.ensemble_size,
            'steps': all_steps,
        }, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved to {out_path} ({size_mb:.1f} MB)")

    env.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Collect Q-value data for oracle analysis")
    parser.add_argument("--env", type=str, default="Hopper-v2",
                        choices=ENVS)
    parser.add_argument("--algo", type=str, default=None,
                        choices=ALGOS + ["baselines", "new"],
                        help="Single algo, 'baselines' (sac/dsac/td3), or 'new' (v4-v6b)")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--all", action="store_true",
                        help="Collect for all 4 envs")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or gpu (parsed early for JAX init)")
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]
    if args.algo == "baselines":
        algos = ["sac", "dsac", "td3"]
    elif args.algo == "new":
        algos = ["resac_v4", "resac_v5", "resac_v5b", "resac_v6b"]
    elif args.algo:
        algos = [args.algo]
    else:
        algos = ALGOS

    for env_name in envs:
        for algo in algos:
            try:
                collect_for_algo_env(algo, env_name, args.n_episodes, args.seed)
            except Exception as e:
                print(f"  ❌ FAILED: {algo} {env_name}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n✅ Collection complete! Data saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
