"""Main training entry point for JAX-based RE-SAC experiments.

Usage:
    conda run -n jax-rl python -m jax_experiments.train --env Hopper-v2
    conda run -n jax-rl python -m jax_experiments.train --device gpu --env Hopper-v2
"""
import os
import sys
import argparse
import time
import numpy as np

# ── Device selection (must happen before JAX import) ──
# Default: gpu. Use --device cpu to run on CPU.
if "JAX_PLATFORMS" not in os.environ:
    _device = "gpu"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--device" and i + 1 < len(sys.argv):
            _device = sys.argv[i + 1].lower()
            break
    os.environ["JAX_PLATFORMS"] = "cuda" if _device == "gpu" else _device
    if _device == "gpu":
        # Auto-configure LD_LIBRARY_PATH for CUDA libs bundled in pip nvidia packages
        _nvidia_lib = None
        for p in sys.path:
            candidate = os.path.join(p, "nvidia")
            if os.path.isdir(candidate):
                _nvidia_lib = candidate
                break
        if _nvidia_lib is None:
            import site
            for sp in site.getsitepackages():
                candidate = os.path.join(sp, "nvidia")
                if os.path.isdir(candidate):
                    _nvidia_lib = candidate
                    break
        if _nvidia_lib is not None:
            _lib_dirs = []
            for subdir in os.listdir(_nvidia_lib):
                lib_path = os.path.join(_nvidia_lib, subdir, "lib")
                if os.path.isdir(lib_path):
                    _lib_dirs.append(lib_path)
            if _lib_dirs:
                os.environ["LD_LIBRARY_PATH"] = ":".join(_lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config
from jax_experiments.common.replay_buffer import ReplayBuffer
from jax_experiments.common.logging import Logger
from jax_experiments.common.checkpoint import (
    save_checkpoint, load_checkpoint, has_checkpoint, get_checkpoint_iteration
)
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv


def make_algo(obs_dim: int, act_dim: int, config: Config):
    """Instantiate algorithm based on config.algo."""
    if config.algo == "sac":
        from jax_experiments.algos.sac_base import SACBase
        return SACBase(obs_dim, act_dim, config, seed=config.seed)
    elif config.algo == "td3":
        from jax_experiments.algos.td3 import TD3
        return TD3(obs_dim, act_dim, config, seed=config.seed)
    elif config.algo == "dsac":
        from jax_experiments.algos.dsac import DSAC
        return DSAC(obs_dim, act_dim, config, seed=config.seed)
    elif config.algo == "bac":
        from jax_experiments.algos.bac import BAC
        return BAC(obs_dim, act_dim, config, seed=config.seed)
    else:
        from jax_experiments.algos.resac import RESAC
        return RESAC(obs_dim, act_dim, config, seed=config.seed)


def evaluate(agent, env, config: Config, tasks=None, n_episodes: int = 10):
    """Fast GPU-scan eval: deterministic policy, single JIT call for all episodes.
    Uses EMA policy for RE-SAC if available (more stable)."""
    from flax import nnx
    # Use EMA policy if available (RE-SAC), otherwise current policy
    if hasattr(agent, 'ema_policy'):
        policy_params = nnx.state(agent.ema_policy, nnx.Param)
    else:
        policy_params = nnx.state(agent.policy, nnx.Param)

    rng_key = jax.random.PRNGKey(42)  # fixed key for reproducible eval
    n_steps = n_episodes * config.max_episode_steps

    # If tasks provided, pick a representative task for eval
    if tasks is not None:
        env.set_task(tasks[0])

    rew_np, done_np = env.eval_rollout(
        policy_params, n_steps, rng_key, context_params=None)

    # Segment into episodes via done mask
    ep_rewards, ep_r, completed = [], 0.0, 0
    for i in range(n_steps):
        ep_r += rew_np[i]
        if done_np[i] > 0.5:
            ep_rewards.append(ep_r)
            ep_r = 0.0
            completed += 1
            if completed >= n_episodes:
                break

    if not ep_rewards:  # no episode completed (e.g. very early training)
        ep_rewards = [ep_r]

    return float(np.mean(ep_rewards)), float(np.std(ep_rewards))


def collect_samples(agent, env, replay_buffer, config, n_steps: int):
    """Collect n_steps via GPU scan-fused rollout."""
    is_random = replay_buffer.size < config.start_train_steps
    rng_key = jax.random.PRNGKey(config.seed + replay_buffer.size)

    if is_random:
        # Random exploration: sequential API (small overhead ok)
        obs = env.reset()
        episode_rewards = []
        episode_reward = 0.0
        for _ in range(n_steps):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, done, env.current_task_id)
            episode_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs = env.reset()
        return episode_rewards
    else:
        # Scan-fused GPU rollout
        policy_params = nnx.state(agent.policy, nnx.Param)

        prev_task_id = env.current_task_id
        (obs, act, rew, nobs, done), ep_rewards = env.rollout(
            policy_params, n_steps, rng_key, context_params=None)

        # Bulk push — JAX arrays stay on GPU (zero CPU transfer)
        task_ids = jnp.full(n_steps, env.current_task_id, dtype=jnp.int32)
        replay_buffer.push_batch_jax(obs, act, rew.reshape(-1, 1),
                                     nobs, done.reshape(-1, 1), task_ids)

        return ep_rewards


def train(config: Config):
    """Main training loop with checkpoint/resume support."""
    stationary = getattr(config, 'stationary', False)
    resume = getattr(config, 'resume', False)
    print(f"{'='*60}")
    print(f"  Algorithm: {config.algo.upper()}")
    print(f"  Environment: {config.env_name}")
    print(f"  Mode: {'STATIONARY' if stationary else 'Non-stationary'}")
    if not stationary:
        print(f"  Varying: {config.varying_params}")
    print(f"  Seed: {config.seed}")
    print(f"  Updates/iter: {config.updates_per_iter}  Samples/iter: {config.samples_per_iter}")
    print(f"  Ensemble size: {config.ensemble_size}  Hidden dim: {config.hidden_dim}")
    if config.algo == "resac":
        print(f"  beta={config.beta}  beta_ood={config.beta_ood}  weight_reg={config.weight_reg}  beta_bc={config.beta_bc}")
        if config.adaptive_beta:
            print(f"  ★ Adaptive beta: {config.beta_start} → {config.beta_end}  warmup={config.beta_warmup}")
    print(f"  Brax backend: {config.brax_backend}")
    print(f"  JAX devices: {jax.devices()}")
    print(f"{'='*60}")

    # Create environment
    env = NonstationaryEnv(config.env_name, rand_params=config.varying_params,
                           log_scale_limit=config.log_scale_limit, seed=config.seed,
                           backend=config.brax_backend)
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    # Task setup: stationary = no switching, non-stationary = periodic gravity changes
    if stationary:
        # Use default physics — no task perturbation
        train_tasks = [{}]
        test_tasks = [{}]
    else:
        train_tasks = env.sample_tasks(config.task_num)
        test_tasks = env.sample_tasks(config.test_task_num)
        env.set_nonstationary_para(train_tasks, config.changing_period, config.changing_interval)

    # Create agent
    agent = make_algo(obs_dim, act_dim, config)

    # Build scan-fused rollout (compiles policy+physics into one XLA call)
    policy_graphdef = nnx.graphdef(agent.policy)
    env.build_rollout_fn(policy_graphdef, context_graphdef=None)
    print(f"  Built scan-fused rollout for {config.env_name}")

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim, act_dim, capacity=config.replay_size)

    # Logging
    run_name = config.run_name or f"{config.algo}_{config.env_name}_{config.seed}"
    log_dir = os.path.join(config.save_root, run_name, "logs")
    model_dir = os.path.join(config.save_root, run_name, "models")
    ckpt_dir = os.path.join(config.save_root, run_name, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(log_dir)

    # --- Resume from checkpoint ---
    start_iter = 0
    total_steps = 0
    if resume and has_checkpoint(ckpt_dir):
        ckpt_iter = get_checkpoint_iteration(ckpt_dir)
        if ckpt_iter >= config.max_iters - 1:
            print(f"  ✅ Already completed ({ckpt_iter+1}/{config.max_iters} iters). Skipping.")
            env.close()
            return
        start_iter, total_steps = load_checkpoint(
            ckpt_dir, agent, replay_buffer, logger, config.algo)
        # Rebuild scan fn after loading params (graphdefs are the same)
        agent._build_scan_fn()
    elif resume:
        # Check if we can detect completion from logs
        iter_file = os.path.join(log_dir, 'iteration.npy')
        if os.path.exists(iter_file):
            saved_iters = np.load(iter_file)
            if len(saved_iters) > 0 and int(saved_iters[-1]) >= config.max_iters - 1:
                print(f"  ✅ Already completed (logs show iter {int(saved_iters[-1])}). Skipping.")
                env.close()
                return

    print(f"Logging to: {log_dir}")
    if start_iter > 0:
        print(f"Resuming from iteration {start_iter} (total_steps={total_steps})")
    else:
        print(f"Starting training... (first {config.start_train_steps} steps are random exploration)")

    # Separate eval env to avoid polluting training env's state
    eval_env = NonstationaryEnv(config.env_name, rand_params=config.varying_params,
                                log_scale_limit=config.log_scale_limit, seed=config.seed + 1000,
                                backend=config.brax_backend)
    eval_env.build_rollout_fn(policy_graphdef, context_graphdef=None)

    start_time = time.time()

    for iteration in range(start_iter, config.max_iters):
        iter_start = time.time()

        # --- Collect samples ---
        ep_rewards = collect_samples(agent, env, replay_buffer, config, config.samples_per_iter)
        total_steps += config.samples_per_iter

        if len(ep_rewards) > 0:
            logger.log("train_reward_mean", float(np.mean(ep_rewards)))
            logger.log("train_reward_std", float(np.std(ep_rewards)))

        # --- Training updates (fused via lax.scan) ---
        metrics = {}
        if replay_buffer.size >= config.start_train_steps:
            sample_key = jax.random.PRNGKey(config.seed + iteration + total_steps)
            stacked = replay_buffer.sample_stacked(
                config.updates_per_iter, config.batch_size,
                rng_key=sample_key)

            # Adaptive beta_lcb for RE-SAC
            beta_override = None
            if config.algo == "resac" and config.adaptive_beta:
                beta_override = agent.get_adaptive_beta(iteration, config.max_iters)

            metrics = agent.multi_update(stacked, **(
                {"beta_override": beta_override} if beta_override is not None else {}
            ))

            if beta_override is not None:
                metrics["beta_lcb"] = beta_override

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    logger.log(k, v)

        # --- Evaluation (expensive, only every log_interval) ---
        eval_mean = None
        if iteration % config.log_interval == 0:
            eval_mean, eval_std = evaluate(agent, eval_env, config, test_tasks,
                                           n_episodes=config.eval_episodes)
            logger.log("eval_reward", eval_mean)
            logger.log("eval_reward_std", eval_std)
            # Feed eval to RE-SAC for anchoring + performance-aware beta
            if hasattr(agent, 'report_eval'):
                agent.report_eval(eval_mean)
        logger.log("total_steps", total_steps)
        logger.log("iteration", iteration)
        logger.log("mode_id", env.current_task_id)

        # --- Always print status ---
        iter_time = time.time() - iter_start
        q_std_str = ""
        if "q_std_mean" in metrics:
            q_std_str = f" | Q-std: {metrics.get('q_std_mean', 0):.2f}"
        beta_str = ""
        if "beta_lcb" in metrics:
            beta_str = f" | β: {metrics['beta_lcb']:.3f}"
        eval_str = f" | Eval: {eval_mean:.1f}" if eval_mean is not None else ""
        extra = f"TaskID: {env.current_task_id}{q_std_str}{beta_str}{eval_str}"
        extra += f" | {iter_time:.1f}s/iter"
        logger.print_status(iteration, extra)

        # --- Save ---
        if iteration % config.save_interval == 0:
            logger.save()
        # Checkpoint every 200 iters (expensive due to replay buffer)
        if iteration % 200 == 0 and iteration > 0:
            save_checkpoint(ckpt_dir, agent, replay_buffer, logger,
                           iteration, total_steps, config.algo)

    # Final save + checkpoint
    logger.save()
    save_checkpoint(ckpt_dir, agent, replay_buffer, logger,
                   config.max_iters - 1, total_steps, config.algo)
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Results saved to: {log_dir}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="JAX RE-SAC Training")
    parser.add_argument("--algo", type=str, default="resac",
                        choices=["resac", "sac", "td3", "dsac", "bac"],
                        help="Algorithm: resac | sac | td3 | dsac | bac")
    parser.add_argument("--num_quantiles", type=int, default=32,
                        help="Number of quantiles for DSAC (IQN)")
    parser.add_argument("--env", type=str, default="Hopper-v2")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--stationary", action="store_true",
                        help="Use stationary (classic) MuJoCo env — no gravity perturbation")
    parser.add_argument("--varying_params", nargs="+", default=["gravity"])
    parser.add_argument("--task_num", type=int, default=40)
    parser.add_argument("--test_task_num", type=int, default=40)
    parser.add_argument("--save_root", type=str, default="jax_experiments/results")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--samples_per_iter", type=int, default=None)
    parser.add_argument("--updates_per_iter", type=int, default=None)
    parser.add_argument("--backend", type=str, default="spring",
                        choices=["spring", "generalized"],
                        help="Brax physics backend: spring (fast) or generalized (accurate)")
    parser.add_argument("--device", type=str, default="gpu",
                        choices=["cpu", "gpu"],
                        help="JAX device: gpu (default) or cpu")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available. Skip if already completed.")
    # RE-SAC hyperparameters
    parser.add_argument("--beta", type=float, default=None,
                        help="LCB coefficient for policy (default: -2.0)")
    parser.add_argument("--beta_ood", type=float, default=None,
                        help="OOD regularization weight (default: 0.01)")
    parser.add_argument("--weight_reg", type=float, default=None,
                        help="Critic weight regularization (default: 0.01)")
    parser.add_argument("--beta_bc", type=float, default=None,
                        help="Behavior cloning weight (default: 0.001)")
    # Adaptive beta_lcb
    parser.add_argument("--adaptive_beta", action="store_true",
                        help="Enable adaptive beta_lcb annealing: beta_start → beta_end")
    parser.add_argument("--beta_start", type=float, default=None,
                        help="Initial beta_lcb, more pessimistic (default: -2.0)")
    parser.add_argument("--beta_end", type=float, default=None,
                        help="Final beta_lcb, more exploitative (default: -0.5)")
    parser.add_argument("--beta_warmup", type=float, default=None,
                        help="Fraction of max_iters to hold beta_start before annealing (default: 0.2)")
    # EMA and anchoring
    parser.add_argument("--ema_tau", type=float, default=None,
                        help="EMA policy smoothing coefficient (default: 0.005)")
    parser.add_argument("--anchor_lambda", type=float, default=None,
                        help="Policy anchoring penalty weight (default: 0.1, 0=disabled)")
    # Ant-stability fixes
    parser.add_argument("--lcb_normalize", action="store_true",
                        help="Normalize LCB penalty by Q magnitude")
    parser.add_argument("--q_std_clip", type=float, default=None,
                        help="Clip Q-std as fraction of |Q_mean| (default: 0=disabled)")
    parser.add_argument("--independent_ratio", type=float, default=None,
                        help="Blend ratio: 1.0=all independent, 0.0=all min target (default: 1.0)")

    args = parser.parse_args()

    config = Config()
    config.algo = args.algo
    config.env_name = args.env
    config.seed = args.seed
    config.max_iters = args.max_iters
    config.stationary = args.stationary
    if args.stationary:
        config.varying_params = []  # no perturbation
    else:
        config.varying_params = args.varying_params
    config.task_num = args.task_num
    config.test_task_num = args.test_task_num
    config.save_root = args.save_root
    config.run_name = args.run_name
    config.ensemble_size = args.ensemble_size
    config.hidden_dim = args.hidden_dim
    config.num_quantiles = args.num_quantiles
    # SAC/BAC baseline: force ensemble_size=2 (standard twin critics)
    if args.algo in ("sac", "bac"):
        config.ensemble_size = 2
    # Only override Config defaults when explicitly provided
    if args.samples_per_iter is not None:
        config.samples_per_iter = args.samples_per_iter
    if args.updates_per_iter is not None:
        config.updates_per_iter = args.updates_per_iter
    config.brax_backend = args.backend
    config.resume = args.resume
    # RE-SAC hyperparameter overrides
    if args.beta is not None:
        config.beta = args.beta
    if args.beta_ood is not None:
        config.beta_ood = args.beta_ood
    if args.weight_reg is not None:
        config.weight_reg = args.weight_reg
    if args.beta_bc is not None:
        config.beta_bc = args.beta_bc
    # Adaptive beta
    if args.adaptive_beta:
        config.adaptive_beta = True
    if args.beta_start is not None:
        config.beta_start = args.beta_start
    if args.beta_end is not None:
        config.beta_end = args.beta_end
    if args.beta_warmup is not None:
        config.beta_warmup = args.beta_warmup
    # EMA and anchoring
    if args.ema_tau is not None:
        config.ema_tau = args.ema_tau
    if args.anchor_lambda is not None:
        config.anchor_lambda = args.anchor_lambda
    # Ant-stability fixes
    if args.lcb_normalize:
        config.lcb_normalize = True
    if args.q_std_clip is not None:
        config.q_std_clip = args.q_std_clip
    if args.independent_ratio is not None:
        config.independent_ratio = args.independent_ratio

    train(config)


if __name__ == "__main__":
    main()
