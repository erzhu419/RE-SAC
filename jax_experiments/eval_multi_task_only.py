"""Standalone multi-task eval — loads only `params.pkl`, runs the FIXED env.

Use case: validate the brax_env RAND_PARAMS_MAP fix without retraining.
A buggy ckpt was trained on (effectively) constant gravity, but eval here
runs through the corrected env so per-task gravity actually shifts. Result:
the previously-bit-identical 40 task means become a real distribution.

Usage:
  python -m jax_experiments.eval_multi_task_only \
    --run_name abl_ns_B0_Hopper-v2_8 \
    --algo resac --env Hopper-v2 \
    --ensemble_size 10 --variant B0 \
    --varying_params gravity --task_num 40 --test_task_num 40 \
    --seed 8

Writes `<save_root>/<run_name>/logs/final_multi_task_fixed.npy` (separate
file so we don't clobber the buggy npy that's useful for the diff).
"""
import argparse
import os
import pickle
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# Ensure jax_experiments package is importable when run as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax_experiments  # noqa: F401  -- triggers nnx.List shim
from jax_experiments.envs.brax_env import BraxNonstationaryEnv as NonstationaryEnv
from jax_experiments.configs.default import Config
from jax_experiments.train import evaluate_multi_task, make_algo


def build_config(args) -> Config:
    config = Config(env_name=args.env, algo=args.algo)
    # Apply args that affect agent construction
    for k in ['seed', 'ensemble_size', 'beta', 'beta_ood', 'weight_reg',
              'beta_bc', 'independent_ratio', 'ema_tau', 'anchor_lambda',
              'adaptive_beta', 'beta_start', 'beta_end', 'beta_warmup',
              'variant', 'log_scale_limit', 'brax_backend',
              'obs_noise_std', 'reward_noise_std']:
        v = getattr(args, k, None)
        if v is not None:
            setattr(config, k, v)
    return config


def main():
    parser = argparse.ArgumentParser()
    # Identification
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--save_root', default='jax_experiments/results')
    parser.add_argument('--out_suffix', default='fixed',
                        help="Output npy suffix. Writes final_multi_task_<suffix>.npy.")
    # Env / algo (must match the trained ckpt)
    parser.add_argument('--algo', required=True,
                        choices=['resac', 'sac', 'td3', 'dsac', 'bac', 'redq', 'sacn', 'tqc'])
    parser.add_argument('--env', required=True)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--ensemble_size', type=int, default=10)
    parser.add_argument('--brax_backend', default='spring')
    parser.add_argument('--log_scale_limit', type=float, default=3.0)
    # RE-SAC variant + IPM regs (must match training to construct agent shape)
    parser.add_argument('--variant', default='B0')
    parser.add_argument('--beta', type=float, default=-2.0)
    parser.add_argument('--beta_ood', type=float, default=0.0)
    parser.add_argument('--weight_reg', type=float, default=0.0)
    parser.add_argument('--beta_bc', type=float, default=0.0)
    parser.add_argument('--independent_ratio', type=float, default=0.75)
    parser.add_argument('--ema_tau', type=float, default=0.005)
    parser.add_argument('--anchor_lambda', type=float, default=0.01)
    parser.add_argument('--adaptive_beta', action='store_true')
    parser.add_argument('--beta_start', type=float, default=-2.0)
    parser.add_argument('--beta_end', type=float, default=-1.0)
    parser.add_argument('--beta_warmup', type=float, default=0.2)
    # Non-stationary task spec — MUST match training so test_tasks rng matches
    parser.add_argument('--varying_params', nargs='+', default=['gravity'])
    parser.add_argument('--task_num', type=int, default=40)
    parser.add_argument('--test_task_num', type=int, default=40)
    parser.add_argument('--n_episodes', type=int, default=5)
    parser.add_argument('--obs_noise_std', type=float, default=0.0)
    parser.add_argument('--reward_noise_std', type=float, default=0.0)
    args = parser.parse_args()

    config = build_config(args)
    config.varying_params = args.varying_params

    # ---- Build env (the eval env in train.py uses seed+1000) ----
    eval_env = NonstationaryEnv(
        args.env, rand_params=args.varying_params,
        log_scale_limit=args.log_scale_limit, seed=args.seed + 1000,
        backend=args.brax_backend)

    # Sample test_tasks via the SAME training env's seed so we hit the same
    # 40 tasks the original training was supposed to be evaluated on.
    train_env = NonstationaryEnv(
        args.env, rand_params=args.varying_params,
        log_scale_limit=args.log_scale_limit, seed=args.seed,
        backend=args.brax_backend)
    _ = train_env.sample_tasks(args.task_num)        # consume train_tasks rng
    test_tasks = train_env.sample_tasks(args.test_task_num)

    # ---- Build agent (must match training arch) ----
    agent = make_algo(eval_env.obs_dim, eval_env.act_dim, config)
    policy_graphdef = nnx.graphdef(agent.policy)
    eval_env.build_rollout_fn(policy_graphdef, context_graphdef=None)

    # ---- Load params.pkl and update policy ----
    params_path = os.path.join(args.save_root, args.run_name, 'checkpoints', 'params.pkl')
    if not os.path.exists(params_path):
        print(f"FATAL: {params_path} missing")
        sys.exit(2)
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    policy_params = jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
        params['policy'])
    nnx.update(agent.policy, policy_params)
    if 'ema_policy' in params and hasattr(agent, 'ema_policy'):
        ema_params = jax.tree.map(
            lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x,
            params['ema_policy'])
        nnx.update(agent.ema_policy, ema_params)

    # ---- Run multi-task eval ----
    print(f"Running multi-task eval over {len(test_tasks)} tasks...")
    out = evaluate_multi_task(agent, eval_env, config, test_tasks,
                              n_episodes=args.n_episodes)

    arr = np.array(out['per_task'])
    out_dir = os.path.join(args.save_root, args.run_name, 'logs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'final_multi_task_{args.out_suffix}.npy')
    np.save(out_path, arr)

    print(f"\n=== {args.run_name} ===")
    print(f"  N tasks: {out['n_tasks']}")
    print(f"  unique  : {len(np.unique(np.round(arr, 2)))}")
    print(f"  mean    : {out['mean']:.2f}")
    print(f"  std     : {out['std']:.2f}")
    print(f"  min     : {out['min']:.2f}")
    print(f"  worst-Q : {out['worst_quartile']:.2f}")
    print(f"  → wrote {out_path}")

    # Also load + print buggy comparison if present
    old_path = os.path.join(out_dir, 'final_multi_task.npy')
    if os.path.exists(old_path):
        old = np.load(old_path)
        print(f"  buggy   : N={len(old)} mean={old.mean():.2f} std={old.std():.2f}  "
              f"(unique={len(np.unique(np.round(old,2)))})")


if __name__ == '__main__':
    main()
