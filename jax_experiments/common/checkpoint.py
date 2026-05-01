"""Checkpoint save/load for resumable training.

Saves:
  - Model params (policy, critic, target_critic, etc.) via flax state dicts
  - Optimizer states
  - Replay buffer data
  - Training state (iteration, total_steps, log_alpha, update_count, logger data)
"""
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


def _to_numpy_tree(pytree):
    """Convert a JAX pytree (nnx.State / optax state) to numpy for serialization."""
    return jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, pytree)


def _to_jax_tree(pytree):
    """Convert numpy pytree back to JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, pytree)


def save_checkpoint(ckpt_dir: str, agent, replay_buffer, logger,
                    iteration: int, total_steps: int, algo: str):
    """Save a full training checkpoint to disk.
    
    Args:
        ckpt_dir: Directory to save checkpoint files.
        agent: Algorithm instance (SACBase, TD3, DSAC, or RESAC).
        replay_buffer: ReplayBuffer instance.
        logger: Logger instance.
        iteration: Current iteration number.
        total_steps: Total environment steps collected.
        algo: Algorithm name string.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- Model params ---
    params = {}
    params['policy'] = _to_numpy_tree(nnx.state(agent.policy, nnx.Param))
    
    if algo == 'td3':
        params['critic'] = _to_numpy_tree(nnx.state(agent.critic, nnx.Param))
        params['target_critic'] = _to_numpy_tree(nnx.state(agent.target_critic, nnx.Param))
        params['target_policy'] = _to_numpy_tree(nnx.state(agent.target_policy, nnx.Param))
        params['policy_opt_state'] = _to_numpy_tree(agent.policy_opt_state)
        params['critic_opt_state'] = _to_numpy_tree(agent.critic_opt_state)
    elif algo == 'dsac':
        params['twin_critic'] = _to_numpy_tree(nnx.state(agent.twin_critic, nnx.Param))
        params['target_twin_critic'] = _to_numpy_tree(nnx.state(agent.target_twin_critic, nnx.Param))
        params['log_alpha'] = np.array(agent.log_alpha)
        params['policy_opt_state'] = _to_numpy_tree(agent.policy_opt_state)
        params['critic_opt_state'] = _to_numpy_tree(agent.critic_opt_state)
        params['alpha_opt_state'] = _to_numpy_tree(agent.alpha_opt_state)
    else:
        # SAC / RESAC / BAC
        params['critic'] = _to_numpy_tree(nnx.state(agent.critic, nnx.Param))
        params['target_critic'] = _to_numpy_tree(nnx.state(agent.target_critic, nnx.Param))
        params['log_alpha'] = np.array(agent.log_alpha)
        params['policy_opt_state'] = _to_numpy_tree(agent.policy_opt_state)
        params['critic_opt_state'] = _to_numpy_tree(agent.critic_opt_state)
        params['alpha_opt_state'] = _to_numpy_tree(agent.alpha_opt_state)
    
    params['update_count'] = agent.update_count

    # RE-SAC private state (Apr 2026 fix). Without these, --resume silently
    # restarts EMA, anchor snapshot, sigma EMA, hash counter, and beta
    # history at iter 0 — making resumed runs not equivalent to fresh ones.
    if algo == 'resac':
        if hasattr(agent, 'ema_policy'):
            params['ema_policy'] = _to_numpy_tree(
                nnx.state(agent.ema_policy, nnx.Param))
        if getattr(agent, '_anchor_params', None) is not None:
            params['anchor_params'] = _to_numpy_tree(agent._anchor_params)
        params['best_eval'] = float(getattr(agent, '_best_eval', -float('inf')))
        params['eval_history'] = list(getattr(agent, '_eval_history', []))
        params['current_beta'] = float(getattr(agent, '_current_beta', 0.0))
        if hasattr(agent, '_sigma_ema'):
            params['sigma_ema'] = float(np.array(agent._sigma_ema))
        if getattr(agent, '_hash_state', None) is not None:
            params['hash_counts'] = np.array(agent._hash_state.counts)

    # Save params
    with open(os.path.join(ckpt_dir, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    
    # --- Replay buffer ---
    buf_data = replay_buffer.to_numpy()
    np.savez_compressed(os.path.join(ckpt_dir, 'replay_buffer.npz'), **buf_data)
    
    # --- Training state ---
    train_state = {
        'iteration': iteration,
        'total_steps': total_steps,
        'logger_data': dict(logger.data),  # defaultdict -> dict for pickle
    }
    with open(os.path.join(ckpt_dir, 'train_state.pkl'), 'wb') as f:
        pickle.dump(train_state, f)
    
    print(f"  💾 Checkpoint saved: iter={iteration}, steps={total_steps}, "
          f"buf={replay_buffer.size}")


def load_checkpoint(ckpt_dir: str, agent, replay_buffer, logger, algo: str):
    """Load a training checkpoint. Returns (start_iteration, total_steps).
    
    Args:
        ckpt_dir: Directory containing checkpoint files.
        agent: Algorithm instance (already constructed with correct architecture).
        replay_buffer: Empty ReplayBuffer instance.
        logger: Logger instance.
        algo: Algorithm name string.
    
    Returns:
        (start_iteration, total_steps) to resume from.
    """
    params_path = os.path.join(ckpt_dir, 'params.pkl')
    buf_path = os.path.join(ckpt_dir, 'replay_buffer.npz')
    state_path = os.path.join(ckpt_dir, 'train_state.pkl')
    
    if not all(os.path.exists(p) for p in [params_path, buf_path, state_path]):
        print(f"  ⚠️  Incomplete checkpoint in {ckpt_dir}, starting fresh")
        return 0, 0
    
    # --- Load params ---
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    
    # Restore policy
    policy_params = _to_jax_tree(params['policy'])
    nnx.update(agent.policy, policy_params)
    
    if algo == 'td3':
        nnx.update(agent.critic, _to_jax_tree(params['critic']))
        nnx.update(agent.target_critic, _to_jax_tree(params['target_critic']))
        nnx.update(agent.target_policy, _to_jax_tree(params['target_policy']))
        agent.policy_opt_state = _to_jax_tree(params['policy_opt_state'])
        agent.critic_opt_state = _to_jax_tree(params['critic_opt_state'])
    elif algo == 'dsac':
        nnx.update(agent.twin_critic, _to_jax_tree(params['twin_critic']))
        nnx.update(agent.target_twin_critic, _to_jax_tree(params['target_twin_critic']))
        agent.log_alpha = jnp.array(params['log_alpha'])
        agent.policy_opt_state = _to_jax_tree(params['policy_opt_state'])
        agent.critic_opt_state = _to_jax_tree(params['critic_opt_state'])
        agent.alpha_opt_state = _to_jax_tree(params['alpha_opt_state'])
    else:
        # SAC / RESAC / BAC
        nnx.update(agent.critic, _to_jax_tree(params['critic']))
        nnx.update(agent.target_critic, _to_jax_tree(params['target_critic']))
        agent.log_alpha = jnp.array(params['log_alpha'])
        agent.policy_opt_state = _to_jax_tree(params['policy_opt_state'])
        agent.critic_opt_state = _to_jax_tree(params['critic_opt_state'])
        agent.alpha_opt_state = _to_jax_tree(params['alpha_opt_state'])
    
    agent.update_count = params['update_count']

    # Restore RE-SAC private state if present (Apr 2026 fix)
    if algo == 'resac':
        from collections import deque
        if 'ema_policy' in params and hasattr(agent, 'ema_policy'):
            nnx.update(agent.ema_policy, _to_jax_tree(params['ema_policy']))
        if 'anchor_params' in params:
            agent._anchor_params = _to_jax_tree(params['anchor_params'])
        if 'best_eval' in params:
            agent._best_eval = params['best_eval']
        if 'eval_history' in params:
            agent._eval_history = deque(params['eval_history'], maxlen=10)
        if 'current_beta' in params:
            agent._current_beta = params['current_beta']
        if 'sigma_ema' in params and hasattr(agent, '_sigma_ema'):
            agent._sigma_ema = jnp.array(params['sigma_ema'])
        if 'hash_counts' in params and getattr(agent, '_hash_state', None) is not None:
            from jax_experiments.algos.ablation_utils import HashCounterState
            agent._hash_state = HashCounterState(
                W_hash=agent._hash_state.W_hash,
                counts=jnp.array(params['hash_counts']),
                hash_dim=agent._hash_state.hash_dim,
            )

    # --- Load replay buffer ---
    buf = np.load(buf_path)
    replay_buffer.from_numpy(buf)
    
    # --- Load training state ---
    with open(state_path, 'rb') as f:
        train_state = pickle.load(f)
    
    start_iter = train_state['iteration'] + 1  # resume from NEXT iteration
    total_steps = train_state['total_steps']
    
    # Restore logger data
    from collections import defaultdict
    logger.data = defaultdict(list, train_state['logger_data'])
    
    print(f"  ✅ Checkpoint loaded: resuming from iter={start_iter}, "
          f"steps={total_steps}, buf={replay_buffer.size}")
    
    return start_iter, total_steps


def has_checkpoint(ckpt_dir: str) -> bool:
    """Check if a valid checkpoint exists."""
    return (os.path.exists(os.path.join(ckpt_dir, 'params.pkl')) and
            os.path.exists(os.path.join(ckpt_dir, 'replay_buffer.npz')) and
            os.path.exists(os.path.join(ckpt_dir, 'train_state.pkl')))


def get_checkpoint_iteration(ckpt_dir: str) -> int:
    """Get the iteration number from a checkpoint. Returns -1 if no checkpoint."""
    state_path = os.path.join(ckpt_dir, 'train_state.pkl')
    if not os.path.exists(state_path):
        return -1
    try:
        with open(state_path, 'rb') as f:
            train_state = pickle.load(f)
        return train_state['iteration']
    except Exception:
        return -1
