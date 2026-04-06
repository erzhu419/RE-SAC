"""TD3: Twin Delayed DDPG — scan-fused JAX implementation.

Key differences from SAC:
  - Deterministic policy (no entropy, no log_prob, no alpha)
  - Delayed policy updates (every 2 critic steps)
  - Target policy smoothing (clipped Gaussian noise on target actions)
  - Exploration via Gaussian noise during collection
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from copy import deepcopy
import numpy as np

from jax_experiments.networks.deterministic_policy import DeterministicPolicy
from jax_experiments.networks.ensemble_critic import EnsembleCritic


class TD3:
    """TD3 with scan-fused gradient updates."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        # Networks
        self.policy = DeterministicPolicy(
            obs_dim, act_dim, config.hidden_dim,
            n_layers=2, rngs=self.rngs)
        self.target_policy = deepcopy(self.policy)

        self.critic = EnsembleCritic(
            obs_dim, act_dim, config.hidden_dim,
            ensemble_size=2, n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        # Optimizers
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)

        self.policy_opt_state = self.policy_opt.init(
            nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(
            nnx.state(self.critic, nnx.Param))

        self.update_count = 0
        self._build_scan_fn()

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        policy_noise = getattr(self.config, 'td3_target_noise', 0.2)
        noise_clip = getattr(self.config, 'td3_noise_clip', 0.5)
        policy_delay = getattr(self.config, 'td3_policy_delay', 2)

        gd_policy = nnx.graphdef(self.policy)
        gd_tgt_policy = nnx.graphdef(self.target_policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_tgt_critic = nnx.graphdef(self.target_critic)

        p_opt = self.policy_opt
        c_opt = self.critic_opt

        @jax.jit
        def _scan_update(critic_params, tgt_critic_params,
                         policy_params, tgt_policy_params,
                         c_opt_state, p_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         rng_key, start_step):
            def body_fn(carry, batch_data):
                (c_p, tc_p, p_p, tp_p, c_os, p_os, key, step) = carry
                (obs, act, rew, next_obs, done) = batch_data
                key, k1, k2 = jax.random.split(key, 3)

                # === Critic update (every step) ===
                def critic_loss_fn(cp):
                    # Target policy smoothing
                    tgt_pm = nnx.merge(gd_tgt_policy, tp_p)
                    next_act = tgt_pm(next_obs)
                    noise = jax.random.normal(k1, next_act.shape) * policy_noise
                    noise = jnp.clip(noise, -noise_clip, noise_clip)
                    next_act = jnp.clip(next_act + noise, -1.0, 1.0)

                    # Min of twin target Q
                    tgt_cm = nnx.merge(gd_tgt_critic, tc_p)
                    tq = tgt_cm(next_obs, next_act).min(axis=0)
                    tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq

                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs, act)
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Delayed policy update ===
                do_policy_update = (step % policy_delay) == 0

                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    new_act = pm(obs)
                    # Use first critic for policy gradient
                    qv = cm(obs, new_act)
                    return -qv.mean(axis=0).mean()

                p_loss, p_grads = jax.value_and_grad(policy_loss_fn)(p_p)
                p_upd, new_p_os_candidate = p_opt.update(p_grads, p_os, p_p)
                new_p_p_candidate = optax.apply_updates(p_p, p_upd)

                # Conditionally apply policy update
                new_p_p = jax.tree.map(
                    lambda new, old: jnp.where(do_policy_update, new, old),
                    new_p_p_candidate, p_p)
                new_p_os = jax.tree.map(
                    lambda new, old: jnp.where(do_policy_update, new, old),
                    new_p_os_candidate, p_os)

                # Soft target updates (only when policy updates)
                new_tc_p = jax.tree.map(
                    lambda tp, cp: jnp.where(do_policy_update,
                                             tp * (1 - tau) + cp * tau, tp),
                    tc_p, new_c_p)
                new_tp_p = jax.tree.map(
                    lambda tp, pp: jnp.where(do_policy_update,
                                             tp * (1 - tau) + pp * tau, tp),
                    tp_p, new_p_p)

                new_carry = (new_c_p, new_tc_p, new_p_p, new_tp_p,
                             new_c_os, new_p_os, key, step + 1)
                metrics = (c_loss, p_loss, pq.mean(), pq.std(axis=0).mean())
                return new_carry, metrics

            init = (critic_params, tgt_critic_params,
                    policy_params, tgt_policy_params,
                    c_opt_state, p_opt_state, rng_key, start_step)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done)
            return jax.lax.scan(body_fn, init, batches)

        self._scan_update = _scan_update

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs) if obs.ndim == 1 else obs
        if obs_jax.ndim == 1:
            obs_jax = obs_jax[None]
        if deterministic:
            action = self.policy(obs_jax)
        else:
            key = self.rngs.params()
            action = self.policy.noisy_action(obs_jax, key, noise_std=0.1)
        return np.array(action[0])

    def multi_update(self, stacked_batch: dict, **kwargs):
        """Fused N-step update. Returns averaged metrics."""
        rng_key = self.rngs.params()

        c_p = nnx.state(self.critic, nnx.Param)
        tc_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)
        tp_p = nnx.state(self.target_policy, nnx.Param)

        # Data is already on device (GPU replay buffer) — no conversion needed
        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        final, metrics = self._scan_update(
            c_p, tc_p, p_p, tp_p,
            self.critic_opt_state, self.policy_opt_state,
            obs, act, rew, nobs, done, rng_key,
            jnp.array(self.update_count))

        (new_c, new_tc, new_p, new_tp,
         self.critic_opt_state, self.policy_opt_state, _, _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_tc)
        nnx.update(self.policy, new_p)
        nnx.update(self.target_policy, new_tp)

        n = obs.shape[0]
        self.update_count += n

        c_loss, p_loss, q_mean, q_std = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
        }
