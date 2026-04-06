"""DSAC: Distributional SAC with IQN — scan-fused JAX (optimized).

Optimizations vs v1:
  1. Pre-sample ALL tau values outside scan body → no random key splitting inside
  2. Shared tau between critic and policy loss → 2 fewer IQN forward passes
  3. num_quantiles=8 default (was 32) → 16x less cross-quantile work
  4. No LayerNorm in critic (handled by quantile_critic.py)
  5. Simplified quantile regression loss
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from copy import deepcopy
import numpy as np

from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.networks.quantile_critic import TwinQuantileCritic


class DSAC:
    """DSAC with IQN critics, scan-fused gradient updates (optimized)."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        num_quantiles = getattr(config, 'num_quantiles', 8)

        # Networks
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            n_layers=2, rngs=self.rngs)
        self.twin_critic = TwinQuantileCritic(
            obs_dim, act_dim, config.hidden_dim,
            n_layers=2, embedding_size=32,
            num_quantiles=num_quantiles, rngs=self.rngs)
        self.target_twin_critic = deepcopy(self.twin_critic)

        # Auto-alpha
        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

        # Optimizers
        self.policy_opt = optax.adam(config.lr)
        self.critic_opt = optax.adam(config.lr)
        self.alpha_opt = optax.adam(config.lr)

        self.policy_opt_state = self.policy_opt.init(
            nnx.state(self.policy, nnx.Param))
        self.critic_opt_state = self.critic_opt.init(
            nnx.state(self.twin_critic, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        self.update_count = 0
        self.num_quantiles = num_quantiles
        self._build_scan_fn()

    @staticmethod
    def _make_tau(key, batch_size, num_quantiles):
        """IQN-style random quantile fractions (static method for pre-sampling)."""
        presum_tau = jax.random.uniform(key, (batch_size, num_quantiles)) + 0.1
        presum_tau = presum_tau / presum_tau.sum(axis=-1, keepdims=True)
        tau = jnp.cumsum(presum_tau, axis=1)
        tau_hat = jnp.zeros_like(tau)
        tau_hat = tau_hat.at[:, 0:1].set(tau[:, 0:1] / 2.0)
        tau_hat = tau_hat.at[:, 1:].set((tau[:, 1:] + tau[:, :-1]) / 2.0)
        return tau_hat, presum_tau

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau_polyak = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.twin_critic)
        gd_target = nnx.graphdef(self.target_twin_critic)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, c_opt_state, p_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         all_tau_hat_tgt, all_presum_tau_tgt,
                         all_tau_hat_cur, all_presum_tau_cur,
                         rng_key):
            """
            Pre-sampled tau arrays are passed as scan inputs:
              all_tau_hat_tgt:    (N_steps, batch, T)
              all_presum_tau_tgt: (N_steps, batch, T)
              all_tau_hat_cur:    (N_steps, batch, T)
              all_presum_tau_cur: (N_steps, batch, T)
            """

            def quantile_regression_loss(pred, target, tau_hat, presum_tau):
                """Quantile Huber loss. All shapes: (batch, T)."""
                diff = target[:, None, :] - pred[:, :, None]  # (N, T, T)
                huber = jnp.where(jnp.abs(diff) <= 1.0,
                                  0.5 * diff ** 2,
                                  jnp.abs(diff) - 0.5)
                sign = jax.lax.stop_gradient(
                    jnp.sign(pred[:, :, None] - target[:, None, :]) / 2.0 + 0.5)
                tau_e = tau_hat[:, :, None]
                weight_e = jax.lax.stop_gradient(presum_tau[:, None, :])
                rho = jnp.abs(tau_e - sign) * huber * weight_e
                return rho.sum(axis=-1).mean()

            def body_fn(carry, scan_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done,
                 tau_hat_tgt, presum_tau_tgt,
                 tau_hat_cur, presum_tau_cur) = scan_data

                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # === Critic loss ===
                def critic_loss_fn(cp):
                    pm = nnx.merge(gd_policy, p_p)
                    na, nlp = pm.sample(next_obs, k1)

                    # Target Q (use pre-sampled tau)
                    tm = nnx.merge(gd_target, t_p)
                    tz1 = tm.zf1(next_obs, na, tau_hat_tgt)
                    tz2 = tm.zf2(next_obs, na, tau_hat_tgt)
                    tz = jnp.minimum(tz1, tz2) - alpha * nlp[:, None]
                    z_target = rew.squeeze(-1)[:, None] + \
                               gamma * (1 - done.squeeze(-1))[:, None] * tz

                    # Current Q (use pre-sampled tau)
                    cm = nnx.merge(gd_critic, cp)
                    z1 = cm.zf1(obs, act, tau_hat_cur)
                    z2 = cm.zf2(obs, act, tau_hat_cur)

                    loss1 = quantile_regression_loss(
                        z1, z_target, tau_hat_cur, presum_tau_tgt)
                    loss2 = quantile_regression_loss(
                        z2, z_target, tau_hat_cur, presum_tau_tgt)
                    return loss1 + loss2, z1

                (c_loss, z1), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy loss (reuse tau_hat_cur) ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2)
                    z1_new = cm.zf1(obs, na, tau_hat_cur)
                    z2_new = cm.zf2(obs, na, tau_hat_cur)
                    q1 = (presum_tau_cur * z1_new).sum(axis=1)
                    q2 = (presum_tau_cur * z2_new).sum(axis=1)
                    q = jnp.minimum(q1, q2)
                    return (jnp.exp(la) * lp - q).mean(), lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                # === Alpha ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Target update ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau_polyak) + cp * tau_polyak,
                    t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_la,
                             new_c_os, new_p_os, new_a_os, key)
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           z1.mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    log_alpha, c_opt_state, p_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_tau_hat_tgt, all_presum_tau_tgt,
                       all_tau_hat_cur, all_presum_tau_cur)
            return jax.lax.scan(body_fn, init, batches)

        self._scan_update = _scan_update

    @property
    def alpha(self):
        return jnp.exp(self.log_alpha)

    def select_action(self, obs, deterministic=False):
        obs_jax = jnp.array(obs) if obs.ndim == 1 else obs
        if obs_jax.ndim == 1:
            obs_jax = obs_jax[None]
        if deterministic:
            action = self.policy.deterministic(obs_jax)
        else:
            key = self.rngs.params()
            action, _ = self.policy.sample(obs_jax, key)
        return np.array(action[0])

    def multi_update(self, stacked_batch: dict, **kwargs):
        """Fused N-step update with pre-sampled tau."""
        rng_key = self.rngs.params()

        c_p = nnx.state(self.twin_critic, nnx.Param)
        t_p = nnx.state(self.target_twin_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)

        # Data is already on device (GPU replay buffer) — no conversion needed
        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        n_steps = obs.shape[0]
        batch_size = obs.shape[1]

        # Pre-sample ALL tau values outside the scan (one bulk JAX call)
        tau_key = self.rngs.params()
        keys = jax.random.split(tau_key, n_steps * 2)
        keys_tgt = keys[:n_steps]
        keys_cur = keys[n_steps:]

        # Vectorized tau sampling: vmap over steps
        tau_hat_tgt, presum_tau_tgt = jax.vmap(
            lambda k: self._make_tau(k, batch_size, self.num_quantiles)
        )(keys_tgt)
        tau_hat_cur, presum_tau_cur = jax.vmap(
            lambda k: self._make_tau(k, batch_size, self.num_quantiles)
        )(keys_cur)

        final, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done,
            tau_hat_tgt, presum_tau_tgt,
            tau_hat_cur, presum_tau_cur,
            rng_key)

        (new_c, new_t, new_p, new_la,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _) = final
        nnx.update(self.twin_critic, new_c)
        nnx.update(self.target_twin_critic, new_t)
        nnx.update(self.policy, new_p)
        self.log_alpha = new_la

        self.update_count += n_steps

        c_loss, p_loss, alpha_val, q_mean, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "alpha": float(alpha_val[-1]),
            "q_mean": float(q_mean.mean()),
            "log_prob": float(lp.mean()),
        }
