"""Core SAC with lax.scan-fused gradient updates — base for RE-SAC.

All gradient steps are fused into a single @jax.jit call via jax.lax.scan,
eliminating per-step Python overhead (nnx.state extraction, float() sync, etc).
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from copy import deepcopy
import numpy as np

from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.networks.ensemble_critic import EnsembleCritic


class SACBase:
    """Vanilla SAC with ensemble critic — scan-fused gradient updates."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        # Networks
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            n_layers=2, rngs=self.rngs)
        self.critic = EnsembleCritic(
            obs_dim, act_dim, config.hidden_dim,
            ensemble_size=config.ensemble_size, n_layers=3, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

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
            nnx.state(self.critic, nnx.Param))
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        self.update_count = 0
        self._build_scan_fn()

    def _build_scan_fn(self):
        """Build JIT-compiled scan function for fused multi-step updates."""
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)

        # Capture graphdefs at build time (static structure, never changes)
        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)

        # Capture optimizer objects (stateless transformation descriptions)
        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, c_opt_state, p_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         rng_key):
            """Fused N-step gradient update via lax.scan."""

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done) = batch_data

                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # Pre-merge models that don't need gradients in critic loss
                # (avoids re-tracing inside value_and_grad closure)
                t_model = nnx.merge(gd_target, t_p)
                p_model_frozen = nnx.merge(gd_policy, p_p)
                na, nlp = p_model_frozen.sample(next_obs, k1)
                tq = t_model(next_obs, na).min(axis=0) - alpha * nlp
                tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq

                # === Critic loss + grad (only critic params differentiated) ===
                def critic_loss_fn(cp):
                    c_model = nnx.merge(gd_critic, cp)
                    pq = c_model(obs, act)
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy loss + grad (only policy params differentiated) ===
                # Merge updated critic ONCE outside the closure
                cm = nnx.merge(gd_critic, new_c_p)

                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    na, lp = pm.sample(obs, k2)
                    qv = cm(obs, na)
                    return (jnp.exp(la) * lp - qv.mean(axis=0)).mean(), lp

                (p_loss, lp), p_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True)(p_p)
                p_upd, new_p_os = p_opt.update(p_grads, p_os, p_p)
                new_p_p = optax.apply_updates(p_p, p_upd)

                # === Alpha update ===
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # === Soft target update ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_la,
                             new_c_os, new_p_os, new_a_os, key)
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init_carry = (critic_params, target_params, policy_params,
                          log_alpha, c_opt_state, p_opt_state, a_opt_state,
                          rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done)
            final_carry, all_metrics = jax.lax.scan(body_fn, init_carry, batches)
            return final_carry, all_metrics

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
        """Fused N-step update via lax.scan. Returns averaged metrics dict."""
        rng_key = self.rngs.params()

        # Extract params
        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)

        # Data is already on device (GPU replay buffer) — no conversion needed
        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        # Single JIT call for all N gradient steps
        final, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, rng_key)

        # Unpack final state back to modules
        (new_c, new_t, new_p, new_la,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        self.log_alpha = new_la

        n = obs.shape[0]
        self.update_count += n

        # Unpack metrics: one float() per metric (not 250)
        c_loss, p_loss, alpha, q_mean, q_std, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(lp.mean()),
        }
