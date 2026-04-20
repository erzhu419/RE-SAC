"""REDQ: Randomized Ensembled Double Q-Learning — scan-fused.

Key differences from SAC:
- N ensemble Q-networks (default N=10)
- Target uses min over random subset of M=2 (not all N)
- Policy uses mean over random subset of M=2
- High UTD ratio (default 20, but we match others at 250 updates/iter)

Reference: Chen et al., "Randomized Ensembled Double Q-Learning:
Learning Fast Without a Model" (ICLR 2021)
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import numpy as np

from jax_experiments.algos.sac_base import SACBase


class REDQ(SACBase):
    """REDQ: SAC + large ensemble + random subset min target."""

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        M = getattr(self.config, 'redq_m', 2)  # subset size for min
        N = self.config.ensemble_size

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, c_opt_state, p_opt_state, a_opt_state,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         rng_key):

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done) = batch_data
                key, k1, k2, k3 = jax.random.split(key, 4)
                alpha = jnp.exp(la)

                # === Target: random subset min ===
                t_model = nnx.merge(gd_target, t_p)
                p_model = nnx.merge(gd_policy, p_p)
                na, nlp = p_model.sample(next_obs, k1)
                tq_all = t_model(next_obs, na)  # [N, batch]

                # Random subset of M indices
                indices = jax.random.choice(k3, N, shape=(M,), replace=False)
                tq_subset = tq_all[indices]  # [M, batch]
                tq = tq_subset.min(axis=0) - alpha * nlp  # [batch]
                tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq

                # === Critic: all N heads trained on shared target ===
                def critic_loss_fn(cp):
                    c_model = nnx.merge(gd_critic, cp)
                    pq = c_model(obs, act)  # [N, batch]
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy: use mean of all N heads (standard SAC) ===
                cm = nnx.merge(gd_critic, new_c_p)

                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    na, lp = pm.sample(obs, k2)
                    qv = cm(obs, na)  # [N, batch]
                    return (jnp.exp(la) * lp - qv.mean(axis=0)).mean(), lp

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

                # === Target soft update ===
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
            return jax.lax.scan(body_fn, init_carry, batches)

        self._scan_update = _scan_update
