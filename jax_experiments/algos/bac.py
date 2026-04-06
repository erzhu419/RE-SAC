"""BAC: Blended Actor-Critic (simplified) — scan-fused JAX.

Implements the core BEE (Blended Exploitation and Exploration) operator
from "Seizing Serendipity" (Ji et al., ICML 2024) on top of SAC.

Key difference from SAC:
  Bellman target y = r + γ * [λ * max_{a∈D} Q(s',a) + (1-λ) * Q(s',a')] - α * logπ

where λ ∈ [0,1] controls exploitation vs exploration blend, and the max
is over a random subset of actions sampled from the replay buffer.
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from copy import deepcopy
import numpy as np

from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.networks.ensemble_critic import EnsembleCritic


class BAC:
    """Simplified BAC with BEE operator, scan-fused gradient updates."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        # BAC-specific hyperparams
        self.lam = getattr(config, 'bac_lambda', 0.5)
        self.n_candidate = getattr(config, 'bac_n_candidate', 10)

        # Networks (same as SAC with twin critics)
        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            n_layers=2, rngs=self.rngs)
        self.critic = EnsembleCritic(
            obs_dim, act_dim, config.hidden_dim,
            ensemble_size=2, n_layers=3, rngs=self.rngs)
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
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        lam = self.lam
        n_candidate = self.n_candidate

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
                         all_candidate_acts, rng_key):
            """BEE-modified scan update.

            all_candidate_acts: [N_steps, batch, n_candidate, act_dim]
            """

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done, cand_acts) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # BEE target: blend exploitation and exploration
                t_model = nnx.merge(gd_target, t_p)
                p_model_frozen = nnx.merge(gd_policy, p_p)

                # Exploration: a' ~ π(·|s')
                na, nlp = p_model_frozen.sample(next_obs, k1)
                q_explore = t_model(next_obs, na).min(axis=0) - alpha * nlp

                # Exploitation: max_{a∈candidates} Q(s', a)
                batch_size = next_obs.shape[0]
                next_obs_exp = jnp.broadcast_to(
                    next_obs[:, None, :],
                    (batch_size, n_candidate, next_obs.shape[-1]))
                no_flat = next_obs_exp.reshape(-1, next_obs.shape[-1])
                ca_flat = cand_acts.reshape(-1, cand_acts.shape[-1])
                q_cand = t_model(no_flat, ca_flat).min(axis=0)
                q_cand = q_cand.reshape(batch_size, n_candidate)
                q_exploit = q_cand.max(axis=1)

                # BEE blend
                q_target = lam * q_exploit + (1 - lam) * q_explore
                tv = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * q_target

                # Critic loss
                def critic_loss_fn(cp):
                    c_model = nnx.merge(gd_critic, cp)
                    pq = c_model(obs, act)
                    return jnp.mean((pq - tv[None]) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # Policy loss (standard SAC)
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

                # Alpha
                a_grad = -(lp.mean() + target_entropy)
                a_upd, new_a_os = a_opt.update(a_grad, a_os, la)
                new_la = jnp.where(auto_alpha, la + a_upd, la)
                new_a_os = jax.tree.map(
                    lambda n, o: jnp.where(auto_alpha, n, o), new_a_os, a_os)

                # Target update
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
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_candidate_acts)
            return jax.lax.scan(body_fn, init_carry, batches)

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
        """Fused N-step update with BEE operator."""
        rng_key = self.rngs.params()

        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)

        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        n_steps = obs.shape[0]
        batch_size = obs.shape[1]

        # Generate candidate actions (random uniform as proxy for buffer sampling)
        cand_key = self.rngs.params()
        candidate_acts = jax.random.uniform(
            cand_key, (n_steps, batch_size, self.n_candidate, self.act_dim),
            minval=-1.0, maxval=1.0)

        final, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, candidate_acts, rng_key)

        (new_c, new_t, new_p, new_la,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        self.log_alpha = new_la

        self.update_count += n_steps

        c_loss, p_loss, alpha, q_mean, q_std, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(lp.mean()),
        }
