"""TQC: Truncated Quantile Critics — scan-fused JAX.

N critics, each outputs K quantile values. For the target:
- Gather all N*K atoms from target critics
- Sort and drop top d atoms per critic (truncation)
- Remaining atoms form the distributional target

Reference: Kuznetsov et al., "Controlling Overestimation Bias with Truncated
Mixture of Continuous Distributional Quantile Critics" (ICML 2020)

Default: N=5, K=25, d=2 (drop top 2 quantiles per critic → keep 23*5=115 atoms)
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from copy import deepcopy
import numpy as np

from jax_experiments.networks.policy import GaussianPolicy
from jax_experiments.networks.n_quantile_critic import NQuantileCritic


class TQC:
    """TQC with N quantile critics and truncated distributional target."""

    def __init__(self, obs_dim: int, act_dim: int, config, seed: int = 0):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.rngs = nnx.Rngs(seed)

        self.n_critics = getattr(config, 'tqc_n_critics', 5)
        self.num_quantiles = getattr(config, 'tqc_n_quantiles', 25)
        self.top_quantiles_to_drop = getattr(config, 'tqc_drop', 2)

        self.policy = GaussianPolicy(
            obs_dim, act_dim, config.hidden_dim,
            n_layers=2, rngs=self.rngs)
        self.critic = NQuantileCritic(
            obs_dim, act_dim, config.hidden_dim,
            n_critics=self.n_critics, n_layers=2, embedding_size=32,
            num_quantiles=self.num_quantiles, rngs=self.rngs)
        self.target_critic = deepcopy(self.critic)

        self.log_alpha = jnp.array(jnp.log(config.alpha))
        self.target_entropy = -float(act_dim)

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

    @staticmethod
    def _make_tau(key, batch_size, num_quantiles):
        """IQN-style random quantile fractions."""
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
        n_critics = self.n_critics
        n_quantiles = self.num_quantiles
        d = self.top_quantiles_to_drop
        # Number of atoms to keep per critic
        n_keep = n_quantiles - d

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
                         all_tau_hat, all_presum_tau,
                         rng_key):

            def quantile_huber_loss(pred, target, tau_hat):
                """Quantile Huber loss.
                pred: (batch, K), target: (batch, K'), tau_hat: (batch, K)
                """
                diff = target[:, None, :] - pred[:, :, None]  # (B, K, K')
                huber = jnp.where(jnp.abs(diff) <= 1.0,
                                  0.5 * diff ** 2,
                                  jnp.abs(diff) - 0.5)
                sign = jax.lax.stop_gradient(
                    (diff < 0).astype(jnp.float32))
                tau_e = tau_hat[:, :, None]  # (B, K, 1)
                rho = jnp.abs(tau_e - sign) * huber
                return rho.sum(axis=-1).mean()

            def body_fn(carry, scan_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done,
                 tau_hat, presum_tau) = scan_data

                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # === Target: truncated distribution ===
                pm_frozen = nnx.merge(gd_policy, p_p)
                na, nlp = pm_frozen.sample(next_obs, k1)

                tm = nnx.merge(gd_target, t_p)
                tq_all = tm(next_obs, na, tau_hat)  # (N, batch, K)

                # Sort each critic's quantiles and drop top d
                tq_sorted = jnp.sort(tq_all, axis=-1)  # sort along K
                tq_truncated = tq_sorted[:, :, :n_keep]  # (N, batch, K-d)

                # Flatten across critics: (batch, N*(K-d))
                tq_flat = tq_truncated.transpose(1, 0, 2).reshape(
                    tq_truncated.shape[1], -1)

                # Subtract entropy
                tq_target = tq_flat - alpha * nlp[:, None]

                # Bellman target: (batch, N*(K-d))
                z_target = rew.squeeze(-1)[:, None] + \
                    gamma * (1 - done.squeeze(-1))[:, None] * tq_target

                # === Critic loss: each of N critics against truncated target ===
                def critic_loss_fn(cp):
                    cm = nnx.merge(gd_critic, cp)
                    z_pred = cm(obs, act, tau_hat)  # (N, batch, K)
                    # Loss per critic head
                    total_loss = 0.0
                    for i in range(n_critics):
                        total_loss = total_loss + quantile_huber_loss(
                            z_pred[i], z_target, tau_hat)
                    return total_loss / n_critics, z_pred

                (c_loss, z_pred), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy: maximize mean of all N*K quantile atoms ===
                def policy_loss_fn(pp):
                    pm = nnx.merge(gd_policy, pp)
                    cm = nnx.merge(gd_critic, new_c_p)
                    na, lp = pm.sample(obs, k2)
                    z_all = cm(obs, na, tau_hat)  # (N, batch, K)
                    # Mean over all atoms
                    q = z_all.mean(axis=(0, 2))  # (batch,)
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
                q_mean = z_pred.mean()
                q_std = z_pred.std(axis=0).mean()
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           q_mean, q_std, lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    log_alpha, c_opt_state, p_opt_state, a_opt_state, rng_key)
            batches = (all_obs, all_act, all_rew, all_next_obs, all_done,
                       all_tau_hat, all_presum_tau)
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
            return np.array(self.policy.deterministic(obs_jax)[0])
        key = self.rngs.params()
        action, _ = self.policy.sample(obs_jax, key)
        return np.array(action[0])

    def multi_update(self, stacked_batch: dict, **kwargs):
        """Fused N-step update with pre-sampled tau."""
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

        # Pre-sample tau for all steps
        tau_key = self.rngs.params()
        keys = jax.random.split(tau_key, n_steps)
        tau_hat, presum_tau = jax.vmap(
            lambda k: self._make_tau(k, batch_size, self.num_quantiles)
        )(keys)

        final, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done,
            tau_hat, presum_tau, rng_key)

        (new_c, new_t, new_p, new_la,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        self.log_alpha = new_la

        self.update_count += n_steps

        c_loss, p_loss, alpha_val, q_mean, q_std, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "alpha": float(alpha_val[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(lp.mean()),
        }
