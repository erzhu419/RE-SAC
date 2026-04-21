"""RE-SAC: SAC + ensemble + OOD reg + LCB policy — scan-fused.

Changes vs. original:
1. weight_reg removed from actor loss (was double-counting with critic target).
2. beta passed as dynamic arg to scan (not closed-over), enabling adaptive schedule.
3. Independent critic targets (no min across ensemble).
4. EMA policy for stable evaluation.
5. Policy anchoring (KL penalty toward best policy).
6. Performance-aware adaptive beta.
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import numpy as np
from copy import deepcopy
from collections import deque

from jax_experiments.algos.sac_base import SACBase


class RESAC(SACBase):
    """RE-SAC with LCB-based policy loss. Overrides scan body."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- EMA policy (Polyak-averaged actor for stable eval) ---
        self.ema_policy = deepcopy(self.policy)
        self.ema_tau = getattr(self.config, 'ema_tau', 0.005)

        # --- Policy anchoring ---
        self._anchor_params = nnx.state(self.policy, nnx.Param)
        self._best_eval = -float('inf')
        self.anchor_lambda = getattr(self.config, 'anchor_lambda', 0.1)

        # --- Performance-aware beta ---
        self._eval_history = deque(maxlen=10)
        self._current_beta = self.config.beta

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        # Ant-stability: captured as compile-time constants
        ind_ratio = self.config.independent_ratio
        lcb_norm = self.config.lcb_normalize
        std_clip = self.config.q_std_clip

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
                         rng_key, beta_lcb, anchor_params, anchor_lambda):
            """beta_lcb, anchor_params, anchor_lambda are dynamic args."""

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os, key) = carry
                (obs, act, rew, next_obs, done) = batch_data
                key, k1, k2 = jax.random.split(key, 3)
                alpha = jnp.exp(la)

                # Pre-merge models not differentiated in critic loss
                tm = nnx.merge(gd_target, t_p)
                pm_frozen = nnx.merge(gd_policy, p_p)
                na, nlp = pm_frozen.sample(next_obs, k1)

                # --- Target: blend independent and min (#4) ---
                tq_all = tm(next_obs, na)             # [K, batch]
                tq_min = tq_all.min(axis=0)           # [batch]
                # ind_ratio=1.0 → pure independent; 0.0 → pure min
                tq_blend = ind_ratio * tq_all + (1 - ind_ratio) * tq_min[None]
                tq_blend = tq_blend - alpha * nlp     # [K, batch]
                tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_blend

                # === Critic (only critic params differentiated) ===
                def critic_loss_fn(cp):
                    """Critic loss L_Q — paper Eq. (13).

                    Paper form:
                        L_Q(φ_k) = E[(Q_φk - y_k)^2] + λ_ale Σ‖W_l^(k)‖_1
                                   + β_ood · σ_ens(s,a)
                    This returns only the MSE term. The λ_ale‖W‖_1 penalty is
                    applied via optax weight-decay in the critic optimizer;
                    the β_ood σ_ens OOD term is folded into the actor's LCB
                    objective below, not the critic loss.
                    """
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs, act)                 # [K, batch]
                    return jnp.mean((pq - tv_all) ** 2), pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # === Policy (LCB + anchor KL) ===
                cm = nnx.merge(gd_critic, new_c_p)

                def policy_loss_fn(pp):
                    """Policy loss L_π — paper Eq. (14).

                    Paper form:
                        L_π(θ) = E[α log π(a|s) - (Q̄(s,a) + β_lcb σ_ens(s,a))]
                    Here qm = Q̄, qs = σ_ens, and lcb = Q̄ + β_lcb σ_ens. The
                    anchor term below (L2 to a frozen best-policy snapshot)
                    is an additional regularizer not in Eq. (14).
                    """
                    pm = nnx.merge(gd_policy, pp)
                    na, lp = pm.sample(obs, k2)
                    qv = cm(obs, na)
                    qm = qv.mean(axis=0)
                    qs = qv.std(axis=0)

                    # (#2) Q-std clipping: cap std as fraction of |mean|
                    if std_clip > 0:
                        qs = jnp.minimum(qs, std_clip * jnp.maximum(jnp.abs(qm), 1.0))

                    # (#1) Normalized LCB: penalty relative to Q magnitude
                    if lcb_norm:
                        q_scale = jnp.maximum(jnp.abs(qm), 1.0)
                        lcb = qm + beta_lcb * qs / q_scale * jnp.abs(qm)
                    else:
                        lcb = qm + beta_lcb * qs

                    base_loss = (jnp.exp(la) * lp - lcb).mean()

                    # Policy anchoring: L2 distance to best policy params
                    anchor_dist = jax.tree.map(
                        lambda p, a: jnp.sum((p - a) ** 2), pp, anchor_params)
                    anchor_loss = anchor_lambda * sum(
                        jax.tree.leaves(anchor_dist))

                    return base_loss + anchor_loss, lp

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

                # === Target ===
                new_t_p = jax.tree.map(
                    lambda tp, cp: tp * (1 - tau) + cp * tau, t_p, new_c_p)

                new_carry = (new_c_p, new_t_p, new_p_p, new_la,
                             new_c_os, new_p_os, new_a_os, key)
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    log_alpha, c_opt_state, p_opt_state, a_opt_state, rng_key)
            return jax.lax.scan(body_fn, init,
                               (all_obs, all_act, all_rew, all_next_obs, all_done))

        self._scan_update = _scan_update

    # ------------------------------------------------------------------
    # Performance-aware adaptive beta
    # ------------------------------------------------------------------
    def get_adaptive_beta(self, iteration: int, max_iters: int) -> float:
        """
        Performance-aware β_lcb schedule:
        - [0, warmup): hold beta_start (safe exploration)
        - [warmup, end]: linear baseline from beta_start → beta_end,
          but gate on recent eval trend — tighten if performance drops.
        """
        cfg = self.config
        if not cfg.adaptive_beta:
            return cfg.beta

        warmup_iters = int(cfg.beta_warmup * max_iters)
        if iteration < warmup_iters:
            return cfg.beta_start

        # Linear baseline
        progress = (iteration - warmup_iters) / max(1, max_iters - warmup_iters)
        progress = min(1.0, progress)
        beta_linear = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * progress

        # Performance gating: if eval is declining, pull β back toward start
        if len(self._eval_history) >= 3:
            recent = list(self._eval_history)
            # Compare last 3 evals to the 3 before that
            if len(recent) >= 6:
                later = np.mean(recent[-3:])
                earlier = np.mean(recent[-6:-3])
            else:
                later = np.mean(recent[-2:])
                earlier = np.mean(recent[:-2]) if len(recent) > 2 else later

            if earlier > 0 and later < earlier * 0.9:
                # Performance dropped >10%: tighten β halfway back to start
                beta_linear = (beta_linear + cfg.beta_start) / 2.0

        self._current_beta = beta_linear
        return beta_linear

    def report_eval(self, eval_reward: float):
        """Called by training loop after each evaluation.
        Updates anchor if new best, feeds performance-aware beta."""
        self._eval_history.append(eval_reward)

        if eval_reward > self._best_eval:
            self._best_eval = eval_reward
            # Update anchor to current best policy
            self._anchor_params = nnx.state(self.policy, nnx.Param)

    # ------------------------------------------------------------------
    # EMA policy update
    # ------------------------------------------------------------------
    def _update_ema_policy(self):
        """Soft-update EMA policy params from current policy."""
        ema_p = nnx.state(self.ema_policy, nnx.Param)
        cur_p = nnx.state(self.policy, nnx.Param)
        new_ema = jax.tree.map(
            lambda e, c: e * (1 - self.ema_tau) + c * self.ema_tau, ema_p, cur_p)
        nnx.update(self.ema_policy, new_ema)

    def select_action_ema(self, obs, deterministic=True):
        """Select action using EMA policy (for stable evaluation)."""
        obs_jax = jnp.array(obs) if obs.ndim == 1 else obs
        if obs_jax.ndim == 1:
            obs_jax = obs_jax[None]
        if deterministic:
            return np.array(self.ema_policy.deterministic(obs_jax)[0])
        key = self.rngs.params()
        action, _ = self.ema_policy.sample(obs_jax, key)
        return np.array(action[0])

    # ------------------------------------------------------------------
    # Override multi_update to pass beta_lcb + anchor dynamically
    # ------------------------------------------------------------------
    def multi_update(self, stacked_batch: dict, beta_override: float = None, **kwargs):
        """Fused N-step update with EMA policy update and anchoring."""
        rng_key = self.rngs.params()

        c_p = nnx.state(self.critic, nnx.Param)
        t_p = nnx.state(self.target_critic, nnx.Param)
        p_p = nnx.state(self.policy, nnx.Param)

        obs = stacked_batch["obs"]
        act = stacked_batch["act"]
        rew = stacked_batch["rew"]
        nobs = stacked_batch["next_obs"]
        done = stacked_batch["done"]

        beta_val = beta_override if beta_override is not None else self.config.beta
        beta_lcb = jnp.array(beta_val, dtype=jnp.float32)
        anchor_lambda = jnp.array(self.anchor_lambda, dtype=jnp.float32)

        final, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            obs, act, rew, nobs, done, rng_key, beta_lcb,
            self._anchor_params, anchor_lambda)

        # Unpack
        (new_c, new_t, new_p, new_la,
         self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
         _) = final
        nnx.update(self.critic, new_c)
        nnx.update(self.target_critic, new_t)
        nnx.update(self.policy, new_p)
        self.log_alpha = new_la

        # EMA policy update
        self._update_ema_policy()

        n = obs.shape[0]
        self.update_count += n

        c_loss, p_loss, alpha, q_mean, q_std, lp = metrics
        return {
            "critic_loss": float(c_loss.mean()),
            "policy_loss": float(p_loss.mean()),
            "alpha": float(alpha[-1]),
            "q_mean": float(q_mean.mean()),
            "q_std_mean": float(q_std.mean()),
            "log_prob": float(lp.mean()),
        }
