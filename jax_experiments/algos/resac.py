"""RE-SAC: SAC + ensemble + OOD reg + LCB policy — scan-fused.

Changes vs. original:
1. weight_reg removed from actor loss (was double-counting with critic target).
2. beta passed as dynamic arg to scan (not closed-over), enabling adaptive schedule.
3. Independent critic targets (no min across ensemble).
4. EMA policy for stable evaluation.
5. Policy anchoring (KL penalty toward best policy).
6. Performance-aware adaptive beta.

Algorithm ablations (paper §6.1.6) gated by config flags, defaults all off so
the baseline path is unchanged:
  - config.use_spectral_norm   (Variant A): hard Lipschitz constraint via
    per-head spectral-norm rescaling after each critic step.
  - config.state_dep_beta      (Variant B): β_lcb scales per-state with σ_ens
    relative to its EMA baseline.
  - config.hash_count_bonus    (Variant C): σ_epi = σ_ens + α/√N(hash(s,a)).
"""
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import numpy as np
from copy import deepcopy
from collections import deque

from jax_experiments.algos.sac_base import SACBase
from jax_experiments.algos.ablation_utils import (
    apply_spectral_norm_to_critic_params,
    state_dep_beta,
    HashCounterState,
    hash_state_action,
    update_counts,
    count_bonus,
)


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

        # --- Ablation: Variant C (hash count bonus) auxiliary state ---
        if getattr(self.config, "hash_count_bonus", False):
            self._hash_state = HashCounterState.init(
                self.obs_dim, self.act_dim,
                hash_dim=int(self.config.hash_dim), seed=int(self.config.seed))
        else:
            self._hash_state = None

        # --- Ablation: Variant B (state-dep β) running σ_ema baseline ---
        self._sigma_ema = jnp.array(1.0)
        # --- Adaptive λ_ale: running EMA of TD residual variance ---
        self._td_var_ema = jnp.array(0.0)

    def _build_scan_fn(self):
        gamma = self.config.gamma
        tau = self.config.tau
        auto_alpha = self.config.auto_alpha
        target_entropy = jnp.array(self.target_entropy)
        # Ant-stability: captured as compile-time constants
        ind_ratio = self.config.independent_ratio
        lcb_norm = self.config.lcb_normalize
        std_clip = self.config.q_std_clip
        # Real l1/sigma_ood penalties (the paper's Eq. (13) terms that were
        # silently absent from the critic loss until Apr 2026 fix).
        weight_reg = float(getattr(self.config, "weight_reg", 0.0))
        beta_ood = float(getattr(self.config, "beta_ood", 0.0))

        # Adaptive λ_ale modes (paper §4.5): "off" / "td_ema" / "probe" /
        # "posterior". When mode != off, weight_reg above is overridden by
        # an estimate of aleatoric noise level computed inside the scan.
        adaptive_mode = str(getattr(self.config, "adaptive_reg_mode", "off"))
        adaptive_base = float(getattr(self.config, "adaptive_reg_base", 0.01))
        adaptive_thr = float(getattr(self.config, "adaptive_reg_threshold", 1.0))
        adaptive_scale = float(getattr(self.config, "adaptive_reg_scale", 1.0))
        adaptive_decay = float(getattr(self.config, "adaptive_reg_decay", 0.99))
        # Ablation flags (compile-time constants — JIT recompiles per variant)
        use_specnorm = bool(getattr(self.config, "use_spectral_norm", False))
        specnorm_c = float(getattr(self.config, "spectral_norm_value", 1.0))
        use_state_beta = bool(getattr(self.config, "state_dep_beta", False))
        state_beta_cap = float(getattr(self.config, "state_dep_beta_cap", 3.0))
        sigma_ema_decay = float(getattr(self.config, "state_dep_beta_ema", 0.99))
        use_count_bonus = bool(getattr(self.config, "hash_count_bonus", False))
        count_alpha = float(getattr(self.config, "hash_count_alpha", 0.5))
        # W_hash / hash_dim captured as compile-time constants when active.
        # _build_scan_fn is called by SACBase.__init__ before our subclass
        # init body runs, so create _hash_state lazily here if needed.
        if use_count_bonus:
            if not getattr(self, "_hash_state", None):
                self._hash_state = HashCounterState.init(
                    self.obs_dim, self.act_dim,
                    hash_dim=int(self.config.hash_dim),
                    seed=int(self.config.seed))
            W_hash = self._hash_state.W_hash
            hash_dim_const = self._hash_state.hash_dim
        else:
            W_hash = None
            hash_dim_const = 0

        gd_policy = nnx.graphdef(self.policy)
        gd_critic = nnx.graphdef(self.critic)
        gd_target = nnx.graphdef(self.target_critic)

        p_opt = self.policy_opt
        c_opt = self.critic_opt
        a_opt = self.alpha_opt

        @jax.jit
        def _scan_update(critic_params, target_params, policy_params,
                         log_alpha, c_opt_state, p_opt_state, a_opt_state,
                         sigma_ema, counts, td_var_ema,
                         all_obs, all_act, all_rew, all_next_obs, all_done,
                         rng_key, beta_lcb, anchor_params, anchor_lambda):
            """beta_lcb, anchor_params, anchor_lambda are dynamic args.
            sigma_ema (Variant B), counts (Variant C), and td_var_ema
            (adaptive λ_ale mode 'td_ema') are also threaded."""

            def body_fn(carry, batch_data):
                (c_p, t_p, p_p, la, c_os, p_os, a_os,
                 sig_ema, cnts, td_var, key) = carry
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

                # ── Aleatoric channel (bus-PyTorch style, paper §4.2.1) ──
                # Per-head pessimism shift in Bellman target, scaled by the
                # target critic's weight magnitude. Replaces the static
                # `weight_reg` with `weight_reg_eff` when an adaptive mode
                # is selected (paper §4.5).
                #
                # Adaptive modes:
                #   "off"      : weight_reg_eff = weight_reg (static)
                #   "td_ema"   : weight_reg_eff = base · σ((td_var_ema - thr)/scale)
                #                — TD-residual variance estimates aleatoric noise
                #   "posterior": weight_reg_eff = base · σ((within-head TD var - thr)/scale)
                #                — explicit aleatoric vs epistemic decomposition
                #   "probe"    : weight_reg is fixed at training start by train.py
                #                from start_train_steps random-rollout reward std
                use_aleatoric = (weight_reg > 0) or (adaptive_mode != "off")
                if use_aleatoric:
                    # Compute target reg_norm[k] = Σ_l ‖W_l^(k)‖_1 always
                    reg_norm_per_head = jnp.zeros(tq_blend.shape[0])
                    for leaf in jax.tree.leaves(t_p):
                        if leaf.ndim == 3 and leaf.shape[1] > 1:
                            reg_norm_per_head = reg_norm_per_head + \
                                jnp.sum(jnp.abs(leaf), axis=(1, 2))

                    # Pick effective coefficient
                    if adaptive_mode == "td_ema":
                        weight_reg_eff = adaptive_base * jax.nn.sigmoid(
                            (td_var - adaptive_thr) / jnp.maximum(adaptive_scale, 1e-6))
                    elif adaptive_mode == "posterior":
                        # Within-head TD residual variance (proper aleatoric estimate)
                        # — averaged across the K heads of the *online* critic
                        # via tq_blend itself isn't a TD residual yet, so we use
                        # the previous step's td_var (already EMA'd).
                        weight_reg_eff = adaptive_base * jax.nn.sigmoid(
                            (jnp.sqrt(jnp.maximum(td_var, 0.0)) - adaptive_thr)
                            / jnp.maximum(adaptive_scale, 1e-6))
                    else:
                        weight_reg_eff = jnp.float32(weight_reg)

                    tq_blend = tq_blend - weight_reg_eff * reg_norm_per_head[:, None]

                tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_blend

                # === Critic (only critic params differentiated) ===
                def critic_loss_fn(cp):
                    """Critic loss L_Q — bus-PyTorch matched (Apr 2026 fix).

                    Mirror of sac_v2_bus_ensemble.py:319-332:
                        loss = MSE(Q, y) + β_ood · mean(σ_ens(s,a))
                    where y already has the aleatoric pessimism shift
                    `- λ_ale · reg_norm(target_net)` applied above.

                    The aleatoric channel is therefore in the TARGET
                    (constant per backup), not as gradient-flowing l1 here.
                    Earlier attempt to put λ_ale ‖W‖_1 directly into critic
                    loss (which IS what paper Eq. (13) literally writes)
                    catastrophically over-regularized on deterministic
                    MuJoCo; the bus-style target shift is the empirically
                    successful version that the LSTM-RL bus experiments
                    validate.
                    """
                    cm = nnx.merge(gd_critic, cp)
                    pq = cm(obs, act)                 # [K, batch]
                    mse = jnp.mean((pq - tv_all) ** 2)
                    # β_ood · σ_ens(s,a): penalize ensemble disagreement on
                    # in-distribution batch (bus PyTorch line 331).
                    if beta_ood > 0:
                        ood_pen = jnp.mean(pq.std(axis=0))
                        mse = mse + beta_ood * ood_pen
                    return mse, pq

                (c_loss, pq), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True)(c_p)
                c_upd, new_c_os = c_opt.update(c_grads, c_os, c_p)
                new_c_p = optax.apply_updates(c_p, c_upd)

                # --- Adaptive λ_ale: update TD residual variance EMA ---
                # The leftover Bellman residual after a trained critic is
                # the aleatoric noise (epistemic gets fitted away). For
                # "td_ema" we average the per-head batch variance; for
                # "posterior" we take the within-head variance averaged
                # across K (Lakshminarayanan-style decomposition).
                if adaptive_mode == "td_ema":
                    td_var_batch = jnp.mean((tv_all - pq) ** 2)
                elif adaptive_mode == "posterior":
                    # Var per head over batch, mean across heads
                    td_var_batch = jnp.mean(jnp.var(tv_all - pq, axis=1))
                else:
                    td_var_batch = td_var
                new_td_var = adaptive_decay * td_var + (1 - adaptive_decay) * td_var_batch

                # --- Variant A: spectral-norm rescale critic kernels ---
                if use_specnorm:
                    new_c_p = apply_spectral_norm_to_critic_params(
                        new_c_p, c=specnorm_c, n_iters=1)

                # === Policy (LCB + anchor KL) ===
                cm = nnx.merge(gd_critic, new_c_p)

                def policy_loss_fn(pp):
                    """Policy loss L_π — paper Eq. (14).

                    Paper form:
                        L_π(θ) = E[α log π(a|s) - (Q̄(s,a) + β_lcb σ_ens(s,a))]
                    Here qm = Q̄, qs = σ_ens, and lcb = Q̄ + β_lcb σ_ens. The
                    anchor term below (L2 to a frozen best-policy snapshot)
                    is an additional regularizer not in Eq. (14).

                    Variant B (state_dep_beta) replaces fixed β_lcb with a
                    per-state β scaled by σ_ens / σ_ema.

                    Variant C (hash_count_bonus) augments σ_ens with a
                    1/sqrt(N(s,a)) exploration bonus before the LCB.
                    """
                    pm = nnx.merge(gd_policy, pp)
                    na, lp = pm.sample(obs, k2)
                    qv = cm(obs, na)
                    qm = qv.mean(axis=0)
                    qs = qv.std(axis=0)

                    # (#2) Q-std clipping: cap std as fraction of |mean|
                    if std_clip > 0:
                        qs = jnp.minimum(qs, std_clip * jnp.maximum(jnp.abs(qm), 1.0))

                    # --- Variant C: add hash-based count bonus to σ ---
                    if use_count_bonus:
                        buckets = hash_state_action(obs, na, W_hash, hash_dim_const)
                        bonus = count_bonus(cnts, buckets, alpha=count_alpha)
                        qs = qs + bonus

                    # --- LCB: state-dependent β (Variant B) or fixed ---
                    if use_state_beta:
                        beta_eff = state_dep_beta(qs, sig_ema, beta_lcb,
                                                  cap=state_beta_cap)
                        lcb = qm + beta_eff * qs
                    elif lcb_norm:
                        # (#1) Normalized LCB: penalty relative to Q magnitude
                        q_scale = jnp.maximum(jnp.abs(qm), 1.0)
                        lcb = qm + beta_lcb * qs / q_scale * jnp.abs(qm)
                    else:
                        lcb = qm + beta_lcb * qs

                    base_loss = (jnp.exp(la) * lp - lcb).mean()

                    # Policy anchoring: parameter-space L2 to a frozen
                    # snapshot of the best policy. Normalised by total
                    # parameter count so anchor_lambda has the same effect
                    # regardless of network width (previous implementation
                    # summed unnormalized squares — anchor strength scaled
                    # quadratically with model size).
                    anchor_sq = jax.tree.map(
                        lambda p, a: jnp.sum((p - a) ** 2), pp, anchor_params)
                    anchor_n = jax.tree.map(lambda p: float(p.size), pp)
                    total_sq = sum(jax.tree.leaves(anchor_sq))
                    total_n = sum(jax.tree.leaves(anchor_n))
                    anchor_loss = anchor_lambda * total_sq / jnp.maximum(total_n, 1.0)

                    # Aux: return σ_ens batch-mean for sigma_ema update
                    return base_loss + anchor_loss, (lp, qs.mean())

                (p_loss, (lp, qs_mean_step)), p_grads = jax.value_and_grad(
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

                # --- Variant B: update σ_ema baseline ---
                if use_state_beta:
                    new_sig_ema = (sigma_ema_decay * sig_ema
                                   + (1 - sigma_ema_decay) * qs_mean_step)
                else:
                    new_sig_ema = sig_ema  # carried unchanged

                # --- Variant C: update count table from this batch's (s, a) ---
                if use_count_bonus:
                    bks = hash_state_action(obs, act, W_hash, hash_dim_const)
                    new_cnts = update_counts(cnts, bks)
                else:
                    new_cnts = cnts

                new_carry = (new_c_p, new_t_p, new_p_p, new_la,
                             new_c_os, new_p_os, new_a_os,
                             new_sig_ema, new_cnts, new_td_var, key)
                metrics = (c_loss, p_loss, jnp.exp(new_la),
                           pq.mean(), pq.std(axis=0).mean(), lp.mean())
                return new_carry, metrics

            init = (critic_params, target_params, policy_params,
                    log_alpha, c_opt_state, p_opt_state, a_opt_state,
                    sigma_ema, counts, td_var_ema, rng_key)
            (final_c_p, final_t_p, final_p_p, final_la,
             final_c_os, final_p_os, final_a_os,
             final_sig_ema, final_cnts, final_td_var, _), metrics = jax.lax.scan(
                body_fn, init,
                (all_obs, all_act, all_rew, all_next_obs, all_done))
            # Return same shape as before plus the three ablation states
            return ((final_c_p, final_t_p, final_p_p, final_la,
                     final_c_os, final_p_os, final_a_os, _),
                    final_sig_ema, final_cnts, final_td_var, metrics)

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

        # Ablation state — fed to scan, returned and persisted across calls.
        sigma_ema_in = self._sigma_ema
        if self._hash_state is not None:
            counts_in = self._hash_state.counts
        else:
            # Dummy 1-element array to keep the JIT signature concrete.
            counts_in = jnp.zeros((1,), dtype=jnp.int32)
        # Adaptive λ_ale state (TD residual variance EMA)
        td_var_in = getattr(self, "_td_var_ema", jnp.array(0.0))

        final, sigma_ema_out, counts_out, td_var_out, metrics = self._scan_update(
            c_p, t_p, p_p, self.log_alpha,
            self.critic_opt_state, self.policy_opt_state, self.alpha_opt_state,
            sigma_ema_in, counts_in, td_var_in,
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

        # Persist ablation state across step calls.
        self._sigma_ema = sigma_ema_out
        self._td_var_ema = td_var_out
        if self._hash_state is not None:
            self._hash_state = HashCounterState(
                W_hash=self._hash_state.W_hash,
                counts=counts_out,
                hash_dim=self._hash_state.hash_dim,
            )

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
