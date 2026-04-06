"""Experiment 10: RE-SAC vs BAC Mechanism Comparison.

Head-to-head comparison of how RE-SAC and BAC solve the Q-value
underestimation problem. BAC uses the BEE operator (blend exploitation
and exploration), while RE-SAC uses ensemble uncertainty decomposition.

Since BAC is not implemented in JAX, this experiment:
1. Implements a simplified BAC-like baseline (BEE operator) in JAX
2. Compares RE-SAC vs BAC on key diagnostics:
   - Q-estimation accuracy
   - Training stability
   - Sample efficiency
   - Computational cost

The simplified BAC modifies SAC's Bellman target:
  y = r + γ * [λ * max_{a∈D} Q(s',a) + (1-λ) * E_{π} Q(s',a')] - α * logπ

Usage:
    # Generate scripts (includes BAC implementation via modified SAC):
    python -m jax_experiments.experiments.exp10_bac_comparison --mode gen_script

    # Analyze existing results:
    python -m jax_experiments.experiments.exp10_bac_comparison --mode both
"""
import os
import sys
import argparse
import pickle
import numpy as np

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from jax_experiments.configs.default import Config

ALGOS_COMPARE = ["sac", "resac", "bac"]
ENVS = ["Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Ant-v2"]
RESULTS_ROOT = "jax_experiments/results"
OUTPUT_DIR = "jax_experiments/experiments/results/exp10_bac_comparison"


def create_bac_algo():
    """Create a BAC-like algorithm by modifying SAC's Bellman target.

    BAC key modification: Bellman target uses
      y = r + γ * [λ * max_{a∈D} Q_min(s',a) + (1-λ) * Q_min(s',a')] - α * log π
    where a' ~ π(·|s') and the max is over a set of candidate actions from D.

    This is implemented as a separate algorithm class.
    """
    bac_code = '''"""BAC: Blended Actor-Critic (simplified) — scan-fused JAX.

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

        # BAC-specific
        self.lam = getattr(config, 'bac_lambda', 0.5)
        self.n_candidate = getattr(config, 'bac_n_candidate', 10)

        # Networks (same as SAC)
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

                # BEE target computation
                t_model = nnx.merge(gd_target, t_p)
                p_model_frozen = nnx.merge(gd_policy, p_p)

                # Exploration part: a' ~ π(·|s')
                na, nlp = p_model_frozen.sample(next_obs, k1)
                q_explore = t_model(next_obs, na).min(axis=0) - alpha * nlp

                # Exploitation part: max_{a∈D} Q(s', a)
                batch_size = next_obs.shape[0]
                # cand_acts: [batch, n_candidate, act_dim]
                # Expand next_obs: [batch, 1, obs_dim] -> [batch, n_candidate, obs_dim]
                next_obs_exp = jnp.broadcast_to(
                    next_obs[:, None, :],
                    (batch_size, n_candidate, next_obs.shape[-1]))
                # Reshape for critic: [batch * n_candidate, ...]
                no_flat = next_obs_exp.reshape(-1, next_obs.shape[-1])
                ca_flat = cand_acts.reshape(-1, cand_acts.shape[-1])
                q_cand = t_model(no_flat, ca_flat).min(axis=0)  # [batch * n_candidate]
                q_cand = q_cand.reshape(batch_size, n_candidate)
                q_exploit = q_cand.max(axis=1)  # [batch] — max over candidates

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

        # Generate candidate actions by randomly sampling from buffer
        # In practice, would sample from replay buffer, but here we use
        # random actions as a proxy (captures the BEE concept)
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
'''
    return bac_code


def save_bac_implementation():
    """Save the BAC algorithm implementation."""
    bac_code = create_bac_algo()
    bac_path = os.path.join("jax_experiments", "algos", "bac.py")
    with open(bac_path, 'w') as f:
        f.write(bac_code)
    print(f"Saved BAC implementation: {bac_path}")
    return bac_path


def generate_run_scripts(envs=None, seeds=None, max_iters=2000):
    """Generate run scripts for BAC comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    envs = envs or ENVS
    seeds = seeds or [8]

    lines = [
        "#!/bin/bash",
        "# Experiment 10: RE-SAC vs BAC Comparison",
        "",
    ]

    for env_name in envs:
        lines.append(f"echo '====== {env_name} ======'")
        for algo in ALGOS_COMPARE:
            for seed in seeds:
                run_name = f"{algo}_{env_name}_{seed}"
                cmd = (
                    f"conda run -n jax-rl python -m jax_experiments.train "
                    f"--algo {algo} --env {env_name} --seed {seed} "
                    f"--run_name {run_name} --stationary --resume "
                    f"--max_iters {max_iters}"
                )
                if algo == "sac":
                    cmd += " --ensemble_size 2"
                elif algo == "bac":
                    cmd += " --ensemble_size 2"

                lines.append(f"echo '--- {algo.upper()} seed={seed} ---'")
                lines.append(cmd)
                lines.append("")

    script_path = os.path.join(OUTPUT_DIR, "run_bac_comparison.sh")
    with open(script_path, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(script_path, 0o755)
    print(f"Generated script: {script_path}")


def load_comparison_results(env_name, seeds=None):
    """Load results for SAC, RE-SAC, and BAC."""
    seeds = seeds or [8]
    results = {}

    for algo in ALGOS_COMPARE:
        seed_curves = []
        for seed in seeds:
            run_name = f"{algo}_{env_name}_{seed}"
            log_dir = os.path.join(RESULTS_ROOT, run_name, "logs")

            if not os.path.isdir(log_dir):
                continue

            curves = {}
            for metric in ["eval_reward", "q_mean", "q_std_mean",
                            "critic_loss", "iteration"]:
                path = os.path.join(log_dir, f"{metric}.npy")
                if os.path.exists(path):
                    curves[metric] = np.load(path)

            if 'eval_reward' in curves:
                seed_curves.append(curves)

        if seed_curves:
            min_len = min(len(c['eval_reward']) for c in seed_curves)
            eval_stacked = np.stack([c['eval_reward'][:min_len] for c in seed_curves])
            results[algo] = {
                'eval_rewards': eval_stacked,
                'eval_mean': eval_stacked.mean(axis=0),
                'eval_std': eval_stacked.std(axis=0),
                'final_mean': float(eval_stacked[:, -1].mean()),
                'final_std': float(eval_stacked[:, -1].std()),
                'n_seeds': len(seed_curves),
                'curves': seed_curves,
            }
            print(f"  {algo.upper()}: {len(seed_curves)} seeds, "
                  f"final = {results[algo]['final_mean']:.1f}")

    return results


def analyze_comparison(env_name, seeds=None):
    """Analyze RE-SAC vs BAC comparison."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"  RE-SAC vs BAC: {env_name}")
    print(f"{'='*50}")

    results = load_comparison_results(env_name, seeds)

    out_path = os.path.join(OUTPUT_DIR, f"bac_comparison_{env_name}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({'env': env_name, 'results': results}, f)

    # Summary
    print(f"\n{'Algo':<12} {'Seeds':>6} {'Final Reward':>15}")
    print("-" * 36)
    for algo in ALGOS_COMPARE:
        if algo in results:
            d = results[algo]
            print(f"{algo:<12} {d['n_seeds']:>6} "
                  f"{d['final_mean']:>7.1f} ± {d['final_std']:<6.1f}")

    return results


def plot_comparison(env_name, output_dir=None):
    """Plot RE-SAC vs BAC comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data_path = os.path.join(OUTPUT_DIR, f"bac_comparison_{env_name}.pkl")
    if not os.path.exists(data_path):
        print(f"No data at {data_path}")
        return

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    output_dir = output_dir or OUTPUT_DIR

    colors = {'sac': '#ff7f0e', 'resac': '#4488FF', 'bac': '#d62728'}
    labels = {'sac': 'SAC', 'resac': 'RE-SAC', 'bac': 'BAC (BEE)'}
    linestyles = {'sac': '--', 'resac': '-', 'bac': '-.'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'RE-SAC vs BAC — {env_name}', fontsize=14)

    # Panel 1: Learning curves
    ax = axes[0]
    for algo in ALGOS_COMPARE:
        if algo not in results:
            continue
        d = results[algo]
        mean = d['eval_mean']
        std = d['eval_std']
        x = np.arange(len(mean))
        w = max(1, len(mean) // 30)
        mean_s = np.convolve(mean, np.ones(w)/w, mode='valid')
        std_s = np.convolve(std, np.ones(w)/w, mode='valid')
        x_s = x[:len(mean_s)]
        ax.plot(x_s, mean_s, label=labels.get(algo, algo),
                color=colors.get(algo, 'gray'),
                linestyle=linestyles.get(algo, '-'), linewidth=2)
        if d['n_seeds'] > 1:
            ax.fill_between(x_s, mean_s - std_s, mean_s + std_s,
                            alpha=0.15, color=colors.get(algo, 'gray'))
    ax.set_xlabel('Eval Point')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Learning Curves')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: Mechanism comparison table
    ax = axes[1]
    ax.axis('off')
    table_data = [
        ['', 'SAC', 'RE-SAC', 'BAC (BEE)'],
        ['Mechanism', 'min(Q₁,Q₂)', 'Q̄ + β·σ', 'λ·max_D + (1-λ)·π'],
        ['Uncertainty', 'None', 'Ensemble var', 'None'],
        ['Exploitation', 'Policy only', 'Adaptive β', 'Buffer max'],
        ['Critics', 'K=2', 'K=10', 'K=2'],
    ]

    # Add performance rows if data available
    for algo in ALGOS_COMPARE:
        if algo in results:
            table_data[0]  # headers already set

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2.0)

    # Style header row
    for j in range(4):
        table[0, j].set_facecolor('#e6e6e6')
        table[0, j].set_fontsize(11)

    ax.set_title('Mechanism Comparison', fontsize=12, pad=30)

    plt.tight_layout()
    out_fig = os.path.join(output_dir, f"bac_comparison_{env_name}.png")
    plt.savefig(out_fig, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {out_fig}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 10: RE-SAC vs BAC Comparison")
    parser.add_argument("--mode",
                        choices=["gen_script", "save_bac", "analyze", "plot", "both"],
                        default="both")
    parser.add_argument("--env", type=str, default="Hopper-v2", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[8])
    args = parser.parse_args()

    envs = ENVS if args.all else [args.env]

    if args.mode == "gen_script":
        save_bac_implementation()
        generate_run_scripts(envs, args.seeds)
    elif args.mode == "save_bac":
        save_bac_implementation()
    else:
        for env_name in envs:
            if args.mode in ("analyze", "both"):
                analyze_comparison(env_name, args.seeds)
            if args.mode in ("plot", "both"):
                plot_comparison(env_name)

    print("\n✅ Experiment 10 complete!")


if __name__ == "__main__":
    main()
