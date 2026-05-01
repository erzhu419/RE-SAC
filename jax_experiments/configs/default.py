"""Hyperparameter configuration dataclass for JAX-based RE-SAC experiments."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Environment
    env_name: str = "Hopper-v2"
    brax_backend: str = "spring"  # 'spring' (fast) or 'generalized' (accurate)
    varying_params: List[str] = field(default_factory=lambda: ["gravity"])
    log_scale_limit: float = 3.0
    task_num: int = 40
    test_task_num: int = 40
    changing_period: int = 20000  # task switches every THIS many env steps
    changing_interval: int = 4000  # align with samples_per_iter

    # Algorithm
    algo: str = "resac"
    seed: int = 8
    gamma: float = 0.99
    tau: float = 0.005  # soft target update
    alpha: float = 0.2  # SAC entropy weight (initial)
    auto_alpha: bool = True
    lr: float = 3e-4
    batch_size: int = 256
    replay_size: int = 1_000_000
    hidden_dim: int = 256
    max_iters: int = 2000
    samples_per_iter: int = 4000  # env steps collected per iteration
    updates_per_iter: int = 250  # gradient steps per iteration
    start_train_steps: int = 10_000  # random exploration before training
    max_episode_steps: int = 1000

    # Ensemble (RE-SAC)
    ensemble_size: int = 10
    beta: float = -2.0  # LCB coefficient for policy
    beta_ood: float = 0.01  # OOD regularization weight
    beta_bc: float = 0.001  # behavior cloning weight
    weight_reg: float = 0.01  # critic regularization weight

    # Adaptive beta_lcb: anneal from beta_start → beta_end over training
    adaptive_beta: bool = False
    beta_start: float = -2.0   # initial (more pessimistic)
    beta_end: float = -1.0     # final (more exploitative, was -0.5)
    beta_warmup: float = 0.2   # fraction of max_iters to hold beta_start

    # EMA policy (Polyak-averaged actor for stable evaluation)
    ema_tau: float = 0.005     # EMA smoothing coefficient (same as target critic)

    # Policy anchoring: penalize deviation from best-so-far policy
    anchor_lambda: float = 0.1  # anchor KL penalty weight (0 = disabled)

    # Ant-stability fixes (high-dim env robustness)
    lcb_normalize: bool = False   # normalize LCB penalty by Q magnitude (CV-based)
    q_std_clip: float = 0.0      # clip Q-std as fraction of |Q_mean| (0 = disabled)
    independent_ratio: float = 1.0  # blend: 1.0=all independent, 0.0=all min target

    # Algorithm ablations (paper §6.1.6)
    # Variant A: hard Lipschitz control via spectral normalization on each critic
    # layer's [K, in, out] kernel (per-head). Replaces the never-actually-applied
    # λ_ale ‖W‖_1 soft penalty implied by paper Eq. (13).
    use_spectral_norm: bool = False
    spectral_norm_value: float = 1.0  # constrain σ_max(W_l) ≤ this per layer

    # Variant B: state-dependent β_lcb. Replaces fixed β with
    # β_eff(s,a) = -|β_0| · clip(σ_ens(s,a) / σ_ema, max=ratio_cap)
    # so pessimism shrinks in familiar (low-σ) states.
    state_dep_beta: bool = False
    state_dep_beta_cap: float = 3.0  # cap on σ/σ_ema ratio
    state_dep_beta_ema: float = 0.99  # EMA decay for σ_ema baseline

    # Variant C: hash-based count bonus added to σ_ens.
    # σ_epi(s,a) = σ_ens(s,a) + count_alpha / sqrt(1 + N(hash(s,a)))
    # Provides a count-based epistemic floor that doesn't suffer from
    # ensemble collapse (heads agreeing in OOD).
    hash_count_bonus: bool = False
    hash_count_alpha: float = 0.5
    hash_dim: int = 14  # 2^14 = 16384 buckets

    # Tag for the ablation variant being run (for logging only)
    variant: str = "B0"

    # Aleatoric noise injection (paper §6.1.X IPM validation on MuJoCo).
    # Adds Gaussian noise to observation/reward during training rollout
    # only — eval is always clean. Use to test whether weight_reg helps in
    # noisy MuJoCo (matching the bus environment's stochasticity).
    obs_noise_std: float = 0.0
    reward_noise_std: float = 0.0

    # ── Adaptive λ_ale (paper §4.5, three modes) ────────────────────────
    # "off"      : use config.weight_reg directly (current behavior)
    # "td_ema"   : λ_ale_eff = base · sigmoid((TD-residual-Var EMA - thr) / scale)
    #              — aleatoric noise estimated from leftover Bellman residual
    # "probe"    : measure reward std during start_train_steps random phase,
    #              fix λ_ale = base · sigmoid((reward_std - thr) / scale) once
    # "posterior": λ_ale_eff = base · sqrt(within-head TD var avg)
    #              — explicit aleatoric vs epistemic decomposition à la BAPR-HRO
    adaptive_reg_mode: str = "off"
    adaptive_reg_base: float = 0.01      # max λ_ale value when fully on
    adaptive_reg_threshold: float = 1.0  # below this aleatoric estimate, λ_ale ≈ 0
    adaptive_reg_scale: float = 1.0      # sigmoid sharpness
    adaptive_reg_decay: float = 0.99     # EMA decay for td_ema mode

    # REDQ
    redq_m: int = 2  # subset size for random min target

    # SAC-N
    sacn_beta_ucb: float = 1.0  # UCB exploration coefficient (positive = optimistic)

    # TQC
    tqc_n_critics: int = 5       # number of quantile critics
    tqc_n_quantiles: int = 25    # quantiles per critic
    tqc_drop: int = 2            # top quantiles to drop per critic (truncation)

    # DSAC (IQN)
    num_quantiles: int = 8

    # TD3
    td3_target_noise: float = 0.2
    td3_noise_clip: float = 0.5
    td3_policy_delay: int = 2

    # BAC (BEE operator)
    bac_lambda: float = 0.5   # blend factor: 0=pure exploration, 1=pure exploitation
    bac_n_candidate: int = 10  # number of candidate actions for max_D

    # Logging
    save_root: str = "jax_experiments/results"
    run_name: str = ""
    log_interval: int = 5  # eval every N iterations
    save_interval: int = 50  # save model every N iterations
    eval_episodes: int = 5

    # Supported environments
    ENVS: List[str] = field(default_factory=lambda: [
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2"
    ])
