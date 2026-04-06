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
