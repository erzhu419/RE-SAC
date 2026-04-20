# RE-SAC Paper Experiment Plan

## P0: Multi-seed runs (3-5 seeds)

Best config per env, seeds = [8, 42, 123, 256, 512]

| Env | Config | Key params |
|-----|--------|-----------|
| Hopper | v5 | ratio=1.0, anchor=0.1, beta_end=-1.0 |
| Walker2d | v5b | ratio=1.0, anchor=0.01, beta_end=-0.5 |
| HalfCheetah | v5 | ratio=1.0, anchor=0.1, beta_end=-1.0 |
| Ant | v6b | ratio=0.5, anchor=0.01, beta_end=-1.0 |

Also need multi-seed for baselines (SAC, DSAC, TD3) for fair comparison.
Total: (4 RE-SAC + 3 baselines) * 4 envs * 4 new seeds = 112 runs
Can parallelize: 4 envs at a time, ~5h each = ~140h GPU total

## P0: Formal ablation table

Base = v5 best config. Remove one component at a time, run on all 4 envs, seed=8.

| Ablation | What changes |
|----------|-------------|
| Full (v5/v6b) | baseline |
| - independent target | ratio=0.0 (pure min, like v1) |
| - EMA policy | ema_tau=0 (eval with live policy) |
| - anchor | anchor_lambda=0 |
| - adaptive beta | fixed beta=-2.0 |
| - performance gating | (disable eval-triggered beta tightening) |

Total: 5 ablations * 4 envs = 20 runs

## P1: REDQ baseline

Implement REDQ (Chen et al., 2021):
- Ensemble of N=10 Q-networks (same as RE-SAC)
- Random subset of M=2 for min target (key difference)
- UTD ratio = 20 (high update-to-data ratio)
- No LCB in policy loss, just mean of random subset

File: jax_experiments/algos/redq.py
Reference: "Randomized Ensembled Double Q-Learning: Learning Fast Without a Model"

Total: 4 envs * 5 seeds = 20 runs

## P1: Computation cost analysis

Collect wall-clock time from existing logs:
- Per-iteration time (already logged as "X.Xs/iter")
- Total training time (already logged)
- FLOPs estimate: ensemble_size * critic_forward_cost

Produce table:
| Algorithm | K | Time/iter (s) | Total (h) | Final Return |
|-----------|---|---------------|-----------|-------------|

Also: Return-per-wallclock curve (eval return vs clock time, not env steps)

No new runs needed - extract from existing logs.

## P2: Non-stationary environment experiments

Use existing BraxNonstationaryEnv with varying_params:
- gravity changes every 20k steps (40 tasks, log_scale_limit=3.0)
- Run best RE-SAC config + baselines

Show:
1. Q-std spike at regime change → ensemble detects non-stationarity
2. Performance recovery after change
3. Comparison with SAC/DSAC which lack change detection

Total: 7 algos * 4 envs * 3 seeds = 84 runs (but can start with 1 seed)

## P2: Hyperparameter sensitivity

Sweep one param at a time, other params at default, seed=8, on HalfCheetah + Ant:

1. anchor_lambda: [0, 0.001, 0.01, 0.1, 0.5, 1.0]
2. beta_end: [-2.0, -1.5, -1.0, -0.5, 0.0]
3. independent_ratio: [0.0, 0.25, 0.5, 0.75, 1.0]
4. ensemble_size: [2, 5, 10, 20]
5. ema_tau: [0, 0.001, 0.005, 0.01, 0.05]

Total: (6+5+5+4+5) * 2 envs = 50 runs

## P3: Additional baselines

### TQC (Truncated Quantile Critics)
- Distributional + ensemble hybrid
- N=5 critics, each outputs 25 quantiles, drop top 2 atoms
- Reference: Kuznetsov et al., 2020

### SAC-N / SUNRISE
- Simple ensemble SAC with N critics, UCB exploration bonus
- Reference: Lee et al., 2021

### CQL (if claiming OOD robustness)
- Conservative Q-Learning, adds penalty for OOD actions in Q-loss
- Reference: Kumar et al., 2020
- May be more relevant for offline RL angle

Implementation priority: TQC > REDQ (P1) > SAC-N > CQL

## Execution Order

1. **P1 compute cost** - no GPU needed, extract from logs NOW
2. **P0 ablation** (20 runs, seed=8) - first batch when GPU free
3. **P0 multi-seed** RE-SAC best configs (16 runs) - second batch
4. **P0 multi-seed** baselines (48 runs) - can reuse seeds 42/123/256/512
5. **P1 REDQ** implement + run (20 runs)
6. **P2 sensitivity** (50 runs)
7. **P2 non-stationary** (12-84 runs)
8. **P3 TQC/SAC-N** if time permits

## Estimated GPU Time

| Task | Runs | Hours/run | Total GPU-h |
|------|------|-----------|-------------|
| P0 ablation | 20 | 5 | 100 |
| P0 multi-seed RE-SAC | 16 | 5 | 80 |
| P0 multi-seed baselines | 48 | 5-10 | 360 |
| P1 REDQ | 20 | 5 | 100 |
| P2 sensitivity | 50 | 5 | 250 |
| P2 non-stationary | 84 | 5 | 420 |
| P3 TQC/SAC-N | 40 | 5 | 200 |
| **Total** | **278** | | **~1510** |

With 4 parallel jobs: ~378h = ~16 days continuous.
Can reduce by: fewer seeds (3), fewer sensitivity points, 1 seed for non-stationary.
Minimum viable: P0 + P1 = ~640h = ~7 days with 4 parallel.
