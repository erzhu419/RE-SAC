# RE-SAC JAX 代码复盘与修改建议

本文只复盘当前 JAX 实现：

- 代码目录：`/home/erzhu419/mine_code/RE-SAC/jax_experiments`
- 主要实现：`jax_experiments/algos/resac.py`
- 旧的 `reference/` PyTorch 代码不作为本次结论依据。

## 总结

**当前 RE-SAC 的效果不好，不是单纯调参问题。** JAX 版实现有几处实质性错位：配置里声明的 `weight_reg / beta_ood / beta_bc` 没有真正进入对应 loss；论文/证明里强调的 frozen Bellman penalty 没有进入 critic target；`ema_tau=0` 的消融语义是错的；非平稳评估只测 `test_tasks[0]`。这些问题会直接影响实验结论和论文叙事。

更准确地说，现在的主路径更像：

```text
SAC + ensemble critic + independent/min blended target
    + actor-side LCB
    + EMA policy eval
    + best-policy parameter L2 anchor
    + performance-aware beta schedule
```

它不是论文中声称的完整 `epistemic + aleatoric` 解耦鲁棒 Bellman operator。

## 关键问题

### 1. `weight_reg / beta_ood / beta_bc` 基本是空参数

配置和 CLI 都有这些参数：

- `Config.weight_reg / beta_ood / beta_bc`：`jax_experiments/configs/default.py:38-40`
- CLI override：`jax_experiments/train.py:367-374`, `jax_experiments/train.py:447-452`
- 训练启动日志也打印：`jax_experiments/train.py:174-177`

但 RE-SAC 的 critic loss 只返回 MSE：

```python
return jnp.mean((pq - tv_all) ** 2), pq
```

位置：`jax_experiments/algos/resac.py:137-150`

注释说 `weight_reg` 通过 optax weight decay 实现，但基础优化器是普通 Adam：

```python
self.policy_opt = optax.adam(config.lr)
self.critic_opt = optax.adam(config.lr)
self.alpha_opt = optax.adam(config.lr)
```

位置：`jax_experiments/algos/sac_base.py:39-42`

所以：

- `weight_reg` 没有变成 AdamW/weight decay；
- `beta_ood` 没有作为 critic OOD regularization；
- `beta_bc` 没有 behavior cloning loss；
- 当前 aleatoric regularization claim 站不住，除非启用额外 ablation 的 spectral norm variant，但默认主实验没有启用。

建议：要么删掉这些字段和论文 claim，要么真正接入 loss/optimizer。论文主线如果保留 `lambda_ale ||W||`，至少需要把 critic optimizer 改成 AdamW 或显式加正则项。

### 2. Bellman target 和论文/Lean 证明不一致

JAX RE-SAC target 是：

```python
tq_all = tm(next_obs, na)
tq_min = tq_all.min(axis=0)
tq_blend = ind_ratio * tq_all + (1 - ind_ratio) * tq_min[None]
tq_blend = tq_blend - alpha * nlp
tv_all = rew.squeeze(-1) + gamma * (1 - done.squeeze(-1)) * tq_blend
```

位置：`jax_experiments/algos/resac.py:128-134`

这里没有把 epistemic penalty 或 aleatoric penalty 放进 frozen Bellman target。epistemic 只在 actor loss 里做 LCB：

```python
qm = qv.mean(axis=0)
qs = qv.std(axis=0)
lcb = qm + beta_lcb * qs
base_loss = (jnp.exp(la) * lp - lcb).mean()
```

位置：`jax_experiments/algos/resac.py:180-208`

这会导致一个论文风险：Lean 证明若证明的是带 frozen penalty 的 Bellman operator contraction，实验代码却没有训练同一个 operator。论文必须二选一：

1. 把 frozen penalty 放进 critic target；
2. 或者把论文叙事改成 actor-side risk-sensitive policy improvement，不再声称 critic Bellman operator 已实现该 penalty。

### 3. `ema_tau=0` 消融是错的

`run_ablation.sh` 写着：

```bash
# ema_tau=0 means EMA = live policy (no smoothing)
```

位置：`jax_experiments/run_ablation.sh:91-94`

但实现里：

```python
new_ema = e * (1 - self.ema_tau) + c * self.ema_tau
```

位置：`jax_experiments/algos/resac.py:325-331`

如果 `ema_tau=0`，EMA policy 永远停在初始化状态。更严重的是评估默认优先用 EMA policy：

```python
if hasattr(agent, 'ema_policy'):
    policy_params = nnx.state(agent.ema_policy, nnx.Param)
```

位置：`jax_experiments/train.py:87-95`

所以 `abl_noema` 不是“用 live policy 评估”，而是“用初始化附近的 EMA policy 评估”。这会污染 EMA 消融结果，也能解释部分 `abl_noema` 极差结果。

建议修法：

```python
if hasattr(agent, "ema_policy") and config.ema_tau > 0:
    policy_params = nnx.state(agent.ema_policy, nnx.Param)
else:
    policy_params = nnx.state(agent.policy, nnx.Param)
```

或者在 `RESAC.__init__` 中当 `ema_tau <= 0` 时不创建 `ema_policy`。

### 4. Policy anchor 不是 KL，而且尺度可能过大

注释说 anchoring 是 KL penalty，但实现是参数 L2：

```python
anchor_dist = jax.tree.map(lambda p, a: jnp.sum((p - a) ** 2), pp, anchor_params)
anchor_loss = anchor_lambda * sum(jax.tree.leaves(anchor_dist))
```

位置：`jax_experiments/algos/resac.py:210-214`

问题：

- 这不是 action distribution KL；
- 没有按参数量归一化；
- `anchor_lambda=0.1` 对整个参数树求和，尺度可能非常大；
- anchor 的正负效果会强烈依赖网络大小和环境。

建议：如果论文写 KL，就用 batch 上的 policy distribution KL；如果保留参数 L2，需要改名为 parameter anchor，并除以参数总数。

### 5. Adaptive beta 的 performance gate 太脆

实现逻辑：

```python
if earlier > 0 and later < earlier * 0.9:
    beta_linear = (beta_linear + cfg.beta_start) / 2.0
```

位置：`jax_experiments/algos/resac.py:294-309`

问题：

- 只有 `earlier > 0` 才启用，负回报或低回报阶段不工作；
- eval 每 5 iter、每次 5 episodes，噪声很大；
- beta gate 可能在后期被少量 eval 波动反复拉回。

建议：使用平滑后的 eval trend、bootstrap CI 或固定 warmup schedule；至少记录触发次数，避免 beta 变化不可解释。

### 6. 非平稳评估只测第一个 test task

`evaluate()` 收到 `tasks` 后只设置：

```python
env.set_task(tasks[0])
```

位置：`jax_experiments/train.py:100-105`

`eval_rollout()` 本身也明确不切换 task：

```python
"""Deterministic eval rollout — does NOT update step counter or switch task."""
```

位置：`jax_experiments/envs/brax_env.py:392-415`

所以非平稳实验的 eval 不是对 `test_task_num=40` 做平均，而是固定测 `test_tasks[0]`。这会让 nonstationary robustness 的结论非常依赖第一个测试任务。

建议：非平稳评估至少循环多个 test tasks，记录：

- mean over tasks；
- worst quartile；
- per-task std；
- default task 单独作为 stationary sanity check。

### 7. Rollout 是每个 iteration 从 fresh reset 开始，不是连续 stream

训练采样的 scan rollout 每次都会：

```python
init_state = self._reset_fn(self._current_sys, init_key)
```

位置：`jax_experiments/envs/brax_env.py:366-372`

虽然最后保存了 `self._state = final_state`，下一次 rollout 并不用它作为初始状态。task switch 也在 rollout 结束后才影响下一段：

```python
self._step_counter += n_steps
self._check_switch()
self._state = final_state
```

位置：`jax_experiments/envs/brax_env.py:386-390`

这未必是 bug，但它不是连续非平稳数据流，而是每个 iteration 一个 fresh-start chunk。论文描述如果写 continuous nonstationarity，需要改表述或改实现。

### 8. Checkpoint 没保存 RE-SAC 私有状态

checkpoint 保存了 policy/critic/target/log_alpha/optimizer/replay/logger：

位置：`jax_experiments/common/checkpoint.py:42-68`, `jax_experiments/common/checkpoint.py:116-142`

但没有保存：

- `ema_policy`
- `_anchor_params`
- `_best_eval`
- `_eval_history`
- `_current_beta`
- `_sigma_ema`
- `_hash_state.counts`

如果用 `--resume`，这些状态会重置，恢复后的训练轨迹和完整训练不等价。

建议：为 `algo == "resac"` 单独保存/恢复这些字段；或者论文实验禁止 resume 并在日志中声明。

### 9. 大并发 ablation 脚本不可作为完成证据

`run_ablation.sh` 一次启动 20 个 GPU job，且设置：

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
```

位置：`jax_experiments/run_ablation.sh:36-38`, `jax_experiments/run_ablation.sh:126-149`

`jax_experiments/logs/ablation_full.log` 显示：

```text
20 experiment(s) failed.
```

后续一些 result dir 可能来自 serial rerun 或手工重跑。结论表应以每个 run 的 `eval_reward.npy` 和完整 log 为准，不要把 `ablation_full.log` 当作成功运行记录。

## 当前结果读法

下面是本地 `eval_reward.npy` 的最后 10 次 eval 平均值。多数是 seed=8，不能当多 seed 论文结论。

### Stationary

| Env | SAC | DSAC | REDQ | SAC-N | RE-SAC v5 | RE-SAC B0/abl |
|---|---:|---:|---:|---:|---:|---:|
| Hopper | 624.7 | 903.7 | - | - | 666.6 | 468.6 |
| Walker2d | 308.1 | 488.6 | - | - | 311.5 | 298.0 |
| HalfCheetah | 13339.5 | 12785.5 | 25167.5 | 22525.4 | 22377.6 | 25043.6 |
| Ant | 10256.5 | 20931.3 | 12874.7 | - | 1641.5 | 24665.1 |

读法：

- Hopper/Walker2d 上 RE-SAC 没明显赢 DSAC。
- HalfCheetah 上 RE-SAC v5 接近 SAC-N，但不如 REDQ 和 B0。
- Ant 上 RE-SAC v5 崩了，但 B0 很强，说明结果高度依赖 EMA/anchor/beta/ratio 组合，不是一个稳健主算法。

### Nonstationary

| Env | SAC | DSAC | RE-SAC | RE-SAC B0 |
|---|---:|---:|---:|---:|
| Hopper | 598.7 | 668.5 | 659.5 | 541.8 |
| Walker2d | 356.6 | 385.2 | 402.2 | 306.2 |
| HalfCheetah | 19908.7 | 17497.2 | 12126.9 | 13865.1 |
| Ant | 5352.5 | 2651.4 | 20710.4 | 20234.9 |

读法：

- Ant 非平稳结果好，但 evaluation 只测 `test_tasks[0]`，需要重评。
- HalfCheetah 非平稳 RE-SAC 明显差于 SAC/DSAC。
- Hopper/Walker2d 的差距不大，难以支撑强 claim。

## 优先修改路线

### P0：先修会污染结论的 bug

1. 修 `ema_tau=0`：评估 live policy，或禁用 `ema_policy`。
2. 修 checkpoint：保存/恢复 EMA、anchor、beta history、sigma/count state。
3. 修 nonstationary eval：对多个 test tasks 求平均，而不是只测 `tasks[0]`。
4. 清理空参数：`beta_ood / beta_bc / weight_reg` 要么接入实现，要么从实验和论文主 claim 移除。

### P1：让算法和论文对齐

如果论文保留 robust Bellman operator：

```python
tq_all = target_critic(next_obs, next_action)
q_mean = tq_all.mean(axis=0)
q_std = tq_all.std(axis=0)
target_q = q_mean + beta_target * q_std - lambda_ale * kappa
target = reward + gamma * (1 - done) * (target_q - alpha * next_logp)
```

注意：`kappa` 必须是 frozen/local penalty，不能是随便的全局常数。

如果论文改成 actor-side LCB：

- 明确 critic target 仍是 SAC/ensemble target；
- 理论证明改成 policy improvement 或 risk-sensitive objective；
- 不再把 Lean contraction 证明包装成当前训练代码的证明。

### P2：提高 ensemble uncertainty 的可信度

当前 `independent_ratio=1.0` 时每个 head 用自己的 target，但还是同 replay batch、同 policy、同 optimizer schedule。ensemble diversity 未必可靠。

建议试：

- bootstrap masks；
- randomized prior functions；
- REDQ-style random subset min target；
- 记录 head correlation、`q_std` 与 OOD/task shift 的相关性；
- 对 actor LCB 的 `q_std` 做合理 clipping/normalization，但要写清楚。

### P3：重新跑最小可信实验

建议先不要铺太大表，先跑一个小而干净的矩阵：

| 组别 | 内容 |
|---|---|
| SAC | 标准 twin critic |
| REDQ | 强 ensemble baseline |
| RE-SAC actor-LCB | 只保留 actor LCB，无 EMA/anchor |
| RE-SAC + EMA | 修正后的 EMA |
| RE-SAC + anchor | normalized parameter L2 或 true KL |
| RE-SAC full | 最终组合 |

每个环境至少 5 seeds。报告 last-10 eval mean/std、best、AUC，并单独报告 nonstationary multi-task mean/worst-quartile。

## 建议的最小代码改动

### EMA 修复

```python
def evaluate(agent, env, config, tasks=None, n_episodes=10):
    use_ema = (
        hasattr(agent, "ema_policy")
        and getattr(config, "ema_tau", 0.0) > 0
    )
    policy = agent.ema_policy if use_ema else agent.policy
    policy_params = nnx.state(policy, nnx.Param)
```

### Anchor 归一化

```python
sq = jax.tree.map(lambda p, a: jnp.sum((p - a) ** 2), pp, anchor_params)
num = jax.tree.map(lambda p: p.size, pp)
anchor_loss = anchor_lambda * sum(jax.tree.leaves(sq)) / sum(jax.tree.leaves(num))
```

### AdamW 接入 `weight_reg`

```python
self.critic_opt = optax.adamw(config.lr, weight_decay=config.weight_reg)
```

如果论文坚持 L1，就不要用 AdamW，要显式写：

```python
l1 = sum(jnp.sum(jnp.abs(x)) for x in jax.tree.leaves(cp))
return mse + lambda_ale * l1, pq
```

### 多 task eval

```python
scores = []
for task in tasks:
    env.set_task(task)
    mean, _ = eval_one_task(...)
    scores.append(mean)
return float(np.mean(scores)), float(np.std(scores))
```

## 验证记录

本次只做了静态代码复盘和本地结果文件读取，没有改 JAX 实验代码。

- 语法检查：对 `resac.py / sac_base.py / train.py / brax_env.py / checkpoint.py / default.py` 做了 Python `compile()` 检查，结果 OK。
- 结果读取：从 `jax_experiments/results/*/logs/eval_reward.npy` 统计最后 10 次 eval。
- 风险：现有结果主要是单 seed，且 nonstationary eval 只测第一个 test task，不能直接作为论文强结论。

## 我的判断

RE-SAC 这条线不是完全不能救，但当前 JAX 版还不能支撑论文的主要 claim。最先要修的是实现和实验口径，不是继续加新技巧。修完 P0 后，如果 HalfCheetah/Ant 仍然表现分裂，就应该把论文重心从“统一鲁棒 Bellman operator”降级为“actor-side ensemble LCB + stabilization tricks 的经验方法”，并用 REDQ/DSAC 做强基线对照。
