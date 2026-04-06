"""Brax GPU-accelerated Non-stationary Environments.

V4: Everything fused in lax.scan — zero Python loop overhead.
- step_fn with explicit sys arg → compiled ONCE
- rollout_with_policy: fuses policy(obs)→action→physics→auto-reset in lax.scan
- sys is passed as pytree arrays → JAX traces once, reuses for all tasks
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict
from brax import envs
from brax.envs.base import State


RAND_PARAMS_MAP = {
    'gravity': 'opt.gravity',
    'body_mass': 'body_mass',
    'dof_damping': 'dof_damping',
    'body_inertia': 'body_inertia',
}

BRAX_ENV_MAP = {
    'Hopper-v2': 'hopper', 'Hopper-v4': 'hopper', 'Hopper-v5': 'hopper',
    'HalfCheetah-v2': 'halfcheetah', 'HalfCheetah-v4': 'halfcheetah', 'HalfCheetah-v5': 'halfcheetah',
    'Walker2d-v2': 'walker2d', 'Walker2d-v4': 'walker2d', 'Walker2d-v5': 'walker2d',
    'Ant-v2': 'ant', 'Ant-v4': 'ant', 'Ant-v5': 'ant',
}


def _build_core_fns(env, env_name):
    """Build pure functions for env physics, reward, obs, reset."""
    pipeline = env._pipeline
    n_frames = env._n_frames
    debug = env._debug
    dt = env.dt
    brax_name = BRAX_ENV_MAP.get(env_name, env_name.lower())

    fwd_w = getattr(env, '_forward_reward_weight', 1.0)
    ctrl_w = getattr(env, '_ctrl_cost_weight', 1e-3)
    healthy_r = getattr(env, '_healthy_reward', 1.0)
    terminate = getattr(env, '_terminate_when_unhealthy', True)
    hz_range = getattr(env, '_healthy_z_range', (0.7, float('inf')))
    ha_range = getattr(env, '_healthy_angle_range', (-0.2, 0.2))
    hs_range = getattr(env, '_healthy_state_range', (-100.0, 100.0))
    exclude_pos = getattr(env, '_exclude_current_positions_from_observation', True)
    rns = getattr(env, '_reset_noise_scale', 5e-3)

    def physics_step(sys, ps, action):
        def f(s, _):
            return pipeline.step(sys, s, action, debug), None
        return jax.lax.scan(f, ps, (), n_frames)[0]

    def reward_obs(ps0, ps1, action):
        x_vel = (ps1.x.pos[0, 0] - ps0.x.pos[0, 0]) / dt
        if brax_name == 'halfcheetah':
            reward = fwd_w * x_vel - ctrl_w * jnp.sum(jnp.square(action))
            done = jnp.float32(0.0)
            pos = ps1.q[1:] if exclude_pos else ps1.q
            obs = jnp.concatenate([pos, ps1.qd])
        elif brax_name in ('hopper', 'walker2d'):
            z = ps1.x.pos[0, 2]
            angle = ps1.q[2]
            if brax_name == 'hopper':
                sv = jnp.concatenate([ps1.q[2:], ps1.qd])
                is_h = jnp.all(jnp.logical_and(hs_range[0] < sv, sv < hs_range[1]))
                is_h = is_h & (hz_range[0] < z) & (z < hz_range[1])
                is_h = is_h & (ha_range[0] < angle) & (angle < ha_range[1])
            else:
                is_h = ((z > hz_range[0]) & (z < hz_range[1]) &
                        (angle > ha_range[0]) & (angle < ha_range[1]))
            h_rew = jnp.where(terminate, healthy_r, healthy_r * is_h)
            reward = fwd_w * x_vel + h_rew - ctrl_w * jnp.sum(jnp.square(action))
            done = jnp.where(terminate, 1.0 - is_h, 0.0)
            pos = ps1.q.at[1].set(z)
            vel = jnp.clip(ps1.qd, -10, 10)
            if exclude_pos: pos = pos[1:]
            obs = jnp.concatenate([pos, vel])
        elif brax_name == 'ant':
            vel = (ps1.x.pos[0] - ps0.x.pos[0]) / dt
            z = ps1.x.pos[0, 2]
            is_h = jnp.where(z < hz_range[0], 0.0, 1.0)
            is_h = jnp.where(z > hz_range[1], 0.0, is_h)
            h_rew = jnp.where(terminate, healthy_r, healthy_r * is_h)
            reward = vel[0] + h_rew - ctrl_w * jnp.sum(jnp.square(action))
            done = jnp.where(terminate, 1.0 - is_h, 0.0)
            qpos = ps1.q[2:] if exclude_pos else ps1.q
            obs = jnp.concatenate([qpos, ps1.qd])
        else:
            raise ValueError(f"Unsupported: {brax_name}")
        return reward, obs, done

    def init_obs(ps):
        if brax_name == 'halfcheetah':
            pos = ps.q[1:] if exclude_pos else ps.q
            return jnp.concatenate([pos, ps.qd])
        elif brax_name in ('hopper', 'walker2d'):
            pos = ps.q.at[1].set(ps.x.pos[0, 2])
            vel = jnp.clip(ps.qd, -10, 10)
            if exclude_pos: pos = pos[1:]
            return jnp.concatenate([pos, vel])
        elif brax_name == 'ant':
            qpos = ps.q[2:] if exclude_pos else ps.q
            return jnp.concatenate([qpos, ps.qd])
        return jnp.concatenate([ps.q, ps.qd])

    def reset_state(sys, rng):
        r1, r2 = jax.random.split(rng)
        q = sys.init_q + jax.random.uniform(r1, (sys.q_size(),), minval=-rns, maxval=rns)
        qd = jax.random.uniform(r2, (sys.qd_size(),), minval=-rns, maxval=rns)
        ps = pipeline.init(sys, q, qd, debug=debug)
        return State(pipeline_state=ps, obs=init_obs(ps),
                     reward=jnp.float32(0.0), done=jnp.float32(0.0),
                     metrics={}, info={})

    return physics_step, reward_obs, reset_state


class BraxNonstationaryEnv:
    """GPU-accelerated non-stationary env.

    The key optimization: build a SINGLE lax.scan rollout function that fuses
    policy inference + physics + auto-reset. Compiled ONCE, reused for all tasks
    because sys is passed as a normal pytree argument.
    """

    def __init__(self, env_name: str, rand_params: List[str] = None,
                 log_scale_limit: float = 3.0, seed: int = 0,
                 backend: str = 'spring'):
        brax_name = BRAX_ENV_MAP.get(env_name, env_name.lower())
        self.env = envs.get_environment(brax_name, backend=backend)
        self.base_sys = self.env.sys
        self.env_name = env_name
        self.backend = backend

        self.rand_params = rand_params if rand_params is not None else ['gravity']
        self.log_scale_limit = log_scale_limit
        self.seed = seed
        self.rng = jax.random.PRNGKey(seed)

        self.obs_dim = self.env.observation_size
        self.act_dim = self.env.action_size
        self.dt = self.env.dt

        self._base_values = {}
        for param in self.rand_params:
            key = RAND_PARAMS_MAP.get(param, param)
            if '.' in key:
                val = self.base_sys
                for p in key.split('.'): val = getattr(val, p)
            else:
                val = getattr(self.base_sys, key)
            self._base_values[param] = jnp.array(val)

        self.current_task_id = 0
        self._tasks = None
        self._task_sys_list = None
        self._changing_interval = 10
        self._changing_period = 100
        self._step_counter = 0
        self._current_sys = self.base_sys
        self._state = None

        # Build core physics functions
        self._physics_step, self._reward_obs, self._reset_state = \
            _build_core_fns(self.env, env_name)

        # Build JIT'd step/reset (for sequential API)
        physics_step = self._physics_step
        reward_obs = self._reward_obs
        reset_state = self._reset_state

        @jax.jit
        def _step(sys, state, action):
            ps0 = state.pipeline_state
            ps1 = physics_step(sys, ps0, action)
            r, o, d = reward_obs(ps0, ps1, action)
            return state.replace(pipeline_state=ps1, obs=o, reward=r, done=d)

        @jax.jit
        def _reset(sys, rng):
            return reset_state(sys, rng)

        self._step_fn = _step
        self._reset_fn = _reset

    # --- Task management ---

    def _set_sys(self, sys):
        self._current_sys = sys

    def sample_tasks(self, n_tasks: int) -> List[Dict]:
        tasks = []
        rng = np.random.RandomState(self.seed + 42)
        for _ in range(n_tasks):
            task = {}
            for param in self.rand_params:
                base = np.array(self._base_values[param])
                log_scale = rng.uniform(-self.log_scale_limit, self.log_scale_limit,
                                       size=base.shape)
                task[param] = base * np.exp(log_scale).astype(np.float32)
            tasks.append(task)
        return tasks

    def set_task(self, task: Dict):
        replacements = {}
        for param, value in task.items():
            key = RAND_PARAMS_MAP.get(param, param)
            replacements[key] = jnp.array(value)
        self._set_sys(self.base_sys.tree_replace(replacements))

    def set_nonstationary_para(self, tasks, changing_period=100, changing_interval=10):
        self._tasks = tasks
        self._changing_period = changing_period
        self._changing_interval = changing_interval
        self._step_counter = 0
        self._task_sys_list = []
        for task in tasks:
            replacements = {}
            for param, value in task.items():
                key = RAND_PARAMS_MAP.get(param, param)
                replacements[key] = jnp.array(value)
            self._task_sys_list.append(self.base_sys.tree_replace(replacements))
        self.current_task_id = 0
        self._set_sys(self._task_sys_list[0])

    def _check_switch(self):
        if (self._tasks is not None and
                self._step_counter % self._changing_interval == 0):
            idx = int(self._step_counter / self._changing_period) % len(self._tasks)
            if idx != self.current_task_id:
                self.current_task_id = idx
                self._set_sys(self._task_sys_list[idx])

    # --- Sequential API ---

    def reset(self):
        self.rng, key = jax.random.split(self.rng)
        self._state = self._reset_fn(self._current_sys, key)
        return np.array(self._state.obs)

    def step(self, action):
        self._step_counter += 1
        self._check_switch()
        action_jax = jnp.array(action)
        self._state = self._step_fn(self._current_sys, self._state, action_jax)
        return np.array(self._state.obs), float(self._state.reward), \
               bool(self._state.done), {}

    def close(self):
        pass

    @property
    def action_space(self):
        class _AS:
            def __init__(s, dim): s.shape = (dim,)
            def sample(s): return np.random.uniform(-1, 1, size=s.shape).astype(np.float32)
        return _AS(self.act_dim)

    # --- Scan-fused rollout with policy ---

    def build_rollout_fn(self, policy_graphdef, context_graphdef=None):
        """Build JIT'd scan rollouts: stochastic (training) + deterministic (eval).

        Args:
            policy_graphdef: nnx.graphdef(agent.policy)
            context_graphdef: optional nnx.graphdef(agent.context_net) for ESCP/BAPR
        """
        from flax import nnx

        physics_step = self._physics_step
        reward_obs = self._reward_obs
        reset_state = self._reset_state
        has_context = context_graphdef is not None

        # --- Stochastic rollout (training) ---
        @jax.jit
        def _rollout_scan(sys, policy_params, context_params, init_state, keys):
            def scan_body(carry, key):
                state = carry
                key1, key2 = jax.random.split(key)

                pre_obs = state.obs
                policy = nnx.merge(policy_graphdef, policy_params)

                if has_context:
                    ctx_net = nnx.merge(context_graphdef, context_params)
                    ep = ctx_net(pre_obs[None])
                    action, _ = policy.sample(pre_obs[None], key1, ep)
                else:
                    action, _ = policy.sample(pre_obs[None], key1)
                action = action[0]

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, action)
                reward, post_obs, done = reward_obs(ps0, ps1, action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs, reward=reward, done=done)

                reset_st = reset_state(sys, key2)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(done, r, n), reset_st, next_state)

                transition = (pre_obs, action, reward, post_obs, done)
                return out_state, transition

            final_state, transitions = jax.lax.scan(
                scan_body, init_state, keys)
            return final_state, transitions

        # --- Deterministic rollout (eval): tanh(mean), no PRNG per step ---
        @jax.jit
        def _rollout_scan_det(sys, policy_params, context_params, init_state, reset_keys):
            """Eval rollout — uses policy mean (no exploration noise).

            reset_keys: [N, 2] keys used only for auto-reset sampling.
            """
            def scan_body(carry, reset_key):
                state = carry
                pre_obs = state.obs
                policy = nnx.merge(policy_graphdef, policy_params)

                if has_context:
                    ctx_net = nnx.merge(context_graphdef, context_params)
                    ep = ctx_net(pre_obs[None])
                    action = policy.deterministic(pre_obs[None], ep)
                else:
                    action = policy.deterministic(pre_obs[None])
                action = action[0]

                ps0 = state.pipeline_state
                ps1 = physics_step(sys, ps0, action)
                reward, post_obs, done = reward_obs(ps0, ps1, action)
                next_state = state.replace(
                    pipeline_state=ps1, obs=post_obs, reward=reward, done=done)

                reset_st = reset_state(sys, reset_key)
                out_state = jax.tree.map(
                    lambda r, n: jnp.where(done, r, n), reset_st, next_state)

                return out_state, (reward, done)

            final_state, (rewards, dones) = jax.lax.scan(
                scan_body, init_state, reset_keys)
            return final_state, (rewards, dones)

        self._rollout_scan = _rollout_scan
        self._rollout_scan_det = _rollout_scan_det
        self._has_context = has_context

    def rollout(self, policy_params, n_steps: int, rng_key, context_params=None):
        """Run n_steps using the pre-built scan rollout.

        Returns JAX arrays (stays on GPU) + episode reward list (CPU).

        Args:
            policy_params: nnx.State(agent.policy, nnx.Param)
            n_steps: number of steps
            rng_key: PRNG key
            context_params: optional nnx.State(agent.context_net, nnx.Param)

        Returns:
            (obs, act, rew, nobs, done): JAX arrays on device
            ep_rewards: list of floats (CPU)
        """
        rng_key, init_key, roll_key = jax.random.split(rng_key, 3)
        init_state = self._reset_fn(self._current_sys, init_key)
        keys = jax.random.split(roll_key, n_steps)

        # Single JIT call for all N steps — outputs stay on GPU
        final_state, (obs, act, rew, nobs, done) = self._rollout_scan(
            self._current_sys, policy_params, context_params, init_state, keys)

        # Episode rewards: only rew and done need CPU transfer (small: [N] floats)
        rew_np = np.array(rew)
        done_np = np.array(done)
        ep_boundaries = np.where(done_np > 0.5)[0]
        if len(ep_boundaries) > 0:
            cumrew = np.cumsum(rew_np)
            ends = cumrew[ep_boundaries]
            starts = np.concatenate([[0.0], ends[:-1]])
            ep_rewards = (ends - starts).tolist()
        else:
            ep_rewards = []

        # Update step counter and trigger task switch for NEXT rollout
        self._step_counter += n_steps
        self._check_switch()   # ← switches _current_sys so next rollout uses new task
        self._state = final_state
        return (obs, act, rew, nobs, done), ep_rewards

    def eval_rollout(self, policy_params, n_steps: int, rng_key, context_params=None):
        """Deterministic eval rollout — does NOT update step counter or switch task.

        Uses tanh(mean) policy (no exploration noise). ~10-50x faster than the
        sequential evaluate() loop because all steps run in one GPU scan call.

        Args:
            policy_params: nnx.State(agent.policy, nnx.Param)
            n_steps: total steps to run (e.g. n_episodes * max_episode_steps)
            rng_key: PRNG key (only used for auto-reset state sampling)
            context_params: optional nnx.State(agent.context_net, nnx.Param)

        Returns:
            (rew_np, done_np): per-step reward and done arrays [n_steps]
        """
        rng_key, init_key, reset_key = jax.random.split(rng_key, 3)
        sys = self._current_sys
        init_state = self._reset_fn(sys, init_key)
        reset_keys = jax.random.split(reset_key, n_steps)

        _, (rew_jax, done_jax) = self._rollout_scan_det(
            sys, policy_params, context_params, init_state, reset_keys)

        return np.array(rew_jax), np.array(done_jax)
