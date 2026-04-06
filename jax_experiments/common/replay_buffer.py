"""GPU-native JAX replay buffer — all data lives on device.

Key optimization: eliminates the GPU→CPU→GPU round-trip that the old numpy
buffer required every iteration:
  OLD: rollout(GPU) → np.array (CPU) → buffer(CPU) → np.random idx (CPU) → jnp.array (GPU)
  NEW: rollout(GPU) → buffer(GPU) → jax.random idx (GPU) → train(GPU)

The buffer uses pre-allocated JAX arrays and in-place updates via .at[].set().
Sampling uses jax.random.randint + jnp.take for fully on-device indexing.

Falls back to CPU-side numpy for:
  - Checkpoint save/load (unavoidable disk I/O)
  - Random exploration phase (sequential env.step returns numpy)
"""
import jax
import jax.numpy as jnp
import numpy as np


class ReplayBuffer:
    """Fixed-size replay buffer with JAX device arrays.

    All storage arrays live on the default JAX device (GPU if available).
    push_batch_jax() and sample_stacked_jax() are fully on-device.
    push() for single transitions (random exploration) still works via CPU→GPU.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ptr = 0
        self.size = 0

        # Pre-allocate on device
        self.obs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.act = jnp.zeros((capacity, act_dim), dtype=jnp.float32)
        self.rew = jnp.zeros((capacity, 1), dtype=jnp.float32)
        self.next_obs = jnp.zeros((capacity, obs_dim), dtype=jnp.float32)
        self.done = jnp.zeros((capacity, 1), dtype=jnp.float32)
        self.task_id = jnp.zeros((capacity,), dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Single-transition push (random exploration phase — infrequent)
    # ------------------------------------------------------------------
    def push(self, obs, act, rew, next_obs, done, task_id=0):
        """Push one transition. Accepts numpy arrays (auto-converts)."""
        i = self.ptr
        self.obs = self.obs.at[i].set(jnp.asarray(obs, dtype=jnp.float32))
        self.act = self.act.at[i].set(jnp.asarray(act, dtype=jnp.float32))
        self.rew = self.rew.at[i].set(jnp.float32(rew))
        self.next_obs = self.next_obs.at[i].set(jnp.asarray(next_obs, dtype=jnp.float32))
        self.done = self.done.at[i].set(jnp.float32(done))
        self.task_id = self.task_id.at[i].set(jnp.int32(task_id))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Batch push — accepts JAX arrays directly (zero-copy from rollout)
    # ------------------------------------------------------------------
    def push_batch_jax(self, obs, act, rew, next_obs, done, task_id=None):
        """Push a batch of transitions from JAX arrays (no CPU transfer).

        Args:
            obs, next_obs: [N, obs_dim] jax arrays
            act: [N, act_dim] jax array
            rew: [N, 1] or [N] jax array
            done: [N, 1] or [N] jax array
            task_id: [N] jax/numpy int array, or None
        """
        n = obs.shape[0]
        rew = rew.reshape(-1, 1) if rew.ndim == 1 else rew
        done = done.reshape(-1, 1) if done.ndim == 1 else done
        if task_id is None:
            task_id = jnp.zeros(n, dtype=jnp.int32)
        else:
            task_id = jnp.asarray(task_id, dtype=jnp.int32)

        # Compute insertion indices (handles wrap-around)
        idx = (jnp.arange(n) + self.ptr) % self.capacity

        self.obs = self.obs.at[idx].set(obs)
        self.act = self.act.at[idx].set(act)
        self.rew = self.rew.at[idx].set(rew)
        self.next_obs = self.next_obs.at[idx].set(next_obs)
        self.done = self.done.at[idx].set(done)
        self.task_id = self.task_id.at[idx].set(task_id)

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    # ------------------------------------------------------------------
    # Legacy batch push (numpy) — used by checkpoint restore
    # ------------------------------------------------------------------
    def push_batch(self, obs, act, rew, next_obs, done, task_id=None):
        """Push a batch of numpy transitions (converts to JAX)."""
        self.push_batch_jax(
            jnp.asarray(obs), jnp.asarray(act),
            jnp.asarray(rew), jnp.asarray(next_obs),
            jnp.asarray(done),
            jnp.asarray(task_id) if task_id is not None else None)

    # ------------------------------------------------------------------
    # Sampling — fully on device
    # ------------------------------------------------------------------
    def sample_stacked(self, n_batches: int, batch_size: int,
                       rng_key=None):
        """Sample n_batches × batch_size transitions, fully on GPU.

        Returns dict of JAX arrays with shape [n_batches, batch_size, ...].
        Used with jax.lax.scan — output goes directly to _scan_update.
        """
        if rng_key is None:
            # Fallback: generate a JAX key from numpy random state
            rng_key = jax.random.PRNGKey(np.random.randint(0, 2**31))

        # [n_batches, batch_size] random indices — all on device
        idx = jax.random.randint(
            rng_key, (n_batches, batch_size), 0, self.size)

        return {
            "obs": self.obs[idx],          # [N, B, obs_dim]
            "act": self.act[idx],          # [N, B, act_dim]
            "rew": self.rew[idx],          # [N, B, 1]
            "next_obs": self.next_obs[idx],  # [N, B, obs_dim]
            "done": self.done[idx],        # [N, B, 1]
            "task_id": self.task_id[idx],  # [N, B]
        }

    def sample(self, batch_size: int, rng: np.random.Generator = None):
        """Sample one batch. Returns dict of JAX arrays [B, ...]."""
        key = jax.random.PRNGKey(
            np.random.randint(0, 2**31) if rng is None
            else int(rng.integers(0, 2**31)))
        idx = jax.random.randint(key, (batch_size,), 0, self.size)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "task_id": self.task_id[idx],
        }

    # ------------------------------------------------------------------
    # Numpy conversion for checkpoint save/load
    # ------------------------------------------------------------------
    def to_numpy(self):
        """Export buffer contents as numpy dict (for checkpoint save)."""
        s = self.size
        return {
            'obs': np.array(self.obs[:s]),
            'act': np.array(self.act[:s]),
            'rew': np.array(self.rew[:s]),
            'next_obs': np.array(self.next_obs[:s]),
            'done': np.array(self.done[:s]),
            'task_id': np.array(self.task_id[:s]),
            'ptr': self.ptr,
            'size': self.size,
        }

    def from_numpy(self, buf_dict):
        """Restore buffer contents from numpy dict (for checkpoint load)."""
        s = int(buf_dict['size'])
        # Write into pre-allocated device arrays
        idx = jnp.arange(s)
        self.obs = self.obs.at[idx].set(jnp.array(buf_dict['obs']))
        self.act = self.act.at[idx].set(jnp.array(buf_dict['act']))
        self.rew = self.rew.at[idx].set(jnp.array(buf_dict['rew']))
        self.next_obs = self.next_obs.at[idx].set(jnp.array(buf_dict['next_obs']))
        self.done = self.done.at[idx].set(jnp.array(buf_dict['done']))
        self.task_id = self.task_id.at[idx].set(jnp.array(buf_dict['task_id']))
        self.ptr = int(buf_dict['ptr'])
        self.size = s

    def __len__(self):
        return self.size
