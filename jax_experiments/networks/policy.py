"""Squashed Gaussian Policy (TanhNormal) for SAC / RE-SAC."""
import jax
import jax.numpy as jnp
from flax import nnx

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nnx.Module):
    """SAC-style squashed Gaussian policy.

    Pure observation-conditioned (no context vector).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, *, rngs: nnx.Rngs):
        input_dim = obs_dim

        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nnx.Linear(in_d, hidden_dim, rngs=rngs))
            in_d = hidden_dim
        self.layers = nnx.List(layers)
        self.n_hidden = n_layers

        self.mean_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.log_std_head = nnx.Linear(hidden_dim, act_dim, rngs=rngs)
        self.act_dim = act_dim

    def __call__(self, obs):
        """Returns (mean, log_std) of the squashed Gaussian."""
        x = obs

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.n_hidden:
                x = nnx.relu(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs, key):
        """Sample action via reparameterization trick. Returns (action, log_prob)."""
        mean, log_std = self(obs)
        std = jnp.exp(log_std)

        noise = jax.random.normal(key, mean.shape)
        z = mean + std * noise  # pre-tanh
        action = jnp.tanh(z)

        # Log-prob with tanh squashing correction
        log_prob = -0.5 * (((z - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        log_prob = log_prob.sum(axis=-1)
        log_prob = log_prob - jnp.sum(jnp.log(1 - action ** 2 + 1e-6), axis=-1)
        return action, log_prob

    def deterministic(self, obs):
        """Deterministic action (mean, tanh-squashed)."""
        mean, _ = self(obs)
        return jnp.tanh(mean)
