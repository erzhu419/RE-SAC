"""Utilities for the RE-SAC algorithmic ablations (paper §6.1.6).

Three primitives, used independently or in combination by resac.py:

  - spectral_normalize_kernel(W)       — Variant A: per-head, per-layer σ_max
    rescaling. Power iteration estimates σ_max for each [in, out] slice of
    a [K, in, out] tensor, then divides by σ_max so each slice has σ_max ≤ c.

  - StateDepBetaState                  — Variant B: tracks σ_ema(σ_ens) so the
    LCB coefficient can scale per-state with current vs typical ensemble std.

  - HashCounter                        — Variant C: SimHash-style state-action
    hashing with a bounded count table. Adds an exploration-style bonus
    α/√N(hash(s,a)) to the ensemble std.

All three are pure JAX so they can live inside the JIT-compiled scan body.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────────
# Variant A: Spectral normalization for vectorized critic layers
# ─────────────────────────────────────────────────────────────────────────

def _power_iter_kernel(kernel: jnp.ndarray, n_iters: int = 1):
    """Estimate σ_max for each [in, out] slice of a [K, in, out] tensor.

    Returns (sigma, u, v) where sigma is [K], u is [K, out], v is [K, in].
    Uses one (or n_iters) power iteration steps. We initialize u with a fixed
    PRNG so the operation is deterministic w.r.t. the current weights.
    """
    K, in_d, out_d = kernel.shape
    key = jax.random.PRNGKey(0)
    u = jax.random.normal(key, (K, out_d))
    u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-12)
    v = jnp.zeros((K, in_d))
    for _ in range(n_iters):
        # v = W u (per-head: [K, in, out] @ [K, out, 1] -> [K, in, 1])
        v = jnp.einsum("kio,ko->ki", kernel, u)
        v = v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
        # u = W^T v
        u = jnp.einsum("kio,ki->ko", kernel, v)
        u = u / (jnp.linalg.norm(u, axis=-1, keepdims=True) + 1e-12)
    sigma = jnp.einsum("kio,ki,ko->k", kernel, v, u)
    return sigma, u, v


def spectral_normalize_kernel(kernel: jnp.ndarray, c: float = 1.0,
                              n_iters: int = 1) -> jnp.ndarray:
    """Rescale a [K, in, out] kernel so each [in, out] slice has σ_max ≤ c.

    Returns the rescaled kernel. No-op on biases (caller should not pass them).
    Hard constraint: any slice with σ > c is divided by σ/c; slices already
    under c are left untouched.
    """
    sigma, _, _ = _power_iter_kernel(kernel, n_iters=n_iters)
    scale = c / jnp.maximum(sigma, c)
    # broadcast [K] -> [K, 1, 1]
    return kernel * scale[:, None, None]


def apply_spectral_norm_to_critic_params(critic_params, c: float = 1.0,
                                         n_iters: int = 1):
    """Walk the critic param tree and spectral-normalize every kernel slot.

    EnsembleCritic stores [K, in, out] kernels and [K, 1, out] biases — both
    3D. We rescale only the kernels. Disambiguation is by shape: bias has
    middle dim == 1, kernel has middle dim > 1. (Avoids depending on nnx
    PyTree path strings, which differ across flax versions.)
    """
    def _maybe_norm(leaf):
        if leaf.ndim == 3 and leaf.shape[1] > 1:
            return spectral_normalize_kernel(leaf, c=c, n_iters=n_iters)
        return leaf
    return jax.tree.map(_maybe_norm, critic_params)


# ─────────────────────────────────────────────────────────────────────────
# Variant B: state-dependent β_lcb
# ─────────────────────────────────────────────────────────────────────────

def state_dep_beta(qs: jnp.ndarray, sigma_ema: jnp.ndarray,
                   beta0: float, cap: float = 3.0) -> jnp.ndarray:
    """Per-state β_eff = -|β0| · clip(σ_ens / σ_ema, max=cap).

    qs:        [B] per-state ensemble std (σ_ens(s,a))
    sigma_ema: scalar (or [1]) running EMA baseline of σ_ens.mean()
    beta0:     base coefficient (paper's β_end)
    cap:       max relative inflation
    """
    eps = 1e-6
    ratio = qs / jnp.maximum(sigma_ema, eps)
    ratio = jnp.minimum(ratio, cap)
    beta_eff = -jnp.abs(beta0) * ratio
    return beta_eff


# ─────────────────────────────────────────────────────────────────────────
# Variant C: hash-based count bonus
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class HashCounterState:
    """Pure-JAX SimHash state-action counter.

    Stores random projection W_hash [d_in, hash_dim] and a counts vector
    of length 2^hash_dim. All updates are jit-friendly via scatter_add.
    """
    W_hash: jnp.ndarray           # [obs_dim + act_dim, hash_dim]
    counts: jnp.ndarray           # [2 ** hash_dim]
    hash_dim: int

    @classmethod
    def init(cls, obs_dim: int, act_dim: int, hash_dim: int = 14, *,
             seed: int = 0) -> "HashCounterState":
        key = jax.random.PRNGKey(seed)
        W = jax.random.normal(key, (obs_dim + act_dim, hash_dim))
        counts = jnp.zeros((2 ** hash_dim,), dtype=jnp.int32)
        return cls(W_hash=W, counts=counts, hash_dim=hash_dim)


def hash_state_action(obs: jnp.ndarray, act: jnp.ndarray,
                      W_hash: jnp.ndarray, hash_dim: int) -> jnp.ndarray:
    """SimHash: project (s, a) and threshold to a binary code, then to int.

    obs: [B, obs_dim], act: [B, act_dim], W_hash: [obs_dim+act_dim, hash_dim]
    Returns: [B] int32 hash bucket in [0, 2^hash_dim).
    """
    sa = jnp.concatenate([obs, act], axis=-1)         # [B, obs_dim+act_dim]
    z = (sa @ W_hash > 0).astype(jnp.int32)           # [B, hash_dim]
    powers = (1 << jnp.arange(hash_dim, dtype=jnp.int32))  # [hash_dim]
    return jnp.sum(z * powers, axis=-1)               # [B]


def update_counts(counts: jnp.ndarray, buckets: jnp.ndarray) -> jnp.ndarray:
    """Increment counts[buckets] by 1, returning new counts. JIT-safe."""
    return counts.at[buckets].add(1)


def count_bonus(counts: jnp.ndarray, buckets: jnp.ndarray,
                alpha: float = 0.5) -> jnp.ndarray:
    """Bonus α / sqrt(1 + N(b)) per state in batch. Returns [B]."""
    n = counts[buckets].astype(jnp.float32)
    return alpha / jnp.sqrt(1.0 + n)
