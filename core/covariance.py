"""Block-diagonal covariance accumulation.

Accumulates A = X^T X and B = X^T Y into (num_blocks, block_size, block_size)
matrices so we never cache the full (B, S, D) activations.
"""

import jax.numpy as jnp
from jax import Array


def to_blocks(X: Array, block_size: int = 64) -> Array:
    """Reshape (...) D -> (... num_blocks, block_size); D must divide block_size."""
    *leading, D = X.shape
    assert D % block_size == 0, f"D={D} not divisible by block_size={block_size}"
    num_blocks = D // block_size
    return X.reshape(*leading, num_blocks, block_size)


def accumulate_covariance(
    X: Array,
    Y: Array,
    mask: Array,
    A_acc: Array,
    B_acc: Array,
    block_size: int = 64,
) -> tuple[Array, Array, int]:
    """Add one batch of masked outer products to the running covariances
    (division by M happens once in the solver, not here)."""

    X_b = to_blocks(X, block_size)
    Y_b = to_blocks(Y, block_size)

    num_blocks = X_b.shape[-2]
    M_total = X_b.size // (num_blocks * block_size)
    X_flat = X_b.reshape(M_total, num_blocks, block_size)
    Y_flat = Y_b.reshape(M_total, num_blocks, block_size)
    mask_flat = mask.reshape(M_total)

    A_delta = jnp.einsum("m,mnb,mnc->nbc", mask_flat, X_flat, X_flat, precision="highest")
    B_delta = jnp.einsum("m,mnb,mnc->nbc", mask_flat, X_flat, Y_flat, precision="highest")

    M_count = jnp.sum(mask)

    return A_acc + A_delta, B_acc + B_delta, M_count


def init_accumulators(num_blocks: int, block_size: int) -> tuple[Array, Array]:
    """Zero-initialised covariance accumulators."""
    shape = (num_blocks, block_size, block_size)
    return jnp.zeros(shape, dtype=jnp.float32), jnp.zeros(shape, dtype=jnp.float32)


def finalise_covariance(
    A_acc: Array,
    B_acc: Array,
    M_total: int,
    G: Array,
    gamma: float = 0.01,
    lambda_reg: float = 0.01,
) -> tuple[Array, Array]:
    """Turn raw sums into regularised, IA-augmented matrices ready for the
    ridge solve: A_IA = (1/M) A_acc + (γ+λ) I, B_IA = (1/M) B_acc + γ G."""
    bs = A_acc.shape[-1]
    I = jnp.eye(bs, dtype=jnp.float32)
    I_blocks = jnp.broadcast_to(I[None], A_acc.shape)

    M_float = jnp.maximum(M_total, 1.0)
    A_IA = A_acc / M_float + (gamma + lambda_reg) * I_blocks
    B_IA = B_acc / M_float + gamma * G

    return A_IA, B_IA