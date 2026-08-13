"""Closed-form block-diagonal ridge solver.

Computes the exact weights W^T = A_IA^{-1} B_IA in a single algebraic step
(no iteration, no learning rate) and solves all blocks in parallel.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import Array


def block_ridge_solve(A_IA: Array, B_IA: Array) -> Array:
    """Solve A W = B per block; Cholesky path (never forms the inverse)."""
    solve_pos_fn = lambda a, b: jsl.solve(a, b, assume_a='pos')
    return jax.vmap(solve_pos_fn)(A_IA, B_IA)


def compute_backward_projection(
    Y: Array,
    X: Array,
    mask: Array,
    lambda_reg: float = 0.01,
    gamma: float = 0.01,
    block_size: int = 64,
) -> Array:
    """Backward-projection matrix G, the (Y -> X) ridge solve, for target
    propagation (Y_{L-1} = Y_L @ G_L^T)."""
    from core.covariance import to_blocks, init_accumulators

    D = Y.shape[-1]
    num_blocks = D // block_size

    Y_flat = to_blocks(Y, block_size).reshape(-1, num_blocks, block_size)
    X_flat = to_blocks(X, block_size).reshape(-1, num_blocks, block_size)
    mask_flat = mask.reshape(-1)

    M = jnp.maximum(jnp.sum(mask), 1.0)
    bs = block_size
    I = jnp.eye(bs)
    I_blocks = jnp.broadcast_to(I[None], (num_blocks, bs, bs))

    A_YY = jnp.einsum("m,mnb,mnc->nbc", mask_flat, Y_flat, Y_flat, precision="highest") / M + (gamma + lambda_reg) * I_blocks
    B_YX = jnp.einsum("m,mnb,mnc->nbc", mask_flat, Y_flat, X_flat, precision="highest") / M + gamma * I_blocks

    solve_pos_fn = lambda a, b: jsl.solve(a, b, assume_a='pos')
    return jax.vmap(solve_pos_fn)(A_YY, B_YX)


def block_matmul(X: Array, W_T: Array) -> Array:
    """Apply block-diagonal weights: per block, out[..., n*bs:(n+1)*bs] = X[..., n*bs:(n+1)*bs] @ W_T[n]."""
    *leading, D = X.shape
    block_size = W_T.shape[-1]
    num_blocks = W_T.shape[-3]

    X_b = X.reshape(*leading, num_blocks, block_size)
    out_b = jnp.einsum("...nb,...nbc->...nc", X_b, W_T)
    return out_b.reshape(*leading, D)


def init_layer_weights(key: Array, num_blocks: int, block_size: int) -> dict:
    """Initialise forward weights W and backward projection G (overwritten on first solve)."""
    key_w, key_g = jax.random.split(key)
    scale = 0.01
    shape = (num_blocks, block_size, block_size)

    W_T = jax.random.normal(key_w, shape) * scale
    G = jax.random.normal(key_g, shape) * scale

    return {"W_T": W_T, "G": G}
