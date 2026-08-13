"""
core/ridge_solver.py — Closed-Form Block-Diagonal Ridge Regression

THE heart of the Swarm Engine. Computes exact optimal weights in a
single algebraic step — no iteration, no learning rate.

Mathematical basis: §3.1 of whitepaper.

Given finalised (A_IA, B_IA) from the covariance accumulator:
    W^T = A_IA^{-1} · B_IA

where:
    A_IA = (1/M) X^T X + (γ+λ) I     [positive-definite by construction]
    B_IA = (1/M) X^T Y + γ · G

We solve 64 independent (64×64) linear systems in parallel using
jax.vmap(jnp.linalg.solve), which XLA compiles to a single batched
cuSOLVER call.

Memory cost: 3 × NB × bs² × 4B = 3 × 64 × 64 × 64 × 4 = 3 MB total.
"""

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import Array


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def block_ridge_solve(A_IA: Array, B_IA: Array) -> Array:
    """
    Solve the block-diagonal Ridge system in parallel.

    W^T[n] = A_IA[n]^{-1} · B_IA[n]   for n = 0..num_blocks-1

    Uses jnp.linalg.solve (not inv) for numerical stability — it
    computes A W = B via LU factorisation without forming A^{-1}.

    Args:
        A_IA: Float[Array, "num_blocks block_size block_size"]
              — positive-definite auto-covariance (already regularised)
        B_IA: Float[Array, "num_blocks block_size block_size"]
              — cross-covariance (already IA-augmented)

    Returns:
        W_T: Float[Array, "num_blocks block_size block_size"]
             — solution satisfying A_IA @ W_T = B_IA per block
    """
    # vmap over the leading num_blocks dimension
    # jsl.solve(A, B, assume_a='pos') uses symmetric Cholesky solve for positive-definite A
    solve_pos_fn = lambda a, b: jsl.solve(a, b, assume_a='pos')
    W_T = jax.vmap(solve_pos_fn)(A_IA, B_IA)
    return W_T


# ---------------------------------------------------------------------------
# Backward projection (G update)
# ---------------------------------------------------------------------------

def compute_backward_projection(
    Y: Array,          # Float["B S D"] — layer targets
    X: Array,          # Float["B S D"] — layer inputs
    mask: Array,       # Float["B S"]
    lambda_reg: float = 0.01,
    gamma: float = 0.01,
    block_size: int = 64,
) -> Array:
    """
    Compute the backward projection matrix G used for target propagation.

    G is the (Y → X) direction ridge solution:
        A_YY = (1/M) Y^T Y + (γ+λ) I
        B_YX = (1/M) Y^T X + γ I
        G^T  = A_YY^{-1} · B_YX

    This gives us a learned linear path from targets back to inputs,
    used to propagate targets layer-by-layer in the training loop:
        Y_{L-1} = Y_L @ G_L^T

    Args:
        Y: Float[Array, "B S D"] — target activations
        X: Float[Array, "B S D"] — input activations
        mask: Float[Array, "B S"]
        lambda_reg, gamma: regularisation hyperparams
        block_size: int

    Returns:
        G: Float[Array, "num_blocks block_size block_size"]
    """
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
    G_T = jax.vmap(solve_pos_fn)(A_YY, B_YX)
    return G_T


# ---------------------------------------------------------------------------
# Block-diagonal forward pass (apply weights to activations)
# ---------------------------------------------------------------------------

def block_matmul(X: Array, W_T: Array) -> Array:
    """
    Apply block-diagonal weight matrix to activations.

    For each block n:  out[..., n*bs:(n+1)*bs] = X[..., n*bs:(n+1)*bs] @ W_T[n]

    Args:
        X:   Float[Array, "... D"]
        W_T: Float[Array, "num_blocks block_size block_size"]

    Returns:
        Float[Array, "... D"]
    """
    *leading, D = X.shape
    block_size = W_T.shape[-1]
    num_blocks = W_T.shape[-3]

    # Reshape: (..., D) → (..., NB, bs)
    X_b = X.reshape(*leading, num_blocks, block_size)

    # Apply each block: (..., NB, bs) @ (..., NB, bs, bs) → (..., NB, bs)
    # einsum: "...nb,...nbc->...nc"
    out_b = jnp.einsum("...nb,...nbc->...nc", X_b, W_T)

    # Reshape back: (..., NB, bs) → (..., D)
    return out_b.reshape(*leading, D)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_layer_weights(key: Array, num_blocks: int, block_size: int) -> dict:
    """
    Initialise W (forward weights) and G (backward projection) for one layer.

    Args:
        key: JAX PRNG key
        num_blocks: int — D // block_size
        block_size: int

    Returns:
        dict: {"W_T": Array[NB, bs, bs], "G": Array[NB, bs, bs]}
    """
    key_w, key_g = jax.random.split(key)
    scale = 0.01  # small init — will be overwritten after first solve
    shape = (num_blocks, block_size, block_size)

    W_T = jax.random.normal(key_w, shape) * scale
    G   = jax.random.normal(key_g, shape) * scale

    return {"W_T": W_T, "G": G}
