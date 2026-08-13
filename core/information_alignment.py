"""
core/information_alignment.py — Information Alignment (IA) Augmented Objective

IA is the mechanism that bounds the drift caused by local target
propagation. Without it, each layer optimises its own pseudo-target
independently, and errors accumulate with depth.

Mathematical basis: §3.5 of whitepaper.

The augmented system is:
    A_IA = (1/M) X^T X + (γ + λ) I
    B_IA = (1/M) X^T Y + γ · G

The extra γ·G term in B_IA "pulls" the forward weights W toward
the backward projection G, creating a soft coupling between the
forward and backward information paths.

Key properties proven in §3.5:
  - A_IA is still positive-definite (γ only raises the diagonal)
  - The IA objective is jointly convex in (W, G)
  - Setting γ=0 recovers standard block Ridge regression

The G matrix itself is updated after the forward solve using the
(Y → X) ridge problem (see ridge_solver.compute_backward_projection).
"""

import jax.numpy as jnp
from jax import Array


def ia_regularise(
    A_raw: Array,       # Float["NB bs bs"] — raw (1/M) X^T X
    B_raw: Array,       # Float["NB bs bs"] — raw (1/M) X^T Y
    G: Array,           # Float["NB bs bs"] — backward projection
    gamma: float = 0.01,
    lambda_reg: float = 0.01,
) -> tuple[Array, Array]:
    """
    Apply Information Alignment regularisation to raw covariance matrices.

    A_IA = A_raw + (γ + λ) I
    B_IA = B_raw + γ · G

    Args:
        A_raw: Float[Array, "NB bs bs"] — (1/M) X^T X (already divided by M)
        B_raw: Float[Array, "NB bs bs"] — (1/M) X^T Y (already divided by M)
        G:     Float[Array, "NB bs bs"] — backward projection matrix
        gamma:      float — IA coupling strength
        lambda_reg: float — ridge regularisation

    Returns:
        (A_IA, B_IA) — regularised system ready for block_ridge_solve
    """
    bs = A_raw.shape[-1]
    I = jnp.eye(bs, dtype=jnp.float32)
    I_blocks = jnp.broadcast_to(I[None], A_raw.shape)

    A_IA = A_raw + (gamma + lambda_reg) * I_blocks
    B_IA = B_raw + gamma * G

    return A_IA, B_IA


def check_positive_definite(A: Array, tol: float = 1e-6) -> Array:
    """
    Check whether A is positive-definite via minimum eigenvalue.

    Args:
        A: Float[Array, "NB bs bs"]
        tol: minimum eigenvalue threshold

    Returns:
        Bool[Array, "NB"] — True if block n is PD
    """
    # Compute eigenvalues for each block
    eigvals = jnp.linalg.eigvalsh(A)        # (NB, bs)
    return jnp.all(eigvals > tol, axis=-1)  # (NB,)


def condition_numbers(A: Array) -> Array:
    """
    Compute condition numbers of each block (for monitoring).
    High condition number → near-singular → might need larger λ.

    Args:
        A: Float[Array, "NB bs bs"]

    Returns:
        Float[Array, "NB"] — condition number per block
    """
    sv = jnp.linalg.svd(A, compute_uv=False)   # (NB, bs)
    return sv[:, 0] / (sv[:, -1] + 1e-12)      # σ_max / σ_min
