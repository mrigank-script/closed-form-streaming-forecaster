"""Information Alignment (IA) augmenting objective.

Adds a soft coupling to the backward projection G to bound the drift of
local target propagation: B_IA = B_raw + γ·G while A_IA = A_raw + (γ+λ) I.
γ=0 recovers plain block ridge regression.
"""

import jax.numpy as jnp
from jax import Array


def ia_regularise(
    A_raw: Array,
    B_raw: Array,
    G: Array,
    gamma: float = 0.01,
    lambda_reg: float = 0.01,
) -> tuple[Array, Array]:
    """Apply IA regularisation: A_IA = A_raw + (γ+λ) I, B_IA = B_raw + γ·G."""
    bs = A_raw.shape[-1]
    I = jnp.eye(bs, dtype=jnp.float32)
    I_blocks = jnp.broadcast_to(I[None], A_raw.shape)

    A_IA = A_raw + (gamma + lambda_reg) * I_blocks
    B_IA = B_raw + gamma * G

    return A_IA, B_IA


def check_positive_definite(A: Array, tol: float = 1e-6) -> Array:
    """Per-block positive-definiteness check via minimum eigenvalue."""
    eigvals = jnp.linalg.eigvalsh(A)
    return jnp.all(eigvals > tol, axis=-1)


def condition_numbers(A: Array) -> Array:
    """Per-block condition number (σ_max / σ_min) for monitoring."""
    sv = jnp.linalg.svd(A, compute_uv=False)
    return sv[:, 0] / (sv[:, -1] + 1e-12)