"""
core/lru.py — Complex-Diagonal Linear Recurrent Unit (LRU)

Implements the parallel O(log L) sequence processing via JAX's
associative scan. Mathematical basis: §2.1, §3.7 of whitepaper.

Key insight: The diagonal structure of Λ means the combine operator
is element-wise, making the scan trivially parallelisable.

CRITICAL CONSTRAINT: r_k < 1.0 ALWAYS. r_k = 1.0 → undamped integrator
→ NaN explosion. Initialise with r ∈ [0.9, 0.999].
"""

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Complex, Float


# ---------------------------------------------------------------------------
# Binary associative operator for jax.lax.associative_scan
# ---------------------------------------------------------------------------

def lru_combine(carry_a, carry_b):
    """
    Binary associative operator for the LRU parallel scan.

    Given two (h, λ) pairs, produces the combined pair as:
        h_new    = λ_b * h_a + h_b
        λ_new    = λ_b * λ_a

    This is associative because:
        ((h_c, λ_c) ∘ (h_b, λ_b)) ∘ (h_a, λ_a)
        = (λ_c*λ_b*h_a + λ_c*h_b + h_c, λ_c*λ_b*λ_a)
        = (h_c, λ_c) ∘ ((h_b, λ_b) ∘ (h_a, λ_a))  ✓

    Args:
        carry_a: (h_a: Array[D] complex, lambda_a: Array[D] complex)
        carry_b: (h_b: Array[D] complex, lambda_b: Array[D] complex)
    Returns:
        (h_new: Array[D] complex, lambda_new: Array[D] complex)
    """
    h_a, lambda_a = carry_a
    h_b, lambda_b = carry_b
    h_new = lambda_b * h_a + h_b
    lambda_new = lambda_b * lambda_a
    return (h_new, lambda_new)


# ---------------------------------------------------------------------------
# LRU forward pass
# ---------------------------------------------------------------------------

def lru_forward(x_sequence: Array, lambda_diag: Array, B: Array) -> Array:
    """
    Full LRU forward pass using parallel associative scan (O(log L)).

    Recurrence:  h_t = Λ h_{t-1} + B x_t
    where Λ = diag(λ_1, ..., λ_D) ∈ ℂ^{D×D} (stored as D-vector).

    The parallel form builds all h_t simultaneously via a binary
    reduction tree. For L=1024 tokens: ⌈log₂(1024)⌉ = 10 parallel steps
    instead of 1024 sequential steps.

    Args:
        x_sequence: Float[Array, "L D_in"]  — input sequence (real)
        lambda_diag: Complex[Array, "D"]     — complex diagonal eigenvalues
        B: Float[Array, "D D_in"]            — input projection matrix (real)

    Returns:
        Float[Array, "L D"] — real part of hidden state sequence
    """
    L = x_sequence.shape[0]

    # Project inputs: (L, D_in) → (L, D) — keep as float, then promote to complex64.
    # Cast B and x to float32 first.
    B32 = B.astype(jnp.float32)
    x32 = x_sequence.astype(jnp.float32)
    if B.ndim == 1:
        # Diagonal B: element-wise product
        h_init = (B32 * x32).astype(jnp.complex64)
    else:
        # Full dense B matrix: projection
        h_init = jnp.einsum("di,li->ld", B32, x32).astype(jnp.complex64)

    # Broadcast Λ across timesteps: (D,) → (L, D)
    # Explicitly cast to complex64 so both scan arrays have the same dtype.
    lambda_c64 = lambda_diag.astype(jnp.complex64)
    lambdas = jnp.broadcast_to(lambda_c64[None, :], (L, lambda_c64.shape[0]))
    # Make a concrete copy (broadcast_to produces a read-only view)
    lambdas = jnp.array(lambdas)

    # Pack as tuple of (h_elements, lambda_elements) — both (L, D) complex64
    elements = (h_init, lambdas)

    # Run parallel associative scan — O(log L) depth
    h_all, _ = jax.lax.associative_scan(lru_combine, elements, axis=0)

    # Return real part: (L, D) float32
    return jnp.real(h_all)


# ---------------------------------------------------------------------------
# LRU initialisation helpers
# ---------------------------------------------------------------------------

def init_lru_params(key: Array, D: int, D_in: int, r_min: float = 0.9, r_max: float = 0.999):
    """
    Initialise LRU parameters.

    Eigenvalue magnitudes are drawn from [r_min, r_max] to ensure
    the recurrence is stable (all |λ_k| < 1.0).
    Phases are drawn uniformly from [0, 2π).
    B is Xavier-initialised (scale 1/√D_in).

    Args:
        key: JAX PRNG key
        D: Hidden dimension (size of Λ diagonal)
        D_in: Input dimension (columns of B)
        r_min: Minimum eigenvalue magnitude (default 0.9)
        r_max: Maximum eigenvalue magnitude (default 0.999)

    Returns:
        dict: {
            "lambda_diag": Complex[Array, "D"],
            "B": Float[Array, "D D_in"]
        }
    """
    key_r, key_phi, key_B = jax.random.split(key, 3)

    # Stable eigenvalue magnitudes: r ∈ [r_min, r_max)
    r = jax.random.uniform(key_r, (D,), minval=r_min, maxval=r_max)
    # Ensure strictly less than 1.0 (belt-and-suspenders)
    r = jnp.clip(r, 0.0, 0.9999)

    # Random phases: φ ∈ [0, 2π)
    phi = jax.random.uniform(key_phi, (D,), minval=0.0, maxval=2.0 * jnp.pi)

    # Complex diagonal: λ_k = r_k * e^{iφ_k}
    lambda_diag = r * jnp.exp(1j * phi)

    # Xavier-like B initialisation
    B = jax.random.normal(key_B, (D, D_in)) / jnp.sqrt(float(D_in))

    return {"lambda_diag": lambda_diag, "B": B}


# ---------------------------------------------------------------------------
# Batched LRU (vmap over batch dimension)
# ---------------------------------------------------------------------------

def lru_forward_batched(x_batch: Array, lambda_diag: Array, B: Array) -> Array:
    """
    Batched LRU forward pass.

    Args:
        x_batch: Float[Array, "B L D_in"]
        lambda_diag: Complex[Array, "D"]
        B: Float[Array, "D D_in"]

    Returns:
        Float[Array, "B L D"]
    """
    return jax.vmap(lambda x: lru_forward(x, lambda_diag, B))(x_batch)
