"""
core/covariance.py — Block-Diagonal Covariance Accumulation

Accumulates the auto-covariance A and cross-covariance B matrices
across batches. These are the raw material for the Ridge solver.

Mathematical basis: §3.1, §3.2 of whitepaper.

A_IA = (1/M) Σ X^T X + (γ + λ) I    (accumulated auto-covariance)
B_IA = (1/M) Σ X^T Y + γ · G          (accumulated cross-covariance)

Key memory win: instead of caching (B, S, D) activations we accumulate
into (num_blocks, block_size, block_size) matrices and discard X, Y.
For D=4096, block_size=64: 64 × 64 × 64 × 4B = 1 MB per matrix.
"""

import jax
import jax.numpy as jnp
from jax import Array


# ---------------------------------------------------------------------------
# Helpers: reshape activations into blocks
# ---------------------------------------------------------------------------

def to_blocks(X: Array, block_size: int = 64) -> Array:
    """
    Reshape (..., D) → (..., num_blocks, block_size).

    D must be divisible by block_size.

    Args:
        X: Float[Array, "... D"]
        block_size: int

    Returns:
        Float[Array, "... num_blocks block_size"]
    """
    *leading, D = X.shape
    assert D % block_size == 0, f"D={D} not divisible by block_size={block_size}"
    num_blocks = D // block_size
    return X.reshape(*leading, num_blocks, block_size)


# ---------------------------------------------------------------------------
# Single-batch covariance accumulation
# ---------------------------------------------------------------------------

def accumulate_covariance(
    X: Array,          # Float["B S D"]
    Y: Array,          # Float["B S D"]
    mask: Array,       # Float["B S"]   (1=valid, 0=padded)
    A_acc: Array,      # Float["num_blocks bs bs"] — running auto-cov
    B_acc: Array,      # Float["num_blocks bs bs"] — running cross-cov
    block_size: int = 64,
    use_pallas_kernels: bool = False,
) -> tuple[Array, Array, int]:
    """
    Accumulate outer products from one batch into the covariance matrices.

    A_acc += Σ_{b,s} [mask_{b,s}] · X_block^T · X_block
    B_acc += Σ_{b,s} [mask_{b,s}] · X_block^T · Y_block

    (Division by M happens once in the solver, not here, to preserve
     integer-friendly accumulation semantics.)

    Args:
        X, Y:  Float[Array, "B S D"]
        mask:  Float[Array, "B S"]      binary (not bool)
        A_acc: Float[Array, "NB bs bs"] running accumulator
        B_acc: Float[Array, "NB bs bs"] running accumulator
        block_size: int
        use_pallas_kernels: bool

    Returns:
        (A_acc_new, B_acc_new, M_count)
        where M_count is the number of valid (unmasked) tokens.
    """
    # Reshape to blocks: (..., D) -> (..., NB, bs)
    X_b = to_blocks(X, block_size)
    Y_b = to_blocks(Y, block_size)
    
    num_blocks = X_b.shape[-2]

    # Flatten all leading dimensions into M_total
    M_total = X_b.size // (num_blocks * block_size)
    X_flat = X_b.reshape(M_total, num_blocks, block_size)
    Y_flat = Y_b.reshape(M_total, num_blocks, block_size)
    mask_flat = mask.reshape(M_total)

    pallas_enabled = use_pallas_kernels
        
    if pallas_enabled:
        # Guard the optional Pallas kernel path: hard-fail with a clear message
        # if kernels.covariance_pallas is not importable, rather than a cryptic
        # ModuleNotFoundError deep inside the expression stack.
        try:
            from kernels.covariance_pallas import pallas_accumulate_covariance
        except ImportError as e:
            raise ImportError(
                "use_pallas_kernels=True but kernels.covariance_pallas is not "
                "available. Install the kernels package, or set "
                "use_pallas_kernels=False to use the reference einsum path."
            ) from e
        # Pallas kernel handles the mask implicitly to prevent HBM round-trips for zero-padded regions
        A_delta, B_delta = pallas_accumulate_covariance(X_flat, Y_flat, mask_flat)
    else:
        A_delta = jnp.einsum("m,mnb,mnc->nbc", mask_flat, X_flat, X_flat, precision="highest")
        B_delta = jnp.einsum("m,mnb,mnc->nbc", mask_flat, X_flat, Y_flat, precision="highest")

    M_count = jnp.sum(mask)

    return A_acc + A_delta, B_acc + B_delta, M_count


# ---------------------------------------------------------------------------
# Full-epoch accumulator (reset → accumulate → finalise)
# ---------------------------------------------------------------------------

def init_accumulators(num_blocks: int, block_size: int) -> tuple[Array, Array]:
    """
    Create zero-initialised covariance accumulators.

    Returns:
        (A_acc, B_acc) both Float[Array, "num_blocks block_size block_size"]
    """
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
    """
    Convert raw accumulators into the Information-Alignment-augmented
    A_IA and B_IA matrices ready for the Ridge solver.

    A_IA = (1/M) A_acc + (γ + λ) I
    B_IA = (1/M) B_acc + γ · G

    Args:
        A_acc: Float[Array, "NB bs bs"] — raw auto-covariance sum
        B_acc: Float[Array, "NB bs bs"] — raw cross-covariance sum
        M_total: int — total valid token count this epoch
        G: Float[Array, "NB bs bs"] — backward projection matrix
        gamma: float — IA strength
        lambda_reg: float — ridge regularisation

    Returns:
        (A_IA, B_IA) both Float[Array, "NB bs bs"]
    """
    bs = A_acc.shape[-1]
    I = jnp.eye(bs, dtype=jnp.float32)     # (bs, bs)
    I_blocks = jnp.broadcast_to(I[None], A_acc.shape)  # (NB, bs, bs)

    M_float = jnp.maximum(M_total, 1.0)
    A_IA = A_acc / M_float + (gamma + lambda_reg) * I_blocks
    B_IA = B_acc / M_float + gamma * G

    return A_IA, B_IA
