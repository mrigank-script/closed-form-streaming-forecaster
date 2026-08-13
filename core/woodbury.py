"""Online RLS / subspace tracking: Gram-Schmidt, Woodbury correction, GHA.

The Woodbury update keeps the whole thing O(D^2) per step and the memory
footprint O(D) by only materialising the block-diagonal correction.
"""

import jax
import jax.numpy as jnp
from jax import Array


def gram_schmidt(A: Array) -> Array:
    """CGS2 (double re-orthogonalization) without cusolver_geqrf, so Q stays
    orthonormal to machine precision."""
    dim, r = A.shape
    def step(carry, i):
        Q = carry
        v = A[:, i]
        proj1 = jnp.dot(Q, jnp.dot(Q.T, v, precision="highest"), precision="highest")
        v1 = v - proj1
        proj2 = jnp.dot(Q, jnp.dot(Q.T, v1, precision="highest"), precision="highest")
        v_ortho = v1 - proj2
        v_norm = v_ortho / (jnp.linalg.norm(v_ortho) + 1e-12)
        return Q.at[:, i].set(v_norm), None
    Q_init = jnp.zeros((dim, r), dtype=A.dtype)
    Q_final, _ = jax.lax.scan(step, Q_init, jnp.arange(r))
    return Q_final


def block_woodbury_correction(A_inv_blocks: Array, W_local: Array, U: Array, V: Array, P: Array, Q: Array) -> Array:
    """Block-diagonal slice of the Woodbury global correction:
    W = W_local - Z C V^T W_local + (Z_P - Z C_P) Q^T, with the (r,r) core
    C = I + V^T A_inv U inverted exactly and per-block terms kept sparse."""
    num_blocks, block_size, _ = A_inv_blocks.shape
    r = U.shape[1]
    r_B = P.shape[1]

    U_b = U.reshape(num_blocks, block_size, r)
    V_b = V.reshape(num_blocks, block_size, r)
    P_b = P.reshape(num_blocks, block_size, r_B)
    Q_b = Q.reshape(num_blocks, block_size, r_B)

    Z_b = jnp.einsum('nbc,ncr->nbr', A_inv_blocks, U_b)
    V_T_Z = jnp.einsum('nbr,nbc->rc', V_b, Z_b)
    core = jnp.eye(r, dtype=jnp.float32) + V_T_Z
    core_inv = jnp.linalg.inv(core)

    Q_w = jnp.einsum('nbr,nbc->rnc', V_b, W_local)
    K_b = jnp.einsum('rx,xnc->rnc', core_inv, Q_w)
    correction_1 = jnp.einsum('nbr,rnc->nbc', Z_b, K_b)

    Z_P = jnp.einsum('nbc,ncx->nbx', A_inv_blocks, P_b)
    V_T_Z_P = jnp.einsum('nbr,nbx->rx', V_b, Z_P)
    C_P = jnp.einsum('rx,xy->ry', core_inv, V_T_Z_P)
    Z_C_P = jnp.einsum('nbr,ry->nby', Z_b, C_P)
    correction_2 = jnp.einsum('nby,ncy->nbc', Z_P - Z_C_P, Q_b)

    return W_local - correction_1 + correction_2


def streaming_gha(U: Array, V: Array, X: Array, Y: Array, mask: Array, lr: float = 1e-4) -> tuple[Array, Array]:
    """Online generalized Hebbian (Oja) update tracking the principal
    subspaces of X (in U) and Y (in V), with CGS2 re-orthogonalisation.
    Uses X^T (X U)/M so the D×D covariance is never formed."""
    X_flat = X.reshape(mask.size, -1)
    Y_flat = Y.reshape(mask.size, -1)
    mask_flat = mask.flatten()

    M = jnp.maximum(jnp.sum(mask_flat), 1.0)
    X_masked = X_flat * mask_flat[:, None]
    Y_masked = Y_flat * mask_flat[:, None]

    XU = X_masked @ U
    CX_U = (X_masked.T @ XU) / M

    YV = Y_masked @ V
    CY_V = (Y_masked.T @ YV) / M

    penalty_U = U @ (U.T @ CX_U)
    U_new = gram_schmidt(U + lr * (CX_U - penalty_U))

    penalty_V = V @ (V.T @ CY_V)
    V_new = gram_schmidt(V + lr * (CY_V - penalty_V))

    return U_new, V_new


def streaming_svd_gha(P: Array, Q: Array, X: Array, Y: Array, mask: Array, lr: float = 1e-4) -> tuple[Array, Array]:
    """Coupled Oja update tracking the left/right singular subspaces of the
    cross-covariance B = X^T Y, again without forming B."""
    X_flat = X.reshape(mask.size, -1)
    Y_flat = Y.reshape(mask.size, -1)
    mask_flat = mask.flatten()

    M = jnp.maximum(jnp.sum(mask_flat), 1.0)
    X_masked = X_flat * mask_flat[:, None]
    Y_masked = Y_flat * mask_flat[:, None]

    Y_Q = Y_masked @ Q
    B_Q = (X_masked.T @ Y_Q) / M

    X_P = X_masked @ P
    B_T_P = (Y_masked.T @ X_P) / M

    penalty_P = P @ (P.T @ B_Q)
    P_new = gram_schmidt(P + lr * (B_Q - penalty_P))

    penalty_Q = Q @ (Q.T @ B_T_P)
    Q_new = gram_schmidt(Q + lr * (B_T_P - penalty_Q))

    return P_new, Q_new


def woodbury_online_update(
    A_inv: Array,
    W_E: Array,
    W_P: Array,
    phi: Array,
    y_E: float,
    y_P: float,
    gamma: float = 1.0
) -> tuple[Array, Array, Array]:
    """Online RLS via the Sherman-Morrison rank-1 update: refreshes A_inv and
    both readouts (W_E, W_P) in O(D^2) per timestep."""
    v = A_inv @ phi
    c = jnp.dot(phi, v)
    k = v / (1.0 + gamma * c)

    A_inv_new = A_inv - gamma * jnp.outer(k, v)

    pred_E = jnp.dot(phi, W_E.squeeze())
    pred_P = jnp.dot(phi, W_P.squeeze())
    err_E = y_E - pred_E
    err_P = y_P - pred_P

    W_E_new = W_E + (k * err_E).reshape(W_E.shape)
    W_P_new = W_P + (k * err_P).reshape(W_P.shape)

    return A_inv_new, W_E_new, W_P_new