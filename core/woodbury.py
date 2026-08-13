import jax
import jax.numpy as jnp
from jax import Array

def gram_schmidt(A: Array) -> Array:
    """
    Double re-orthogonalization Gram-Schmidt (CGS2) for machine-precision orthogonality.
    Bypasses Kaggle's missing cusolver_geqrf_ffi handler while ensuring ||Q^T Q - I|| = O(eps).
    """
    dim, r = A.shape
    def step(carry, i):
        Q = carry
        v = A[:, i]
        # First orthogonalization pass
        proj1 = jnp.dot(Q, jnp.dot(Q.T, v, precision="highest"), precision="highest")
        v1 = v - proj1
        # Second re-orthogonalization pass (CGS2) for machine-precision orthogonality
        proj2 = jnp.dot(Q, jnp.dot(Q.T, v1, precision="highest"), precision="highest")
        v_ortho = v1 - proj2
        v_norm = v_ortho / (jnp.linalg.norm(v_ortho) + 1e-12)
        return Q.at[:, i].set(v_norm), None
    Q_init = jnp.zeros((dim, r), dtype=A.dtype)
    Q_final, _ = jax.lax.scan(step, Q_init, jnp.arange(r))
    return Q_final

def block_woodbury_correction(A_inv_blocks: Array, W_local: Array, U: Array, V: Array, P: Array, Q: Array) -> Array:
    """
    Computes the block-diagonal portion of the Woodbury global correction.
    
    W_global = W_local - A_inv @ U @ (I + V^T A_inv U)^{-1} @ V^T @ W_local
               + (A_inv @ P - A_inv @ U @ (I + V^T A_inv U)^{-1} @ V^T @ A_inv @ P) @ Q^T
               
    To maintain the strict O(D) memory footprint, we extract only the 
    block-diagonal elements of the correction.
    """
    num_blocks, block_size, _ = A_inv_blocks.shape
    r = U.shape[1]
    r_B = P.shape[1]
    
    # Reshape U, V, P, Q into blocks
    U_b = U.reshape(num_blocks, block_size, r)
    V_b = V.reshape(num_blocks, block_size, r)
    P_b = P.reshape(num_blocks, block_size, r_B)
    Q_b = Q.reshape(num_blocks, block_size, r_B)
    
    # 1. Compute Z_b = A_inv_blocks @ U_b -> (num_blocks, block_size, r)
    Z_b = jnp.einsum('nbc,ncr->nbr', A_inv_blocks, U_b)
    
    # 2. Compute the core matrix C = I_r + V^T A_inv U
    V_T_Z = jnp.einsum('nbr,nbc->rc', V_b, Z_b) # (r, r)
    core = jnp.eye(r, dtype=jnp.float32) + V_T_Z
    
    # 3. Invert the small (r, r) core
    core_inv = jnp.linalg.inv(core) # C: (r, r)
    
    # 4. First term: Z C V^T W_local
    Q_w = jnp.einsum('nbr,nbc->rnc', V_b, W_local) # (r, num_blocks, block_size)
    K_b = jnp.einsum('rx,xnc->rnc', core_inv, Q_w) # C @ V^T W_local
    correction_1 = jnp.einsum('nbr,rnc->nbc', Z_b, K_b) # Z @ C @ V^T W_local
    
    # 5. Second term: (Z_P - Z C_P) Q^T
    # Z_P = A_inv_blocks @ P_b
    Z_P = jnp.einsum('nbc,ncx->nbx', A_inv_blocks, P_b) # (num_blocks, block_size, r_B)
    
    # C_P = C @ V^T @ Z_P
    V_T_Z_P = jnp.einsum('nbr,nbx->rx', V_b, Z_P) # (r, r_B)
    C_P = jnp.einsum('rx,xy->ry', core_inv, V_T_Z_P) # (r, r_B)
    
    # Z_C_P = Z_b @ C_P -> (num_blocks, block_size, r_B)
    Z_C_P = jnp.einsum('nbr,ry->nby', Z_b, C_P)
    
    # The term to multiply by Q^T is (Z_P - Z_C_P)
    Target_Term = Z_P - Z_C_P # (num_blocks, block_size, r_B)
    
    # Target_Term @ Q^T block diagonally
    # Q_b is (num_blocks, block_size, r_B). We need Target_Term @ Q_b^T per block
    # Target_Term is (n, b, y), Q_b is (n, c, y), output is (n, b, c)
    correction_2 = jnp.einsum('nby,ncy->nbc', Target_Term, Q_b)
    
    return W_local - correction_1 + correction_2


def streaming_gha(U: Array, V: Array, X: Array, Y: Array, mask: Array, lr: float = 1e-4) -> tuple[Array, Array]:
    """
    Generalized Hebbian Algorithm (Oja's Rule) to update U and V online.
    U tracks the principal components of X (auto-covariance).
    V tracks the principal components of Y (auto-covariance of target).
    
    This uses the stabilized_gha_step with geometric retraction.
    """
    X_flat = X.reshape(mask.size, -1)
    Y_flat = Y.reshape(mask.size, -1)
    mask_flat = mask.flatten()
    
    M = jnp.maximum(jnp.sum(mask_flat), 1.0)
    X_masked = X_flat * mask_flat[:, None]
    Y_masked = Y_flat * mask_flat[:, None]
    
    # Instead of explicitly forming the DxD covariance matrices (which takes O(D^2 N) = 68 GFLOPs per step),
    # we compute (X^T X / M) @ U optimally as X^T (X @ U) / M, which takes O(D N r) = 0.5 GFLOPs.
    XU = X_masked @ U
    CX_U = (X_masked.T @ XU) / M
    
    YV = Y_masked @ V
    CY_V = (Y_masked.T @ YV) / M
    
    # Stabilized GHA Step for U
    penalty_U = U @ (U.T @ CX_U)
    U_tilde = U + lr * (CX_U - penalty_U)
    U_new = gram_schmidt(U_tilde)
    
    # Stabilized GHA Step for V
    penalty_V = V @ (V.T @ CY_V)
    V_tilde = V + lr * (CY_V - penalty_V)
    V_new = gram_schmidt(V_tilde)
    
    return U_new, V_new

def streaming_svd_gha(P: Array, Q: Array, X: Array, Y: Array, mask: Array, lr: float = 1e-4) -> tuple[Array, Array]:
    """
    Coupled Oja's rule to track the SVD of the asymmetric cross-covariance B = X^T Y.
    P tracks the left singular vectors, Q tracks the right singular vectors.
    """
    X_flat = X.reshape(mask.size, -1)
    Y_flat = Y.reshape(mask.size, -1)
    mask_flat = mask.flatten()
    
    M = jnp.maximum(jnp.sum(mask_flat), 1.0)
    X_masked = X_flat * mask_flat[:, None]
    Y_masked = Y_flat * mask_flat[:, None]
    
    # B = (X^T Y) / M
    # Rather than forming DxD B, we compute B Q and B^T P directly
    # B Q = X^T (Y Q) / M
    Y_Q = Y_masked @ Q
    B_Q = (X_masked.T @ Y_Q) / M
    
    # B^T P = Y^T (X P) / M
    X_P = X_masked @ P
    B_T_P = (Y_masked.T @ X_P) / M
    
    # Stabilized SVD step for P
    penalty_P = P @ (P.T @ B_Q)
    P_tilde = P + lr * (B_Q - penalty_P)
    P_new = gram_schmidt(P_tilde)
    
    # Stabilized SVD step for Q
    penalty_Q = Q @ (Q.T @ B_T_P)
    Q_tilde = Q + lr * (B_T_P - penalty_Q)
    Q_new = gram_schmidt(Q_tilde)
    
    return P_new, Q_new


def woodbury_online_update(
    A_inv: Array,       # (D, D) inverse covariance matrix P
    W_E: Array,         # (D, 1) or (D,) energy weight vector
    W_P: Array,         # (D, 1) or (D,) pose quality weight vector
    phi: Array,         # (D,) feature vector
    y_E: float,         # target normalized energy
    y_P: float,         # target pose quality
    gamma: float = 1.0  # learning rate / discount factor
) -> tuple[Array, Array, Array]:
    """
    Online Recursive Least Squares (RLS) via Sherman-Morrison Woodbury rank-1 update.
    Simultaneously updates inverse covariance A_inv and both readout heads (W_E, W_P) in O(D^2) time.
    """
    v = A_inv @ phi               # (D,)
    c = jnp.dot(phi, v)           # scalar phi^T A_inv phi
    k = v / (1.0 + gamma * c)     # (D,) Kalman gain vector
    
    # Update inverse covariance matrix
    A_inv_new = A_inv - gamma * jnp.outer(k, v)
    
    # Prediction errors (residuals)
    pred_E = jnp.dot(phi, W_E.squeeze())
    pred_P = jnp.dot(phi, W_P.squeeze())
    err_E = y_E - pred_E
    err_P = y_P - pred_P
    
    # Update weight vectors
    W_E_new = W_E + (k * err_E).reshape(W_E.shape)
    W_P_new = W_P + (k * err_P).reshape(W_P.shape)
    
    return A_inv_new, W_E_new, W_P_new
