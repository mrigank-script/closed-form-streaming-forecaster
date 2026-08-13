"""Algebraic-identity tests for the core solver.

Each test cross-checks a core module against an independent (dense NumPy /
manual) reference, valid on CPU and GPU (float32 tolerance).

Run:  python -m pytest tests/ -x -q
"""

import numpy as np
import jax
import jax.numpy as jnp

from core import covariance, ridge_solver, information_alignment, hopfield, lru, woodbury
from core.hopfield import _D4_FORWARD, _D4_INVERSE


def _rng(seed):
    return {"key": jax.random.PRNGKey(seed)}


def test_block_ridge_solve_is_exact():
    nb, bs = 8, 16
    A = jax.random.normal(jax.random.PRNGKey(0), (nb, bs, bs))
    A = A @ jnp.swapaxes(A, -1, -2) + bs * jnp.eye(bs)  # PD block
    B = jax.random.normal(jax.random.PRNGKey(1), (nb, bs, bs))
    W_T = ridge_solver.block_ridge_solve(A, B)
    # strongest identity: Cholesky (pos) solve == dense LU solve
    W_ref = jnp.linalg.solve(A, B)
    assert float(jnp.max(jnp.abs(W_T - W_ref))) < 1e-3
    # sanity: A @ W_T ~ B at fp32 tolerance
    resid = jnp.max(jnp.abs(jnp.einsum("nbc,ncd->nbd", A, W_T) - B))
    assert float(resid) < 1e-2


def test_ia_regulise_preserves_pd():
    nb, bs = 4, 8
    A_raw = jax.random.normal(jax.random.PRNGKey(0), (nb, bs, bs))
    A_raw = A_raw @ jnp.swapaxes(A_raw, -1, -2)  # PSD
    B_raw = jax.random.normal(jax.random.PRNGKey(1), (nb, bs, bs))
    G = jax.random.normal(jax.random.PRNGKey(2), (nb, bs, bs))
    A_IA, _ = information_alignment.ia_regularise(A_raw, B_raw, G, gamma=0.01, lambda_reg=0.01)
    pd = information_alignment.check_positive_definite(A_IA)
    assert bool(jnp.all(pd))


def test_rls_matches_batch_ridge():
    """Online Woodbury RLS with gamma=1 must equal the batch ridge solution."""
    D, N = 16, 30
    X = jax.random.normal(jax.random.PRNGKey(0), (N, D))
    y = jax.random.normal(jax.random.PRNGKey(1), (N, 1))
    lam = 1.0

    W = jnp.zeros((D, 1))
    A_inv = jnp.eye(D)
    for i in range(N):
        A_inv, W, _ = woodbury.woodbury_online_update(
            A_inv, W, jnp.zeros_like(W), X[i], float(y[i, 0]), float(y[i, 0]), gamma=1.0
        )

    A_b = X.T @ X + lam * jnp.eye(D)
    W_b = jnp.linalg.solve(A_b, X.T @ y)
    assert float(jnp.max(jnp.abs(W - W_b))) < 1e-3


def test_woodbury_inv_update_identity():
    """Sherman-Morrison: A_inv after one update == inv(A + phi phi^T)."""
    D = 8
    A = jax.random.normal(jax.random.PRNGKey(0), (D, D))
    A = A @ A.T + D * jnp.eye(D)
    A_inv = jnp.linalg.inv(A)
    phi = jax.random.normal(jax.random.PRNGKey(1), (D,))
    A_inv_new, _, _ = woodbury.woodbury_online_update(
        A_inv, jnp.zeros((D, 1)), jnp.zeros((D, 1)), phi, 1.0, 1.0, gamma=1.0
    )
    ref = jnp.linalg.inv(A + jnp.outer(phi, phi))
    assert float(jnp.max(jnp.abs(A_inv_new - ref))) < 1e-3


def test_gram_schmidt_orthonormal():
    A = jax.random.normal(jax.random.PRNGKey(0), (20, 6))
    Q = woodbury.gram_schmidt(A)
    M = Q.T @ Q
    off = jnp.max(jnp.abs(M - jnp.eye(6)))
    assert float(off) < 1e-3


def _ref_gha_step(U, X_np, lr):
    """Dense NumPy reference for the stabilized GHA update (masked data)."""
    M = X_np.shape[0]
    CXU = (X_np.T @ (X_np @ U)) / M
    U = U + lr * (CXU - U @ (U.T @ CXU))
    # modified Gram-Schmidt w/ re-orthogonalization, matching core.woodbury
    Q = np.zeros_like(U)
    for j in range(U.shape[1]):
        v = U[:, j].copy()
        for _ in range(2):
            for k in range(j):
                v = v - (Q[:, k] @ v) * Q[:, k]
        Q[:, j] = v / (np.linalg.norm(v) + 1e-12)
    return Q


def test_streaming_gha_matches_numpy_reference():
    """GHA converges to the top-r PCA subspace, matching a dense NumPy reference."""
    D, r, N, steps = 12, 3, 200, 40
    key = jax.random.PRNGKey(3)
    # Rank-signal data: strong low-rank core + smaller isotropic noise
    core = jax.random.normal(key, (D, r)) @ jax.random.normal(jax.random.PRNGKey(4), (r, N))
    noise = 0.05 * jax.random.normal(jax.random.PRNGKey(5), (D, N))
    X = (core + noise).T        # (N, D)
    X_np = np.asarray(X)
    U = jax.random.normal(jax.random.PRNGKey(6), (D, r)) / jnp.sqrt(D)

    U_ref = np.array(U)
    mask = jnp.ones((N, 1))
    lr = 1e-4 * 8
    for _ in range(steps):
        U, V = woodbury.streaming_gha(U, jnp.zeros((D, r)), X, jnp.zeros((N, D)), mask, lr=lr)
        U_ref = _ref_gha_step(U_ref, X_np, lr)
    U = np.asarray(U)
    mask2 = np.array(mask)
    Xm = X_np * mask2

    assert np.max(np.abs(U.T @ U - np.eye(r))) < 1e-3            # orthonormality
    # both should capture ~identical total variance of the top-r subspace
    U_ref_orth = U_ref / (np.linalg.norm(U_ref, axis=0, keepdims=True) + 1e-12)
    e_jax = np.linalg.norm(Xm @ U)
    e_ref = np.linalg.norm(Xm @ U_ref_orth)
    assert abs(e_jax - e_ref) / max(e_ref, 1e-9) < 1e-2
    # subspace alignment: columns of U should span the same space as U_ref
    proj = U.T @ U_ref_orth
    assert float(np.min(np.linalg.svd(proj, compute_uv=False)) > 0.999)


def test_lru_matches_sequential_recurrence():
    L, D_in, D = 8, 4, 6
    x = jax.random.normal(jax.random.PRNGKey(0), (L, D_in))
    phi = jax.random.uniform(jax.random.PRNGKey(1), (D,), minval=0.0, maxval=2 * jnp.pi)
    r = 0.95 * jnp.ones((D,))
    lambda_diag = r * jnp.exp(1j * phi)
    B = jax.random.normal(jax.random.PRNGKey(2), (D, D_in)) / jnp.sqrt(D_in)

    h_all = lru.lru_forward(x, lambda_diag, B)  # real part, (L, D)

    h = jnp.zeros(D, dtype=jnp.complex64)
    seq = []
    for t in range(L):
        h = lambda_diag * h + (B @ x[t]).astype(jnp.complex64)
        seq.append(jnp.real(h))
    seq = jnp.stack(seq, axis=0)
    assert float(jnp.max(jnp.abs(h_all - seq))) < 1e-4


def test_hopfield_single_step_matches_reference():
    D, K, S = 12, 5, 4
    R = jax.random.normal(jax.random.PRNGKey(0), (S, D))
    MK = jax.random.normal(jax.random.PRNGKey(1), (K, D)) / jnp.sqrt(D)
    MV = jax.random.normal(jax.random.PRNGKey(2), (K, D)) / jnp.sqrt(D)
    beta = 0.1
    R1 = hopfield.hopfield_refinement(R, MK, MV, beta=beta, H=1)
    scores = beta * (R @ MK.T)
    ref = jax.nn.softmax(scores, axis=-1) @ MV
    assert float(jnp.max(jnp.abs(R1 - ref))) < 1e-4


def test_d4_inverse_pairs_recover_input():
    x = jax.random.normal(jax.random.PRNGKey(0), (6, 6, 3))  # square H=W (D4 requires it)
    for T, T_inv in zip(_D4_FORWARD, _D4_INVERSE):
        y = T_inv(T(x))
        assert float(jnp.max(jnp.abs(y - x))) < 1e-5


def test_covariance_accumulation_matches_reference():
    B, S, D, bs = 2, 4, 16, 8
    X = jax.random.normal(jax.random.PRNGKey(0), (B, S, D))
    Y = jax.random.normal(jax.random.PRNGKey(1), (B, S, D))
    mask = jnp.array([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    A_acc, B_acc = covariance.init_accumulators(D // bs, bs)
    A_acc, B_acc, M = covariance.accumulate_covariance(X, Y, mask, A_acc, B_acc, bs)

    nb = D // bs
    X_b = covariance.to_blocks(X, bs).reshape(B * S, nb, bs)
    Y_b = covariance.to_blocks(Y, bs).reshape(B * S, nb, bs)
    m = mask.reshape(B * S)
    A_ref = jnp.einsum("m,mnb,mnc->nbc", m, X_b, X_b, precision="highest")
    B_ref = jnp.einsum("m,mnb,mnc->nbc", m, X_b, Y_b, precision="highest")
    assert float(jnp.max(jnp.abs(A_acc - A_ref))) < 1e-3
    assert float(jnp.max(jnp.abs(B_acc - B_ref))) < 1e-3
    assert int(M) == int(jnp.sum(mask))


def test_block_matmul_roundtrip():
    nb, bs = 4, 8
    D = nb * bs
    X = jax.random.normal(jax.random.PRNGKey(0), (6, D))
    W_T = jax.random.normal(jax.random.PRNGKey(1), (nb, bs, bs)) * 0.1
    out = ridge_solver.block_matmul(X, W_T)
    X_b = X.reshape(6, nb, bs)
    ref = jnp.einsum("...nb,...nbc->...nc", X_b, W_T).reshape(6, D)
    assert float(jnp.max(jnp.abs(out - ref))) < 1e-4


def test_backward_projection_shape_symmetry():
    B, S, D = 2, 4, 16
    bs = 8
    X = jax.random.normal(jax.random.PRNGKey(0), (B, S, D))
    Y = jax.random.normal(jax.random.PRNGKey(1), (B, S, D))
    mask = jnp.ones((B, S))
    G_T = ridge_solver.compute_backward_projection(Y, X, mask, block_size=bs)
    assert G_T.shape == (D // bs, bs, bs)
    assert bool(jnp.all(jnp.isfinite(G_T)))


def test_features_slice_matches_full_build():
    """Chunked causal feature stream must equal the full build on same rows."""
    from experiments import features as F

    T, S, L = 40, 3, 8          # small enough to build the full tensor
    rng = np.random.RandomState(0)
    X = rng.randn(T, S).astype(np.float64)
    tod = 6
    Phi, _ = F.build_features(X, tod, lookback=L)

    t0, n = 10, 20              # rows [10, 30)
    Phi_slice = F.features_slice(X, tod, t0, n, lookback=L)
    eb = F.eff_bins(L)
    assert Phi_slice.shape == (n, S, 6 + L + eb + 2)
    ref = Phi[t0:t0 + n]
    got = Phi_slice.astype(np.float64)
    assert np.max(np.abs(got - ref)) < 1e-12, np.max(np.abs(got - ref))


def test_static_head_matches_dense_least_squares():
    """Step-r closed-form ridge predicts like dense per-channel LS."""
    from experiments import features as F
    from experiments.eval_protocol import _static_head

    T, S, L, r, lam = 300, 2, 8, 3, 1e-2
    rng = np.random.RandomState(1)
    X = rng.randn(T, S).astype(np.float64)
    tod = 6
    tr_end = 240
    W, _ = _static_head(X, tod, tr_end, r, L, lam, chunk_t=16)

    Phi, _ = F.build_features(X, tod, lookback=L)
    # held-out origins (test-region like)
    t = np.arange(250, T - r)
    Phi_t = Phi[t].astype(np.float64)          # (n, S, F)
    y_t = X[t + r]                             # (n, S)
    for s in range(S):
        tt = np.arange(L - 1, tr_end - r)
        A2 = Phi[tt, s].T @ Phi[tt, s]
        B2 = Phi[tt, s].T @ X[tt + r, s]
        # scale-invariant ridge as in eval_protocol._static_head
        A2 = A2 + lam * (np.trace(A2) / Phi.shape[-1]) * np.eye(Phi.shape[-1])
        w_dense = np.linalg.solve(A2, B2)
        pred_jax = np.asarray(W[s]) @ Phi_t[:, s].T
        pred_dense = w_dense @ Phi_t[:, s].T
        scale = np.max(np.abs(pred_dense)) + 1e-9
        assert np.max(np.abs(pred_jax - pred_dense)) / scale < 1e-2, (
            np.max(np.abs(pred_jax - pred_dense)) / scale)