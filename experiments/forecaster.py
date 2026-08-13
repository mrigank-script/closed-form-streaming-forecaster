"""Closed-form streaming forecaster (S0/S1).

1. warmup: per-channel ridge fit on the train segment via the core solver
   (information_alignment.ia_regularise + symmetric-Cholesky solve).
2. online: at each live time t, fold the now-observed target pair
   (phi[t-h], X[t]) into a vectorised Woodbury RLS state (O(D^2)), then emit
   the h-step-ahead forecast phi[t] @ W -> X[t+h].

Both stages run on GPU; the online stage is a single lax.scan over (S, D)
states, chunked so the full (T, S, D) input never materialises on GPU.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import linalg as jsl

from core import information_alignment as ia


def ridge_warmup(Phi: jnp.ndarray, y: jnp.ndarray,
                 lam: float = 1e-3, gamma: float = 0.0, bs: int = 64):
    """Per-channel ridge fit: Phi (T, S, F), y (T, S) -> W (S, F), A_inv (S, F, F).

    A_IA = Phi^T Phi / M + (gamma+lam) I ; B = Phi^T y / M, via core.ia_regularise
    (gamma=0 recovers plain ridge; the machinery is shared with the full solver).
    """
    T = Phi.shape[0]
    y2 = jnp.where(jnp.isnan(y), 0.0, y)
    m = jnp.sum(jnp.isfinite(y), axis=0)          # (S,)
    M = jnp.maximum(m, 1.0)[None, :]

    A = jnp.einsum("tsf,tsg->sfg", Phi, Phi)          # (S, F, F)
    B = jnp.einsum("tsf,ts->sf", Phi, y2)             # (S, F)
    A_raw = A / M.T[:, None, None]
    B_raw = B / M.T[:, None]

    A_IA, _ = ia.ia_regularise(A_raw, jnp.zeros_like(B_raw[..., None]),
                               jnp.zeros_like(A_raw), gamma=gamma, lambda_reg=lam)
    # single-target ridge: solve A_IA W = B
    W = jsl.solve(A_IA, B_raw[..., None], assume_a="pos")[..., 0]   # (S, F)
    A_inv = jnp.linalg.inv(A_IA)
    return W, A_inv


def predict_static(Phi: jnp.ndarray, W: jnp.ndarray) -> jnp.ndarray:
    """S0: fixed-weight forecast. Phi (T, S, F) -> (T, S)."""
    return jnp.einsum("tsf,sf->ts", Phi, W)


def _online_step(carry, xs):
    A_inv, W = carry              # (S, F, F), (S, F)
    phi_prev, y_obs, phi_now = xs # (S, F), (S,), (S, F)
    # fold observed target pair (phi[t-h], X[t]) into RLS (gamma=1)
    v = jnp.einsum("sba,sa->sb", A_inv, phi_prev)
    c = jnp.einsum("sa,sa->s", phi_prev, v)
    k = v / (1.0 + c[:, None])
    A_inv = A_inv - jnp.einsum("sa,sb->sab", k, v)
    pred_prev = jnp.einsum("sa,sa->s", phi_prev, W)
    W = W + k * (y_obs - pred_prev)[:, None]
    # now forecast h-ahead using the just-updated weights
    pred = jnp.einsum("sa,sa->s", phi_now, W)
    return (A_inv, W), pred


def run_online(Phi_prev, y_obs, Phi_now, W0, A_inv0, chunk: int = 1024):
    """Vectorised RLS online forecasting.

    Phi_prev (L, S, F) = phi[t-h] observed inputs; y_obs (L, S) = X[t] targets;
    Phi_now (L, S, F) = phi[t] -> returns the h-step forecasts
    phi[t] @ W_after_update, scored in chunks.
    """
    state = (A_inv0, W0)

    def chunk_scan(carry, xs):
        return lax.scan(_online_step, carry, xs)

    chunk_scan = jax.jit(chunk_scan)
    preds = []
    L = Phi_prev.shape[0]
    for i in range(0, L, chunk):
        xs = (Phi_prev[i:i+chunk], y_obs[i:i+chunk], Phi_now[i:i+chunk])
        state, pred = chunk_scan(state, xs)
        preds.append(pred)
    return jnp.concatenate(preds, axis=0)


def pad_features(F: np.ndarray, bs: int = 64):
    """Pad the feature dim up to a multiple of bs (for block-diagonal fitting)."""
    T, S, F = F.shape
    Fp = -(-F // bs) * bs
    if Fp == F:
        return F.astype(np.float32), F
    out = np.zeros((T, S, Fp), dtype=np.float32)
    out[:, :, :F] = np.nan_to_num(F, nan=0.0)
    return out, F