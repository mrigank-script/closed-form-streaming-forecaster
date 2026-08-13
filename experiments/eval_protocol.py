"""Official-benchmark evaluation harness.

Implements the no-information-leakage protocol of the online forecasting
literature (OneNet / DSOF): DSOF-parity splits (20/5/75, classic ETT borders),
per-series normalization fit on the train segment only, and cumulative
MSE/MAE over every (origin, step) in the test segment. Features stream in
chunks so Traffic never materialises a (T, S, F) tensor.

Models: static (closed-form per-step ridge, a DLinear-class batch row) and
rls (per-step online RLS head per r: ridge warmup on train, then
update-on-now-observed-target -> predict-future).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)  # fp64 only for the tiny per-channel solve

from experiments import features as F


# GPU accumulation / scan primitives

@jax.jit
def _acc_train_cov(phi, y, A, B):
    """Accumulate A += sum phi^T phi and B += sum phi^T y over the chunk."""
    A = A + jnp.einsum("nsf,nsg->sfg", phi, phi)
    B = B + jnp.einsum("nsf,ns->sf", phi, y)
    return A, B


def _online_chunk(carry, xs):
    """One online step for all series at once (batched RLS, the S1 head)."""
    A_inv, W = carry                                   # (S,F,F),(S,F)
    phi_update, y_obs, phi_pred = xs                   # (S,F),(S,),(S,F)
    v = jnp.einsum("sfa,sa->sf", A_inv, phi_update)
    c = jnp.einsum("sf,sf->s", phi_update, v)
    k = v / (1.0 + c[:, None])
    A_inv = A_inv - jnp.einsum("sf,sg->sfg", k, v)
    pred_prev = jnp.einsum("sf,sf->s", phi_update, W)
    W = W + k * (y_obs - pred_prev)[:, None]
    pred = jnp.einsum("sf,sf->s", phi_pred, W)
    return (A_inv, W), pred


# Train-phase closed-form solve

def _static_head(X, tod, tr_end, r, seq_len, lam, chunk_t, H_lru=None):
    """Per-channel ridge weights for step r: phi(t) -> x[t+r].

    Train set = origins t in [seq_len-1, tr_end-r), i.e. targets still inside
    the train segment. `lam` is a RELATIVE ridge: the effective regulariser is
    lam * trace(A)/F per channel, so closed-form weights stay bounded along
    near-null directions of the Gram (96 correlated lags) instead of blowing
    up like an unregularised solve. H_lru optionally appends the S2 LRU
    context to the features.
    """
    S = X.shape[1]
    F_base = 6 + seq_len + F.eff_bins(seq_len) + 2
    F_lru = 0 if H_lru is None else H_lru.shape[2]
    F_all = F_base + F_lru
    A = jnp.zeros((S, F_all, F_all), dtype=jnp.float32)
    B = jnp.zeros((S, F_all), dtype=jnp.float32)
    t = seq_len - 1
    while t + r < tr_end:
        n = min(chunk_t, tr_end - r - t)
        phi = F.features_slice(X, tod, t, n, seq_len).astype(np.float32)
        if H_lru is not None:
            phi = F.with_lru_context(phi, H_lru, t, n)
        y = X[t + r: t + r + n].astype(np.float32)
        A, B = _acc_train_cov(jnp.asarray(phi), jnp.asarray(y), A, B)
        t += n
    # scale-invariant ridge: lambda = lam * trace(A)/F  per channel
    trace = jnp.trace(A, axis1=1, axis2=2) / F_all            # (S,)
    I = jnp.eye(F_all, dtype=jnp.float32)
    A = A + lam * trace[:, None, None] * I
    # solve in float64 for stability; weights back to float32 afterwards
    A64 = A.astype(jnp.float64)
    B64 = B.astype(jnp.float64)[..., None]
    W = jnp.linalg.solve(A64, B64)[..., 0].astype(jnp.float32)
    A_inv = jnp.linalg.inv(A64).astype(jnp.float32)
    return W, A_inv


# Full protocol

def evaluate(ds, pred_len: int, model: str = "static",
             lam: float = 1e-3, seq_len: int = 96,
             chunk_t: int = 512, chunk_online: int = 1536,
             return_by_step: bool = True,
             return_chunk_errs: bool = False,
             n_lru_modes: int = 0, lru_seed: int = 0) -> dict:
    """Run the protocol for (dataset, pred_len, model).

    model: "static" | "rls" | "s2" (static + LRU context) | "s2rls".

    With return_chunk_errs the dict also carries `chunk_errs` — a few hundred
    roughly-iid error blocks from the online sweep. For a deterministic
    closed-form head there is no weight-init randomness to average over, so
    block-bootstrap/jackknife over these chunks is the honest measure of
    sampling variability.
    """
    name, X = ds["name"], ds["X"]
    X = np.asarray(X, dtype=np.float64)
    T, S = X.shape
    tod = F.CADENCE[name]
    te_start = ds["borders"]["test"][0]
    tr_end = ds["borders"]["train"][1]
    lam = float(lam)

    use_lru = model in ("s2", "s2rls")
    if model in ("s2", "s2rls"):
        n_lru_modes = n_lru_modes or 8
    H_lru = F.lru_context(X, n_modes=n_lru_modes, seed=lru_seed) if use_lru else None

    is_online = model in ("rls", "s2rls")
    mse_sum = 0.0
    mae_sum = 0.0
    n_total = 0
    step_err = {}
    chunk_errs = []          # (sq_sum, ab_sum, n) per online chunk

    scan_jit = None
    if is_online:
        scan_jit = jax.jit(lambda c, xs: lax.scan(_online_chunk, c, xs))

    for r in range(1, pred_len + 1):
        W, A_inv = _static_head(X, tod, tr_end, r, seq_len, lam, chunk_t, H_lru)
        sq = 0.0
        ab = 0.0
        n = 0
        if is_online:
            state = (A_inv, W)

        t = te_start
        while t + r < T:
            nstep = min(chunk_online, T - r - t)
            if nstep <= 0:
                break
            if model == "static":
                phi = F.features_slice(X, tod, t, nstep, seq_len).astype(np.float32)
                y = X[t + r: t + r + nstep].astype(np.float32)
                pred = jnp.einsum("nsf,sf->ns", jnp.asarray(phi), W)
                pred = np.asarray(pred, dtype=np.float32)
                d_ = pred - y
            elif model == "s2":
                phi = F.features_slice(X, tod, t, nstep, seq_len).astype(np.float32)
                phi = F.with_lru_context(phi, H_lru, t, nstep)
                y = X[t + r: t + r + nstep].astype(np.float32)
                pred = jnp.einsum("nsf,sf->ns", jnp.asarray(phi), W)
                pred = np.asarray(pred, dtype=np.float32)
                d_ = pred - y
            else:
                # update head r on the target that just became observed:
                # features ending at (t_origin - r), target x[t_origin]
                phi_up = F.features_slice(X, tod, t - r, nstep, seq_len).astype(np.float64)
                # predict x[tau+r] using ONLY info up to origin tau
                phi_pr = F.features_slice(X, tod, t, nstep, seq_len).astype(np.float64)
                if use_lru:
                    phi_up = F.with_lru_context(phi_up, H_lru, t - r, nstep)
                    phi_pr = F.with_lru_context(phi_pr, H_lru, t, nstep)
                y_obs = X[t: t + nstep].astype(np.float64)
                y_true = X[t + r: t + r + nstep].astype(np.float64)
                state = (np.asarray(state[0], dtype=np.float64),
                         np.asarray(state[1], dtype=np.float64))
                state, pred = scan_jit(
                    state,
                    (jnp.asarray(phi_up), jnp.asarray(y_obs), jnp.asarray(phi_pr)),
                )
                state = (state[0].astype(np.float32), state[1].astype(np.float32))
                pred = np.asarray(pred, dtype=np.float64)
                d_ = pred - y_true
            sq += float(np.sum(d_ * d_))
            ab += float(np.sum(np.abs(d_)))
            n += d_.size
            if d_.size > 0:
                chunk_errs.append((float(np.sum(d_ * d_)),
                                   float(np.sum(np.abs(d_))),
                                   int(d_.size)))
            t += nstep

        mse_sum += sq
        mae_sum += ab
        n_total += n
        step_err[r] = (sq / max(n, 1), ab / max(n, 1), n)

    out = {
        "dataset": name, "pred_len": pred_len, "model": model,
        "mse": mse_sum / max(n_total, 1),
        "mae": mae_sum / max(n_total, 1),
        "n": n_total,
    }
    if return_by_step:
        out["by_step"] = {
            str(r): [float(step_err[r][0]), float(step_err[r][1]), int(step_err[r][2])]
            for r in step_err
        }
    if return_chunk_errs:
        out["chunk_errs"] = chunk_errs
    return out