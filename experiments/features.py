"""experiments/features.py — per-channel sliding-window feature building.

For each series independently at each time t we build a small feature vector
(linear + stats + spectral + circadian), so the closed-form head is a small
per-channel ridge/RLS model — the classic streaming-forecasting decomposition
(Microprediction, statistical streamers) and the natural fit for the core
solver's block-diagonal head.

Cadences (periodicity for sine/cosine time-of-day features):
  electricity: hourly   (24)   traffic: hourly   (24)
  exchange:    daily    (7)    etth1/2: hourly   (24)
  ettm1:       15-min   (96)   weather: 10-min   (144)
"""

from functools import lru_cache
import warnings
import numpy as np
import jax
import jax.numpy as jnp

from core.lru import init_lru_params, lru_forward

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

CADENCE = {
    "electricity": 24,
    "traffic": 24,
    "exchange": 7,
    "etth1": 24,
    "etth2": 24,
    "ettm1": 96,
    "weather": 144,
}

DEFAULT_LOOKBACK = 24
FFT_BINS = 8


def eff_bins(lookback: int, bins: int = FFT_BINS) -> int:
    """Number of FFT bins that fit in a `lookback`-length window."""
    return min(bins, max(lookback - 1, 1))


@lru_cache(maxsize=8)
def day_phase_tri(t_mod: int, n: int) -> np.ndarray:
    """Per-timestep (n, 2) sine/cosine of the daily phase for the given cadence."""
    tt = np.arange(max(n, t_mod))
    phase = 2 * np.pi * (tt % t_mod) / t_mod
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1)[:n]


def _window_stats(X, lookback: int = DEFAULT_LOOKBACK):
    """Lagged stats computed causally with np.lib.stride_tricks.

    X: (T_c) 1-D series. Returns arrays of length T_c (NaN-padded at start).
    """
    L = lookback
    T = X.shape[0]
    pad = np.full((L,), np.nan, dtype=np.float64)
    xp = np.concatenate([pad[:-1], X])  # xp[t] aligned so window ends at t
    # build windows (T, L); window t uses x[t-L+1..t]
    w = np.lib.stride_tricks.sliding_window_view(xp, L)  # (T, L)
    mean = np.nanmean(w, axis=1)
    std = np.nanstd(w, axis=1)
    mini = np.nanmin(w, axis=1)
    maxi = np.nanmax(w, axis=1)
    d = np.diff(w, axis=1)                       # (T, L-1)
    ad = np.nanmean(np.abs(d), axis=1)           # volatility
    trend = np.gradient(w, axis=1)[:, -1]        # last-step slope approx
    return mean, std, mini, maxi, ad, trend


def _spectral_bins(X, lookback: int = DEFAULT_LOOKBACK, bins: int = FFT_BINS):
    """Normalised FFT magnitudes of the last `lookback` samples (causal), (T, b)."""
    L = lookback
    b = eff_bins(L, bins)
    T = X.shape[0]
    pad = np.full((L - 1,), np.nan, dtype=np.float64)
    xp = np.concatenate([pad, X])
    w = np.lib.stride_tricks.sliding_window_view(xp, L)   # (T, L)
    fft = np.fft.fft(np.nan_to_num(w), axis=1)
    mag = np.abs(fft[:, 1:b + 1])
    eps = np.sum(mag, axis=1, keepdims=True) + 1e-9
    return mag / eps                                    # scale-free spectral shape


def _feature_columns(x, todv, lookback: int, bins: int = FFT_BINS):
    """Per-series feature columns for a whole slice, causal.

    x:     (T_c,) 1-D series.
    todv:  (T_c, 2) per-row time-of-day phase (already sliced to the SAME rows).
    Returns (T_c, F_all) float64 with the full build_features column layout.
    """
    L = lookback
    b = eff_bins(L, bins)
    mean, std, mini, maxi, ad, trend = _window_stats(x, L)
    spec = _spectral_bins(x, L, bins)
    pad = np.full((L - 1,), np.nan, dtype=np.float64)
    xl = np.concatenate([pad, x])                       # aligned history window
    w = np.lib.stride_tricks.sliding_window_view(xl, L)  # (T_c, L)
    F_all = 6 + L + b + 2
    Fs = np.column_stack([mean, std, mini, maxi, ad, trend,
                          w, spec, todv])
    return np.asarray(Fs, dtype=np.float64).reshape(-1, F_all)


def build_features(X_all: np.ndarray, tod: int, lookback: int = DEFAULT_LOOKBACK):
    """Build per-channel feature matrix for a dataset.

    Args:
        X_all: (T, S) float array (series as columns).
        tod:   cadence period for time-of-day features.

    Returns:
        F: (T, S, F) float64 feature matrix (start rows NaN-flagged where
           the lookback window is incomplete).
        colnames: list[str]
    """
    T, S = X_all.shape
    F_all = 6 + lookback + eff_bins(lookback) + 2
    F = np.full((T, S, F_all), np.nan, dtype=np.float64)
    todv = day_phase_tri(tod, T)                    # (T, 2) -- extra rows -> slice

    for s in range(S):
        F[:, s, :] = _feature_columns(X_all[:, s], todv, lookback)

    colnames = (["mean", "std", "min", "max", "avgdiff", "trend"]
                + [f"x_lag{i}" for i in range(lookback)]
                + [f"fft{i}" for i in range(FFT_BINS)]
                + ["sin_tod", "cos_tod"])
    return F, colnames


def features_slice(X_all: np.ndarray, tod: int, t0: int, n: int,
                   lookback: int = DEFAULT_LOOKBACK):
    """Causal feature stream rows [t0, t0+n) for a dataset.

    Matches build_features exactly on the same rows, but computes only the
    minimal contiguous slice [t0-lookback+1, t0+n) so the full (T, S, F)
    tensor never materialises (required for Traffic: 17544 x 862 x 112).

    Args:
        X_all: (T, S) float array.
        tod:   cadence period.
        t0, n: build features for global rows t0 .. t0+n-1.
        lookback: window length.

    Returns:
        F: (n, S, F_all) float64 -- row r corresponds to global t0 + r.
    """
    L = lookback
    T, S = X_all.shape
    assert n >= 0 and 0 <= t0 and t0 + n <= T, f"slice {t0}:{t0+n} outside 0:{T}"
    start = t0 - L + 1
    end = t0 + n
    T_ctx = end - start
    xs = np.asarray(X_all[max(start, 0):end], dtype=np.float64)  # (T_have, S)
    tctx = xs.shape[0]
    if tctx < L:
        raise ValueError(
            f"features_slice needs at least L={L} context rows; got {tctx} "
            f"(t0={t0}, n={n}); callers must start at t0 >= L-1")

    # todv aligned to the GLOBAL rows [start, end); we then shift below.
    todv = day_phase_tri(tod, max(end, tod))[start:end]          # (T_ctx, 2)

    F_slice = np.empty((tctx, S, 6 + L + eff_bins(L) + 2), dtype=np.float64)
    for s in range(S):
        F_slice[:, s, :] = _feature_columns(xs[:, s], todv, L)

    # local row r corresponds to global t = start + r; the tool helpers put
    # row r's window END at global time start+r, so output rows for global
    # t in [t0, t0+n) are local rows [t0-start, t0-start+n) = [L-1, L-1+n).
    return F_slice[L - 1:L - 1 + n]


def causal_fill(X, y=None):
    """Forward-fill NaN rows generated by finite lookback (causality preserved)."""
    Xn = X.copy()
    idx = np.arange(Xn.shape[0])
    for s in range(Xn.shape[1]):
        good = np.flatnonzero(np.isfinite(Xn[:, s, 0]) if Xn.ndim == 3 else np.isfinite(Xn[:, s]))
        if len(good) == 0:
            continue
        first = good[0]
        if Xn.ndim == 3:
            Xn[:first, s] = Xn[first, s]
        else:
            Xn[:first, s] = Xn[first, s]
    return Xn


# ---------------------------------------------------------------------------
# S2: LRU reservoir-context features
# ---------------------------------------------------------------------------

def lru_context(X: np.ndarray, n_modes: int = 8, seed: int = 0,
                r_min: float = 0.5, r_max: float = 0.995) -> np.ndarray:
    """Fixed random LRU context (S2), causal per series, (T, S, 2*n_modes).

    Drives a bank of damped complex oscillators (the LRU diagonal) with the
    standardized series, then appends the real and imaginary parts of the
    hidden state as context features. The map is FIXED (no training): the
    closed-form ridge readout learns the mixing, exactly the next-gen-
    reservoir decomposition — this is what gives the head long-memory
    context beyond the 96-lag window.

    Causality: h_t = Λ h_{t-1} + B x_t uses only x_0..x_t, so states are
    leakage-free. Precompute once per dataset, then slice rows to match
    features_slice.
    """
    T, S = X.shape
    D = n_modes
    key = jax.random.PRNGKey(seed)
    p = init_lru_params(key, D, 1, r_min=r_min, r_max=r_max)
    lam = p["lambda_diag"].astype(jnp.complex64)
    B = p["B"].astype(jnp.float32)                      # (D, 1)

    def one_series(xs):                                 # (T,)
        x32 = xs.astype(jnp.float32)
        # h_t = Λ h_{t-1} + B x_t, B applied as flat (D,) vector
        Bh = x32[:, None] * B[:, 0][None, :]            # (T, D)
        Bh = Bh.astype(jnp.complex64)
        lam_t = lam.astype(jnp.complex64)                  # (D,)
        def step(c, xb):
            nc = c * lam_t + xb
            return nc, nc
        _, Hc = jax.lax.scan(step, jnp.zeros((D,), jnp.complex64), Bh)
        return Hc                                        # (T, D) complex

    Hc = jax.vmap(one_series, in_axes=1, out_axes=1)(jnp.asarray(X.astype(jnp.float32)))
    Hr = np.asarray(jnp.real(Hc), dtype=np.float32)
    Hi = np.asarray(jnp.imag(Hc), dtype=np.float32)
    return np.concatenate([Hr, Hi], axis=-1)            # (T, S, 2D)


def with_lru_context(F_slice: np.ndarray, H: np.ndarray, t0: int, n: int):
    """Append LRU context rows [t0, t0+n) to a feature slice (n, S, F)."""
    return np.concatenate([F_slice, H[t0:t0 + n].astype(np.float32)], axis=-1)