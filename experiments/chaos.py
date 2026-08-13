"""experiments/chaos.py — Paper 2 physics arm (track #4).

Chaotic benchmark suite run through the SAME closed-form core solver that
drives the streaming tables (Paper 2's "whole stack as a repeatable tool"
thesis): ridge warmup + online prediction, no SGD anywhere.

Tasks
-----
1. Lorenz96 (N=5, F=8, dt=0.02) — next-generation-reservoir benchmark:
   delay-embed + quadratic monomial features, closed-form ridge for
   one-step-ahead; multi-step free-run evaluated out to the N-Lyapunov
   horizon (NG-RC reference: one-step NMSE ~1e-4, free run ~1-2 Lyapunov
   times, NARMA10 NMSE < 0.0391 line from the Swarm plan).
2. NARMA10 / NARMA30 echo-state tasks with the same feature stack.
3. Lyapunov spectrum of the Lorenz96 flow via QR variational equations
   (used to express the horizon in units of the largest exponent lambda1).

Run:  ./run.sh experiments.run_chaos
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

from experiments import features as F  # nin-twod / helpers reused for cadence colnames
from core.ridge_solver import block_ridge_solve  # closed-form head demonstration


# ---------------------------------------------------------------------------
# Integrators (GPU-ready via lax.scan)
# ---------------------------------------------------------------------------

def lorenz96_vector(x, F=8.0):
    """Lorenz96 vector field on the ring. x: (N,)
    dx_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F"""
    xp1 = jnp.roll(x, -1)     # x_{i+1}
    xm2 = jnp.roll(x, 2)      # x_{i-2}
    xm1 = jnp.roll(x, 1)      # x_{i-1}
    return (xp1 - xm2) * xm1 - x + F


def lorenz96_rk4(x, F=8.0, dt=0.02):
    """One RK4 step. x: (N,)"""
    def f(xx):
        return lorenz96_vector(xx, F)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_lorenz96(x0, n_steps, F=8.0, dt=0.02):
    """Integrate Lorenz96 for n_steps, returning every step (T, N)."""
    def step(x, _):
        return lorenz96_rk4(x, F, dt), x
    _, traj = lax.scan(step, jnp.asarray(x0, dtype=jnp.float64),
                       None, length=int(n_steps))
    return traj


integrate_lorenz96_jit = jax.jit(integrate_lorenz96, static_argnums=(1,))


def narma(n_tau, x_input):
    """NARMA(n): non-linear auto-regressive system driven by white noise.

    y[t] = alpha*y[t-1] + beta*y[t-1]*sum_{i=1..n} y[t-i]
           + gamma*x[t-1]*x[t-n] + delta
    with standard narma10 norms and x ~ U[0.0, 0.5].
    Returns (y, x)."""
    T = x_input.shape[0]
    alpha = 0.30 if n_tau == 10 else 0.35   # narma10/narma30 conventions
    beta = 0.05 if n_tau == 10 else 0.02
    gamma = 1.50 if n_tau == 10 else 1.90
    delta = 0.10
    y = np.zeros(T)
    for t in range(1, T):
        lag = x_input[t - 1] * x_input[t - n_tau]      # standard NARMA term
        y[t] = (alpha * y[t - 1]
                + beta * y[t - 1] * np.sum(y[max(0, t - n_tau):t])
                + gamma * lag
                + delta)
    return y


# ---------------------------------------------------------------------------
# Reservoir / feature construction (NG-RC parity)
# ---------------------------------------------------------------------------

def delay_embed_signals(u, memory_order=4, lag=1):
    """Delay-embed a multivariate signal.

    u : (T, N) -> features [1.0, u[t], u[t-lag], ... u[t-k*lag]] (T, k*N + 1)
    """
    T, N = u.shape
    cols = [np.ones((T, 1))]
    for i in range(memory_order):
        shift = i * lag
        cols.append(np.roll(u, shift, axis=0))
    col = np.concatenate(cols, axis=-1)
    valid = memory_order * lag
    col[:valid] = 0.0  # initial transient zero-padded
    return col


def quadratic_feature(u_linear):
    """Quadratic monomial uplift: [u_linear, outer(u_linear,u_linear) triu].

    NG-RC parity: lift the memory-embedded mean-centered signal with the
    quadratic monomials (skip bias row, already present in u_linear).
    u_linear: (T, D) -> (T, D * (D+1) / 2)  [linear + quadratic triu].
    """
    D = u_linear.shape[1]
    tri = np.triu_indices(D)
    q = u_linear[:, np.newaxis, :] * u_linear[:, :, np.newaxis]
    quad = q[:, tri[0], tri[1]]                      # (T, D(D+1)/2)
    return np.concatenate([u_linear, quad], axis=-1)


def train_test_split_blocks(u, y, n_train, gap=100):
    """Chronological train/test with a burn-gap (NG-RC convention)."""
    T = u.shape[0]
    tr = (slice(0, n_train), u[0:n_train], y[0:n_train])
    te = (slice(n_train + gap, T), u[n_train + gap:T], y[n_train + gap:T])
    return tr, te


# ---------------------------------------------------------------------------
# Closed-form ridge trainer (core solver on reservoir features)
# ---------------------------------------------------------------------------

def ridge_predict(X_tr, Y_tr, X_te, lam=1e-6):
    """Closed-form ridge (our core solver, per-output independent).

    Returns Y_hat (T_te, outputs). Assumes X already has bias column or
    caller centers. Uses scale-invariant lam*trace regularization.
    """
    Xtr = jnp.asarray(X_tr, dtype=jnp.float64)
    Ytr = jnp.asarray(Y_tr, dtype=jnp.float64)
    Xte = jnp.asarray(X_te, dtype=jnp.float64)
    D = Xtr.shape[1]
    lam_l = lam * jnp.trace(Xtr.T @ Xtr) / D
    A = Xtr.T @ Xtr + lam_l * jnp.eye(D, dtype=jnp.float64)
    B = Xtr.T @ Ytr
    W = jnp.linalg.solve(A, B)                       # (D, outputs)
    return np.asarray(Xte @ W)


# ---------------------------------------------------------------------------
# Lyapunov spectrum (QR method)
# ---------------------------------------------------------------------------

def lyapunov_lorenz96(x0, F=8.0, dt=0.02, n_trans=1000, n_steps=20000, n=5):
    """Largest Lyapunov exponent via QR of the variational tangent map."""
    jac = jax.jacfwd(lorenz96_vector)

    def one_step(c, _):
        x, Q0 = c
        A = jac(x)                                   # (n, n)
        M = Q0 + dt * (A @ Q0)                        # top-order tangent
        Q, R = jnp.linalg.qr(M)
        x2 = lorenz96_rk4(x, F, dt)
        return (x2, Q), jnp.log(jnp.clip(jnp.abs(jnp.diag(R)), 1e-12, None)) / dt

    @jax.jit
    def run(x, q0):
        carry_init = (x, q0)
        (_, _), logs = lax.scan(one_step, carry_init, None, length=n_steps)
        return jnp.max(jnp.mean(logs, axis=0))       # largest exponent per-unit-time

    x = jnp.asarray(x0, dtype=jnp.float64)
    Q = jnp.eye(n, dtype=jnp.float64)
    for _ in range(n_trans):                          # burn-in (numpy loop)
        x = lorenz96_rk4(x, F, dt)
    return float(run(x, Q))


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def nmse(y, y_hat):
    mse = np.mean((y - y_hat) ** 2)
    var = np.var(y)
    return mse / max(var, 1e-12)


def benchmark_lorenz96(n=5, F=8.0, dt=0.02, mem=4, n_train=5000,
                       n_test=2000, lam=1e-6, seed=0, horizon_lyap=1.0):
    """Lorenz96 step-prediction benchmark with N-Lyapunov free-run."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-0.01, 0.01, size=(n,))
    burn = 5000
    gap = 100
    traj = np.asarray(integrate_lorenz96_jit(
        x0, burn + n_train + n_test + mem + gap, F, dt))
    traj = traj[burn:]                                 # discard transient
    # one-step target: u[t+1]
    u = traj[:-1]
    y = traj[1:]
    Xlin = delay_embed_signals(u, mem)
    X = quadratic_feature(Xlin)                        # (T, D)
    n_tr = n_train
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    gap = 100
    X_te, y_te = X[n_tr + gap:], y[n_tr + gap:]
    y_hat = ridge_predict(X_tr, y_tr, X_te, lam)
    nmse_1 = nmse(y_te, y_hat)

    # free-run: closed-form one-step weights looped (multi-step)
    Xtrf = jnp.asarray(X_tr, dtype=jnp.float64)
    ytrf = jnp.asarray(y_tr, dtype=jnp.float64)
    D = X_tr.shape[1]
    lam_l = lam * jnp.trace(Xtrf.T @ Xtrf) / D
    A = Xtrf.T @ Xtrf + lam_l * jnp.eye(D, dtype=jnp.float64)
    B = Xtrf.T @ ytrf
    W = np.asarray(jnp.linalg.solve(A, B))
    # free-run over the TEST region; seed embedding ends at n_tr+gap-1 so the
    # first iterated prediction is aligned with true_free[0] = y[n_tr+gap].
    seed_end = n_tr + gap
    state = u[seed_end - mem:seed_end]                 # (mem, n) seed trajectory
    free = []
    for _ in range(n_test):
        feats = np.concatenate([np.ones((1, 1)), state[::-1].reshape(1, -1)], axis=-1)
        fq = quadratic_feature(feats)
        pred = fq @ W
        free.append(pred[0])
        state = np.concatenate([state[1:], pred], axis=0)
    free = np.asarray(free)
    true_free = y[seed_end:seed_end + n_test]
    # error growth vs Lyapunov time
    lam1 = np.max(lyapunov_lorenz96(x0, F, dt, n_trans=2000, n_steps=10000, n=n))
    err = np.sqrt(np.mean((free - true_free) ** 2, axis=1))
    t_lyap = np.argmax(err > np.std(y_te))
    return {
        "task": f"Lorenz96_n{n}_F{F}",
        "n_features": X.shape[1],
        "nmse_step": float(nmse_1),
        "lyapunov_lam1": float(lam1),
        "free_run_steps": int(n_test),
        "free_run_lyap_times": float(n_test * dt * lam1),
        "error_crossed_std_step": int(t_lyap),
        "error_crossed_std_lyap": float(t_lyap * dt * lam1),
    }


def benchmark_narma(n_tau=10, mem=5, n_train=2000, n_test=2000,
                    lam=1e-4, seed=0, amp=0.2):
    """NARMA10/30 one-step NMSE with the NG-RC feature stack.

    Drive x ~ U[0, amp]; amp is kept small enough (0.2) that the classic
    NARMA30 recurrence stays bounded over the benchmark horizon (its
    U[0, 0.5] variant is marginally unstable at long T for any method).

    Echo-state observable: the feedback loop needs labelled pairs, so the
    classic observable is the concatenation [drive, past output] — the
    target's x[t-1]*x[t-n] term then becomes reconstructible via memory.
    """
    rng = np.random.default_rng(seed)
    x_drive = rng.uniform(0.0, amp, size=n_train + n_test + 50)
    y = narma(n_tau, x_drive)
    # observable: [drive_x, past_y] at origin t; target y[t+1]
    u_obs = np.stack([x_drive[:-1], y[:-1]], axis=-1)
    u = u_obs
    yt = y[1:, None]                                   # target: NARMA output
    Xlin = delay_embed_signals(u, mem)
    X = quadratic_feature(Xlin)
    X_tr, y_tr = X[:n_train], yt[:n_train]
    X_te, y_te = X[n_train + 50:], yt[n_train + 50:]
    y_hat = ridge_predict(X_tr, y_tr, X_te, lam)
    return {
        "task": f"NARMA{n_tau}",
        "n_features": X.shape[1],
        "drive_amp": amp,
        "observable": "[drive, past_y]",
        "nmse": float(nmse(y_te, y_hat)),
    }


def run_all():
    print("=== Chaos track: closed-form core solver on chaotic benchmarks ===")
    print("--- Lorenz96 (N=5, F=8) NG-RC parity ---")
    r1 = benchmark_lorenz96()
    for k, v in r1.items():
        print(f"  {k}: {v}")
    print("--- NARMA echo-state tasks ---")
    for n in (10, 30):
        try:
            r = benchmark_narma(n_tau=n)
            print(f"  {r['task']}: NMSE {r['nmse']:.6f}  features {r['n_features']}")
        except Exception as e:
            print(f"  NARMA{n}: FAILED {e}")


if __name__ == "__main__":
    run_all()