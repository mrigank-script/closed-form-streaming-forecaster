"""Chaos track figures.

Reproduces a Lorenz96 run through the closed-form core: true-vs-prediction
traces, free-run error growth in Lyapunov times (with the λ1 annotation),
step-NMSE across seeds, and NARMA10/30 reconstructions. Only curves recomputed
from experiments.chaos, consistent with data/proc/chaos_results.json.

Output: figures/06_chaos/<name>.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figuregen import data as D
from figuregen.style import (PALETTE, format_axis, save_fig, sci)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "06_chaos")


def _run_l96(seed=0):
    """Re-run benchmark_lorenz96 and also capture free-run & truth traces."""
    from experiments.chaos import (benchmark_lorenz96, integrate_lorenz96_jit,
                                   delay_embed_signals, quadratic_feature,
                                   lyapunov_lorenz96, ridge_predict, nmse)
    import numpy as np
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    n, F, dt = 5, 8.0, 0.02
    x0 = rng.uniform(-0.01, 0.01, size=(n,))
    burn, n_train, n_test, mem, gap = 5000, 5000, 2000, 4, 100
    traj = np.asarray(integrate_lorenz96_jit(
        x0, burn + n_train + n_test + mem + gap, F, dt))[burn:]
    u = traj[:-1]
    y = traj[1:]
    Xlin = delay_embed_signals(u, mem)
    X = quadratic_feature(Xlin)
    X_tr, y_tr = X[:n_train], y[:n_train]
    seed_end = n_train + gap
    X_te, y_te = X[seed_end:], y[seed_end:]
    y_hat = ridge_predict(X_tr, y_tr, X_te, 1e-6)
    # free run
    Xtrf = jnp.asarray(X_tr, dtype=jnp.float64)
    ytrf = jnp.asarray(y_tr, dtype=jnp.float64)
    dd = X_tr.shape[1]
    lam_l = 1e-6 * jnp.trace(Xtrf.T @ Xtrf) / dd
    A = Xtrf.T @ Xtrf + lam_l * jnp.eye(dd, dtype=jnp.float64)
    W = np.asarray(jnp.linalg.solve(A, Xtrf.T @ ytrf))
    state = u[seed_end - mem:seed_end]
    free = []
    for _ in range(n_test):
        feats = np.concatenate([np.ones((1, 1)), state[::-1].reshape(1, -1)], axis=-1)
        pred = quadratic_feature(feats) @ W          # (1, N)
        pred_row = pred[0]
        free.append(pred_row)
        state = np.concatenate([state[1:], pred_row[None, :]], axis=0)
    free = np.asarray(free, dtype=np.float64)
    # chaotic free-runs diverge and overflow to inf/NaN late on; cut the trace
    # at the first non-finite or >1e3 outlier so the growth panel stays legible
    finite = np.isfinite(free) & (np.abs(free) < 1e3)
    if not finite.all():
        bad = np.flatnonzero(~finite.any(axis=1))
        cut = int(bad[0]) if len(bad) else len(free)
        free = free[:max(cut, 2)]
    return {
        "task": f"Lorenz96_n{n}_F{F}", "seed": seed,
        "true_step": y_te[:, 0], "pred_step": y_hat[:, 0],
        "true_free": y[seed_end:seed_end + n_test, 0][:len(free)],
        "pred_free": free[:, 0],
        "nmse_step": nmse(y_te, y_hat),
        "lam1": lyapunov_lorenz96(x0, F, dt, n_trans=2000, n_steps=10000, n=n),
        "std_y": float(np.std(y_te)),
    }


def fig_l96_traces():
    """True vs predicted: one-step (zoom) and free-run over a window."""
    r = _run_l96(0)
    w = 400
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=False)
    ax = axes[0]
    ax.plot(r["true_step"][:w], color=PALETTE["ink"], lw=1.0, label="truth")
    ax.plot(r["pred_step"][:w], color=PALETTE["orange"], lw=1.2, ls="--",
            label="one-step pred")
    format_axis(ax, title="Lorenz 96 one-step closed-form prediction "
                          f"(step NMSE {sci(r['nmse_step'])})",
                ylabel=r"$x_1$")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.plot(r["true_free"][:w], color=PALETTE["ink"], lw=1.0, label="truth")
    ax.plot(r["pred_free"][:w], color=PALETTE["red"], lw=1.1, ls="--",
            label="free-run (closed-form loop)")
    format_axis(ax, title="Free-run vs truth (first Lyapunov time)",
                xlabel="step", ylabel=r"$x_1$")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "l96_traces.png"), panels=2, caption="one-step + free-run traces")


def fig_l96_free_run_error():
    """RMSE growth of the free run vs Lyapunov time; mark 1 Lyap crossing."""
    r = _run_l96(0)
    cross = np.argmax(np.abs(r["pred_free"] - r["true_free"]) > r["std_y"])
    cross_lyap = cross * 0.02 * r["lam1"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ts = np.arange(len(r["pred_free"])) * 0.02 * r["lam1"]
    d = np.abs(r["pred_free"] - r["true_free"])
    ax.plot(ts, d, color=PALETTE["orange"], lw=1.4)
    ax.axhline(r["std_y"], color=PALETTE["red"], ls="--", lw=1.1,
               label=r"$\sigma_y$ (error reference)")
    ax.axvline(cross_lyap, color=PALETTE["red"], ls=":", lw=1.2,
               label=f"first exceed σ at {cross_lyap:.1f} Lyap times")
    exp_h = np.clip(d[0] * np.exp(r["lam1"] * ts), 1e-12, d.max() * 2)
    ax.plot(ts, exp_h, color=PALETTE["muted"], lw=1.0, ls="-",
            label=f"e^{{λ1 t}} growth (λ1={r['lam1']:.2f})")
    ax.set_yscale("symlog", linthresh=np.std(r["true_free"]))
    format_axis(ax, title="Free-run error growth in Lyapunov time units",
                xlabel="Lyapunov times", ylabel="|error|")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "l96_free_run_error.png"))


def fig_l96_seed_stats():
    """Step NMSE and λ1 across the 3 seeds from chaos_results.json."""
    c = D.load_chaos()
    blocks = [
        (c["lorenz96_seed_stats"]["nmse_step"], r"one-step NMSE", True),
        (c["lorenz96_seed_stats"]["lyapunov_lam1"], r"$\lambda_1$ (max Lyap)", False),
        (c["lorenz96_seed_stats"]["error_crossed_std_lyap"],
         "free-run horizon (Lyap times)", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (stat, lab, logy) in zip(axes, blocks):
        mean = stat["%s_mean" % _key_of(lab)]
        std = stat["%s_std" % _key_of(lab)]
        ax.bar(["mean", "std"], [mean, std],
               color=[PALETTE["orange"], PALETTE["blue"]],
               edgecolor=PALETTE["ink"], lw=0.6)
        for i, v in enumerate([mean, std]):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
        format_axis(ax, title=lab, ylabel="n=3 seeds", logy=logy)
    fig.suptitle("Lorenz96 seed spread (genuine IC randomness — SE meaningful)",
                 fontweight="bold", y=1.05)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "l96_seed_stats.png"), panels=3, caption="NMSE, lambda1, horizon across seeds")


def _key_of(lab):
    if "NMSE" in lab:
        return "nmse_step"
    if "lambda" in lab:
        return "lyapunov_lam1"
    return "error_crossed_std_lyap"


def fig_phase_portrait():
    """Lorenz96 phase portrait (x1 vs x2) of truth and 1-step pred."""
    r = _run_l96(0)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    w = 1500
    # adjacent-in-time pairs (x_i, x_{i+1}) only — no roll wrap, so the
    # closing wrap segment can't draw a straight chord across the attractor.
    ts = r["true_step"][:w]
    ax.plot(ts[:-1], ts[1:],
            color=PALETTE["ink"], lw=0.7, alpha=0.75,
            label=r"truth ($x_1$, $x_2$)")
    tf = r["true_free"][:300]
    ax.plot(tf[:-1], tf[1:],
            color=PALETTE["orange"], lw=0.9, label="first free-run steps")
    format_axis(ax, title="Lorenz 96 $x_1$ vs $x_2$",
                xlabel=r"$x_1$", ylabel=r"$x_2$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "l96_phase_portrait.png"), panels=1,
             caption="phase portrait: attractor ring (truth) + free-run arc")


def _seed_nmse_spread():
    from experiments.chaos import benchmark_lorenz96
    vals = []
    for s in range(3):
        try:
            vals.append(benchmark_lorenz96(n_train=5000, n_test=2000, seed=s))
        except Exception:
            continue
    return vals


def fig_seed_traces():
    """Step-NMSE spread across 3 ICs + the summary values from JSON."""
    c = D.load_chaos()
    sm = c["lorenz96_seed_stats"]["nmse_step"]
    total = sm["nmse_step_n"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(["mean", "std"], [sm["nmse_step_mean"], sm["nmse_step_std"]],
           color=PALETTE["orange"], edgecolor=PALETTE["ink"], lw=0.6)
    for i, v in enumerate([sm["nmse_step_mean"], sm["nmse_step_std"]]):
        ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7.5)
    ax.set_yscale("log")
    format_axis(ax, title=f"Lorenz96 step NMSE across {total} seeds",
                ylabel="NMSE")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "l96_nmse_seeds.png"), panels=1,
             caption="step-NMSE mean/std across seeds (JSON)")


def _run_narma(n_tau, mem, amp):
    from experiments.chaos import benchmark_narma, narma, delay_embed_signals, \
        quadratic_feature, ridge_predict, nmse
    import numpy as np
    n_train, n_test = 5000, 2000
    rng = np.random.default_rng(0)
    x_drive = rng.uniform(0.0, amp, size=n_train + n_test + 50)
    y = narma(n_tau, x_drive)
    u = np.stack([x_drive[:-1], y[:-1]], axis=-1)
    yt = y[1:, None]
    Xlin = delay_embed_signals(u, mem)
    X = quadratic_feature(Xlin)
    X_tr, y_tr = X[:n_train], yt[:n_train]
    X_te, y_te = X[n_train + 50:], yt[n_train + 50:]
    y_hat = ridge_predict(X_tr, y_tr, X_te, 1e-4)
    return {"tau": n_tau, "amp": amp, "y": y_te[:, 0], "yhat": y_hat[:, 0],
            "nmse": nmse(y_te, y_hat)}


def fig_narma():
    """NARMA10/30 reconstructions + NMSE bars with the 0.0391 parity line."""
    r10 = _run_narma(10, 12, 0.5)
    r30 = _run_narma(30, 35, 0.2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for ax, r, n in [(axes[0, 0], r10, 10), (axes[0, 1], r30, 30),
                     (axes[1, 0], r10, 10), (axes[1, 1], r30, 30)]:
        if n == 10:
            ax.plot(r["y"][:300], color=PALETTE["ink"], lw=1.0, label="truth")
            ax.plot(r["yhat"][:300], color=PALETTE["orange"], lw=1.1, ls="--",
                    label="pred")
        else:
            ax.plot(r["y"][:300], color=PALETTE["ink"], lw=1.0, label="truth")
            ax.plot(r["yhat"][:300], color=PALETTE["red"], lw=1.1, ls="--",
                    label="pred")
        format_axis(ax, title=f"NARMA{n} (amp={r['amp']}) — NMSE {sci(r['nmse'])}",
                    ylabel="y")
        ax.legend(fontsize=7)
    axes[1, 0].plot(r10["y"][300:900] - r10["yhat"][300:900],
                    color=PALETTE["muted"], lw=0.8)
    axes[1, 0].set_ylabel("residual")
    axes[1, 1].plot(r30["y"][300:900] - r30["yhat"][300:900],
                    color=PALETTE["muted"], lw=0.8)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "narma_recon.png"), panels=4, caption="reconstruction + residuals for NARMA10/30")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    nms = [r10["nmse"], r30["nmse"]]
    ax.bar(["NARMA10", "NARMA30"], nms, color=[PALETTE["orange"], PALETTE["red"]],
           edgecolor=PALETTE["ink"], lw=0.6)
    ax.axhline(0.0391, color=PALETTE["blue"], ls="--", lw=1.2,
               label="NG-RC parity line 0.0391")
    ax.set_yscale("log")
    for i, v in enumerate(nms):
        ax.text(i, v, sci(v), ha="center", va="bottom", fontsize=7.5)
    format_axis(ax, title="NARMA one-step NMSE vs parity",
                ylabel="NMSE")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "narma_nmse.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_l96_traces()
    fig_l96_free_run_error()
    fig_l96_seed_stats()
    fig_phase_portrait()
    fig_seed_traces()
    fig_narma()
    print("[s06_chaos] done")


if __name__ == "__main__":
    main()