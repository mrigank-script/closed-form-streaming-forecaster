"""figuregen/s05_s2_lru.py — S2 (LRU context) analysis figures.

Shows where the LRU reservoir sits in the feature space, its eigenvalue
spectrum, memory decay timescales, and the honest clean-data result: S2RLS
matches RLS within block-bootstrap noise on ETTm1-H24, and damps Electricity.

Output: figures/05_s2_lru/<name>.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figuregen import data as D
from figuregen.style import (PALETTE, MODEL_COLORS, format_axis, save_fig,
                             sci)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "05_s2_lru")


def fig_lru_spectrum():
    """Fixed random diagonal bank: |lambda| radii & decay timescales (D=8)."""
    rng_min, rng_max = 0.5, 0.995
    D8 = 8
    radii = np.linspace(rng_min, rng_max, D8)
    tau = -1.0 / np.log(np.maximum(radii, 1e-12))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1))
    ax = axes[0]
    u, v = np.cos(2 * np.pi * np.arange(0, 36) / 36), np.sin(2 * np.pi * np.arange(0, 36) / 36)
    ax.plot(u, v, ls="--", color=PALETTE["grey"], lw=0.8)
    ax.plot([-1, 1], [0, 0], color=PALETTE["grey"], lw=0.6)
    ax.plot([0, 0], [-1, 1], color=PALETTE["grey"], lw=0.6)
    for r in radii:
        a = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(a), r * np.sin(a), color=PALETTE["violet"], lw=0.5,
                alpha=0.5)
    ax.scatter(radii * np.cos(np.linspace(0, 2 * np.pi, D8, endpoint=False)),
               radii * np.sin(np.linspace(0, 2 * np.pi, D8, endpoint=False)),
               s=34, color=PALETTE["orange"], edgecolor=PALETTE["ink"], zorder=5)
    ax.set_aspect("equal")
    format_axis(ax, title="Fixed random LRU eigenvalues $\\lambda$ (D=8)",
                xlabel=r"$\mathrm{Re}(\lambda)$",
                ylabel=r"$\mathrm{Im}(\lambda)$")
    ax = axes[1]
    ax.semilogy(radii, tau, "o-", color=PALETTE["orange"],
                markerfacecolor=PALETTE["orange"], markeredgecolor=PALETTE["ink"])
    ax.set_xlabel(r"$|\lambda|$")
    ax.set_ylabel("decay timescale $\\tau$ (steps)")
    format_axis(ax, title="Memory decay $\\tau=-1/\\ln|\\lambda|$")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "lru_spectrum.png"), panels=2, caption="eigenvalues + decay timescale")


def fig_impulse_response():
    """Impulse responses of the 8 oscillators: h_t = lambda^t.

    Each curve = one fixed oscillator's memory of a unit spike at step 0,
    decaying as |lambda|^t. Colour encodes |lambda| (light = fast decay,
    dark = long memory) so the 8 curves stay distinct on the log axis
    instead of collapsing into one overlapping blob.
    """
    import matplotlib as mpl
    radii = np.linspace(0.5, 0.995, 8)
    t = np.arange(0, 301)
    norm = mpl.colors.Normalize(vmin=radii[0], vmax=radii[-1])
    cmap = plt.get_cmap("plasma_r")
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for r in radii:
        ax.plot(t, r ** t, color=cmap(norm(r)), lw=1.4,
                label=f"|$\\lambda$|={r:.2f}")
    ax.plot(t, np.exp(-t / 96), color=PALETTE["ink"], lw=2.2, ls="--",
            label="cutoff at lookback=96")
    format_axis(ax, title="LRU bank impulse responses vs 96-lag cutoff",
                xlabel="step", ylabel="|h_t| (log)", logy=True)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "lru_impulse.png"), panels=1,
             caption="impulse response |lambda|^t per oscillator + 96-step "
                     "window cutoff")


def fig_ettm1_rls_vs_s2rls():
    """Honest clean-data comparison: RLS vs S2RLS with SE bands (ETTm1-H24)."""
    seeds, _ = D.load_seeds()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, metric in zip(axes, ("mse", "mae")):
        vals = {}
        for m in ("rls", "s2rls"):
            r = seeds.get(("ettm1", 24, m, 0.0))
            if r:
                vals[m] = (r[metric], r[f"{metric}_se"] if metric == "mse"
                           else r["mae_se"])
        names = list(vals)
        y = [vals[n][0] for n in names]
        e = [vals[n][1] for n in names]
        ax.bar(names, y, yerr=e, color=[MODEL_COLORS[n] for n in names],
               capsize=4)
        for n, v, se in zip(names, y, e):
            ax.annotate(f"{v:.4f}±{se:.4f}", xy=(names.index(n), v + se),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.8)
        format_axis(ax, title=f"ETTm1-H24 {metric.upper()} (block-bootstrap)",
                    ylabel=metric.upper())
    fig.suptitle("S2 on clean data: matches RLS within noise (no big win, "
                 "no regression)", fontweight="bold", y=1.05)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "ettm1_rls_vs_s2rls.png"), panels=2,
             caption="MSE + MAE with SE")


def fig_s2_across_datasets():
    """S2RLS-vs-RLS MSE delta across every dataset we have seeds for."""
    seeds, _ = D.load_seeds()
    keys = sorted({(ds, H) for (ds, H, m, clip) in seeds
                   if m in ("rls", "s2rls") and clip == 0.0})
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    x = np.arange(len(keys))
    for i, (ds, H) in enumerate(keys):
        r_rls = seeds.get((ds, H, "rls", 0.0))
        r_s2 = seeds.get((ds, H, "s2rls", 0.0))
        if not r_rls or not r_s2:
            continue
        d_mse = (r_s2["mse"] - r_rls["mse"]) / r_rls["mse"] * 100
        se = np.sqrt(r_rls["mse_se"] ** 2 + r_s2["mse_se"] ** 2) \
            / r_rls["mse"] * 100
        ax.errorbar(i, d_mse, yerr=se, fmt="o", color=PALETTE["violet"],
                    capsize=3)
        ax.text(i, d_mse, f"{ds}-H{H}", ha="center", va="bottom" if d_mse >= 0
                else "top", fontsize=6.3, rotation=45, color=PALETTE["muted"])
    ax.axhline(0, color=PALETTE["ink"], lw=1.0)
    ax.set_xticks([])
    format_axis(ax, title="S2RLS vs RLS — % MSE change (bootstrap SE)",
                xlabel="", ylabel="Δ MSE (%)")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "s2_across_datasets.png"), panels=1,
             caption="percentage MSE change of S2RLS vs RLS across datasets")

def main():
    os.makedirs(OUT, exist_ok=True)
    fig_lru_spectrum()
    fig_impulse_response()
    fig_ettm1_rls_vs_s2rls()
    fig_s2_across_datasets()
    print("[s05_s2_lru] done")


if __name__ == "__main__":
    main()