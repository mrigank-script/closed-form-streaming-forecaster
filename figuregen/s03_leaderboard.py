"""Ours vs published DSOF Table-2 cells.

For every rank-valid (dataset, H) we render correlated panels:
  * grouped bars   : ours (static/rls/s2/s2rls) vs the 14 published cells
  * histogram      : distribution of the published values, our best marked
  * scatter/bubble : teacher x {batch vs dsof}, ours as star
  * heatmap        : relative improvement (pub - ours)/pub per cell
  * horizon curve  : per-step MSE/MAE from by_step (H24/H48)
  * bootstrap SE   : block-bootstrap bars with p5/p95 whiskers

Electricity is intentionally excluded here (corrected row lives in 04).
Output: figures/03_leaderboard/<ds>_H<H>_<kind>.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figuregen import data as D
from figuregen.style import (MODEL_ORDER, MODEL_COLORS, PALETTE, format_axis,
                             panel_label, save_fig, PUBLISHED_COLOR)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "03_leaderboard")

RANK_DSETS = ["etth2", "ettm1", "exchange", "weather", "traffic"]
ALL_H = [1, 24, 48]

DS_NAME = {"etth2": "ETTh2", "ettm1": "ETTm1", "exchange": "Exchange",
           "weather": "Weather", "traffic": "Traffic", "electricity": "Electricity"}


def load_all():
    rows, files = D.load_results()
    seeds, _ = D.load_seeds()
    return rows, seeds


def fig_bars(rows, seeds, ds, H):
    name = DS_NAME.get(ds, ds)
    pub = D.published_cells(ds, H)
    teachers = sorted({c[0] for c in pub})
    pub_by = {(c[0], c[1]): c[2] for c in pub}
    ours = {m: D.row_value(rows, ds, H, m) for m in MODEL_ORDER}
    ours = {k: v for k, v in ours.items() if v is not None}
    ours_mae = {m: D.row_value(rows, ds, H, m, "mae") for m in ours}
    best = D.ours_best(rows, ds, H)

    def whisk(ax, xs, mm, field):
        for xi, m in zip(xs, mm):
            r = seeds.get((ds, H, m, 0.0))
            if r is None:
                continue
            v = r[field]
            lo = r[f"{field}_p5"]
            hi = r[f"{field}_p95"]
            ax.errorbar(xi, v, yerr=[[v - lo], [hi - v]], fmt="none",
                        ecolor=PALETTE["muted"], elinewidth=0.8, capsize=2)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    g = np.arange(len(teachers))

    # --- MSE: dot plot (no bars on log) — published block + ours cluster ---
    ax = axes[0]
    x0 = g[-1] + 1.15
    xours = x0 + np.arange(len(ours)) * 1.15
    bv = [pub_by.get((t, "batch")) for t in teachers]
    dv = [pub_by.get((t, "dsof")) for t in teachers]
    ax.scatter(g - 0.18, bv, marker="o", s=17, edgecolor=PUBLISHED_COLOR,
               facecolor="white", lw=0.9, zorder=4)
    ax.scatter(g + 0.18, dv, marker="o", s=17, color=PUBLISHED_COLOR,
               zorder=4)
    for m, xp in zip(ours, xours):
        v = ours[m]
        r = seeds.get((ds, H, m, 0.0))
        if r:
            ax.errorbar(xp, v, marker="o", ms=5, color=MODEL_COLORS[m],
                        yerr=[[max(v - r["mse_p5"], 0)],
                              [max(r["mse_p95"] - v, 0)]],
                        ecolor=PALETTE["muted"], elinewidth=0.8, capsize=2,
                        zorder=5)
        else:
            ax.plot(xp, v, "o", color=MODEL_COLORS[m], ms=5, zorder=5)
    ax.axvline(x0 - 0.6, color=PALETTE["border"], lw=0.7)
    if best:
        ax.axhline(best[1], color=PALETTE["muted"], ls=":", lw=1.1)
    ax.set_xlim(g[0] - 0.8, xours[-1] + 0.8)
    ax.set_xticks(list(g) + list(xours))
    ax.set_xticklabels(teachers + [m for m in ours])
    ax.tick_params(axis="x", labelsize=7.5)
    format_axis(ax, title=f"{name} H{H} MSE", ylabel="MSE", logy=True)
    handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=PUBLISHED_COLOR,
               ms=5, label="published batch"),
        Line2D([], [], marker="o", ls="", color=PUBLISHED_COLOR, ms=5,
               label="published dsof"),
    ]
    for m in ours:
        handles.append(Line2D([], [], marker="o", ls="", color=MODEL_COLORS[m],
                              ms=5, label=m))
    ax.legend(handles=handles, fontsize=6.5, ncol=2, loc="upper left")

    # --- MAE: ours only (published MAE not tabulated) ---
    ax = axes[1]
    xm = np.arange(len(ours))
    ax.bar(xm, [ours_mae[m] for m in ours], 0.55,
           color=[MODEL_COLORS[m] for m in ours], label=[m for m in ours])
    whisk(ax, xm, list(ours), "mae")
    ax.set_xticks(xm)
    ax.set_xticklabels([m for m in ours])
    format_axis(ax, title=f"{name} H{H} MAE (ours)", ylabel="MAE")
    ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle(f"{name} H{H} — Ours vs Published Cells (7 Teachers x Batch:DSOF)",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_H{H}_bars.png"), panels=2,
             caption="published MSE cells (grey) vs ours with block-bootstrap "
                     "[p5,p95] whiskers; published cells are single-point table "
                     "values (no std supplied)")


def fig_hist(rows, ds, H):
    name = DS_NAME.get(ds, ds)
    pub = D.published_cells(ds, H)
    vals = np.array([c[2] for c in pub])
    ours_mse = {m: D.row_value(rows, ds, H, m) for m in MODEL_ORDER}
    ours_mse = {k: v for k, v in ours_mse.items() if v is not None}
    ours_mae = {m: D.row_value(rows, ds, H, m, "mae") for m in ours_mse}
    best = D.ours_best(rows, ds, H)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, (field, ylab) in zip(axes, [("mse", "MSE"), ("mae", "MAE")]):
        ax.hist(vals, bins=max(6, len(vals)), color=PUBLISHED_COLOR, alpha=0.75,
                edgecolor="#AEB9C6", lw=0.4)
        mine = ours_mse if field == "mse" else ours_mae
        mv = None
        if mine:
            k0 = min(mine, key=mine.get)
            mv = mine[k0]
            ax.axvline(mv, color=MODEL_COLORS[k0], lw=1.4,
                       label=f"ours {k0}={mv:.4f}")
        format_axis(ax,
                    title=f"{name} H{H} {ylab} — Published Dist (n={len(vals)})",
                    xlabel=ylab, ylabel="cells",
                    logx=(field == "mse"))
        ax.legend(fontsize=7)
        if field == "mae":
            ax.text(0.98, 0.92, "pub MAE not tabulated",
                    transform=ax.transAxes, ha="right", fontsize=6.5,
                    color=PALETTE["muted"])
    fig.suptitle(f"{name} H{H} — Where Our Row Sits in the Published Spread",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_H{H}_hist.png"), panels=2,
             caption="histogram of published rows + our marker (MSE, MAE)")


def fig_summary(rows, seeds, ds):
    """Per-dataset summary across H: best-ours, beats-count, and seed bars."""
    Hs = [H for H in ALL_H if (ds, H, "rls") in rows]
    if not Hs:
        return
    name = DS_NAME.get(ds, ds)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    mse_v = [D.row_value(rows, ds, H, "rls") for H in Hs]
    sta_v = [D.row_value(rows, ds, H, "static") for H in Hs]
    ax = axes[0]
    x = np.arange(len(Hs))
    ax.plot(x, sta_v, "o-", color=PALETTE["blue"], ms=5, lw=1.4,
            label="static")
    ax.plot(x, mse_v, "s-", color=PALETTE["orange"], ms=5, lw=1.4,
            label="rls")
    ax.set_xticks(x); ax.set_xticklabels([f"H{h}" for h in Hs])
    format_axis(ax, title="MSE by H", ylabel="MSE", logy=True)
    ax.legend(fontsize=7)
    ax = axes[1]
    beats = [D.beats_count(rows, ds, H) for H in Hs]
    ax.bar(x, [b[0] for b in beats], 0.5, color=PALETTE["green"])
    for i, (b, t, tot) in enumerate(beats):
        ax.text(i, b + 0.3, f"{b}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"H{h}" for h in Hs])
    ax.set_ylim(0, 15)
    format_axis(ax, title="cells beaten / 14 (MSE)")
    ax = axes[2]
    for i, H in enumerate(Hs):
        r = seeds.get((ds, H, "rls", 0.0))
        if not r:
            continue
        ax.errorbar(i, r["mse"], yerr=[[r["mse"] - r["mse_p5"]],
                                       [r["mse_p95"] - r["mse"]]],
                    color=PALETTE["orange"], marker="o", ms=5, lw=1.4,
                    capsize=3)
    ax.set_xticks(range(len(Hs)))
    ax.set_xticklabels([f"H{h}" for h in Hs])
    format_axis(ax, title="RLS bootstrap MSE [p5,p95]", ylabel="MSE",
                logy=True)
    fig.suptitle(f"{name} — Headline Sweep Across Horizons",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_summary.png"), panels=3,
             caption="MSE by H, beats-count, bootstrap intervals (log scale)")


def fig_scatter(rows, ds, H):
    name = DS_NAME.get(ds, ds)
    cells = D.published_cells(ds, H)
    bt = [c for c in cells if c[1] == "batch"]
    dso = [c for c in cells if c[1] == "dsof"]
    best = D.ours_best(rows, ds, H)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.scatter([c[2] for c in bt], [c[2] for c in dso], color=PUBLISHED_COLOR,
               s=30, zorder=4, label="published (batch vs dsof)")
    for c in bt:
        ax.annotate(c[0], (c[2], c[2]), fontsize=6.4, color=PALETTE["muted"],
                    xytext=(2, 2), textcoords="offset points")
    if best:
        ax.axhline(best[1], color=MODEL_COLORS[best[0]], ls=":", lw=1.0)
        ax.axvline(best[1], color=MODEL_COLORS[best[0]], ls=":", lw=1.0)
        ax.scatter([best[1]], [best[1]], color=MODEL_COLORS[best[0]], s=42,
                   marker="D", zorder=7, label=f"ours {best[0]}")
    lim = [0, max(bt[0][2] if bt else 0, *[c for *_ , c in dso], 1e-6)]
    ax.plot(lim, lim, color=PALETTE["muted"], lw=1.0, ls="-", alpha=0.5)
    format_axis(ax, title=f"{name} H{H} — Batch vs DSOF (Diagonal=Fix)",
                xlabel="batch MSE", ylabel="dsof MSE", logx=True, logy=True)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_H{H}_scatter.png"))


def fig_heatmap(rows, ds, H):
    name = DS_NAME.get(ds, ds)
    pub = D.published_cells(ds, H)
    cells = D.published_cells(ds, H)
    best = D.ours_best(rows, ds, H)
    if best is None:
        return
    _, ourv = best
    teachers = sorted({c[0] for c in cells})
    modes = ["batch", "dsof"]
    M = np.zeros((len(teachers), len(modes)))
    for i, t in enumerate(teachers):
        for j, m in enumerate(modes):
            v = [c[2] for c in cells if c[0] == t and c[1] == m][0]
            M[i, j] = (ourv - v) / v * 100        # % better (negative = better)
    fig, ax = plt.subplots(figsize=(5.4, len(teachers) * 0.72 + 0.9))
    im = ax.imshow(M, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(modes))); ax.set_xticklabels(modes)
    ax.set_yticks(range(len(teachers))); ax.set_yticklabels(teachers)
    for i in range(len(teachers)):
        for j in range(len(modes)):
            ax.text(j, i, f"{M[i, j]:+.1f}%", ha="center", va="center",
                    fontsize=8, color=PALETTE["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.04)
    cb.set_label("ours − pub (% of pub)", fontsize=7)
    ax.set_title(f"{name} H{H} — Relative Improvement Over Published",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_H{H}_heatmap.png"))


def fig_horizon(rows, ds, H):
    if H not in (24, 48):
        return
    name = DS_NAME.get(ds, ds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, metric in zip(axes, ("mse", "mae")):
        for m in MODEL_ORDER:
            s = D.by_step_series(rows, ds, H, m)
            if s is None:
                continue
            steps = [x[0] for x in s]
            vals = [x[1] if metric == "mse" else x[2] for x in s]
            ax.plot(steps, vals, color=MODEL_COLORS[m], lw=1.5,
                    marker=".", markersize=2.8, label=m)
        format_axis(ax, title=f"{name} H{H} — Per-Step {metric.upper()}",
                    xlabel="step r", ylabel=metric.upper())
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_H{H}_horizon.png"), panels=2, caption="per-step MSE + MAE")


def fig_seeds(rows, seeds, ds, H):
    name = DS_NAME.get(ds, ds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, metric in zip(axes, ("mse", "mae")):
        xs2 = []
        labs = []
        for i, m in enumerate(MODEL_ORDER):
            r = seeds.get((ds, H, m, 0.0))
            if r is None:
                continue
            xs2.append(i)
            v = r[metric]
            lo = r[metric + "_p5"] if metric == "mse" else r["mae_p5"]
            hi = r[metric + "_p95"] if metric == "mse" else r["mae_p95"]
            ax.bar(i, v, width=0.55, color=MODEL_COLORS[m],
                   yerr=[[min(v - lo, 0)], [max(hi - v, 0)]],
                   capsize=3, error_kw=dict(elinewidth=0.8,
                                            ecolor=PALETTE["muted"]))
            labs.append(m)
        ax.set_xticks(xs2)
        ax.set_xticklabels(labs)
        format_axis(ax, title=f"{name} H{H} — Bootstrap {metric.upper()}",
                    xlabel="model", ylabel=metric.upper())
    fig.suptitle(f"{name} H{H} — Block-Bootstrap [p5,p95] Intervals",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, f"{ds}_H{H}_seeds.png"), panels=2,
             caption="bootstrap MSE and MAE bars with p5-p95 whiskers")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, seeds = load_all()
    n = 0
    for ds in RANK_DSETS:
        fig_summary(rows, seeds, ds)
        for H in ALL_H:
            if (ds, H, "rls") not in rows:
                continue
            fig_bars(rows, seeds, ds, H)
            fig_hist(rows, ds, H)
            fig_scatter(rows, ds, H)
            fig_heatmap(rows, ds, H)
            fig_horizon(rows, ds, H)
            fig_seeds(rows, seeds, ds, H)
            n += 1
    print(f"[s03_leaderboard] rendered leading sheets for {n} (dataset,H)")


if __name__ == "__main__":
    main()