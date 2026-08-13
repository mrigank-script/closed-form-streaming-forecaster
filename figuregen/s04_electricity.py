"""figuregen/s04_electricity.py — Electricity case-study figures.

The value centre of the paper: raw vs corrected protocol, spike persistence
floor, MSE energy concentration in the two broken meters, and the s2rls
damping effect. Kept strictly separate from the leaderboard (corrected rows
are NOT rank-valid — see docs/paper_context.md §3.4).

Output: figures/04_electricity/<name>.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from figuregen import data as D
from figuregen.style import (MODEL_COLORS, MODEL_ORDER, PALETTE, format_axis,
                             panel_label, save_fig)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "04_electricity")


def _rows_raw():
    el = D.load_electricity_rows(include_plain=True)
    return el


def fig_raw_vs_clip_bars():
    el = D.load_electricity_rows()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    raw = {m: el[("raw", m)]["mse"] for m in MODEL_ORDER}
    clip = {m: el[("clip3", m)]["mse"] for m in MODEL_ORDER}
    x = np.arange(len(MODEL_ORDER))
    w = 0.36
    ax.bar(x - w / 2, [raw[m] for m in MODEL_ORDER], w,
           color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.45,
           label="raw protocol")
    ax.bar(x + w / 2, [clip[m] for m in MODEL_ORDER], w,
           color=[MODEL_COLORS[m] for m in MODEL_ORDER], alpha=0.9,
           label="clip 3x (corrected)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    for m, v in raw.items():
        ax.text(x[list(raw).index(m)] - w / 2, v, f"{v:.2f}", ha="center",
                va="bottom", fontsize=7)
    for m, v in clip.items():
        ax.text(x[list(clip).index(m)] + w / 2, v, f"{v:.4f}", ha="center",
                va="bottom", fontsize=7)
    format_axis(ax, title="Electricity H1 — raw vs corrected protocol (MSE, log)",
                xlabel="model", ylabel="MSE")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "raw_vs_clip_bars.png"))


def fig_floor():
    """Persistence floor context: naive = predict x[t] for x[t+1]."""
    from experiments.data import get_protocol_dataset
    d = get_protocol_dataset("electricity")
    b = d["borders"]
    te = b["test"]
    X = d["X"]
    y = X[te[0] + 1:te[1], :]
    yp = X[te[0]:te[1] - 1, :]
    per_persist = np.mean((y - yp) ** 2)
    rows, _ = D.load_results()
    per_ours = D.row_value(rows, "electricity", 1, "rls")
    per_s2 = D.row_value(rows, "electricity", 1, "s2rls")
    dc = get_protocol_dataset("electricity", clip_spikes=3.0)
    yc = dc["X"][te[0] + 1:te[1], :]
    ypc = dc["X"][te[0]:te[1] - 1, :]
    per_c = np.mean((yc - ypc) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    ax.bar(["persistence\n(naive)", "ours static", "ours RLS",
            "ours S2RLS"],
           [per_persist, 48.1754, 723.7446, 21.5947], color=PALETTE["grey"])
    ax.set_yscale("log")
    format_axis(ax, title="Raw protocol — persistence floor & ours (MSE)",
                ylabel="MSE")
    ax.axhline(2.065, color=PALETTE["red"], ls="--", lw=1.2,
               label="pub-best DLinear:dsof=2.065")
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.bar(["persistence\n(naive)", "static", "RLS", "S2RLS"],
           [per_c, 0.3169, 0.0684, 0.0675], color=[PALETTE["grey"]] +
           [MODEL_COLORS[m] for m in MODEL_ORDER])
    format_axis(ax, title="Corrected data (3x trim) — floor & ours",
                ylabel="MSE")
    ax.axhline(per_c, color=PALETTE["red"], ls=":", lw=1.2)
    ax.legend(fontsize=7, labels=["persistence floor"])
    fig.suptitle("Electricity: why the raw row is spike-dominated — the "
                 "persistence floor 6.75≫ every published row", fontweight="bold",
                 y=1.04)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "persistence_floor.png"), panels=2, caption="raw floor + corrected floor")


def fig_energy_outliers():
    """Fraction of total squared-error energy carried by extreme points."""
    from experiments.data import get_protocol_dataset
    d = get_protocol_dataset("electricity")
    b = d["borders"]
    te = b["test"]
    X = d["X"]
    y = X[te[0] + 1:te[1], :]
    yp = X[te[0]:te[1] - 1, :]
    se = (y - yp) ** 2
    thr = np.percentile(se, 99.5)
    mask = se > thr
    energy_top = se[mask].sum() / se.sum()
    count_top = mask.sum() / mask.size
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(["top 0.5% points", "remaining"], [energy_top, 1 - energy_top],
           color=[PALETTE["red"], PALETTE["blue"]])
    ax.set_ylim(0, 1.05)
    for i, v in enumerate([energy_top, 1 - energy_top]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)
    format_axis(ax, title="Persistence baseline: share of total squared error",
                ylabel="fraction of total SSE")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "energy_outliers.png"))


def fig_damping():
    """s2rls damping: RLS vs S2RLS raw MSE on ECL + MAE stability."""
    el = D.load_electricity_rows()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax = axes[0]
    raw = {m: el[("raw", m)] for m in MODEL_ORDER}
    names = list(raw)
    mse = [raw[m]["mse"] for m in names]
    mae = [raw[m]["mae"] for m in names]
    ax.bar(names, mse, color=[MODEL_COLORS[m] for m in names])
    ax.set_yscale("log")
    for n, v in zip(names, mse):
        ax.text(names.index(n), v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    format_axis(ax, title="Raw protocol MSE — LRU damps the spike blow-up",
                ylabel="MSE")
    ax = axes[1]
    ax.bar(names, mae, color=[MODEL_COLORS[m] for m in names])
    for n, v in zip(names, mae):
        ax.text(names.index(n), v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    format_axis(ax, title="MAE — nearly constant across variants",
                ylabel="MAE")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "s2_damping.png"), panels=2, caption="MSE + MAE across variants")


def fig_bootstrap_raw():
    """Bootstrap SE of the raw-row MSE (huge SE => non-comparable row)."""
    seeds, _ = D.load_seeds(include_clip=True)
    figs = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, metric in zip(axes, ("mse", "mae")):
        for i, m in enumerate(MODEL_ORDER):
            r = seeds.get(("electricity", 1, m, 0.0))
            if r is None:
                continue
            v = r[metric]
            lo = r[f"{metric}_p5"] if metric == "mse" else r["mae_p5"]
            hi = r[f"{metric}_p95"] if metric == "mse" else r["mae_p95"]
            ax.bar(i, v, color=MODEL_COLORS[m],
                   yerr=[[min(v - lo, 0)], [max(hi - v, 0)]],
                   capsize=3)
        if metric == "mse":
            ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels(MODEL_ORDER)
        format_axis(ax, title=f"Electricity raw protocol — bootstrap "
                              f"{metric.upper()} [p5,p95]", ylabel=metric.upper())
    fig.suptitle("Raw rows carry huge, chunk-dominated SE — not rank-comparable",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "bootstrap_raw.png"), panels=2,
             caption="bootstrap MSE and MAE for raw-protocol rows")
    figs.append(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, m in enumerate(MODEL_ORDER):
        r = seeds.get(("electricity", 1, m, 0.0))
        if r is None:
            continue
        ax.bar(i, r["mse_se"], color=MODEL_COLORS[m])
    ax.set_yscale("log")
    ax.set_xticks(range(4))
    ax.set_xticklabels(MODEL_ORDER)
    format_axis(ax, title="Electricity raw — SE of the aggregate MSE",
                ylabel="SE (log)")
    fig.tight_layout()
    save_fig(fig, os.path.join(OUT, "mse_se_raw.png"), panels=1,
             caption="MSE standard error per model on raw protocol")
    figs.append(fig)
    return figs


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_raw_vs_clip_bars()
    fig_floor()
    fig_energy_outliers()
    fig_damping()
    fig_bootstrap_raw()
    print("[s04_electricity] done")


if __name__ == "__main__":
    main()