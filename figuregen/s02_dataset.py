"""figuregen/s02_dataset.py — dataset descriptive figures.

Raw-series montages, train/val/test split timelines, cadence periodograms and
the ECL spike-channel (114/146) train-vs-test range conflict that drives the
Electricity case study. Panels within a PNG are correlated (same dataset).

Output: figures/02_dataset/<name>.png
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.data import get_protocol_dataset
from figuregen.style import (PALETTE, format_axis, panel_label, save_fig,
                             sci)

OUT = os.path.join(os.path.dirname(__file__), "..", "figures", "02_dataset")

DATASETS = ["etth2", "ettm1", "exchange", "weather", "electricity", "traffic"]


def _n_show(name, cap=6):
    return {
        "etth2": 7, "ettm1": 7, "exchange": 8, "weather": 6,
        "electricity": 6, "traffic": 6,
    }.get(name, cap)


def fig_raw_montage():
    """One PNG per dataset: N channels raw (natural units) side by side."""
    for name in DATASETS:
        try:
            d = get_protocol_dataset(name)
        except Exception as e:
            print(f"[s02] {name}: load failed {e}")
            continue
        X = d["X"]
        T = X.shape[0]
        S = min(d["meta"]["S"], _n_show(name))
        ncol = S
        fig, axes = plt.subplots(nrow := 1, ncol, figsize=(1.15 * ncol + 0.6, 2.2))
        axes = np.atleast_1d(axes)
        for s in range(ncol):
            ax = axes[s]
            ax.plot(np.arange(T), X[:, s], color=PALETTE["blue"], lw=0.5)
            format_axis(ax, title=f"ch {s}" if name != "electricity" else f"meter {s}")
            if s > 0:
                ax.set_yticks([])
        for s in range(ncol):
            axes[s].set_ylabel("")
        axes[0].set_ylabel("z-score")
        fig.suptitle(f"{name} — z-scored (train-only stats), T={T}",
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        save_fig(fig, os.path.join(OUT, f"raw_{name}.png"), panels=ncol,
                 caption=f"{name}: {S} channel subplots (correlated montage)")
    print("[s02] raw montages done")


def fig_splits():
    """Train/val/test split timeline for the 20/5/75 and ETT protocols."""
    for name in DATASETS:
        try:
            d = get_protocol_dataset(name)
        except Exception as e:
            print(f"[s02] {name}: load failed {e}")
            continue
        b = d["borders"]
        fig, ax = plt.subplots(figsize=(7.5, 1.4))
        blocks = [
            ("train", b["train"], PALETTE["blue"]),
            ("val", b["val"], PALETTE["sky"]),
            ("test", b["test"], PALETTE["grey"]),
        ]
        for lab, (s0, s1), c in blocks:
            ax.barh(0, s1 - s0, left=s0, color=c, edgecolor=PALETTE["ink"],
                    linewidth=0.6, height=0.5)
            ax.text((s0 + s1) / 2, 0, f"{s1 - s0}\n{lab}", ha="center",
                    va="center", fontsize=7.5, color=PALETTE["ink"])
        ax.set_xlim(0, d["meta"]["T"])
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks(np.arange(0, d["meta"]["T"] + 1, max(1, d["meta"]["T"] // 5)))
        ax.set_yticks([])
        ax.set_title(f"{name} — DSOF splits (T={d['meta']['T']})", fontsize=9)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        fig.tight_layout()
        save_fig(fig, os.path.join(OUT, f"splits_{name}.png"))
    print("[s02] splits done")


def fig_cadence():
    """Per-dataset seasonal/cadence view: autocorrelation + spectra of ch 0."""
    for name in DATASETS:
        try:
            d = get_protocol_dataset(name)
        except Exception as e:
            print(f"[s02] {name}: load failed {e}")
            continue
        x = d["X"][:, 0]
        x = x[np.isfinite(x)]
        T = len(x)
        # ACF up to one cadence period * 3
        from experiments.features import CADENCE, DEFAULT_LOOKBACK
        cad = CADENCE[name]
        maxlag = max(cad * 3, 96)
        corr = np.correlate(x - x.mean(), x - x.mean(), "full")[len(x) - 1:len(x) + maxlag]
        corr /= corr[0]
        # periodogram
        freqs = np.fft.rfftfreq(T, d=1.0)
        pw = np.abs(np.fft.rfft(x - x.mean())) ** 2

        fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
        axes[0].plot(np.arange(maxlag + 1), corr[:maxlag + 1],
                     color=PALETTE["blue"], lw=1.1)
        axes[0].axvline(cad, color=PALETTE["orange"], ls="--", lw=1.0,
                        label=f"cadence={cad}")
        format_axis(axes[0], title="autocorrelation", xlabel="lag",
                    ylabel="ACF")
        axes[0].legend(fontsize=7)
        axes[1].plot(freqs[1:], pw[1:], color=PALETTE["orange"], lw=0.8)
        format_axis(axes[1], title="power spectrum (first 1/cad)",
                    xlabel="freq", ylabel="power", logx=True, logy=True)
        fig.suptitle(f"{name} — cadence {cad}", fontweight="bold", y=1.05)
        fig.tight_layout()
        save_fig(fig, os.path.join(OUT, f"cadence_{name}.png"), panels=2, caption="autocorrelation + power spectrum")
    print("[s02] cadence done")


def fig_ecl_spikes():
    """ECL broken meters: train vs test range for channels 114 & 146."""
    d = get_protocol_dataset("electricity")
    X = d["X"]
    b = d["borders"]
    tr = (b["train"][0], b["train"][1])
    te = (b["test"][0], b["test"][1])
    for ch in (114, 146):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.7),
                                 gridspec_kw={"width_ratios": [1, 1, 1.2]})
        axes[0].plot(np.arange(*tr), X[tr[0]:tr[1], ch], color=PALETTE["blue"],
                     lw=0.6)
        format_axis(axes[0], title="train segment", xlabel="t", ylabel="z-score")
        axes[1].plot(np.arange(*te), X[te[0]:te[1], ch], color=PALETTE["red"],
                     lw=0.5)
        format_axis(axes[1], title="test segment", xlabel="t")
        # histogram of |z| on train vs test
        ax = axes[2]
        ztr = np.abs(X[tr[0]:tr[1], ch])
        zte = np.abs(X[te[0]:te[1], ch])
        ax.hist(ztr, bins=60, color=PALETTE["blue"], alpha=0.7, label="train",
                density=True)
        ax.hist(zte, bins=120, color=PALETTE["red"], alpha=0.5, label="test",
                density=True)
        ax.axvline(np.percentile(ztr, 99), color=PALETTE["blue"], ls="--", lw=1)
        format_axis(ax, title="|z| distribution", xlabel="|z|", ylabel="density")
        ax.legend(fontsize=7)
        mx = np.abs(X[:, ch]).max()
        fig.suptitle(f"ECL channel {ch} — test runs {mx:.0f}σ past train "
                     f"range ({X[tr[0]:tr[1], ch].max():.1f}max)",
                     fontweight="bold", y=1.06)
        fig.tight_layout()
        save_fig(fig, os.path.join(OUT, f"ecl_spike_ch{ch}.png"), panels=3, caption="train, test, |z| distribution")
    print("[s02] ECL spikes done")


def fig_ecl_trim():
    """Visualise the 3x clip policy on one healthy + one broken meter."""
    d = get_protocol_dataset("electricity")
    from experiments.data import get_protocol_dataset as gd
    dc = gd("electricity", clip_spikes=3.0)
    for ch in (114, 146):
        fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
        for ax, (Xlab, Xd) in zip(axes, [("raw", d), ("clip 3x", dc)]):
            X = Xd["X"]
            ax.plot(np.arange(X.shape[0]), X[:, ch], color=PALETTE["orange"],
                    lw=0.5)
            format_axis(ax, title=Xlab, xlabel="t", ylabel="z-score")
        fig.suptitle(f"ECL ch {ch}: 3x sign-preserving trim (train untouched)",
                     fontweight="bold", y=1.05)
        fig.tight_layout()
        save_fig(fig, os.path.join(OUT, f"ecl_trim_ch{ch}.png"), panels=2, caption="raw vs clip")
    print("[s02] ECL trim done")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_raw_montage()
    fig_splits()
    fig_cadence()
    fig_ecl_spikes()
    fig_ecl_trim()
    print("[s02_dataset] done")


if __name__ == "__main__":
    main()