"""figuregen/style.py — global A1-grade publication style.

Single source of truth for the paper figure suite. Every plot in figures/
derives its MPL rcParams, colourmap and palette from here so the whole paper
reads as one cohesive, top-tier visual system (Nature / NeurIPS / DeepMind aesthetic):
Clean serif/sans typography, Okabe-Ito & modern scientific color palettes, high DPI,
crisp container cards, elegant badges, and structured layout styling.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Refined A1 Publication Palette (Scientific, contrast-rich, colorblind-safe)
PALETTE = {
    "ink":          "#0F172A",   # Deep slate / near-black text
    "ink_light":    "#334155",   # Body text / secondary labels
    "muted":        "#64748B",   # Subtitles, gridlines, inactive elements
    "paper":        "#FFFFFF",   # Figure canvas background
    "panel_bg":     "#F8FAFC",   # Container background fill
    "card_bg":      "#FFFFFF",   # Inner card fill
    "border":       "#CBD5E1",   # Default border stroke
    "border_dark":  "#94A3B8",   # Emphasized border stroke
    
    # Category Accents (Distinct, vibrant, colorblind-safe)
    "blue":         "#0284C7",   # Data / Input (Sky Blue)
    "blue_light":   "#E0F2FE",
    "blue_border":  "#38BDF8",
    
    "orange":       "#D97706",   # Core Solver / Feature Map (Amber)
    "orange_light": "#FEF3C7",
    "orange_border":"#F59E0B",
    
    "green":        "#059669",   # Online Update / Woodbury RLS (Emerald)
    "green_light":  "#D1FAE5",
    "green_border": "#34D399",
    
    "purple":       "#7C3AED",   # S2 LRU Reservoir / Context Bank (Violet)
    "purple_light": "#EDE9FE",
    "purple_border":"#A78BFA",
    
    "red":          "#DC2626",   # Backprop / Metrics / Baseline (Crimson)
    "red_light":    "#FEE2E2",
    "red_border":   "#F87171",

    "teal":         "#0D9488",   # Multi-horizon / Evaluation (Teal)
    "teal_light":   "#CCFBF1",
    "teal_border":  "#2DD4BF",
    
    "grey":         "#64748B",   # Published benchmark / Neutral
    "grey_light":   "#F1F5F9",
    "grey_border":  "#E2E8F0",
    
    # Aliases for backwards compatibility with section modules
    "sky":          "#38BDF8",
    "violet":       "#8B5CF6",
    "yellow":       "#F59E0B",
}

# Sequential accent colormap for heatmaps / densities
ACCENT_CMAP = "YlGnBu"

MODEL_COLORS = {
    "static": PALETTE["blue"],
    "rls":    PALETTE["orange"],
    "s2":     PALETTE["purple"],
    "s2rls":  PALETTE["green"],
}

PUBLISHED_COLOR = PALETTE["grey"]
OURS_EDGE = PALETTE["ink"]

MODEL_ORDER = ["static", "rls", "s2", "s2rls"]
MODEL_LABELS = {
    "static": "Static (closed-form ridge)",
    "rls":    "Ours-RLS (online Woodbury)",
    "s2":     "Ours-S2 (LRU context)",
    "s2rls":  "Ours-S2RLS",
}


def _load_stix():
    """Return the best available font settings."""
    import matplotlib.font_manager as fm
    known = {f.name for f in fm.fontManager.ttflist}
    stix = "STIX Two Text"
    if stix not in known:
        stix = "STIXGeneral"
    return stix, "DejaVu Sans", "DejaVu Serif"


def configure(dpi: int = 300, font_scale: float = 1.0):
    """Apply global publication rcParams. Call once per process."""
    stix, sans, serif = _load_stix()
    base = 9.0 * font_scale
    mpl.rcParams.update({
        "backend": "Agg",
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "font.family": "sans-serif",
        "font.sans-serif": [sans, "Arial", "Helvetica", "DejaVu Sans"],
        "font.serif": [stix, "Times New Roman", serif],
        "mathtext.fontset": "cm",
        "axes.titlesize": base + 3.0,
        "axes.titleweight": "bold",
        "axes.labelsize": base + 1.5,
        "axes.labelweight": "normal",
        "axes.edgecolor": PALETTE["border_dark"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#EEF2F7",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.fontsize": base,
        "legend.frameon": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": PALETTE["border"],
        "legend.fancybox": True,
        "figure.facecolor": PALETTE["paper"],
        "axes.facecolor": PALETTE["paper"],
        "savefig.facecolor": PALETTE["paper"],
        "image.cmap": ACCENT_CMAP,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "errorbar.capsize": 3.0,
        "hatch.linewidth": 0.8,
    })


def save_fig(fig, path, close: bool = True, panels: int = 1, caption: str = ""):
    """Persist a figure and write sidecar metadata."""
    import json as _json
    fig.savefig(path, dpi=fig.get_dpi())
    meta = {
        "panels": int(panels),
        "caption": caption,
        "file": os.path.basename(path),
    }
    with open(path[:-4] + ".meta.json", "w") as fh:
        _json.dump(meta, fh, indent=2)
    if close:
        plt.close(fig)
    return path


def panel_label(ax, tag):
    """Add bold '(a)' / '(b)' corner tag used in multi-panel figures."""
    ax.text(-0.06, 1.03, tag, transform=ax.transAxes, fontweight="bold",
            fontsize=12, va="bottom", ha="right", color=PALETTE["ink"])


def format_axis(ax, xlabel=None, ylabel=None, title=None, logx=False, logy=False):
    """Consistent axis dressing."""
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def sci(x):
    """Compact scientific string."""
    return f"{x:.2e}"


def pct(x):
    return f"{100 * x:.1f}%"


def setup_matplotlib():
    """Idempotent entrypoint."""
    configure()


if mpl.rcParams.get("backend") != "Agg" or not mpl.rcParams.get("axes.grid"):
    configure()