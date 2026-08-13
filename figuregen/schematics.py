"""figuregen/schematics.py — publication-grade diagram primitives.

Implements clean duo-tone color theory, strict mapping conventions, bus connectors,
and modern paper typography:
- Duo-color scheme: Slate/Grayscale for standard elements + Single Amber Accent for proposed solver
- Custom color overrides for white-fill / black-stroke nodes
- Rectangles = operations/solvers; Rounded capsules = data/tensors
- Left-to-right flow with zero crossing arrows
- Bus connectors for clean tree-like feature aggregation (no arrow chaos)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figuregen.style import save_fig

# Duo-Tone Color Palette
COLOR_NEUTRAL_BG = "#F8FAFC"
COLOR_NEUTRAL_STROKE = "#64748B"
COLOR_NEUTRAL_TEXT = "#0F172A"

COLOR_ACCENT_BG = "#FEF3C7"
COLOR_ACCENT_STROKE = "#D97706"
COLOR_ACCENT_TEXT = "#92400E"

COLOR_CONTAINER_BG = "#F1F5F9"
COLOR_CONTAINER_STROKE = "#E2E8F0"

COLOR_WHITE_BG = "#FFFFFF"
COLOR_BLACK_STROKE = "#0F172A"


def draw_node(ax, x, y, w, h, title, sub=None, shape="rect", accent=False,
              fill_override=None, stroke_override=None, text_override=None,
              title_size=8.5, sub_size=7.2):
    """Draw a diagram node adhering to strict mapping conventions:
    - shape='rect': Operation / module / solver
    - shape='rounded': Data representation / tensor / vector
    - accent=True: Single accent color reserved for the proposed solver contribution
    - fill_override / stroke_override: Optional explicit color overrides (e.g. white fill, black stroke)
    """
    if fill_override:
        fill_color = fill_override
        stroke_color = stroke_override if stroke_override else COLOR_BLACK_STROKE
        text_color = text_override if text_override else COLOR_NEUTRAL_TEXT
        lw = 1.1
    else:
        fill_color = COLOR_ACCENT_BG if accent else COLOR_NEUTRAL_BG
        stroke_color = COLOR_ACCENT_STROKE if accent else COLOR_NEUTRAL_STROKE
        text_color = COLOR_ACCENT_TEXT if accent else COLOR_NEUTRAL_TEXT
        lw = 1.2 if accent else 0.8

    rounding = 0.02 if shape == "rect" else 0.12

    p = FancyBboxPatch((x - w / 2.0, y - h / 2.0), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                       linewidth=lw, edgecolor=stroke_color,
                       facecolor=fill_color, zorder=3)
    ax.add_patch(p)

    if sub:
        ax.text(x, y + 0.15 * h, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold" if accent else "semibold",
                color=text_color, zorder=4)
        ax.text(x, y - 0.20 * h, sub, ha="center", va="center",
                fontsize=sub_size, color=text_color, alpha=0.85, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold" if accent else "semibold",
                color=text_color, zorder=4)
    return p


def draw_arrow(ax, x0, y0, x1, y1, dashed=False, accent=False, label=None, rad=0.0,
               label_dy=0.20, label_dx=0.0):
    """Draw a clean connector arrow maintaining left-to-right flow."""
    color = COLOR_ACCENT_STROKE if accent else COLOR_NEUTRAL_STROKE
    linestyle = "--" if dashed else "-"
    connectionstyle = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0.0"

    a = FancyArrowPatch((x0, y0), (x1, y1),
                        connectionstyle=connectionstyle,
                        arrowstyle="-|>",
                        mutation_scale=12,
                        linewidth=1.0,
                        color=color,
                        linestyle=linestyle,
                        zorder=2)
    ax.add_patch(a)

    if label:
        mx, my = (x0 + x1) / 2.0 + label_dx, (y0 + y1) / 2.0 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
                color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.20", fc="#FFFFFF", ec="none", alpha=0.95),
                zorder=6)
    return a


def draw_bus_manifold(ax, x_in, y_in, x_out, y_nodes, accent=False):
    """Draw a clean bus connector manifold (fork & join) for multi-branch features."""
    color = COLOR_ACCENT_STROKE if accent else COLOR_NEUTRAL_STROKE
    y_min, y_max = min(y_nodes), max(y_nodes)
    x_bus_in = x_in + 0.5
    x_bus_out = x_out - 0.5

    # Input -> Bus line
    ax.plot([x_in, x_bus_in], [y_in, y_in], color=color, lw=1.0, zorder=2)
    ax.plot([x_bus_in, x_bus_in], [y_min, y_max], color=color, lw=1.0, zorder=2)

    # Output Bus line -> Output
    ax.plot([x_bus_out, x_bus_out], [y_min, y_max], color=color, lw=1.0, zorder=2)
    ax.plot([x_bus_out, x_out], [y_in, y_in], color=color, lw=1.0, zorder=2)
    
    # Arrow to final output
    a = FancyArrowPatch((x_bus_out, y_in), (x_out, y_in),
                        arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color=color, zorder=2)
    ax.add_patch(a)


def draw_scope_container(ax, x, y, w, h, title, accent=False):
    """Draw a subtle background region box grouping logical steps."""
    stroke = COLOR_ACCENT_STROKE if accent else COLOR_CONTAINER_STROKE
    fill = COLOR_ACCENT_BG if accent else COLOR_CONTAINER_BG
    alpha = 0.25 if accent else 0.40

    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.03,rounding_size=0.10",
                       linewidth=1.0, edgecolor=stroke,
                       facecolor=fill, alpha=alpha, zorder=0)
    ax.add_patch(p)
    ax.text(x + 0.25, y + h - 0.22, title.upper(), fontsize=8.0, fontweight="bold",
            color=COLOR_ACCENT_TEXT if accent else COLOR_NEUTRAL_STROKE, ha="left", va="center", zorder=1)
    return p