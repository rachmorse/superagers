#!/usr/bin/env python3
"""Figure 3: (A) SFC difference map displayed above LME results for both atlases (B, superager status; C, episodic memory)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plot_sfc_difference as psd
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

LME_CSV = Path("/home/rachel/Desktop/data/lme_results.csv")  # Written by analyses.R
MAIN_ATLAS: psd.Atlas = "schaefer200"  # panel A
OUTPUT_PATH = Path(__file__).parent / "figures" / f"em_sfc_cortex_plot_{MAIN_ATLAS}.png"

# Main and secondary colors
INK = psd.INK
MUTED = "#6b6b6b"
OUTLINE_LW = 1.4  # pFDR cell outline

# Top to bottom
NETWORKS = ["Global", "Heteromodal", "DMN", "ECN", "SN", "Sensory"]
NETWORK_MAP = {  # Network name in analyses.R -> figure label
    "global": "Global",
    "hmod": "Heteromodal",
    "dmn": "DMN",
    "control": "ECN",
    "salience": "SN",
    "sensory": "Sensory",
}
ATLAS_STYLE = {  # Atlas -> (parcel count, CI line style, row offset from network center)
    "schaefer200": ("200", "-", -0.22),
    "schaefer400": ("400", (0, (1, 2)), 0.22),  # Dotted
}
OUTCOMES = {  # Model -> (panel title, color, x limits, x tick step)
    "group": ("B)  Superager status", "#2E6FA3", (-0.2, 0.62), 0.2),
    "memory": ("C)  Episodic memory", "#C1440E", (-0.05, 0.22), 0.1),
}


def load_stats(csv_path: Path, model: str) -> dict[str, dict[str, tuple[float, ...]]]:
    """Read LME stats for one model across atlases.

    Args:
        csv_path: lme_results.csv written by analyses.R.
        model: Model type, "group" or "memory".

    Returns:
        Atlas -> network label -> (beta, ci_lo, ci_hi, p, p_fdr).
    """
    df = pd.read_csv(csv_path)
    df = df[(df["model"] == model) & df["network"].isin(NETWORK_MAP)]
    return {
        atlas: {
            NETWORK_MAP[r.network]: (r.coef, r.ci_lo, r.ci_hi, r.p_value, r.p_fdr)
            for r in df[df["atlas"] == atlas].itertuples()
        }
        for atlas in ATLAS_STYLE
    }


def cell_style(p: float, p_fdr: float, color: str) -> dict[str, object]:
    """Return pFDR cell fill, edge, and text color for one of three significance states.

    Args:
        p: Uncorrected p-value.
        p_fdr: FDR-corrected p-value.
        color: Panel color for significant cells.

    Returns:
        Style dict with fill, edge, and text color.
    """
    if p_fdr < 0.05:  # Survives FDR correction: light tint of the panel color
        fill = tuple(1 - 0.18 * (1 - c) for c in to_rgb(color))
        return {"fill": fill, "edge": color, "text": INK}
    if p < 0.05:  # Nominally significant
        return {"fill": "white", "edge": color, "text": INK}
    return {"fill": "#f4f4f4", "edge": "none", "text": MUTED}


def fmt_pfdr(p_fdr: float) -> str:
    """Format pFDR for a cell."""
    return "<0.001" if p_fdr < 0.001 else f"{p_fdr:.3f}"  # noqa: PLR2004


def label_at(ax: plt.Axes, text: str, y: float, dx: float, **kw: object) -> None:
    """Write text at a data y position, offset dx points left of the axis."""
    ax.annotate(text, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(dx, 0),
                textcoords="offset points", va="center", annotation_clip=False, **kw)


def plot_forest(
    ax: plt.Axes,
    stats: dict[str, dict[str, tuple[float, ...]]],
    color: str,
    xlim: tuple[float, float],
    xstep: float,
) -> None:
    """Draw one forest panel with side-by-side pFDR cells per network.

    Args:
        ax: Axes to draw on.
        stats: Atlas -> network -> (beta, ci_lo, ci_hi, p, p_fdr).
        color: Marker, CI, and significant-cell color.
        xlim: (xmin, xmax) axis limits.
        xstep: x tick spacing.
    """
    # pFDR cells, in inches unless noted
    cell_left_pad = 0.15  # Forest edge to the first cell
    cell_w = 0.69  # Fits "0.044" at 13 pt
    cell_gap = 0.08  # Between the 200 and 400 cells
    cell_h = 0.84  # In data units where one network row is 1.0

    # Label offsets left of the forest in points
    parcel_dx = -8
    network_dx = -64

    # Cells sit right of the axes
    forest_w = ax.get_position().width * ax.figure.get_figwidth()
    cell_tf = blended_transform_factory(ax.transAxes, ax.transData)  # x in axes, y in data
    cell_wf = cell_w / forest_w
    cell_xs = [1 + (cell_left_pad + k * (cell_w + cell_gap)) / forest_w for k in range(len(ATLAS_STYLE))]

    ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
    for i, net in enumerate(NETWORKS):
        label_at(ax, net, i, network_dx, ha="right", fontsize=14, color=INK)
        for (atlas, (parcels, ls, offset)), cell_x in zip(ATLAS_STYLE.items(), cell_xs, strict=True):
            b, lo, hi, p, p_fdr = stats[atlas][net]
            y = i + offset
            ax.plot([lo, hi], [y, y], color=color, lw=2.2, ls=ls, solid_capstyle="round",
                    dash_capstyle="round", zorder=2)
            ax.plot(b, y, marker="o", ms=8, color=color, zorder=3)
            label_at(ax, parcels, y, parcel_dx, ha="right", fontsize=12, color=MUTED)

            style = cell_style(p, p_fdr, color)
            ax.add_patch(Rectangle((cell_x, i - cell_h / 2), cell_wf, cell_h, transform=cell_tf,
                                   facecolor=style["fill"], edgecolor=style["edge"],
                                   lw=OUTLINE_LW, clip_on=False, zorder=2))
            ax.text(cell_x + cell_wf / 2, i, fmt_pfdr(p_fdr), transform=cell_tf, ha="center",
                    va="center", fontsize=13, color=style["text"], clip_on=False, zorder=3)

    # Header rows
    head_y = -0.85
    label_at(ax, "SFC", head_y, network_dx, ha="right", fontsize=12, color=MUTED)
    label_at(ax, "Parcels", head_y, parcel_dx, ha="right", fontsize=12, color=MUTED)
    for (parcels, _, _), cell_x in zip(ATLAS_STYLE.values(), cell_xs, strict=True):
        ax.text(cell_x + cell_wf / 2, head_y, parcels, transform=cell_tf, ha="center",
                va="center", fontsize=12, color=MUTED, clip_on=False)
    ax.text((cell_xs[0] + cell_xs[-1] + cell_wf) / 2, head_y - 0.4, "pFDR", transform=cell_tf,
            ha="center", va="center", fontsize=12, color=MUTED, clip_on=False)

    ax.set_ylim(len(NETWORKS) - 0.5, -0.5)  # First network on top
    ax.set_yticks([])
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(xstep))
    ax.set_xlabel("Standardized β (95% CI)", fontsize=14, color=INK, labelpad=6)
    ax.tick_params(axis="x", labelsize=13, colors=MUTED, labelcolor=INK, length=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def add_legend(fig: plt.Figure) -> None:
    """Add the centered significance key. Atlas line styles are given in the caption."""
    colors = [c for _, c, _, _ in OUTCOMES.values()]

    def swatches(p: float, p_fdr: float, cols: list[str]) -> tuple[Patch, ...]:
        return tuple(Patch(facecolor=s["fill"], edgecolor=s["edge"], lw=OUTLINE_LW)
                     for s in (cell_style(p, p_fdr, c) for c in cols))

    handles = [swatches(0, 0, colors), swatches(0, 1, colors), swatches(1, 1, [MUTED])]
    labels = ["Significant after FDR correction", "Nominally significant", "Not significant"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3, frameon=False,
               fontsize=13, handlelength=2.2, columnspacing=2.0, bbox_to_anchor=(0.5, 0.0),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})


def main() -> None:
    """Assemble panel A above panels B and C and save Figure 3."""
    # Layout, in inches unless noted
    margin = 0.2  # Left edge to panel titles

    # Panel A (top row)
    a_row_span = 0.85  # Brain row width as a fraction of figure width
    a_title_space = 0.35  # Figure top to top of the brains
    a_brain_to_cbar = 0.02  # Bottom of the brains to the colorbar
    a_cbar_space = 0.52  # Colorbar plus its label
    a_to_bc_gap = 0.3  # Colorbar label to the B and C titles

    # Panels B and C (bottom row); each is labels | forest | pFDR cells
    bc_height = 5.8  # B and C titles to the legend
    bc_plot_bottom = 1.2  # Space under the plots for the x-axis and legend
    bc_plot_top = bc_height - 1.05  # Space over the plots for titles and headers
    label_w = 2.35  # Network names and parcel counts
    forest_w = 3.4  # Forest plot
    pfdr_w = 1.65  # Both pFDR cell columns
    panel_gap = 0.05  # Between B and C

    images = psd.brain_views(MAIN_ATLAS)
    panel_w = label_w + forest_w + pfdr_w
    fig_w = margin + 2 * panel_w + panel_gap
    sizes = psd.brain_row_sizes(images, a_row_span * fig_w)
    brain_h = max(h for _, h in sizes)
    fig_h = a_title_space + brain_h + a_cbar_space + a_to_bc_gap + bc_height
    fig = plt.figure(figsize=(fig_w, fig_h))

    def title_at(left: float, top: float, text: str) -> None:
        fig.text(left / fig_w, top / fig_h, text, ha="left", va="top", fontsize=16, color=INK)

    # Panel A: brain row centered across the full width
    brain_bottom = fig_h - a_title_space - brain_h
    row_w = sum(w for w, _ in sizes) + psd.BRAIN_GAP * (len(sizes) - 1)
    psd.draw_brain_row(fig, images, sizes, (fig_w - row_w) / 2, brain_bottom)
    psd.add_colorbar(fig, fig_w / 2, brain_bottom - a_brain_to_cbar - 0.14)
    title_at(margin, fig_h - 0.05,
             "A)  Structure-function coupling difference between superagers and typical older adults")

    # Panels B and C side by side
    for k, (model, (title, color, xlim, xstep)) in enumerate(OUTCOMES.items()):
        panel_left = margin + k * (panel_w + panel_gap)
        ax = fig.add_axes(((panel_left + label_w) / fig_w, bc_plot_bottom / fig_h, forest_w / fig_w,
                           (bc_plot_top - bc_plot_bottom) / fig_h))
        plot_forest(ax, load_stats(LME_CSV, model), color, xlim, xstep)
        title_at(panel_left, bc_height - 0.15, title)

    add_legend(fig)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
