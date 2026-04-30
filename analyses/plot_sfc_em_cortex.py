#!/usr/bin/env python3
import re
from pathlib import Path
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from matplotlib.image import imread
from matplotlib.lines import Line2D
import numpy as np

# Import function from plot_sfc_difference.py 
from plot_sfc_difference import average_session_differences


def load_forest_stats(html_path: Path, region_map: dict):
    """Read sa_stats and em_stats from the LME rows of Table 3 in results.html.

    Returns two dicts mapping network name (beta, ci_lo, ci_hi, p, p_fdr).
    Parses the flextable HTML directly so the figure stays in sync with the
    results document without manual copy-paste.
    """
    soup = BeautifulSoup(html_path.read_text(), "html.parser")
    sa_stats, em_stats = {}, {}
    section  = None   # "sa" superager | "em" episodic memory
    in_slope = False

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue

        text = cells[0].get_text(strip=True)

        # Merged header/type rows have colspan="6" on the single cell
        if cells[0].get("colspan"):
            if "annual change" in text.lower():
                in_slope = True
            elif text == "Repeated measures linear mixed effects analyses":
                in_slope = False
            elif text == "Superager status":
                section = "sa"
            elif text == "Episodic memory":
                section = "em"
            continue

        # Data rows: skip if wrong section or in a slope / annual change results sub-section
        if len(cells) < 4 or text not in region_map or in_slope or section is None:
            continue

        coef_ci  = cells[1].get_text(strip=True)   # 0.2224 (0.0053-0.4395)
        p_raw    = cells[2].get_text(strip=True)    # 0.0449*
        pfdr_raw = cells[3].get_text(strip=True)    # 0.0606

        m = re.match(r"(-?[\d.]+)\s*\((-?[\d.]+)-(-?[\d.]+)\)", coef_ci)
        beta, ci_lo, ci_hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
        p    = float(re.sub(r"\*+", "", p_raw))
        pfdr = float(re.sub(r"\*+", "", pfdr_raw))

        net = region_map[text]
        if section == "sa":
            sa_stats[net] = (beta, ci_lo, ci_hi, p, pfdr)
        else:
            em_stats[net] = (beta, ci_lo, ci_hi, p, pfdr)

    return sa_stats, em_stats


# Forest plot for panel B and C
def plot_forest(
    ax: plt.Axes,
    stats: dict,
    title: str,
    xlabel: str,
    xlim: tuple,
    color: str,
    networks: list,
) -> None:
    """Draw a horizontal forest plot of network-level LME coefficients.

    Args:
        ax: Axes to draw on.
        stats: Dict mapping network name to (beta, ci_lo, ci_hi, p, p_fdr).
        title: Axes title string.
        xlabel: x-axis label.
        xlim: (xmin, xmax) axis limits; should leave room for pFDR annotations.
        color: Hex color string for the markers and error bars.
        networks: Ordered list of network names.
    """
    ys    = np.arange(len(networks))
    # pFDR annotations at a fixed distance right of the plot 
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)

    for i, net in enumerate(networks):
        b, lo, hi, p, pfdr = stats[net]
        fdr_sig = pfdr < 0.05
        nom_sig = p    < 0.05
        face    = color if nom_sig else "white"  # filled = p < .05, open = n.s.

        # Horizontal error bar with centre dot at beta, lines to CI bounds
        ax.errorbar(
            b, i,
            xerr=[[b - lo], [hi - b]],
            fmt="o", color=color,
            markerfacecolor=face, markeredgecolor=color,
            markeredgewidth=1.4, markersize=8,
            capsize=3, elinewidth=1.2, linewidth=1.2,
        )
        # Format pFDR with asterisk if significant, <.001 if very small
        pfdr_txt = "<.001*" if pfdr < 0.001 else (f"{pfdr:.3f}*" if fdr_sig else f"{pfdr:.3f}")
        ax.text(
            1.03, i, pfdr_txt,
            transform=trans,
            va="center", ha="left", fontsize=14,
            color="#444444",
            clip_on=False,
        )

    # Column header for the pFDR annotation column
    ax.text(
        1.03, len(networks) - 0.5, "pFDR",
        transform=trans,
        va="bottom", ha="left", fontsize=14,
        color="#444444",
        clip_on=False,
    )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)  # null effect line
    ax.set_yticks(ys)
    ax.set_yticklabels(networks, fontsize=14)
    # SFC label to sit just above the y-axis tick labels
    ax.text(
        -0.02, 1.02, "SFC",
        transform=ax.transAxes,
        va="bottom", ha="right", fontsize=14,
        clip_on=False,
    )
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title(title, fontsize=16, pad=8)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.6, len(networks) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    """Generate and save the figure.

    Calls average_session_differences to produce the averaged brain
    map (Panel A), loads the resulting PNG, then assembles it with two
    forest plots (Panels B and C) into a single figure.
    """
    session_dirs = {
        "01": Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-01/group_connectivity_matrices"),
        "02": Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-02/group_connectivity_matrices"),
    }
    average_output_dir = Path(
        "/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/average_across_sessions"
    )
    output_path = Path(
        "/home/rachel/Desktop/superagers/analyses/figures/em_sfc_cortex_plot.png"
    )
    label_type = "long"
    vmin       = -0.03
    vmax       =  0.03
    subcortical_name_map = {
        "left hippocampus":  "Left-Hippocampus",
        "left amygdala":     "Left-Amygdala",
        "left pallidum":     "Left-Pallidum",
        "left putamen":      "Left-Putamen",
        "left caudate":      "Left-Caudate",
        "left accumbens":    "Left-Accumbens-area",
        "left thalamus":     "Left-Thalamus",
        "right hippocampus": "Right-Hippocampus",
        "right amygdala":    "Right-Amygdala",
        "right pallidum":    "Right-Pallidum",
        "right putamen":     "Right-Putamen",
        "right caudate":     "Right-Caudate",
        "right accumbens":   "Right-Accumbens-area",
        "right thalamus":    "Right-Thalamus",
    }
    results_html = Path(__file__).parent.parent / "analyses" / "results.html"
    networks = ["Sensory", "SN", "ECN", "DMN", "Heteromodal", "Global"]
    region_map = {
        "Global SFC":      "Global",
        "Heteromodal SFC": "Heteromodal",
        "DMN SFC":         "DMN",
        "ECN SFC":         "ECN",
        "SN SFC":          "SN",
        "Sensory SFC":     "Sensory",
    }

    sa_stats, em_stats = load_forest_stats(results_html, region_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Panel A: generate brain surface map (colorbar added separately below)
    print("Generating brain surface map…")
    average_session_differences(
        session_dirs=session_dirs,
        label_type=label_type,
        vmin=vmin,
        vmax=vmax,
        colorbar_label=None,
        include_subcortical=True,
        subcortical_name_map=subcortical_name_map,
    )
    plt.close("all")

    brain_png = average_output_dir / "visualizations" / "diff_superagers_average.png"
    brain_img = imread(str(brain_png))

    # Assemble combined figure
    # Left column: brain image + horizontal colorbar below.
    # Right column: forest plots B and C stacked.
    brain_ar     = brain_img.shape[0] / brain_img.shape[1]  # H / W
    brain_scale  = 0.85  # factor to scale down the brain image column width
    forest_h     = 2.5   # height of each forest panel
    forest_vgap  = 1.4   # vertical gap between panels B and C
    fig_h        = 2 * forest_h + forest_vgap
    brain_col_w  = (fig_h / brain_ar) * brain_scale
    forest_col_w = 6.0
    fig_w        = brain_col_w + forest_col_w + 5.5

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Set up figure layout with two columns (brain, forest)
    # with the forest column split into two rows (B and C)
    gs_outer = gridspec.GridSpec(
        1, 2,
        figure=fig,
        width_ratios=[brain_col_w, forest_col_w],
        wspace=0.08,
    )
    gs_left = gridspec.GridSpecFromSubplotSpec(
        1, 1,
        subplot_spec=gs_outer[0],
    )
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs_outer[1],
        height_ratios=[1, 1],
        hspace=forest_vgap / forest_h,
    )

    # Panel A
    ax_a = fig.add_subplot(gs_left[0])
    ax_a.imshow(brain_img, aspect="equal")
    ax_a.set_aspect("equal", anchor="W")
    ax_a.axis("off")
    ax_a.set_title(
        "A)  Structure-function coupling difference between superagers and non-superagers",
        fontsize=16, pad=8, loc="left",
    )

    # Horizontal colorbar just below the brain panel
    ax_cbar = ax_a.inset_axes([0.4, -0.07, 0.3, 0.04])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("RdBu_r"), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal", format="%.2f")
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_label("Structure-function coupling difference", fontsize=14, labelpad=6)
    cbar.ax.tick_params(labelsize=14)

    # Panels B and C
    ax_b = fig.add_subplot(gs_right[0])
    ax_c = fig.add_subplot(gs_right[1])

    plot_forest(
        ax_b, sa_stats,
        title="B)  Superager status",
        xlabel="Standardised β (95% CI)",
        xlim=(-0.15, 0.53),
        color="#2E6FA3",
        networks=networks,
    )
    plot_forest(
        ax_c, em_stats,
        title="C)  Episodic memory",
        xlabel="Standardised β (95% CI)",
        xlim=(-0.05, 0.18),
        color="#C1440E",
        networks=networks,
    )

    # Legend below panel C
    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#555555", markeredgecolor="#555555",
               markersize=8, label="p < .05"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="white", markeredgecolor="#555555",
               markeredgewidth=1.4, markersize=8, label="n.s."),
    ]
    ax_c.legend(
        handles=legend_elements,
        fontsize=14,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.28),
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
