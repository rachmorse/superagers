#!/usr/bin/env python3
"""
Plot the superager vs. non-superager SFC difference vector (averaged across
sessions) as a 1x214 heatmap with readable network and hemisphere labels.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
AVERAGE_CSV = (
    BASE_DIR
    / "average_across_sessions"
    / "diff_superagers_vs_non_superagers_average_baseline_followup_average.csv"
)
OUTPUT_PNG = (
    BASE_DIR
    / "average_across_sessions"
    / "visualizations"
    / "diff_superagers_average_vector.png"
)

CMAP = "RdBu_r"
VMIN = -0.03
VMAX = 0.03

# Full network display names (matching plot_feature_importance.py)
NETWORK_LABELS = {
    "Default":     "Default mode",
    "Cont":        "Executive control",
    "DorsAttn":    "Dorsal attention",
    "SalVentAttn": "Salience",
    "SomMot":      "Somatomotor",
    "Limbic":      "Limbic",
    "Vis":         "Visual",
    "Subcortex":   "Subcortical",
}

HEMISPHERE_GROUPS = [
    (0,   99,  "Left Hemisphere"),
    (100, 199, "Right Hemisphere"),
    (200, 213, "Subcortical"),
]


def parse_network(roi_name):
    """Extract network code from an ROI label string.

    Args:
        roi_name (str): ROI label (e.g. "7Networks_LH_Default_PCC" or
            "Subcortical_1_LH_Thal").

    Returns:
        str: Network code (e.g. "Default"), or "Subcortex" /
        "Unknown" when the label cannot be parsed.
    """
    if isinstance(roi_name, str):
        if roi_name.startswith("Subcortical"):
            return "Subcortex"
        if "7Networks" in roi_name:
            parts = roi_name.split("_")
            if len(parts) >= 3:
                return parts[2]
    return "Unknown"


def get_network_groups(roi_labels):
    """Return contiguous network spans from an ordered list of ROI labels.

    Args:
        roi_labels (list[str]): ROI name labels in plotting order.

    Returns:
        list[tuple[int, int, str]]: Each entry is
        (start_idx, end_idx, network_code) with inclusive indices.
    """
    groups = []
    current = None
    start = 0
    for i, name in enumerate(roi_labels):
        network = parse_network(name)
        if network != current:
            if current is not None:
                groups.append((start, i - 1, current))
            current = network
            start = i
    if current is not None:
        groups.append((start, len(roi_labels) - 1, current))
    return groups


def load_roi_labels():
    """Load ROI name labels from the first available subject SFC CSV.

    Searches ses-01/individual_coupling_matrices/ under BASE_DIR for
    any *_structure_function_coupling.csv file that contains a
    ROI_name column.

    Returns:
        list[str]: 214 ROI name labels, or generic "ROI_N" labels if no
        subject CSV is found.
    """
    subject_dir = BASE_DIR / "ses-01" / "individual_coupling_matrices"
    for csv_file in sorted(subject_dir.glob("*_structure_function_coupling.csv")):
        df = pd.read_csv(csv_file)
        if "ROI_name" in df.columns:
            return df["ROI_name"].astype(str).tolist()
    return [f"ROI_{i + 1}" for i in range(214)]


def main():
    """Load the average SFC difference vector and save a clean heatmap figure."""
    df = pd.read_csv(AVERAGE_CSV, index_col=0)
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    roi_labels = load_roi_labels()
    network_groups = get_network_groups(roi_labels)

    # Layout: heatmap row + label row 
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1.5], hspace=0.0)
    ax_heat  = fig.add_subplot(gs[0])
    ax_label = fig.add_subplot(gs[1])

    #  Heatmap 
    im = ax_heat.imshow(
        values.reshape(1, -1),
        aspect="auto",
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
    )
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])

    # Hemisphere dividers
    for start, _, _ in HEMISPHERE_GROUPS[1:]:
        ax_heat.axvline(start - 0.5, color="black", linewidth=2)

    # Colorbar — attach to both axes so they stay the same width
    cbar = fig.colorbar(im, ax=[ax_heat, ax_label], orientation="vertical", fraction=0.015, pad=0.02)
    cbar.set_ticks([VMIN, 0, VMAX])
    cbar.set_label("Structure-function\ncoupling difference", fontsize=22)
    cbar.ax.tick_params(labelsize=16)

    # Label axes 
    ax_label.set_xlim(-0.5, 213.5)
    ax_label.set_ylim(0, 1)
    ax_label.axis("off")

    # Network boundary tick marks at top of label axes
    for start, _, _ in network_groups:
        ax_label.plot([start - 0.5, start - 0.5], [0.75, 1.0],
                      color="gray", linewidth=0.8, transform=ax_label.transData, clip_on=False)
    ax_label.plot([213.5, 213.5], [0.75, 1.0],
                  color="gray", linewidth=0.8, transform=ax_label.transData, clip_on=False)

    # Network labels — alternate y-position so narrow spans don't overlap
    # Skip Subcortex since the hemisphere label below covers it
    for i, (start, end, label) in enumerate(network_groups):
        if label == "Subcortex":
            continue
        mid = (start + end) / 2
        span = end - start + 1
        y = 0.88 if (i % 2 == 0 or span < 6) else 0.65
        display_label = NETWORK_LABELS.get(label, label)
        ax_label.text(mid, y, display_label, ha="center", va="center",
                      fontsize=16, transform=ax_label.transData,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    # Hemisphere bracket lines and labels
    for start, end, label in HEMISPHERE_GROUPS:
        mid = (start + end) / 2
        # Horizontal bracket
        ax_label.plot([start - 0.5, end + 0.5], [0.38, 0.38],
                      color="black", linewidth=1.5, transform=ax_label.transData)
        # End ticks
        for x in [start - 0.5, end + 0.5]:
            ax_label.plot([x, x], [0.30, 0.38], color="black", linewidth=1.5,
                          transform=ax_label.transData)
        ax_label.text(mid, 0.12, label, ha="center", va="center",
                      fontsize=14, fontweight="bold", transform=ax_label.transData)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.2)
    print(f"Saved to {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
