#!/usr/bin/env python3
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    """Load single-subject connectivity data and save the three-panel figure."""
    base_dir = Path("/home/rachel/Desktop/schaefer_analysis")
    subject = "sub-4019"
    ses = "ses-02"

    sc_path = base_dir / f"structural_connectivity/{ses}/individual_connectivity_matrices/{subject}_{ses}_structural_connectivity_matrix.csv"
    fc_path = base_dir / f"functional_connectivity/native_space/{ses}/individual_connectivity_matrices/{subject}_{ses}_functional_connectivity_matrix_fisher_z.csv"
    sfc_path = base_dir / f"structure_function_coupling/{ses}/individual_coupling_matrices/{subject}_{ses}_structure_function_coupling.csv"
    output_png = Path(__file__).parent / "group_connectivity_figure.png"

    cmap = "RdBu_r"
    tick_positions = [0, 25, 50, 75, 100, 125, 150, 175, 200]

    # SC matrix
    sc_df = pd.read_csv(sc_path, index_col=0)
    sc_mat = sc_df.to_numpy()
    np.fill_diagonal(sc_mat, 0)

    # FC matrix
    fc_df = pd.read_csv(fc_path, index_col=0)
    fc_mat = fc_df.to_numpy()

    # SFC vector
    sfc_df = pd.read_csv(sfc_path)
    sfc_vals = sfc_df["pearson_rho"].to_numpy()

    # Figure layout
    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 0.5], hspace=0.35, wspace=0.2)
    ax_sc  = fig.add_subplot(gs[0, 0])
    ax_fc  = fig.add_subplot(gs[0, 1])
    ax_sfc = fig.add_subplot(gs[1, :])

    # SC matrix panel — make_axes_locatable keeps the colorbar flush with the image 
    im_sc = ax_sc.imshow(
        sc_mat,
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=sc_mat.max(),
        interpolation="nearest",
    )
    ax_sc.set_title(
        "Structural connectivity\nDiffusion weighted imaging - tractography",
        fontsize=22, pad=14,
    )
    ax_sc.set_xlabel("Regions", fontsize=22)
    ax_sc.set_ylabel("Regions", fontsize=22)
    ax_sc.set_xticks(tick_positions)
    ax_sc.set_yticks(tick_positions)
    ax_sc.tick_params(labelsize=18)
    divider_sc = make_axes_locatable(ax_sc)
    cax_sc = divider_sc.append_axes("right", size="5%", pad=0.12)
    cbar_sc = fig.colorbar(im_sc, cax=cax_sc)
    cbar_sc.set_label("Normalized white matter tracts", fontsize=22)
    cbar_sc.ax.tick_params(labelsize=18)

    # FC matrix panel
    fc_abs = np.nanmax(np.abs(fc_mat))
    im_fc = ax_fc.imshow(
        fc_mat,
        aspect="equal",
        cmap=cmap,
        vmin=-fc_abs,
        vmax=fc_abs,
        interpolation="nearest",
    )
    ax_fc.set_title(
        "Functional connectivity\nResting-state functional imaging - timeseries",
        fontsize=22, pad=14,
    )
    ax_fc.set_xlabel("Regions", fontsize=22)
    ax_fc.set_ylabel("Regions", fontsize=22)
    ax_fc.set_xticks(tick_positions)
    ax_fc.set_yticks(tick_positions)
    ax_fc.tick_params(labelsize=18)
    divider_fc = make_axes_locatable(ax_fc)
    cax_fc = divider_fc.append_axes("right", size="5%", pad=0.12)
    cbar_fc = fig.colorbar(im_fc, cax=cax_fc)
    cbar_fc.set_label("Fisher-z correlation coefficient", fontsize=22)
    cbar_fc.ax.tick_params(labelsize=18)

    # SFC vector panel
    im_sfc = ax_sfc.imshow(
        sfc_vals.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=np.nanmax(sfc_vals),
        interpolation="nearest",
    )
    ax_sfc.set_title(
        "Structure-function coupling\n"
        "Measure of correlation between each region's structural and functional connectivity",
        fontsize=22, pad=14,
    )
    ax_sfc.set_xlabel("Regions", fontsize=22)
    ax_sfc.set_yticks([])
    ax_sfc.set_xticks(tick_positions)
    ax_sfc.tick_params(labelsize=18)
    cbar_sfc = fig.colorbar(im_sfc, ax=ax_sfc, orientation="vertical", fraction=0.015, pad=0.02)
    cbar_sfc.set_label("Correlation\ncoefficient", fontsize=22)
    cbar_sfc.ax.tick_params(labelsize=18)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
