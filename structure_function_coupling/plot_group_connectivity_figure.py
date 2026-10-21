#!/usr/bin/env python3
"""Plot group-averaged SC, FC and SFC for the SFC explanatory figure."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from run_sfc_pipeline import ROOT, SESSIONS, SUBJECTS

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"

FC_ROOT = ROOT / f"functional_connectivity_{ATLAS}" / "native_space"
STRUCT_ROOT = ROOT / f"structural_connectivity_{ATLAS}"
SFC_ROOT = ROOT / f"structure_function_coupling_{ATLAS}"


def sc_path(subject: str, ses: str) -> Path:
    """Return the structural connectivity matrix path.

    Args:
        subject: Subject ID, e.g. "sub-1023".
        ses: Session ID, e.g. "ses-01".

    Returns:
        Path to the subject's SC matrix CSV.
    """
    filename = f"{subject}_{ses}_structural_connectivity_matrix.csv"
    return STRUCT_ROOT / ses / "individual_connectivity_matrices" / filename


def fc_path(subject: str, ses: str) -> Path:
    """Return the Fisher-z functional connectivity matrix path.

    Args:
        subject: Subject ID, e.g. "sub-1023".
        ses: Session ID, e.g. "ses-01".

    Returns:
        Path to the subject's FC matrix CSV.
    """
    filename = f"{subject}_{ses}_functional_connectivity_matrix_fisher_z.csv"
    return FC_ROOT / ses / "individual_connectivity_matrices" / filename


def sfc_path(subject: str, ses: str) -> Path:
    """Return the structure-function coupling vector path.

    Args:
        subject: Subject ID, e.g. "sub-1023".
        ses: Session ID, e.g. "ses-01".

    Returns:
        Path to the subject's SFC CSV.
    """
    filename = f"{subject}_{ses}_structure_function_coupling.csv"
    return SFC_ROOT / ses / "individual_coupling_matrices" / filename


def load_matrix(path: Path) -> np.ndarray:
    """Load a region-by-region connectivity matrix.

    Args:
        path: Path to a CSV with region labels as the index.

    Returns:
        Connectivity matrix as a 2D array.
    """
    return pd.read_csv(path, index_col=0).to_numpy()


def load_sfc(path: Path) -> np.ndarray:
    """Load regional SFC values.

    Args:
        path: Path to an SFC CSV with a "pearson_rho" column.

    Returns:
        SFC values as a 1D array.
    """
    return pd.read_csv(path)["pearson_rho"].to_numpy()


def group_mean(
    path_fn: Callable[[str, str], Path],
    load_fn: Callable[[Path], np.ndarray],
    label: str,
) -> np.ndarray:
    """Average data across all subjects and sessions, skipping missing files.

    Args:
        path_fn: Maps (subject, session) to a file path.
        load_fn: Loads a file path into an array.
        label: Name used in the printed summary.

    Returns:
        Element-wise mean across all available subject-sessions.
    """
    arrays = [
        load_fn(path_fn(subject, ses)) for ses in SESSIONS for subject in SUBJECTS if path_fn(subject, ses).exists()
    ]
    print(f"{label}: averaged {len(arrays)} subject-sessions")
    return np.nanmean(np.stack(arrays), axis=0)


def load_or_compute_group_means(cache_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached group means, or compute and cache them.

    Delete the cache file to recompute, e.g. after the subject list changes.

    Args:
        cache_path: Path to the .npz cache file.

    Returns:
        Group mean SC matrix, FC matrix and SFC vector.
    """
    if cache_path.exists():
        print(f"Loading cached group means from {cache_path}")
        cached = np.load(cache_path)
        return cached["sc"], cached["fc"], cached["sfc"]

    sc_mat = group_mean(sc_path, load_matrix, "SC")
    fc_mat = group_mean(fc_path, load_matrix, "FC")
    sfc_vals = group_mean(sfc_path, load_sfc, "SFC")
    np.savez(cache_path, sc=sc_mat, fc=fc_mat, sfc=sfc_vals)
    print(f"Cached group means → {cache_path}")
    return sc_mat, fc_mat, sfc_vals


def main() -> None:
    """Average connectivity across all subjects at ses-01 and ses-02 and save the three-panel figure.

    For the SFC explanatory figure finished in Lucidchart.
    """
    output_png = Path(__file__).parent / f"group_mean_connectivity_figure_{ATLAS}.png"
    cache_path = Path(__file__).parent / f"group_means_{ATLAS}.npz"

    cmap = "RdBu_r"
    n_parcels = int(ATLAS.removeprefix("schaefer"))
    tick_positions = np.linspace(0, n_parcels, 9, dtype=int)

    sc_mat, fc_mat, sfc_vals = load_or_compute_group_means(cache_path)
    np.fill_diagonal(sc_mat, 0)

    # Figure layout
    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 0.5], hspace=0.35, wspace=0.2)
    ax_sc  = fig.add_subplot(gs[0, 0])
    ax_fc  = fig.add_subplot(gs[0, 1])
    ax_sfc = fig.add_subplot(gs[1, :])

    # SC matrix panel
    im_sc = ax_sc.imshow(
        sc_mat,
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=np.nanmax(sc_mat),
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
    divider_sfc = make_axes_locatable(ax_sfc)
    cax_sfc = divider_sfc.append_axes("right", size="2%", pad=0.12)
    cbar_sfc = fig.colorbar(im_sfc, cax=cax_sfc, format="%.2f")
    sfc_max = np.nanmax(sfc_vals)
    cbar_sfc.set_ticks([0, sfc_max / 2, sfc_max])  # label the top value
    cbar_sfc.set_label("Correlation\ncoefficient", fontsize=22)
    cbar_sfc.ax.tick_params(labelsize=18)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Saved → {output_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
