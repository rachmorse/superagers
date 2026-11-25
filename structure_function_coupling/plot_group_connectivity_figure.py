#!/usr/bin/env python3
"""Plot group-averaged SC, FC and SFC for the SFC explanatory figure."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FC_MATRIX_FISHER_Z,
    FC_ROOTS,
    INDIVIDUAL_MATRIX_DIR,
    N_PARCELS_BY_ATLAS,
    SC_MATRIX,
    SC_ROOTS,
    SFC_FILE,
    SFC_INDIVIDUAL_DIR,
    SFC_ROOTS,
    SUBJECTS,
)

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer200"
SESSIONS_TO_AVERAGE: List[Literal["ses-01", "ses-02"]] = ["ses-01"]

OUTPUT_PNG = Path(__file__).parent / f"group_mean_connectivity_figure_{ATLAS}.png"
CACHE_PATH = Path(__file__).parent / f"group_means_{ATLAS}.npz"


def sc_path(subject: str, ses: str) -> Path:
    """Return the structural connectivity matrix path.

    Args:
        subject: Subject ID, e.g. "sub-1023".
        ses: Session ID, e.g. "ses-01".

    Returns:
        Path to the subject's SC matrix CSV.
    """
    return SC_ROOTS[ATLAS] / ses / INDIVIDUAL_MATRIX_DIR / SC_MATRIX.format(subject=subject, ses=ses)


def fc_path(subject: str, ses: str) -> Path:
    """Return the Fisher-z functional connectivity matrix path.

    Args:
        subject: Subject ID, e.g. "sub-1023".
        ses: Session ID, e.g. "ses-01".

    Returns:
        Path to the subject's FC matrix CSV.
    """
    return FC_ROOTS[ATLAS] / ses / INDIVIDUAL_MATRIX_DIR / FC_MATRIX_FISHER_Z.format(subject=subject, ses=ses)


def sfc_path(subject: str, ses: str) -> Path:
    """Return the structure-function coupling vector path.

    Args:
        subject: Subject ID, e.g. "sub-1023".
        ses: Session ID, e.g. "ses-01".

    Returns:
        Path to the subject's SFC CSV.
    """
    return SFC_ROOTS[ATLAS] / ses / SFC_INDIVIDUAL_DIR / SFC_FILE.format(subject=subject, ses=ses)


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
        load_fn(path_fn(subject, ses))
        for ses in SESSIONS_TO_AVERAGE
        for subject in SUBJECTS
        if path_fn(subject, ses).exists()
    ]
    print(f"{label}: averaged {len(arrays)} subject-sessions")
    return np.nanmean(np.stack(arrays), axis=0)


def load_or_compute_group_means(cache_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def style_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    image: AxesImage,
    title: str,
    cbar_label: str,
    ticks: np.ndarray,
    is_matrix: bool = True,
) -> Colorbar:
    """Add the title, region axes and colorbar to one panel.

    Args:
        fig: Figure holding the panel.
        ax: Panel axes.
        image: Image drawn on the panel.
        title: Panel title.
        cbar_label: Colorbar label.
        ticks: Region tick positions.
        is_matrix: True for the SC and FC matrices (regions on both axes), False for the SFC vector.

    Returns:
        The panel's colorbar.
    """
    ax.set_title(title, fontsize=22, pad=14)
    ax.set_xlabel("Regions", fontsize=22)
    ax.set_xticks(ticks)
    if is_matrix:
        ax.set_ylabel("Regions", fontsize=22)
        ax.set_yticks(ticks)
    else:
        ax.set_yticks([])
    ax.tick_params(labelsize=18)

    cax = make_axes_locatable(ax).append_axes("right", size="5%" if is_matrix else "2%", pad=0.12)
    cbar = fig.colorbar(image, cax=cax) if is_matrix else fig.colorbar(image, cax=cax, format="%.2f")
    cbar.set_label(cbar_label, fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    return cbar


def main() -> None:
    """Average connectivity across all subjects at ses-01 and save the three-panel figure.

    For the SFC explanatory figure finished in Lucidchart.
    """
    cmap = "RdBu_r"
    tick_positions = np.linspace(0, N_PARCELS_BY_ATLAS[ATLAS], 9, dtype=int)

    sc_mat, fc_mat, sfc_vals = load_or_compute_group_means(CACHE_PATH)
    np.fill_diagonal(sc_mat, 0)

    # Figure layout
    fig = plt.figure(figsize=(25, 15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 0.5], hspace=0.35, wspace=0.2)
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_fc = fig.add_subplot(gs[0, 1])
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
    style_panel(
        fig,
        ax_sc,
        im_sc,
        "Structural connectivity\nDiffusion weighted imaging - tractography",
        "Normalized white matter tracts",
        tick_positions,
    )

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
    style_panel(
        fig,
        ax_fc,
        im_fc,
        "Functional connectivity\nResting-state functional imaging - timeseries",
        "Fisher-z correlation coefficient",
        tick_positions,
    )

    # SFC vector panel
    sfc_max = np.nanmax(sfc_vals)
    im_sfc = ax_sfc.imshow(
        sfc_vals.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=sfc_max,
        interpolation="nearest",
    )
    cbar_sfc = style_panel(
        fig,
        ax_sfc,
        im_sfc,
        "Structure-function coupling\n"
        "Measure of correlation between each region's structural and functional connectivity",
        "Correlation\ncoefficient",
        tick_positions,
        is_matrix=False,
    )
    cbar_sfc.set_ticks([0, sfc_max / 2, sfc_max])  # label the top value

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved → {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
