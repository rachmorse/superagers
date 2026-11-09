#!/usr/bin/env python3
"""Superager vs. typical older adult SFC difference maps, rendered per atlas for Figure 3 and the supplement."""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yabplot as yab
from matplotlib.image import imread
from nilearn import datasets, plotting, surface

Atlas = Literal["schaefer200", "schaefer400"]

ATLASES: list[Atlas] = ["schaefer200", "schaefer400"]
SESSIONS: list[Literal["ses-01", "ses-02"]] = ["ses-01", "ses-02"]
LABEL_TYPE: Literal["long", "tp"] = "long"  # "long" = longitudinal superagers, "tp" = each session's own label
SCHAEFER_ROOT = Path("/home/rachel/Desktop/schaefer_analysis")
SUBJECTS = Path("/home/rachel/Desktop/data/final_cohort_ids.txt").read_text().split()  # Final longitudinal cohort
DEMOGRAPHICS_CSV = Path("/home/rachel/Desktop/data/superager.csv")
AVERAGE_NAME = "diff_superagers_average"

VMIN, VMAX = -0.03, 0.03  # Shared color scale for all maps
CMAP = "RdBu_r"
CORTEX_VIEWS = [("left", "lateral"), ("left", "medial"), ("right", "medial"), ("right", "lateral")]
SUBCORT_VIEWS = ["anterior", "superior"]
SUBCORT_SCALE = 0.85  # Subcortical views relative to true scale
BRAIN_GAP = 0.15  # Inches between views in a row
INK = "#222222"

SUBCORTICAL_NAME_MAP = {  # Lowercase region name -> yabplot aseg name
    "left hippocampus": "Left-Hippocampus",
    "left amygdala": "Left-Amygdala",
    "left pallidum": "Left-Pallidum",
    "left putamen": "Left-Putamen",
    "left caudate": "Left-Caudate",
    "left accumbens": "Left-Accumbens-area",
    "left thalamus": "Left-Thalamus",
    "right hippocampus": "Right-Hippocampus",
    "right amygdala": "Right-Amygdala",
    "right pallidum": "Right-Pallidum",
    "right putamen": "Right-Putamen",
    "right caudate": "Right-Caudate",
    "right accumbens": "Right-Accumbens-area",
    "right thalamus": "Right-Thalamus",
}


def average_dir(atlas: Atlas) -> Path:
    """Return the folder holding an atlas's averaged difference map and renders."""
    return SCHAEFER_ROOT / f"structure_function_coupling_{atlas}" / "average_across_sessions"


def trim_whitespace(img: np.ndarray) -> np.ndarray:
    """Crop white or transparent borders from an image array.

    Args:
        img: Image array of shape (H, W, 3) or (H, W, 4).

    Returns:
        np.ndarray: Cropped image with white borders removed.
    """
    threshold = 250 if img.dtype == np.uint8 else 0.98
    if img.ndim == 3 and img.shape[2] == 4:  # noqa: PLR2004
        # Exclude transparent and opaque-white pixels (nilearn renders white bg as fully opaque)
        alpha_mask = img[:, :, 3] > 0
        color_mask = ~np.all(img[:, :, :3] >= threshold, axis=2)
        mask = alpha_mask & color_mask
    else:
        mask = ~np.all(img[:, :, :3] >= threshold, axis=2)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax + 1, cmin:cmax + 1]


def crop_colorbar(img: np.ndarray) -> np.ndarray:
    """Remove the colorbar strip from the bottom of a yabplot image.

    Args:
        img: yabplot image array of shape (H, W, 3) or (H, W, 4).

    Returns:
        np.ndarray: Cropped image with colorbar removed.
    """
    threshold = 250 if img.dtype == np.uint8 else 0.98
    row_is_white = np.mean(np.all(img[:, :, :3] >= threshold, axis=2), axis=1) > 0.9  # noqa: PLR2004

    # tail -> colorbar -> gap -> brain
    state = "tail"
    for r in range(img.shape[0] - 1, img.shape[0] // 2, -1):
        if state == "tail" and not row_is_white[r]:
            state = "colorbar"
        elif state == "colorbar" and row_is_white[r]:
            state = "gap"
        elif state == "gap" and not row_is_white[r]:
            return img[:r + 1]
    return img


def build_subcortical_data(diff: pd.Series, name_map: dict[str, str]) -> dict[str, float]:
    """Map subcortical index entries to yabplot aseg names.

    Args:
        diff: ROI-indexed series of SFC difference values.
        name_map: Lowercase region suffix (e.g. 'left hippocampus') -> yabplot aseg name
            (e.g. 'Left-Hippocampus').

    Returns:
        dict: yabplot aseg region name -> SFC difference value.
    """
    subcortical = diff[diff.index.str.contains("Subcortical")]
    valid_yab_names = set(yab.get_atlas_regions(atlas="aseg", category="subcortical"))
    data, unmatched, invalid = {}, [], []

    for roi_name, value in subcortical.items():
        region = roi_name.split(": ", 1)[-1].lower()
        yab_name = name_map.get(region)
        if yab_name is None:
            unmatched.append(roi_name)
            continue
        if yab_name not in valid_yab_names:
            invalid.append((roi_name, yab_name))
            continue
        data[yab_name] = value

    if unmatched:
        print(f"  Subcortical unmatched in name_map: {unmatched}")
    if invalid:
        print(f"  Subcortical mapped to invalid yabplot names: {invalid}")
        print(f"  Valid aseg names: {sorted(valid_yab_names)}")
    if not unmatched and not invalid:
        print(f"  {len(data)} subcortical ROIs mapped and validated.")
    return data


def session_difference(atlas: Atlas, ses: str, label_type: str) -> pd.Series:
    """Compute the superager minus typical older adult Fisher-z coupling difference for one session.

    Args:
        atlas: Parcellation, e.g. "schaefer200".
        ses: Session ID (format: ses-01).
        label_type: "long" for longitudinally defined superagers, otherwise that session's label.

    Returns:
        pd.Series: Mean Fisher-z difference per ROI, named after the session.
    """
    individual_dir = SCHAEFER_ROOT / f"structure_function_coupling_{atlas}" / ses / "individual_coupling_matrices"
    coupling = pd.DataFrame({
        subject: pd.read_csv(individual_dir / f"{subject}_{ses}_structure_function_coupling.csv")
        .set_index("ROI_name")["pearson_rho"]
        for subject in SUBJECTS
    }).T
    fisher_z = np.arctanh(coupling)

    demographics = pd.read_csv(DEMOGRAPHICS_CSV)
    demographics["id"] = "sub-" + demographics["id"].astype(str).str.replace("^sub-", "", regex=True)
    label_col = "superager_long" if label_type == "long" else f"superager_tp{ses[-1]}"
    labels = demographics.set_index("id")[label_col].reindex(fisher_z.index)

    print(f"{atlas} {ses}: {int((labels == 1).sum())} superagers, {int((labels == 0).sum())} non-superagers")
    return (fisher_z[labels == 1].mean() - fisher_z[labels == 0].mean()).rename(ses)


def average_session_differences(atlas: Atlas) -> pd.Series:
    """Average the per-session difference maps across timepoints and save them as CSV.

    Args:
        atlas: Parcellation, e.g. "schaefer200".

    Returns:
        pd.Series: Mean difference per ROI across sessions.
    """
    diffs = [session_difference(atlas, ses, LABEL_TYPE) for ses in SESSIONS]
    average = pd.concat(diffs, axis=1).mean(axis=1).rename(AVERAGE_NAME)
    out_dir = average_dir(atlas)
    out_dir.mkdir(parents=True, exist_ok=True)
    average.to_frame().to_csv(out_dir / f"{AVERAGE_NAME}.csv")
    return average


def render_views(diff: pd.Series, atlas: Atlas) -> list[Path]:
    """Render the four cortical and two subcortical views of a difference map.

    Args:
        diff: ROI-indexed difference values, cortical and subcortical.
        atlas: Parcellation used to project cortical ROIs onto fsaverage5.

    Returns:
        list[Path]: PNGs in CORTEX_VIEWS then SUBCORT_VIEWS order.
    """
    out_dir = average_dir(atlas) / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cortex: fill each Schaefer parcel with its value, then project to the surface
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    schaefer = datasets.fetch_atlas_schaefer_2018(
        n_rois=int(atlas.removeprefix("schaefer")), yeo_networks=7, resolution_mm=2
    )
    atlas_img = nib.load(schaefer["maps"])
    atlas_data = atlas_img.get_fdata()
    names = [n.decode() if isinstance(n, bytes) else n for n in schaefer["labels"]]
    vol = np.zeros_like(atlas_data)
    for idx, name in enumerate(names, start=1):
        vol[atlas_data == idx] = diff[name]
    img = nib.Nifti1Image(vol, atlas_img.affine, atlas_img.header)

    pngs = []
    for hemi, view in CORTEX_VIEWS:
        data = surface.vol_to_surf(img, fsaverage[f"pial_{hemi}"], radius=1, n_samples=5)
        data[np.isclose(data, 0)] = np.nan  # Medial wall stays gray
        png = out_dir / f"{AVERAGE_NAME}_{hemi}_{view}.png"
        plotting.plot_surf_stat_map(
            fsaverage[f"pial_{hemi}"], data, hemi=hemi, view=view, colorbar=False, cmap=CMAP,
            vmin=VMIN, vmax=VMAX, threshold=None, bg_map=fsaverage[f"sulc_{hemi}"],
            bg_on_data=True, output_file=png,
        )
        plt.close("all")
        pngs.append(png)

    # Subcortex: yabplot aseg meshes
    sub_data = build_subcortical_data(diff, SUBCORTICAL_NAME_MAP)
    for view in SUBCORT_VIEWS:
        png = out_dir / f"{AVERAGE_NAME}_subcortical_{view}.png"
        yab.plot_subcortical(
            data=sub_data, atlas="aseg", vminmax=[VMIN, VMAX], cmap=CMAP, figsize=(500, 500),
            style="matte", bmesh_color="lightgray", bmesh_alpha=0.15, display_type="object",
            views=[view], export_path=str(png),
        )
        pngs.append(png)
    return pngs


def brain_views(atlas: Atlas) -> list[np.ndarray]:
    """Compute, render, and load the six cropped views for one atlas.

    Args:
        atlas: Parcellation, e.g. "schaefer200".

    Returns:
        list[np.ndarray]: Cropped images in CORTEX_VIEWS then SUBCORT_VIEWS order.
    """
    pngs = render_views(average_session_differences(atlas), atlas)
    n_cortex = len(CORTEX_VIEWS)
    images = [trim_whitespace(imread(p)) for p in pngs[:n_cortex]]
    images += [trim_whitespace(crop_colorbar(imread(p))) for p in pngs[n_cortex:]]
    return images


def brain_row_sizes(images: list[np.ndarray], row_w: float) -> list[tuple[float, float]]:
    """Size the six views on one physical scale so the row spans row_w inches.

    The anterior subcortical view spans brain height, like the cortex. The superior view
    spans brain length, like the width of a lateral cortex view.

    Args:
        images: Cropped views from brain_views.
        row_w: Total row width in inches, gaps included.

    Returns:
        list[tuple[float, float]]: (width, height) in inches per view.
    """
    lateral_ar = images[0].shape[1] / images[0].shape[0]
    rel_h = [1.0] * len(CORTEX_VIEWS) + [SUBCORT_SCALE, SUBCORT_SCALE * lateral_ar]
    unit_w = sum(r * im.shape[1] / im.shape[0] for r, im in zip(rel_h, images, strict=True))
    cortex_h = (row_w - BRAIN_GAP * (len(images) - 1)) / unit_w
    return [(cortex_h * r * im.shape[1] / im.shape[0], cortex_h * r) for r, im in zip(rel_h, images, strict=True)]


def draw_brain_row(
    fig: plt.Figure, images: list[np.ndarray], sizes: list[tuple[float, float]], left: float, bottom: float
) -> None:
    """Place a row of views on a figure, vertically centered on the tallest.

    Args:
        fig: Figure to draw on.
        images: Cropped views from brain_views.
        sizes: (width, height) in inches from brain_row_sizes.
        left: Row left edge in inches.
        bottom: Row bottom edge in inches.
    """
    fig_w, fig_h = fig.get_size_inches()
    row_h = max(h for _, h in sizes)
    x = left
    for img, (w, h) in zip(images, sizes, strict=True):
        ax = fig.add_axes((x / fig_w, (bottom + (row_h - h) / 2) / fig_h, w / fig_w, h / fig_h))
        ax.imshow(img)
        ax.axis("off")
        x += w + BRAIN_GAP


def add_colorbar(fig: plt.Figure, center_x: float, bottom: float, width: float = 3.0) -> None:
    """Add the shared horizontal SFC difference colorbar.

    Args:
        fig: Figure to draw on.
        center_x: Horizontal center in inches.
        bottom: Bar bottom edge in inches.
        width: Bar width in inches.
    """
    fig_w, fig_h = fig.get_size_inches()
    cax = fig.add_axes(((center_x - width / 2) / fig_w, bottom / fig_h, width / fig_w, 0.14 / fig_h))
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(VMIN, VMAX))
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    ticks = [VMIN, 0, VMAX]
    cbar.set_ticks(ticks, labels=[f"{v:.2f}".replace("-", "−") for v in ticks])  # True minus sign
    cbar.outline.set_visible(False)
    cbar.set_label("Structure-function coupling difference", fontsize=14, color=INK, labelpad=6)
    cbar.ax.tick_params(labelsize=13, length=0)


if __name__ == "__main__":
    for atlas in ATLASES:
        brain_views(atlas)
