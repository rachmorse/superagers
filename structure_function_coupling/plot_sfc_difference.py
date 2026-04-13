#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from matplotlib.image import imread
from nilearn import datasets, surface, plotting
from nilearn.datasets import fetch_surf_fsaverage


def trim_whitespace(img: np.ndarray) -> np.ndarray:
    """Crop white or transparent borders from an image array.

    Args:
        img: Image array of shape (H, W, 3) or (H, W, 4).

    Returns:
        np.ndarray: Cropped image with white borders removed.
    """
    threshold = 250 if img.dtype == np.uint8 else 0.98
    if img.ndim == 3 and img.shape[2] == 4:
        # Exclude transparent pixels AND opaque-white pixels (nilearn renders white bg as fully opaque)
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


def visualize_coupling(
    coupling_file,
    group_name,
    output_dir,
    ses,
    vmin=None,
    vmax=None,
    file_suffix="sfc",
    symmetric_scale=False,
    colorbar_label=None,
):
    """Create a four-view brain surface visualization from ROI-level values.

    Plots left lateral, left medial, right medial, and right lateral views
    by projecting parcellated ROI values onto the fsaverage surface using
    the Schaefer 200-ROI atlas.

    Args:
        coupling_file: Directory containing {group_name}_average.csv.
        group_name: Name of the group CSV to load (without _average.csv).
        output_dir: Directory where the combined PNG will be saved.
        ses: Timepoint.
        vmin: Minimum value for colormap scaling. If None, inferred from
            the data.
        vmax: Maximum value for colormap scaling. If None, inferred from
            the data.
        file_suffix: Short suffix appended to the output PNG filename.
        symmetric_scale: If True and vmin/vmax are None,
            forces a symmetric range around zero.
        colorbar_label: Optional label for the colorbar y-axis.

    Returns:
        matplotlib.figure.Figure: The combined four-view figure.
    """
    coupling_csv = Path(f"{coupling_file}/{group_name}.csv")
    coupling_df = pd.read_csv(coupling_csv, index_col=0)
    coupling_df = coupling_df[~coupling_df.index.str.contains("Subcortical")]

    rho_values = coupling_df.iloc[:, 0].values

    if vmin is None or vmax is None:
        if symmetric_scale:
            max_abs = np.nanmax(np.abs(rho_values))
            inferred_vmin, inferred_vmax = -max_abs, max_abs
        else:
            inferred_vmin = np.nanmin(rho_values)
            inferred_vmax = np.nanmax(rho_values)
        if vmin is None:
            vmin = inferred_vmin
        if vmax is None:
            vmax = inferred_vmax

    output_path = Path(output_dir) / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)

    fsaverage = fetch_surf_fsaverage("fsaverage5")
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)

    atlas_img = nib.load(schaefer["maps"])
    atlas_data = atlas_img.get_fdata()
    atlas_roi_names = schaefer["labels"]
    if isinstance(atlas_roi_names[0], bytes):
        atlas_roi_names = [label.decode("utf-8") for label in atlas_roi_names]

    roi_name_to_atlas_idx = {name: i for i, name in enumerate(atlas_roi_names)}

    coupling_vol = np.zeros_like(atlas_data)
    mapped_count = 0
    for roi_name, value in zip(coupling_df.index, rho_values):
        if roi_name in roi_name_to_atlas_idx:
            atlas_idx = roi_name_to_atlas_idx[roi_name]
            coupling_vol[atlas_data == (atlas_idx + 1)] = value
            mapped_count += 1
        else:
            print(f"Warning: Could not find a matching atlas ROI for {roi_name}")

    print(f"Successfully mapped {mapped_count} out of {len(coupling_df.index)} ROIs")

    coupling_img = nib.Nifti1Image(coupling_vol, atlas_img.affine, atlas_img.header)

    surf_data_left = surface.vol_to_surf(coupling_img, fsaverage["pial_left"], radius=3, n_samples=5)
    surf_data_right = surface.vol_to_surf(coupling_img, fsaverage["pial_right"], radius=3, n_samples=5)

    surf_data_left[np.isclose(surf_data_left, 0)] = np.nan
    surf_data_right[np.isclose(surf_data_right, 0)] = np.nan

    temp_output_path = output_path / "temp"
    temp_output_path.mkdir(parents=True, exist_ok=True)

    cold_hot_cmap = plt.get_cmap("RdBu_r")

    views = [
        ("left",  "lateral", surf_data_left,  fsaverage["sulc_left"]),
        ("left",  "medial",  surf_data_left,  fsaverage["sulc_left"]),
        ("right", "medial",  surf_data_right, fsaverage["sulc_right"]),
        ("right", "lateral", surf_data_right, fsaverage["sulc_right"]),
    ]

    # Save all views without a colorbar so every PNG crops consistently
    temp_paths = {}
    for hemi, view, surf_data, sulc in views:
        p = temp_output_path / f"{group_name}_{ses}_{hemi}_{view}.png"
        plotting.plot_surf_stat_map(
            fsaverage[f"pial_{hemi}"],
            surf_data,
            hemi=hemi,
            view=view,
            colorbar=False,
            cmap=cold_hot_cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=None,
            bg_map=sulc,
            bg_on_data=True,
            output_file=p,
        )
        temp_paths[f"{hemi}_{view}"] = p

    row_keys = ["left_lateral", "left_medial", "right_medial", "right_lateral"]
    images = {key: trim_whitespace(imread(temp_paths[key])) for key in row_keys}

    # Scale uniformly so the tallest brain = 3 inches tall
    max_h_px = max(images[k].shape[0] for k in row_keys)
    scale = 3.0 / max_h_px

    brain_widths_in  = [images[k].shape[1] * scale for k in row_keys]
    brain_heights_in = [images[k].shape[0] * scale for k in row_keys]
    row_h_in = max(brain_heights_in)

    gap_in = 0.25  # whitespace between brains
    cbar_w_in = 3.0  # wide enough to fit rotated label without clipping
    total_w_in = sum(brain_widths_in) + gap_in * (len(row_keys) - 1)
    fig_w_in = total_w_in + cbar_w_in
    fig = plt.figure(figsize=(fig_w_in, row_h_in))

    x_cursor = 0.0
    for key, bw_in, bh_in in zip(row_keys, brain_widths_in, brain_heights_in):
        img = images[key]
        y_off = (row_h_in - bh_in) / 2
        ax = fig.add_axes([
            x_cursor / fig_w_in,
            y_off / row_h_in,
            bw_in / fig_w_in,
            bh_in / row_h_in,
        ])
        ax.imshow(img)
        ax.axis("off")
        x_cursor += bw_in + gap_in

    if colorbar_label:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cold_hot_cmap, norm=norm)
        sm.set_array([])
        # Place a narrow colorbar strip, leaving room for the label to the right
        cbar_left = total_w_in / fig_w_in + 0.01
        cbar_ax = fig.add_axes([cbar_left, 0.12, 0.025, 0.76])
        cbar = fig.colorbar(sm, cax=cbar_ax, format="%.2f")
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_label(colorbar_label, fontsize=25, labelpad=16)
        cbar.ax.tick_params(labelsize=23)

    parts = [group_name] + ([f"ses-{ses}"] if ses else []) + ([file_suffix] if file_suffix else [])
    combined_path = output_path / f"{'_'.join(parts)}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"Combined visualization saved to {combined_path}")

    for f in os.listdir(temp_output_path):
        os.remove(os.path.join(temp_output_path, f))
    os.rmdir(temp_output_path)

    return fig


def average_session_differences(
    session_dirs,
    label_type,
    vmin=None,
    vmax=None,
    colorbar_label=None,
):
    """Average difference maps across sessions and plot the result.

    Loads the per-session superager vs. non-superager difference CSVs,
    averages them across timepoints, saves the averaged CSV, and calls
    visualize_coupling to produce the final surface figure.

    Args:
        session_dirs: Dictionary mapping session identifiers (e.g. "01",
            "02") to the Path of that session's group connectivity
            directory.
        label_type: Superager labeling scheme. One of "long", "tp1",
            or "tp2".
        vmin: Minimum value for colormap scaling. If None, inferred
            symmetrically from the data.
        vmax: Maximum value for colormap scaling. If None, inferred
            symmetrically from the data.
        colorbar_label: Optional label for the colorbar y-axis.
    """
    diff_series = []
    for ses, session_dir in session_dirs.items():
        if label_type == "long":
            diff_name = "diff_superagers_vs_non_superagers_long"
        elif ses == "01":
            diff_name = "diff_superagers_vs_non_superagers_tp1"
        else:
            diff_name = "diff_superagers_vs_non_superagers_tp2"

        diff_file = Path(session_dir) / f"{diff_name}_average.csv"
        if not diff_file.exists():
            print(f"Skipping {diff_file}: file not found.")
            continue
        diff_series.append(pd.read_csv(diff_file, index_col=0).iloc[:, 0].rename(ses))

    if len(diff_series) < 2:
        print("Skipping averaged difference plot: need both ses-01 and ses-02 difference files.")
        return

    average_output_dir = Path(next(iter(session_dirs.values()))).parent.parent / "average_across_sessions"
    average_output_dir.mkdir(parents=True, exist_ok=True)

    average_name = "diff_superagers_average"
    average_df = pd.concat(diff_series, axis=1).mean(axis=1).to_frame(name=average_name)
    average_df.to_csv(average_output_dir / f"{average_name}.csv")

    visualize_coupling(
        coupling_file=average_output_dir,
        group_name=average_name,
        output_dir=average_output_dir,
        ses="",
        vmin=vmin,
        vmax=vmax,
        file_suffix="",
        symmetric_scale=True,
        colorbar_label=colorbar_label,
    )


if __name__ == "__main__":
    label_type = "long"

    session_dirs = {
        "01": Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-01/group_connectivity_matrices"),
        "02": Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-02/group_connectivity_matrices"),
    }

    average_session_differences(
        session_dirs=session_dirs,
        label_type=label_type,
        vmin=-0.03,
        vmax=0.03,
        colorbar_label="Structure-function\ncoupling difference",
    )
