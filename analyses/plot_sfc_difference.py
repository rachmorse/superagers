#!/usr/bin/env python3
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from matplotlib.image import imread
from nilearn import datasets, surface, plotting
from nilearn.datasets import fetch_surf_fsaverage
import yabplot as yab


def trim_whitespace(img: np.ndarray) -> np.ndarray:
    """Crop white or transparent borders from an image array.

    Args:
        img: Image array of shape (H, W, 3) or (H, W, 4).

    Returns:
        np.ndarray: Cropped image with white borders removed.
    """
    threshold = 250 if img.dtype == np.uint8 else 0.98
    if img.ndim == 3 and img.shape[2] == 4:        
        # Exclude transparent pixels and opaque-white pixels (nilearn renders white bg as fully opaque)
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
    row_is_white = np.mean(np.all(img[:, :, :3] >= threshold, axis=2), axis=1) > 0.9

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


def build_subcortical_data(diff: pd.Series, name_map: dict) -> dict:
    """Map subcortical index entries to yabplot aseg names.

    Args:
        diff: ROI-indexed series of SFC difference values.
        name_map: Dict mapping the lowercased region suffix (e.g.
            'left hippocampus') to the yabplot aseg name (e.g.
            'Left-Hippocampus').

    Returns:
        dict: Mapping of yabplot aseg region name to SFC difference value.
    """
    subcortical = diff[diff.index.str.contains("Subcortical")]
    valid_yab_names = set(yab.get_atlas_regions(atlas="aseg", category="subcortical"))
    data, unmatched, invalid = {}, [], []

    for roi_name, value in subcortical.items():
        region   = roi_name.split(": ", 1)[-1].lower()
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
    network_filter=None,
    include_subcortical=False,
    subcortical_name_map=None,
):
    """Create a brain surface visualisation from ROI-level SFC values.

    Plots left lateral, left medial, right medial, and right lateral cortical
    views by projecting parcellated ROI values onto the fsaverage surface using
    the Schaefer 200-ROI atlas. When include_subcortical is True, anterior and
    superior subcortical views are rendered with yabplot and added in a
    right-hand column.

    Args:
        coupling_file: Directory containing {group_name}.csv.
        group_name: Stem of the CSV to load (no extension).
        output_dir: Directory where the combined PNG will be saved.
        ses: Session identifier appended to temporary filenames (pass "" if
            not applicable).
        vmin: Minimum value for colormap scaling. If None, inferred from data.
        vmax: Maximum value for colormap scaling. If None, inferred from data.
        file_suffix: Short suffix appended to the output PNG filename.
        symmetric_scale: If True and vmin/vmax are None, forces a symmetric
            range around zero.
        colorbar_label: Label for the colorbar. Pass None to omit the colorbar.
        network_filter: Optional list of network name strings. When provided,
            only ROIs whose names contain one of these strings are colored.
        include_subcortical: If True, render subcortical ROIs with yabplot
            alongside the cortical views.
        subcortical_name_map: Dict mapping lowercase region name (e.g.
            'left hippocampus') to yabplot aseg name (e.g. 'Left-Hippocampus').

    Returns:
        matplotlib.figure.Figure: The assembled figure.
    """
    coupling_csv = Path(coupling_file) / f"{group_name}.csv"
    full_series  = pd.read_csv(coupling_csv, index_col=0).iloc[:, 0]
    cortical   = full_series[~full_series.index.str.contains("Subcortical")]
    rho_values = cortical.values

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
    schaefer  = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)

    atlas_img      = nib.load(schaefer["maps"])
    atlas_data     = atlas_img.get_fdata()
    atlas_roi_names = schaefer["labels"]
    if isinstance(atlas_roi_names[0], bytes):
        atlas_roi_names = [label.decode("utf-8") for label in atlas_roi_names]

    roi_name_to_atlas_idx = {name: i for i, name in enumerate(atlas_roi_names)}

    coupling_vol = np.zeros_like(atlas_data)
    for roi_name, value in cortical.items():
        if network_filter and not any(f in roi_name for f in network_filter):
            continue
        if roi_name in roi_name_to_atlas_idx:
            atlas_idx = roi_name_to_atlas_idx[roi_name]
            coupling_vol[atlas_data == (atlas_idx + 1)] = value
        else:
            print(f"Warning: Could not find a matching atlas ROI for {roi_name}")

    coupling_img = nib.Nifti1Image(coupling_vol, atlas_img.affine, atlas_img.header)

    surf_data_left  = surface.vol_to_surf(coupling_img, fsaverage["pial_left"],  radius=1, n_samples=5)
    surf_data_right = surface.vol_to_surf(coupling_img, fsaverage["pial_right"], radius=1, n_samples=5)

    surf_data_left[np.isclose(surf_data_left,   0)] = np.nan
    surf_data_right[np.isclose(surf_data_right, 0)] = np.nan

    temp_output_path = output_path / "temp"
    temp_output_path.mkdir(parents=True, exist_ok=True)

    cold_hot_cmap = plt.colormaps["RdBu_r"]

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
    scale    = 3.0 / max_h_px

    brain_widths_in  = [images[k].shape[1] * scale for k in row_keys]
    brain_heights_in = [images[k].shape[0] * scale for k in row_keys]
    row_h_in = 3.0  # by construction: max_h_px * scale
    gap_in   = 0.25  # whitespace between cortical brain panels

    # Subcortical slices 
    subcort_coronal_arr = None
    subcort_axial_arr   = None
    if include_subcortical and subcortical_name_map is not None:
        subcortical_data = build_subcortical_data(full_series, subcortical_name_map)

        shared_sub_kw = dict(
            data=subcortical_data, atlas="aseg",
            vminmax=[vmin, vmax], cmap="RdBu_r",
            figsize=(500, 500), style="matte",
            bmesh_color="lightgray",
            bmesh_alpha=0.15, display_type="object",
        )

        subcort_cor_png = temp_output_path / f"{group_name}_subcortical_anterior.png"
        subcort_ax_png  = temp_output_path / f"{group_name}_subcortical_superior.png"

        print("  Rendering subcortical (anterior view)…")
        yab.plot_subcortical(views=["anterior"], export_path=str(subcort_cor_png), **shared_sub_kw)
        print("  Rendering subcortical (superior view)…")
        yab.plot_subcortical(views=["superior"], export_path=str(subcort_ax_png),  **shared_sub_kw)

        subcort_coronal_arr = trim_whitespace(crop_colorbar(imread(str(subcort_cor_png))))
        subcort_axial_arr   = trim_whitespace(crop_colorbar(imread(str(subcort_ax_png))))

    top_row_w  = brain_widths_in[0] + gap_in + brain_widths_in[1]
    bot_row_w  = brain_widths_in[2] + gap_in + brain_widths_in[3]
    left_col_w = max(top_row_w, bot_row_w)

    vert_gap_in = 0.1
    total_h_in  = 2 * row_h_in + vert_gap_in

    def _subcort_width(arr):
        if arr is None:
            return 0.0
        return arr.shape[1] * (row_h_in / arr.shape[0])

    right_col_w  = max(_subcort_width(subcort_coronal_arr), _subcort_width(subcort_axial_arr))
    horiz_gap_in = 0.15
    has_subcort  = right_col_w > 0
    cbar_w_in    = 3.0 if colorbar_label is not None else 0.0
    fig_w_in     = left_col_w + (horiz_gap_in + right_col_w if has_subcort else 0) + cbar_w_in

    fig = plt.figure(figsize=(fig_w_in, total_h_in))

    # Top cortical row
    top_y_base = (vert_gap_in + row_h_in) / total_h_in
    x_cursor   = 0.0
    for i, key in enumerate(["left_lateral", "left_medial"]):
        bw_in, bh_in = brain_widths_in[i], brain_heights_in[i]
        y_off = (row_h_in - bh_in) / 2
        ax = fig.add_axes([
            x_cursor / fig_w_in,
            top_y_base + y_off / total_h_in,
            bw_in / fig_w_in,
            bh_in / total_h_in,
        ])
        ax.imshow(images[key])
        ax.axis("off")
        x_cursor += bw_in + gap_in

    # Bottom cortical row
    x_cursor = 0.0
    for i, key in enumerate(["right_medial", "right_lateral"], start=2):
        bw_in, bh_in = brain_widths_in[i], brain_heights_in[i]
        y_off = (row_h_in - bh_in) / 2
        ax = fig.add_axes([
            x_cursor / fig_w_in,
            y_off / total_h_in,
            bw_in / fig_w_in,
            bh_in / total_h_in,
        ])
        ax.imshow(images[key])
        ax.axis("off")
        x_cursor += bw_in + gap_in

    # Subcortical panels in right column
    coronal_scale = 0.75 # render coronal slightly smaller than axial
    if has_subcort:
        subcort_x0 = (left_col_w + horiz_gap_in) / fig_w_in
        for arr, y_base_in, scale_f in [
            (subcort_coronal_arr, vert_gap_in + row_h_in, coronal_scale),
            (subcort_axial_arr,   0.0,                    1.1),
        ]:
            if arr is None:
                continue
            sh_in = row_h_in * scale_f
            sw_in = arr.shape[1] * (sh_in / arr.shape[0])
            x_off = (right_col_w - sw_in) / 2 # centre horizontally in slot
            y_off = (row_h_in   - sh_in) / 2 # centre vertically in slot
            ax_s = fig.add_axes([
                subcort_x0 + x_off / fig_w_in,
                (y_base_in + y_off) / total_h_in,
                sw_in / fig_w_in,
                sh_in / total_h_in,
            ])
            ax_s.imshow(arr)
            ax_s.axis("off")

    # Add colorbar
    if colorbar_label is not None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm   = plt.cm.ScalarMappable(cmap=cold_hot_cmap, norm=norm)
        sm.set_array([])
        cbar_left = (left_col_w + (horiz_gap_in + right_col_w if has_subcort else 0)) / fig_w_in + 0.01
        cbar_ax   = fig.add_axes([cbar_left, 0.2, 0.025, 0.6])
        cbar = fig.colorbar(sm, cax=cbar_ax, format="%.2f")
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_label(colorbar_label, fontsize=25, labelpad=16)
        cbar.ax.tick_params(labelsize=18)

    parts = [group_name] + ([f"ses-{ses}"] if ses else []) + ([file_suffix] if file_suffix else [])
    combined_path = output_path / f"{'_'.join(parts)}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Combined visualization saved to {combined_path}")

    shutil.rmtree(temp_output_path)

    return fig


def average_session_differences(
    session_dirs,
    label_type,
    vmin=None,
    vmax=None,
    colorbar_label=None,
    network_filter=None,
    network_label=None,
    include_subcortical=False,
    subcortical_name_map=None,
):
    """Average difference maps across sessions and plot the result.

    Loads the per-session superager vs. non-superager difference CSVs,
    averages them across timepoints, saves the averaged CSV, and calls
    visualize_coupling to produce the final surface figure.

    Args:
        session_dirs: Dictionary mapping session identifiers (e.g. "01",
            "02") to the Path of that session's group connectivity directory.
        label_type: Superager labeling scheme where "long" is the longitudinally
            defined superagers.
        vmin: Minimum value for colormap scaling. If None, inferred
            symmetrically from the data.
        vmax: Maximum value for colormap scaling. If None, inferred from data.
        colorbar_label: Label for the colorbar. Pass None to omit the colorbar.
        network_filter: Optional list of network name strings passed through
            to visualize_coupling.
        network_label: Short string appended to the output filename to identify
            the network subset (e.g. "dmn"). Defaults to no suffix.
        include_subcortical: If True, subcortical ROIs are rendered with
            yabplot alongside the cortical surface maps.
        subcortical_name_map: Dict mapping lowercase region name to yabplot
            aseg name, passed through to visualize_coupling.
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
    average_df   = pd.concat(diff_series, axis=1).mean(axis=1).to_frame(name=average_name)
    average_df.to_csv(average_output_dir / f"{average_name}.csv")

    visualize_coupling(
        coupling_file=average_output_dir,
        group_name=average_name,
        output_dir=average_output_dir,
        ses="",
        vmin=vmin,
        vmax=vmax,
        file_suffix=network_label or "",
        symmetric_scale=True,
        colorbar_label=colorbar_label,
        network_filter=network_filter,
        include_subcortical=include_subcortical,
        subcortical_name_map=subcortical_name_map,
    )


if __name__ == "__main__":
    session_dirs = {
        "01": Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-01/group_connectivity_matrices"),
        "02": Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-02/group_connectivity_matrices"),
    }

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

    average_session_differences(
        session_dirs=session_dirs,
        label_type="long",
        vmin=-0.03,
        vmax=0.03,
        colorbar_label="Structure-function\ncoupling difference",
        include_subcortical=True,
        subcortical_name_map=subcortical_name_map,
    )
