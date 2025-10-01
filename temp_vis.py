import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
import os
import glob
from nilearn.datasets import fetch_surf_fsaverage
from matplotlib.image import imread
from nilearn import datasets, surface, plotting
import matplotlib.pyplot as plt
import nibabel as nib
from typing import List  
from fnmatch import fnmatchcase 


def visualize_coupling(coupling_file, group_name, output_dir, ses, vmin=0.15, vmax=0.41, binarize=True, rois_to_plot: List[str] | None = None):
    """
    Create multi-view brain surface visualizations of structure-function coupling from a DataFrame
    
    Args:
        coupling_file (pd.DataFrame): DataFrame with ROI names as index and coupling values in first column
        group_name (str): Name of the group (e.g., 'superagers', 'maintainers')
        output_dir (str or Path): Directory to save visualizations
        vmin, vmax (float): Min and max values for color scaling
        ses (str): Session identifier
        binarize (bool): Whether to binarize the coupling values for visualization
        rois_to_plot (list[str] | None): Optional list of ROI name patterns to plot.
    """
    coupling_csv = Path(f"{coupling_file}/{group_name}_average.csv")
    coupling_df = pd.read_csv(coupling_csv, index_col=0)

    print(f"{coupling_df.head(5)}")

    # Optionally filter to user-specified ROIs 
    if rois_to_plot:
        patterns = [p.lower() for p in rois_to_plot]
        idx = coupling_df.index.tolist()
        keep = [
            any(fnmatchcase(name.lower(), pat) or name.lower() == pat for pat in patterns)
            for name in idx
        ]
        coupling_df = coupling_df.loc[np.array(keep)]
        if coupling_df.empty:
            print("[WARN] None of the requested ROIs matched the DataFrame index.")
        else:
            print(f"Filtered ROIs: kept {len(coupling_df)} / {len(idx)}")

    # Set vmax as the maximum value in the DataFrame
    if vmax is None:
        vmax = coupling_df.iloc[:, 0].max()

    if vmin is None:  
        vmin = coupling_df.iloc[:, 0].min()                                

    # Drop the subcoritical ROIs 
    coupling_df = coupling_df[~coupling_df.index.str.contains('Subcortical')]
    
    # Extract data 
    rho_values = coupling_df.iloc[:, 0].values  # Get the first column's values
    subject = group_name
    
    # Create output directory
    output_path = Path(output_dir) / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get fsaverage5 for visualization
    fsaverage = fetch_surf_fsaverage('fsaverage5')
    
    # Step 1: Get the Schaefer atlas parcellations
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)

    # First, create a volume where ROI value = coupling value
    atlas_img = nib.load(schaefer['maps'])
    atlas_data = atlas_img.get_fdata()
    atlas_roi_names = schaefer['labels']

    # Convert to strings
    if isinstance(atlas_roi_names[0], bytes):
        atlas_roi_names = [label.decode('utf-8') for label in atlas_roi_names]

    # Create a mapping from the order of SFC ROI names to the Schaefer atlas indices (which is a different order)
    roi_name_to_atlas_idx = {}
    for i, atlas_name in enumerate(atlas_roi_names):
        roi_name_to_atlas_idx[atlas_name] = i

    # Now use this mapping for visualization
    coupling_vol = np.zeros_like(atlas_data)

    # Count how many ROIs were successfully mapped
    mapped_count = 0

    for my_roi_name, value in zip(coupling_df.index, rho_values):
        if my_roi_name in roi_name_to_atlas_idx:
            # Get the correct atlas index for this ROI name
            atlas_idx = roi_name_to_atlas_idx[my_roi_name]
            # Atlas uses 1-indexed ROIs in the volume
            coupling_vol[atlas_data == (atlas_idx + 1)] = value
            mapped_count += 1
        else:
            print(f"Warning: Could not find a matching atlas ROI for {my_roi_name}")

    print(f"Successfully mapped {mapped_count} out of {len(coupling_df.index)} ROIs")

    # If binarize is True, convert to binary presence/absence
    if binarize:
        coupling_vol = (coupling_vol > 0).astype(int)
        rho_values = np.unique(coupling_vol)
        vmin, vmax = 0, 1
        
    # Debug output
    print(f"Coupling values shape: {rho_values.shape}, min: {np.min(rho_values)}, max: {np.max(rho_values)}")
        
    # Create a nifti image with the coupling values
    coupling_img = nib.Nifti1Image(coupling_vol, atlas_img.affine, atlas_img.header)
            
    # Project volume to each surface
    surf_data_left = surface.vol_to_surf(
        coupling_img, 
        fsaverage['pial_left'],
        radius=3,
        n_samples=5
    )

    surf_data_right = surface.vol_to_surf(
        coupling_img, 
        fsaverage['pial_right'],
        radius=3,
        n_samples=5
    )

    # Check that the projected ranges are correct
    print(f"Left surface data range: {np.min(surf_data_left)}-{np.max(surf_data_left)}")
    print(f"Right surface data range: {np.min(surf_data_right)}-{np.max(surf_data_right)}")

    # Create masks for vertices with values of zero
    mask_left = np.isclose(surf_data_left, 0)
    mask_right = np.isclose(surf_data_right, 0)

    # Replace zeros with NaN so they appear as grey/transparent in plots
    surf_data_left[mask_left] = np.nan
    surf_data_right[mask_right] = np.nan

    # Verify final data
    print(f"Final left surface data range: {np.nanmin(surf_data_left)}-{np.nanmax(surf_data_left)}")
    print(f"Final right surface data range: {np.nanmin(surf_data_right)}-{np.nanmax(surf_data_right)}")
    
    temp_output_path = output_path / "temp"
    temp_output_path.mkdir(parents=True, exist_ok=True)

    # Set the colormap
    cold_hot_cmap = plt.get_cmap('RdBu_r')

    # Right lateral fig
    right_lat_path = temp_output_path / f"{subject}_{ses}_right_lateral.png"
    plotting.plot_surf_stat_map(
        fsaverage['pial_right'],
        surf_data_right,
        darkness=0.3, 
        hemi='right',
        view='lateral',
        colorbar=True,
        cmap=cold_hot_cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  
        bg_map=fsaverage['sulc_right'],
        bg_on_data=True,
        output_file=right_lat_path
    )
    
    # Right medial fig
    right_med_path = temp_output_path / f"{subject}_{ses}_right_medial.png"
    plotting.plot_surf_stat_map(
        fsaverage['pial_right'],
        surf_data_right, 
        darkness=0.3, 
        hemi='right',
        view='medial',
        colorbar=False,
        cmap=cold_hot_cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  
        bg_map=fsaverage['sulc_right'],
        bg_on_data=True,
        output_file=right_med_path
    )
    
    # Left medial fig
    left_med_path = temp_output_path / f"{subject}_{ses}_left_medial.png"
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        surf_data_left,
        darkness=0.3, 
        hemi='left',
        view='medial',
        colorbar=False,
        cmap=cold_hot_cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  
        bg_map=fsaverage['sulc_left'],
        bg_on_data=True,
        output_file=left_med_path
    )
    
    # Left lateral fig
    left_lat_path = temp_output_path / f"{subject}_{ses}_left_lateral.png"
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        surf_data_left,
        darkness=0.3, 
        hemi='left',
        view='lateral',
        colorbar=False,
        cmap=cold_hot_cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  
        bg_map=fsaverage['sulc_left'],
        bg_on_data=True,
        output_file=left_lat_path
    )
    
    # Now combine the images 
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    plt.subplots_adjust(wspace=-0.3, hspace=0)  # Reduce horizontal and vertical spacing
    plt.suptitle(f"Structure-Function Coupling\n{subject} - ses-{ses}", fontsize=16, y=0.98) 
    # plt.tight_layout()

    # Load and display the images in order: left lateral, left medial, right medial, right lateral    
    axes[0].imshow(imread(left_lat_path))
    axes[0].axis('off')
    
    axes[1].imshow(imread(left_med_path))
    axes[1].axis('off')
    
    axes[2].imshow(imread(right_med_path))
    axes[2].axis('off')
    
    axes[3].imshow(imread(right_lat_path))
    axes[3].axis('off')
    
    # Save the combined figure
    combined_path = output_path / f"{subject}_ses-{ses}_sfc.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved to {combined_path}")
    
    # Remove temp_output_path
    for file in os.listdir(temp_output_path):
        os.remove(os.path.join(temp_output_path, file))
    os.rmdir(temp_output_path)

    return fig

def main(output_directory_group, ses, output_group_connectivity_file):
    output_directory_group = Path(output_directory_group)
    output_directory_group.mkdir(parents=True, exist_ok=True)

    for group_name in group_names:
        # Visualize SFC in selected groups
        visualize_coupling(
            coupling_file=output_group_connectivity_file,
            group_name=group_name,
            output_dir=output_directory_group,
            ses=ses,
            vmin=None, 
            vmax=None,
            binarize=True,  # Set to True to visualize presence/absence
            rois_to_plot=["7Networks_LH_DorsAttn_Post_*", "7Networks_LH_Cont_OFC_*", "7Networks_LH_Cont_Temp_*", "7Networks_RH_Cont_PFCv_*"]  
        )
    

if __name__ == "__main__":
    sessions = ["01"]
    group_names = ["superagers", "non_superagers"] # This is to loop through the visualization

    base_dir = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    src1 = base_dir / "ses-01" / "group_connectivity_matrices"
    src2 = base_dir / "ses-02" / "group_connectivity_matrices"
    dest = base_dir / "ses-avg" / "group_connectivity_matrices"
    dest.mkdir(parents=True, exist_ok=True)

    for g in group_names:
        df1 = pd.read_csv(src1 / f"{g}_average.csv", index_col=0)
        df2 = pd.read_csv(src2 / f"{g}_average.csv", index_col=0)
        avg = pd.concat([df1.iloc[:, 0], df2.iloc[:, 0]], axis=1).mean(axis=1)
        avg.to_frame(name=df1.columns[0]).to_csv(dest / f"{g}_average.csv")

    # Group difference on ses-avg (g1 - g2) 
    g1, g2 = group_names  # assumes exactly two groups  
    df_g1 = pd.read_csv(dest / f"{g1}_average.csv", index_col=0)  
    df_g2 = pd.read_csv(dest / f"{g2}_average.csv", index_col=0)  
    diff_name = f"{g1}_minus_{g2}"  
    (df_g1.iloc[:, 0] - df_g2.iloc[:, 0]).to_frame(name=df_g1.columns[0]).to_csv(  
        dest / f"{diff_name}_average.csv"  
    )
    group_names.append(diff_name)  

    for ses in sessions:
            print("--------------------------")
            print(f"Processing ses-{ses} ...")
            print("--------------------------")

            sfc_df = Path(f"/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-{ses}")
            output_directory_group = Path(f"{sfc_df}/group_connectivity_matrices")
            output_group_connectivity_file = Path(f"{output_directory_group}")

            main(
                output_directory_group=output_directory_group,
                ses=ses,
                output_group_connectivity_file=output_group_connectivity_file
            )