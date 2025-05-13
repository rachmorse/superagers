import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import nibabel as nib
from nilearn import plotting, surface
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
import matplotlib.pyplot as plt


def get_subjects_to_process(output_folder, ses, mask_dir, functional_dir, structural_dir):
    """Generate a list of subjects to process based on whether they have
    a structural and functional martix and do not have coupling for the
    specified timepoint.

    Args:
        output_folder (Path): Path to the directory coupling results
        ses (str): Session ID (format: ses-01).
        functional_dir (Path): Path to the directory containing functional connectivity matrices.
        structural_dir (Path): Path to the directory containing structural connectivity matrices.
        mask_dir (Path): Path to the directory containing the mask files to use to check which subjects to process.

    Returns:
        list: List of subject IDs to process.
    """
    subjects_to_process = []
    already_processed = []

    # Iterate over all possible subject directories
    for subject in os.listdir(mask_dir):
        if not subject.startswith("sub-"):
            continue
        
        # Check if the required directory exists and hasn't been processed yet  
        func_conn_path = functional_dir / f"{subject}_{ses}_functional_connectivity_matrix.csv"
        struct_conn_path = structural_dir / f"{subject}_{ses}_structural_connectivity_matrix.csv"
        output_file_path = output_folder / f"{subject}_{ses}_structure_function_coupling.csv"

        if func_conn_path.exists() and struct_conn_path.exists() and not output_file_path.exists():
            subjects_to_process.append(subject)
        elif output_file_path.exists():
            already_processed.append(subject)

    return subjects_to_process, already_processed

def calculate_structure_function_coupling(structural_dir, functional_dir, subject, ses):
    """
    Calculate structure-function coupling for each ROI
    
    Args:
        subject (str): Subject identifier
        ses (str): Timepoint 
        structural_dir (str or Path): Directory containing structural connectivity matrices
        functional_dir (str or Path): Directory containing functional connectivity matrices
    
    Returns:
        dict: Dictionary containing coupling results
    """
    
    # Define paths using specific directory structure
    func_conn_path = functional_dir / f"{subject}_{ses}_functional_connectivity_matrix.csv"
    struct_conn_path = structural_dir / f"{subject}_{ses}_structural_connectivity_matrix.csv"

    # Load connectivity matrices
    func_conn_df = pd.read_csv(func_conn_path)
    struct_conn_df = pd.read_csv(struct_conn_path)

    # Extract region names from the first column (excluding the header row)
    roi_names = func_conn_df.iloc[1:, 0].tolist()
    
    # Drop the first row and column to get the numeric data only
    func_conn_matrix = func_conn_df.iloc[0:, 1:].to_numpy()
    struct_conn_matrix = struct_conn_df.iloc[0:, 1:].to_numpy()

    # Check matrix dimensions match
    if func_conn_matrix.shape != struct_conn_matrix.shape:
        raise ValueError(f"Matrix dimensions don't match: "
                       f"functional {func_conn_matrix.shape} vs structural {struct_conn_matrix.shape}")
    
    n_rois = 214
    
    # Initialize coupling data structures
    coupling = {
        "subject": subject,
        "ses": ses,
        "rho": np.zeros((n_rois, 1)),
        "roi_names": roi_names
    }
    
    # Calculate structure-function coupling for each ROI
    for m in range(n_rois):
        # Get structural and functional connectivity for this region
        X = struct_conn_matrix[:, m]
        Y = func_conn_matrix[:, m]
        
        # Find indices where both are non-zero
        idx_both_non_zero = (X != 0) & (Y != 0)
        idx_both_non_zero[m] = False  # Set diagonal to False
        
        X_filtered = X[idx_both_non_zero]
        Y_filtered = Y[idx_both_non_zero]
        
        if len(X_filtered) == 0 or len(Y_filtered) == 0:
            coupling["rho"][m,0] = np.nan
        else:
            rho = stats.pearsonr(X_filtered, Y_filtered)[0]
            coupling["rho"][m,0] = rho
    
    return coupling


# def visualize_coupling(coupling_dict, output_dir, cmap='coolwarm'):
#     """
#     Visualize the coupling results as a heatmap and bar plot
    
#     Args:
#         coupling_dict (dict): The coupling dictionary returned by calculate_structure_function_coupling
#         cmap (str): Matplotlib colormap name
#         output_dir (str or Path): Directory to save the visualization
#     """
#     rho_values = coupling_dict["rho"]
    
#     # Create the figure and axis
#     fig, ax = plt.subplots(figsize=(5, 10))
    
#     # Create the heatmap
#     im = ax.imshow(rho_values.reshape(-1, 1), cmap=cmap, aspect='auto')
    
#     # Set title and y-axis label
#     ax.set_title(f"Structure-Function Coupling\n{coupling_dict['subject']} - {coupling_dict['ses']}")
#     ax.set_ylabel("ROI")
    
#     # Remove x-axis ticks and labels
#     ax.set_xticks([])
#     ax.set_xticklabels([])
    
#     # Add colorbar
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("Correlation")
    
#     plt.tight_layout()

#     # Save the figure
#     output_path = Path(output_dir) / "visualizations"
#     output_path.mkdir(parents=True, exist_ok=True)
#     subject = coupling_dict["subject"]
#     ses = coupling_dict["ses"]
#     fig_path = output_path / f"{subject}_{ses}_coupling_visualization.png"
#     fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
#     return fig

def visualize_coupling(coupling_dict, output_dir, cmap='cold_hot', vmin=0, vmax=0.5):
    """
    Create multi-view brain surface visualizations of structure-function coupling
    
    Args:
        coupling_dict (dict): The coupling dictionary with rho values
        output_dir (str or Path): Directory to save visualizations
        cmap (str): Colormap to use ('cold_hot' gives blue-red gradient)
        vmin, vmax (float): Min and max values for color scaling
    """
    from nilearn import datasets, surface
    from nilearn import plotting
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import nibabel as nib
    import os
    import tempfile
    
    # Extract data
    rho_values = coupling_dict["rho"].flatten()
    subject = coupling_dict["subject"]
    ses = coupling_dict["ses"]
    
    # Create output directory
    output_path = Path(output_dir) / "visualizations"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get fsaverage5 for visualization
    from nilearn.datasets import fetch_surf_fsaverage
    fsaverage = fetch_surf_fsaverage('fsaverage5')
    
    # Step 1: Get the Schaefer atlas parcellation
    # Fetch the volumetric Schaefer atlas
    schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    
    # Debug output
    print(f"Schaefer atlas info: {schaefer.keys()}")
    print(f"Coupling values shape: {rho_values.shape}, min: {np.min(rho_values)}, max: {np.max(rho_values)}")
    
    # Step 2: Project the volumetric atlas to the surface
    # We'll need to get surface parcellations directly, but for now let's create a workaround
    
    # Create a nifti image with our coupling values
    # First, create a volume where ROI value = coupling value
    atlas_img = nib.load(schaefer['maps'])
    atlas_data = atlas_img.get_fdata()
    
    # Make sure we print some debug info
    print(f"Atlas data shape: {atlas_data.shape}")
    print(f"Atlas data unique values: {np.unique(atlas_data)[:10]}...")
    
    # Create a volume where each voxel's value is its ROI's coupling value
    coupling_vol = np.zeros_like(atlas_data)
    
    # Map the first 200 ROIs (Schaefer atlas)
    for roi_idx in range(min(200, len(rho_values))):
        # Schaefer ROIs are 1-indexed in the volume
        roi_label = roi_idx + 1
        coupling_vol[atlas_data == roi_label] = rho_values[roi_idx]
    
    # Print some information about the coupling volume
    print(f"Coupling vol unique values: {np.unique(coupling_vol)[:10]}...")
    print(f"Coupling vol non-zero count: {np.count_nonzero(coupling_vol)}")
    
    # Create a nifti image with the coupling values
    coupling_img = nib.Nifti1Image(coupling_vol, atlas_img.affine, atlas_img.header)
    
    # Save the coupling volume for inspection
    # nib.save(coupling_img, os.path.join(output_path, f"{subject}_{ses}_coupling.nii.gz"))
    
    # Project the volumetric data to the surface
    print("Projecting to surface...")
    
    # Project volume to each surface
    surf_data_left = surface.vol_to_surf(
        coupling_img, 
        fsaverage['pial_left'],
        radius=3,  # Use a larger radius to ensure mapping
        n_samples=5  # Sample more points along the normal
    )
    
    surf_data_right = surface.vol_to_surf(
        coupling_img, 
        fsaverage['pial_right'],
        radius=3,
        n_samples=5
    )

    # Create a mask for values that should be grey (those that are exactly 0 or NaN)
    mask_left = np.where(np.isclose(surf_data_left, 0) | np.isnan(surf_data_left))
    mask_right = np.where(np.isclose(surf_data_right, 0) | np.isnan(surf_data_right))

    # Replace those values with NaN which will be rendered as grey
    surf_data_left[mask_left] = np.nan
    surf_data_right[mask_right] = np.nan
    
    # Print info about surface data
    print(f"Left surface data shape: {surf_data_left.shape}")
    print(f"Right surface data shape: {surf_data_right.shape}")
    print(f"Left surface data range: {np.min(surf_data_left)}-{np.max(surf_data_left)}")
    print(f"Right surface data range: {np.min(surf_data_right)}-{np.max(surf_data_right)}")
    
    temp_output_path = output_path / "temp"
    temp_output_path.mkdir(parents=True, exist_ok=True)

    # Right lateral view
    right_lat_path = temp_output_path / f"{subject}_{ses}_right_lateral.png"
    print(f"Creating right lateral view: {right_lat_path}")
    plotting.plot_surf_stat_map(
        fsaverage['pial_right'],
        surf_data_right,
        hemi='right',
        view='lateral',
        colorbar=True,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  # Make sure we don't threshold any data
        bg_map=fsaverage['sulc_right'],
        bg_on_data=True,
        output_file=right_lat_path
    )
    
    # Right medial view
    right_med_path = temp_output_path / f"{subject}_{ses}_right_medial.png"
    print(f"Creating right medial view: {right_med_path}")
    plotting.plot_surf_stat_map(
        fsaverage['pial_right'],
        surf_data_right,
        hemi='right',
        view='medial',
        colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  # Make sure we don't threshold any data
        bg_map=fsaverage['sulc_right'],
        bg_on_data=True,
        output_file=right_med_path
    )
    
    # Left medial view
    left_med_path = temp_output_path / f"{subject}_{ses}_left_medial.png"
    print(f"Creating left medial view: {left_med_path}")
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        surf_data_left,
        hemi='left',
        view='medial',
        colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  # Make sure we don't threshold any data
        bg_map=fsaverage['sulc_left'],
        bg_on_data=True,
        output_file=left_med_path
    )
    
    # Left lateral view
    left_lat_path = temp_output_path / f"{subject}_{ses}_left_lateral.png"
    print(f"Creating left lateral view: {left_lat_path}")
    plotting.plot_surf_stat_map(
        fsaverage['pial_left'],
        surf_data_left,
        hemi='left',
        view='lateral',
        colorbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        threshold=None,  # Make sure we don't threshold any data
        bg_map=fsaverage['sulc_left'],
        bg_on_data=True,
        output_file=left_lat_path
    )
    
    # Now combine the images using matplotlib instead of PIL
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=-0.8, hspace=0)  # Reduce horizontal and vertical spacing
    plt.suptitle(f"Structure-Function Coupling\n{subject} - {ses}", fontsize=16, y=0.98)  # Move title up slightly

    
    # Load and display the images in order: left lateral, left medial, right medial, right lateral
    from matplotlib.image import imread
    
    axes[0].imshow(imread(left_lat_path))
    axes[0].axis('off')
    
    axes[1].imshow(imread(left_med_path))
    axes[1].axis('off')
    
    axes[2].imshow(imread(right_med_path))
    axes[2].axis('off')
    
    axes[3].imshow(imread(right_lat_path))
    axes[3].axis('off')
    
    # Add overall title
    plt.suptitle(f"Structure-Function Coupling\n{subject} - {ses}", fontsize=16)
    plt.tight_layout()
    
    # Save the combined figure
    combined_path = output_path / f"{subject}_{ses}_combined_brain_views.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved to {combined_path}")
    
    # Remove temp_output_path
    for file in os.listdir(temp_output_path):
        os.remove(os.path.join(temp_output_path, file))
    os.rmdir(temp_output_path)

    return fig


def save_coupling_results(coupling_dict, output_path):
    """
    Save coupling results to CSV files
    
    Args:
        coupling_dict (dict): The coupling dictionary returned by calculate_structure_function_coupling
        output_path (str or Path): Directory to save results
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subject = coupling_dict["subject"]
    ses = coupling_dict["ses"]
    
    # Combine ROI index and rho into a single DataFrame
    results_df = pd.DataFrame({
        'region_index': range(len(coupling_dict["rho"])),
        'pearson_rho': coupling_dict["rho"].flatten(),
    })
    
    # Save to CSV
    output_file = output_path / f"{subject}_{ses}_structure_function_coupling.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"Completed processing for {subject}: {output_file}")
    
    return output_file

def main():
    # Define the directories using Path
    ses = "ses-01"
    structural_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/structural_connectivity/{ses}/individual_connectivity_matrices")
    functional_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space/{ses}/individual_connectivity_matrices")
    output_dir = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/individual_coupling_matrices")
    mask_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")
    
    # Make sure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # subjects_to_process, already_processed = get_subjects_to_process(output_dir, ses, mask_dir, functional_dir, structural_dir)
    # print(f"Number of subjects already processed: {len(already_processed)}")
    # print(f"Number of subjects to process: {len(subjects_to_process)}")
    
    subjects_to_process = ["sub-134123"]

    for subject in subjects_to_process:
        # Calculate structure-function coupling for the specified ses
        results = calculate_structure_function_coupling(structural_dir, functional_dir, subject, ses)

        save_coupling_results(results, output_dir)

        visualize_coupling(results, output_dir, cmap='coolwarm')
        
if __name__ == "__main__":
    main()
    