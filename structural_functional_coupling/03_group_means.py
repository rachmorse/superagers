import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union
import os
import glob
from nilearn.datasets import fetch_surf_fsaverage
from matplotlib.image import imread
from nilearn import datasets, surface, plotting
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats
from statsmodels.stats.multitest import multipletests


def fisher_transform(connectivity_file, output_directory, ses):
    """Apply Fisher z-transformation to the correlation coefficients.

    Args:
        correlations (np.ndarray): Correlation coefficients.
        output_dir (Path): Directory to save the transformed data.
        ses (str): Timepoint
        connectivity_file (Path): Path to the CSV file containing the correlation coefficients.

    Returns:
        np.ndarray: Transformed correlation coefficients.
    """
    # Load the correlations from the file
    correlations = pd.read_csv(connectivity_file, index_col=0)
    
    # Apply arctanh 
    fisher_z = pd.DataFrame(np.arctanh(correlations), 
                           index=correlations.index, 
                           columns=correlations.columns)
    
    # Save the transformed data
    output_file = output_directory / f"fisher_z_all_sfc_ses-{ses}.csv"
    fisher_z.to_csv(output_file)
    
    print(f"Fisher Z-transformed data saved to {output_file}")
    return fisher_z


def consolidate_sfc_data(ses, sfc_df):
    """Creates a single DataFrame with all SFC data, where each row is a participant
    and each column is an ROI.
    
    Args:
        ses (str): Timepoint
        sfc_df (Path): Path to the directory containing SFC data
        
    Returns:
        pd.DataFrame: Consolidated DataFrame with participants as rows and ROIs as columns
    """
    # Define paths
    individual_dir = sfc_df / "individual_coupling_matrices"
    output_dir = sfc_df / "all_to_all_roi_matrices"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subject files
    subject_files = glob.glob(str(individual_dir / "sub-*_ses-*_structure_function_coupling.csv"))
    
    if not subject_files:
        print(f"No subject files found in {individual_dir}")
        return None
    
    # Initialize an empty dictionary to store data
    all_data = {}
    roi_names = set()  
    
    # Process each subject file
    for subject_file in subject_files:
        # Extract subject ID from filename
        subject = os.path.basename(subject_file).split('_')[0] 
        
        # Read the subject's data
        try:
            df = pd.read_csv(subject_file)

            # Check the format of the CSV 
            if len(df.columns) == 2:  
                roi_col = df.columns[0]
                val_col = df.columns[1]
                
                # Create a dictionary for this subject's data
                subject_data = dict(zip(df[roi_col], df[val_col]))
                
                # Update the set of all ROI names
                roi_names.update(df[roi_col])
                
            else:
                print(f"Unexpected format in file {subject_file}. Skipping.")
                continue
            
            # Add this subject's data to the main dictionary
            all_data[subject] = subject_data
            
        except Exception as e:
            print(f"Error processing file {subject_file}: {e}")
    
    if not all_data:
        print("No data were successfully processed.")
        return None
    
    # Convert the dictionary to a DataFrame
    consolidated_df = pd.DataFrame.from_dict(all_data, orient='index')
    
    # Save the consolidated DataFrame
    output_file = output_dir / f"all_sfc_data_ses-{ses}.csv"
    consolidated_df.to_csv(output_file)
    
    print(f"Consolidated data saved to {output_file}")
    print(f"DataFrame shape: {consolidated_df.shape}")
    return consolidated_df


def save_group_averages(group_df, group_name, output_file, ses, label_type):
    """Compute the means for a group, drop non-relevant columns, and save to CSV.
    
    Args:
        group_df (pd.DataFrame): DataFrame containing the group data.
        group_name (str): Name of the group.
        output_file (Union[str, Path]): Path to save the output CSV file.
        ses (str): Timepoint
        label_type (str): Type of labeling ("long", "tp1", "tp2") based on superager definition
    """
    mean_vals = group_df.mean(numeric_only=True)
    
    # Prepare and save the result DataFrame
    result_df = pd.DataFrame([mean_vals], index=[group_name])
    if label_type == "long":
        result_df = result_df.drop(columns=['superager_long'], errors='ignore')
    elif ses == "01":
        result_df = result_df.drop(columns=['superager_tp1'], errors='ignore')
    else:
        result_df = result_df.drop(columns=['superager_tp2'], errors='ignore')
    result_df.index.name = 'id'

    # Transpose the DataFrame so ROI names are rows instead of columns
    result_df_transposed = result_df.transpose()
    result_df_transposed.index.name = 'roi'
    result_df_transposed.columns.name = None  # Remove the name from columns
    
    # Print the output path
    print(f"Saving results to {output_file}")
    
    # Save to CSV
    result_df_transposed.to_csv(output_file)


def process_connectivity(connectivity_file: Union[str, Path], superager_file: Union[str, Path], output_files: dict, ses: str, label_type: str):
    """Process and merge connectivity data with superager status, 
    then calculate averages.

    Args:
        connectivity_file (Union[str, Path]): Path to the connectivity CSV file.
        superager_file (Union[str, Path]): Path to the superager status CSV file.
        output_files (dict): Dictionary to save averages for each category.
        ses (str): Timepoint
        label_type (str): Type of labeling ("long", "tp1", "tp2") based on superager definition
    """
    # Load the data
    df_connectivity = pd.read_csv(connectivity_file)
    df_superager = pd.read_csv(superager_file)

    # Rename the first column to id
    df_connectivity = df_connectivity.rename(columns={df_connectivity.columns[0]: 'id'})

    # Ensure necessary columns are present
    if label_type == "long":
        df_superager = df_superager[['id', 'superager_long']]
    elif ses == "01":
        df_superager = df_superager[['id', 'superager_tp1']]
    else:
        df_superager = df_superager[['id', 'superager_tp2']]

    # Ensure 'id' columns have the same data type
    df_connectivity['id'] = df_connectivity['id'].astype(str)
    df_superager['id'] = df_superager['id'].astype(str)
    if not df_superager['id'].str.startswith('sub-').all():
        df_superager['id'] = 'sub-' + df_superager['id'].str.replace('^sub-', '', regex=True)
    df_superager = df_superager.dropna()

    # Merge dataframes on 'id'
    df = pd.merge(df_connectivity, df_superager, on='id', how="inner")
    print(f"Length of data before merge: connectivity={len(df_connectivity)}, superager={len(df_superager)}")
    print(f"Length of data after merge: {len(df)}")

    # First, process the "all subjects" group
    group_name = 'all_subjects'
    if group_name in output_files:
        output_file = output_files[group_name]
        save_group_averages(df, group_name, output_file, ses, label_type)

    # Process individual groups
    if label_type == "long":
        column_prefix = [('superager_long', 'superagers_long')]
    elif ses == "01":
        column_prefix = [('superager_tp1', 'superagers_tp1')]
    else:
        column_prefix = [('superager_tp2', 'superagers_tp2')]

    for column, prefix in column_prefix:
        for label, group_df in df.groupby(column):
            if label == 1:
                group_name = f"{prefix}"
            else:
                group_name = f"non_{prefix}"
            output_file = output_files[group_name]
            save_group_averages(group_df, group_name, output_file, ses, label_type)

    print("CSV files created successfully!")


def convert_edge_table_to_roi_means(connectivity_file: Union[str, Path], output_file: Union[str, Path]):
    """Convert an edge-wise connectivity table to ROI-wise mean connectivity.

    Args:
        connectivity_file: CSV path with one row per subject and columns named
            as ``ROI1-ROI2``.
        output_file: CSV path where ROI-wise subject means are saved. The first
            column is ``id`` and remaining columns are ROI names.

    Returns:
        Path: Path to the written ROI-wise CSV.
    """
    df = pd.read_csv(connectivity_file)
    df = df.rename(columns={df.columns[0]: "id"})
    edge_cols = [col for col in df.columns if col != "id"]

    roi_sums: Dict[str, np.ndarray] = {}
    roi_counts: Dict[str, np.ndarray] = {}
    n_subjects = len(df)

    for col in edge_cols:
        if "-" not in col:
            continue
        roi_1, roi_2 = col.split("-", 1)
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()

        if roi_1 not in roi_sums:
            roi_sums[roi_1] = np.zeros(n_subjects, dtype=float)
            roi_counts[roi_1] = np.zeros(n_subjects, dtype=float)
        if roi_2 not in roi_sums:
            roi_sums[roi_2] = np.zeros(n_subjects, dtype=float)
            roi_counts[roi_2] = np.zeros(n_subjects, dtype=float)

        valid = ~np.isnan(values)
        roi_sums[roi_1][valid] += values[valid]
        roi_sums[roi_2][valid] += values[valid]
        roi_counts[roi_1][valid] += 1
        roi_counts[roi_2][valid] += 1

    roi_means = {"id": df["id"].astype(str)}
    for roi in sorted(roi_sums.keys()):
        means = np.full(n_subjects, np.nan, dtype=float)
        has_values = roi_counts[roi] > 0
        means[has_values] = roi_sums[roi][has_values] / roi_counts[roi][has_values]
        roi_means[roi] = means

    out_df = pd.DataFrame(roi_means)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print(f"Saved ROI means to {output_file}")
    return output_file


def generate_modality_surface_plots(
    modality_name: str,
    modality_slug: str,
    edge_connectivity_file: Union[str, Path],
    superager_file: Union[str, Path],
    output_directory_group: Union[str, Path],
    ses: str,
    label_type: str,
    group_names,
    group_vmin=None,
    group_vmax=None,
    group_symmetric=False,
    diff_vmin=None,
    diff_vmax=None,
    diff_symmetric=True,
):
    """Generate group-average surface plots for FC or SC.

    Args:
        modality_name: Label used in figure titles.
        modality_slug: Short suffix used in output filenames.
        edge_connectivity_file: All-subject edge-wise connectivity CSV.
        superager_file: Superager CSV used for grouping.
        output_directory_group: Directory for group-average CSVs and figures.
        ses: Timepoint identifier without prefix (for example, ``"01"``).
        label_type: Label mode (``"tp1"``, ``"tp2"``, or ``"long"``).
        group_names: Group names to visualize (same convention as SFC).
        group_vmin: Color minimum for standard group maps.
        group_vmax: Color maximum for standard group maps.
        group_symmetric: Whether group map scaling should be symmetric.
        diff_vmin: Color minimum for difference map.
        diff_vmax: Color maximum for difference map.
        diff_symmetric: Whether difference scaling should be symmetric.
    """
    output_directory_group = Path(output_directory_group)
    output_directory_group.mkdir(parents=True, exist_ok=True)

    roi_means_file = output_directory_group / f"all_{modality_slug}_roi_means_ses-{ses}.csv"
    convert_edge_table_to_roi_means(edge_connectivity_file, roi_means_file)

    output_files = {
        "all_subjects": output_directory_group / "all_subjects_average.csv",
        "superagers_tp1": output_directory_group / "superagers_tp1_average.csv",
        "non_superagers_tp1": output_directory_group / "non_superagers_tp1_average.csv",
        "superagers_tp2": output_directory_group / "superagers_tp2_average.csv",
        "non_superagers_tp2": output_directory_group / "non_superagers_tp2_average.csv",
        "superagers_long": output_directory_group / "superagers_long_average.csv",
        "non_superagers_long": output_directory_group / "non_superagers_long_average.csv",
    }
    process_connectivity(roi_means_file, superager_file, output_files, ses, label_type)

    for group_name in group_names:
        visualize_coupling(
            coupling_file=output_directory_group,
            group_name=group_name,
            output_dir=output_directory_group,
            ses=ses,
            vmin=group_vmin,
            vmax=group_vmax,
            figure_label=modality_name,
            file_suffix=modality_slug,
            symmetric_scale=group_symmetric,
        )

    if label_type == "long":
        superager_path = output_files["superagers_long"]
        non_superager_path = output_files["non_superagers_long"]
        diff_name = "diff_superagers_vs_non_superagers_long"
    elif ses == "01":
        superager_path = output_files["superagers_tp1"]
        non_superager_path = output_files["non_superagers_tp1"]
        diff_name = "diff_superagers_vs_non_superagers_tp1"
    else:
        superager_path = output_files["superagers_tp2"]
        non_superager_path = output_files["non_superagers_tp2"]
        diff_name = "diff_superagers_vs_non_superagers_tp2"

    if superager_path.exists() and non_superager_path.exists():
        df_super = pd.read_csv(superager_path, index_col=0)
        df_non = pd.read_csv(non_superager_path, index_col=0)
        diff_values = df_super.iloc[:, 0] - df_non.iloc[:, 0]
        df_diff = pd.DataFrame(diff_values, columns=[diff_name])
        output_diff_file = output_directory_group / f"{diff_name}_average.csv"
        df_diff.to_csv(output_diff_file)
        print(f"Difference saved to {output_diff_file}")

        visualize_coupling(
            coupling_file=output_directory_group,
            group_name=diff_name,
            output_dir=output_directory_group,
            ses=ses,
            vmin=diff_vmin,
            vmax=diff_vmax,
            figure_label=modality_name,
            file_suffix=modality_slug,
            symmetric_scale=diff_symmetric,
        )


def visualize_coupling(
    coupling_file,
    group_name,
    output_dir,
    ses,
    vmin=0,
    vmax=0.47,
    figure_label="Structure-Function Coupling",
    file_suffix="sfc",
    symmetric_scale=False,
):
    """Create multi-view brain surface visualizations from ROI-level values.

    Args:
        coupling_file: Directory containing ``{group_name}_average.csv`` files.
        group_name: Name of the group (for example, ``"superagers_tp1"``).
        output_dir: Directory where visualizations will be written.
        ses: Timepoint identifier without prefix (for example, ``"01"``).
        vmin: Minimum value for color scaling. If ``None``, it is inferred.
        vmax: Maximum value for color scaling. If ``None``, it is inferred.
        figure_label: Figure title prefix used in the combined image.
        file_suffix: Suffix used in the output PNG filename.
        symmetric_scale: Whether to force a symmetric range around zero when
            inferring color limits.
    """
    coupling_csv = Path(f"{coupling_file}/{group_name}_average.csv")
    coupling_df = pd.read_csv(coupling_csv, index_col=0)

    # Drop the subcoritical ROIs 
    coupling_df = coupling_df[~coupling_df.index.str.contains('Subcortical')]
    
    # Extract data 
    rho_values = coupling_df.iloc[:, 0].values  # Get the first column's values
    if vmin is None or vmax is None:
        if symmetric_scale:
            max_abs = np.nanmax(np.abs(rho_values))
            inferred_vmin, inferred_vmax = -max_abs, max_abs
        else:
            inferred_vmin, inferred_vmax = np.nanmin(rho_values), np.nanmax(rho_values)
        if vmin is None:
            vmin = inferred_vmin
        if vmax is None:
            vmax = inferred_vmax

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
    plt.suptitle(f"{figure_label}\n{subject} - ses-{ses}", fontsize=16, y=0.98) 
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
    combined_path = output_path / f"{subject}_ses-{ses}_{file_suffix}.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Combined visualization saved to {combined_path}")
    
    # Remove temp_output_path
    for file in os.listdir(temp_output_path):
        os.remove(os.path.join(temp_output_path, file))
    os.rmdir(temp_output_path)

    return fig


def main(output_directory_group, connectivity_file, superager_file, ses, sfc_df, output_directory, fisher_z_connectivity_file, output_group_connectivity_file, label_type):
    output_directory_group = Path(output_directory_group)
    output_directory_group.mkdir(parents=True, exist_ok=True)

    # Define output file paths for different categories
    output_files = {
        'all_subjects': output_directory_group / "all_subjects_average.csv",
        'superagers_tp1': output_directory_group / "superagers_tp1_average.csv",
        'non_superagers_tp1': output_directory_group / "non_superagers_tp1_average.csv",
        'superagers_tp2': output_directory_group / "superagers_tp2_average.csv",
        'non_superagers_tp2': output_directory_group / "non_superagers_tp2_average.csv",
        'superagers_long': output_directory_group / "superagers_long_average.csv",
        'non_superagers_long': output_directory_group / "non_superagers_long_average.csv",
    }

    # Combine individual data into one df
    consolidated_df = consolidate_sfc_data(ses, sfc_df)
    if consolidated_df is None:
        print(f"Skipping ses-{ses}: no SFC data.")
        return

    # Fisher-z transform the connectivity data
    fisher_transform(connectivity_file, output_directory, ses)

    # Process the connectivity data and save averages
    process_connectivity(fisher_z_connectivity_file, superager_file, output_files, ses, label_type)
    
    for group_name in group_names:
        # Visualize SFC in selected groups
        visualize_coupling(
            coupling_file=output_group_connectivity_file,
            group_name=group_name,
            output_dir=output_directory_group,
            vmax=None,
            ses=ses)
    
    # Calculate and visualize the difference
    if label_type == "long":
        superager_path = output_files['superagers_long']
        non_superager_path = output_files['non_superagers_long']
        diff_name = "diff_superagers_vs_non_superagers_long"
    elif ses == "01":
        superager_path = output_files['superagers_tp1']
        non_superager_path = output_files['non_superagers_tp1']
        diff_name = "diff_superagers_vs_non_superagers_tp1"
    else:
        superager_path = output_files['superagers_tp2']
        non_superager_path = output_files['non_superagers_tp2']
        diff_name = "diff_superagers_vs_non_superagers_tp2"

    if superager_path.exists() and non_superager_path.exists():
        print(f"Calculating difference between {superager_path} and {non_superager_path}...")
        df_super = pd.read_csv(superager_path, index_col=0)
        df_non = pd.read_csv(non_superager_path, index_col=0)
        
        # Calculate difference (Superager - Non-Superager)
        # Use values to avoid column name mismatch
        diff_values = df_super.iloc[:, 0] - df_non.iloc[:, 0]
        df_diff = pd.DataFrame(diff_values, columns=[diff_name])

        # Save the difference
        output_diff_file = output_directory_group / f"{diff_name}_average.csv"
        df_diff.to_csv(output_diff_file)
        print(f"Difference saved to {output_diff_file}")
        
        # Visualize the difference with symmetric scale
        visualize_coupling(
            coupling_file=output_group_connectivity_file,
            group_name=diff_name,
            output_dir=output_directory_group,
            ses=ses,
            vmin=-0.03, 
            vmax=0.03
        )


if __name__ == "__main__":
    sessions = ["01"]
    label_type = "long"  # Options: "tp1", "tp2", "long" based off of which superager definition you want to use
    for ses in sessions:
        if label_type == "long":
            group_names = ["superagers_long", "non_superagers_long"]
        elif ses == "01":
            group_names = ["superagers_tp1", "non_superagers_tp1"] # This is to loop through the visualization
        else:
            group_names = ["superagers_tp2", "non_superagers_tp2"]

    for ses in sessions:
            print("--------------------------")
            print(f"Processing ses-{ses} ...")
            print("--------------------------")

            superager_file = "/home/rachel/Desktop/data/superager.csv"  
            sfc_df = Path(f"/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-{ses}")
            output_directory = Path(f"{sfc_df}/all_to_all_roi_matrices")
            output_directory_group = Path(f"{sfc_df}/group_connectivity_matrices")
            connectivity_file = Path(f"{output_directory}/all_sfc_data_ses-{ses}.csv")
            fisher_z_connectivity_file = Path(f"{output_directory}/fisher_z_all_sfc_ses-{ses}.csv")
            output_group_connectivity_file = Path(f"{output_directory_group}")
            fc_connectivity_file = Path(
                f"/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space/ses-{ses}/all_to_all_roi_matrices/fisher_z_all_to_all_roi_matrix.csv"
            )
            sc_connectivity_file = Path(
                f"/home/rachel/Desktop/schaefer_analysis/structural_connectivity/ses-{ses}/all_to_all_roi_matrices/all_to_all_roi_matrix.csv"
            )
            fc_output_directory_group = Path(
                f"/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space/ses-{ses}/group_connectivity_matrices"
            )
            sc_output_directory_group = Path(
                f"/home/rachel/Desktop/schaefer_analysis/structural_connectivity/ses-{ses}/group_connectivity_matrices"
            )

            # Make sure the output directory exists
            output_directory.mkdir(parents=True, exist_ok=True)
            output_directory_group.mkdir(parents=True, exist_ok=True)
            fc_output_directory_group.mkdir(parents=True, exist_ok=True)
            sc_output_directory_group.mkdir(parents=True, exist_ok=True)

            main(
                output_directory_group=output_directory_group,
                connectivity_file=connectivity_file,
                superager_file=superager_file,
                ses=ses,
                sfc_df=sfc_df,
                output_directory=output_directory,
                fisher_z_connectivity_file=fisher_z_connectivity_file,
                output_group_connectivity_file=output_group_connectivity_file,
                label_type=label_type
            )

            if fc_connectivity_file.exists():
                generate_modality_surface_plots(
                    modality_name="Functional Connectivity (Fisher Z)",
                    modality_slug="fc_fisher_z",
                    edge_connectivity_file=fc_connectivity_file,
                    superager_file=superager_file,
                    output_directory_group=fc_output_directory_group,
                    ses=ses,
                    label_type=label_type,
                    group_names=group_names,
                    group_vmin=None,
                    group_vmax=None,
                    group_symmetric=False,
                    diff_vmin=None,
                    diff_vmax=None,
                    diff_symmetric=True,
                )
            else:
                print(f"Missing FC connectivity file: {fc_connectivity_file}")

            if sc_connectivity_file.exists():
                generate_modality_surface_plots(
                    modality_name="Structural Connectivity",
                    modality_slug="sc",
                    edge_connectivity_file=sc_connectivity_file,
                    superager_file=superager_file,
                    output_directory_group=sc_output_directory_group,
                    ses=ses,
                    label_type=label_type,
                    group_names=group_names,
                    group_vmin=None,
                    group_vmax=None,
                    group_symmetric=False,
                    diff_vmin=None,
                    diff_vmax=None,
                    diff_symmetric=True,
                )
            else:
                print(f"Missing SC connectivity file: {sc_connectivity_file}")
            
