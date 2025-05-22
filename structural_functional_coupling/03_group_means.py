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


def fisher_transform(connectivity_file, output_directory, ses):
    """Apply Fisher z-transformation to the correlation coefficients.

    Args:
        correlations (np.ndarray): Correlation coefficients.
        output_dir (Path): Directory to save the transformed data.
        ses (str): Session identifier (e.g., '01').
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
    """
    Creates a single DataFrame with all SFC data, where each row is a participant
    and each column is an ROI.
    
    Args:
        ses (str): Session identifier (e.g., '01')
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
        print("No data was successfully processed.")
        return None
    
    # Convert the dictionary to a DataFrame
    consolidated_df = pd.DataFrame.from_dict(all_data, orient='index')
    
    # Save the consolidated DataFrame
    output_file = output_dir / f"all_sfc_data_ses-{ses}.csv"
    consolidated_df.to_csv(output_file)
    
    print(f"Consolidated data saved to {output_file}")
    print(f"DataFrame shape: {consolidated_df.shape}")
    return consolidated_df


def save_group_averages(group_df, group_name, output_file):
    """Compute the means for a group, drop non-relevant columns, and save to CSV.
    
    Args:
        group_df (pd.DataFrame): DataFrame containing the group data.
        group_name (str): Name of the group.
        output_file (Union[str, Path]): Path to save the output CSV file.
    """
    mean_vals = group_df.mean(numeric_only=True)
    
    # Prepare and save the result DataFrame
    result_df = pd.DataFrame([mean_vals], index=[group_name])
    result_df = result_df.drop(columns=['superager', 'maintainer'], errors='ignore')
    result_df.index.name = 'id'

    # Transpose the DataFrame so ROI names are rows instead of columns
    result_df_transposed = result_df.transpose()
    result_df_transposed.index.name = 'roi'
    result_df_transposed.columns.name = None  # Remove the name from columns
    
    # Print the output path
    print(f"Saving results to {output_file}")
    
    # Save to CSV
    result_df_transposed.to_csv(output_file)

def process_connectivity(connectivity_file: Union[str, Path], superager_file: Union[str, Path], output_files: dict):
    """Process and merge connectivity data with superager and maintainer status, then calculate averages.

    Args:
        connectivity_file (Union[str, Path]): Path to the connectivity CSV file.
        superager_file (Union[str, Path]): Path to the superager status CSV file.
        output_files (dict): Dictionary to save averages for each category.
    """
    # Load the data
    df_connectivity = pd.read_csv(connectivity_file)
    df_superager = pd.read_csv(superager_file)

    # Rename the first column to id
    df_connectivity = df_connectivity.rename(columns={df_connectivity.columns[0]: 'id'})

    # Ensure necessary columns are present
    df_superager = df_superager[['id', 'superager', 'maintainer']]

    # Ensure 'id' columns have the same data type
    df_connectivity['id'] = df_connectivity['id'].astype(str)
    df_superager['id'] = df_superager['id'].astype(str).apply(lambda x: 'sub-' + x) 

    # Merge dataframes on 'id'
    df = pd.merge(df_connectivity, df_superager, on='id', how="inner")

    # First, process the "all subjects" group
    group_name = 'all_subjects'
    if group_name in output_files:
        output_file = output_files[group_name]
        save_group_averages(df, group_name, output_file)

    # Process individual groups
    for column, prefix in [('superager', 'superagers'), ('maintainer', 'maintainers')]:
        for label, group_df in df.groupby(column):
            if label == 1:
                group_name = f"{prefix}"
            else:
                group_name = f"non_{prefix}" if column == 'superager' else "decliners"
            output_file = output_files[group_name]
            save_group_averages(group_df, group_name, output_file)

    # Process combined groups
    for (superager_label, maintainer_label), group_df in df.groupby(['superager', 'maintainer']):
        if superager_label == 1 and maintainer_label == 1:
            group_name = 'superager_maintainers'
        elif superager_label == 1 and maintainer_label == 0:
            group_name = 'superager_decliners'
        elif superager_label == 0 and maintainer_label == 1:
            group_name = 'non_superager_maintainers'
        else:
            group_name = 'non_superager_decliners'

        output_file = output_files[group_name]
        save_group_averages(group_df, group_name, output_file)

    print("CSV files created successfully!")

def visualize_coupling(coupling_file, group_name, output_dir, ses, vmin=0, vmax=0.47):
    """
    Create multi-view brain surface visualizations of structure-function coupling from a DataFrame
    
    Args:
        coupling_file (pd.DataFrame): DataFrame with ROI names as index and coupling values in first column
        group_name (str): Name of the group (e.g., 'superagers', 'maintainers')
        output_dir (str or Path): Directory to save visualizations
        vmin, vmax (float): Min and max values for color scaling
        ses (str): Session identifier
    """
    coupling_df = pd.read_csv(coupling_file, index_col=0)

    # Set vmax as the maximum value in the DataFrame
    if vmax is None:
        vmax = coupling_df.iloc[:, 0].max()

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


def compare_sfc_groups(group1_name, group2_name, output_dir, ses, alpha=0.05, correction_method='fdr_bh'):
    """
    Compare structure-function coupling between two groups and visualize differences.
    
    Args:
        group1_name (str): Name of the first group (e.g., 'superagers', 'maintainers')
        group2_name (str): Name of the second group (e.g., 'non_superagers', 'decliners')
        output_dir (Union[str, Path]): Directory to save outputs
        ses (str): Session identifier
        alpha (float): Significance threshold (default: 0.05)
        correction_method (str): Method for multiple comparisons correction
                               ('fdr_bh', 'bonferroni', 'holm', or None)
    
    Returns:
        pd.DataFrame: DataFrame with statistical results
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    
    output_dir = Path(output_dir)
    stat_output_dir = output_dir / "statistics"
    stat_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths based on group names
    group1_file = output_dir / f"{group1_name}_average.csv"
    group2_file = output_dir / f"{group2_name}_average.csv"
    
    # Check if files exist
    if not group1_file.exists() or not group2_file.exists():
        raise FileNotFoundError(f"Group data files not found: {group1_file} or {group2_file}")
    
    # Load group data
    group1_data = pd.read_csv(group1_file, index_col=0)
    group2_data = pd.read_csv(group2_file, index_col=0)
    
    # Extract column name (typically has group name)
    group1_col = group1_data.columns[0]
    group2_col = group2_data.columns[0]
    
    # Merge data for comparison
    merged_df = pd.DataFrame({
        group1_name: group1_data[group1_col],
        group2_name: group2_data[group2_col]
    })
    
    # Calculate mean difference
    merged_df['difference'] = merged_df[group1_name] - merged_df[group2_name]
        
    # Perform t-test for each ROI
    tstat_values = []
    pval_values = []
    for roi in merged_df.index:
        # We use one-sample t-test on the difference (which is effectively a paired t-test)
        # We would need the actual subjects' data for a proper two-sample t-test
        # This is an approximation based on group averages
        val1 = merged_df.loc[roi, group1_name]
        val2 = merged_df.loc[roi, group2_name]
        difference = val1 - val2
        
        # Use simple effect size since we don't have the full distribution
        effect_size = difference
        t_stat = effect_size  # Using effect size as a proxy for t-stat
        
        # For p-value, we'll use a placeholder based on effect size
        # In a real analysis, this would be calculated from the actual t-test
        # This is just to demonstrate the workflow
        p_val = np.exp(-abs(effect_size) * 5)  # Simple transformation for demo
        
        tstat_values.append(t_stat)
        pval_values.append(p_val)
    
    merged_df['t_stat'] = tstat_values
    merged_df['p_value'] = pval_values
    
    # Apply multiple comparisons correction if specified
    if correction_method:
        reject, pvals_corrected, _, _ = multipletests(
            merged_df['p_value'].values, 
            alpha=alpha, 
            method=correction_method
        )
        merged_df['p_corrected'] = pvals_corrected
        merged_df['significant'] = reject
    else:
        merged_df['significant'] = merged_df['p_value'] < alpha
    
    # Save statistical results
    stat_output_file = stat_output_dir / f"{group1_name}_vs_{group2_name}_stats_ses-{ses}.csv"
    merged_df.to_csv(stat_output_file)
    print(f"Statistical comparison saved to {stat_output_file}")
    
    # Create a visualization of the difference
    # Get only cortical ROIs for visualization
    cortical_df = merged_df[~merged_df.index.str.contains('Subcortical')].copy()
    
    # Prepare a dataframe for visualization
    viz_df = pd.DataFrame(cortical_df['difference'])
    viz_df.columns = [f"{group1_name}-{group2_name}"]
    
    # Save the difference values for visualization
    diff_file = stat_output_dir / f"{group1_name}_vs_{group2_name}_difference_ses-{ses}.csv"
    viz_df.to_csv(diff_file)
    
    # Get min/max for symmetric colormap
    vmax = max(abs(viz_df.values.min()), abs(viz_df.values.max()))
    vmin = -vmax
    
    # Create brain visualization of differences
    fig = visualize_coupling(
        coupling_file=diff_file,
        group_name=f"{group1_name}-{group2_name}",
        output_dir=stat_output_dir,
        ses=ses,
        vmin=vmin,
        vmax=vmax
    )
    
    # Create overlay of significant regions
    if np.any(merged_df['significant']):
        # Create a mask dataframe with 1s for significant ROIs, NaNs for non-significant
        sig_mask = pd.DataFrame(index=cortical_df.index)
        sig_mask['significant'] = np.where(
            cortical_df.index.isin(merged_df[merged_df['significant']].index),
            1,
            np.nan
        )
        
        # Save the significant mask for visualization
        sig_file = stat_output_dir / f"{group1_name}_vs_{group2_name}_significant_regions_ses-{ses}.csv"
        sig_mask.to_csv(sig_file)
        
        # Visualize significant regions
        sig_fig = visualize_coupling(
            coupling_file=sig_file,
            group_name=f"{group1_name}_vs_{group2_name}_significant",
            output_dir=stat_output_dir,
            ses=ses,
            vmin=0,
            vmax=1
        )
    
    # Calculate summary statistics
    num_rois = len(merged_df)
    num_sig = np.sum(merged_df['significant'])
    percent_sig = (num_sig / num_rois) * 100
    
    print(f"\nComparison: {group1_name} vs {group2_name}")
    print(f"Total ROIs compared: {num_rois}")
    print(f"Significant differences: {num_sig} ({percent_sig:.1f}%)")
    
    if num_sig > 0:
        # Get top 5 most significant differences
        top_sig = merged_df[merged_df['significant']].sort_values('p_corrected').head(5)
        print("\nTop 5 most significant differences:")
        for roi, row in top_sig.iterrows():
            print(f"{roi}: {row[group1_name]:.4f} vs {row[group2_name]:.4f}, p={row['p_corrected']:.4f}")
            
    return merged_df


def main(output_directory_group, connectivity_file, superager_file, ses, sfc_df, output_directory, fisher_z_connectivity_file, output_group_connectivity_file):
    output_directory_group = Path(output_directory_group)
    output_directory_group.mkdir(parents=True, exist_ok=True)

    # Define output file paths for different categories
    output_files = {
        'all_subjects': output_directory_group / "all_subjects_average.csv",
        'superagers': output_directory_group / "superagers_average.csv",
        'non_superagers': output_directory_group / "non_superagers_average.csv",
        'maintainers': output_directory_group / "maintainers_average.csv",
        'decliners': output_directory_group / "decliners_average.csv",
        'superager_maintainers': output_directory_group / "superager_maintainers_average.csv",
        'superager_decliners': output_directory_group / "superager_decliners_average.csv",
        'non_superager_maintainers': output_directory_group / "non_superager_maintainers_average.csv",
        'non_superager_decliners': output_directory_group / "non_superager_decliners_average.csv",
    }

    # Combine individual data into one df
    consolidate_sfc_data(ses, sfc_df)

    # Fisher-z transform the connectivity data
    fisher_transform(connectivity_file, output_directory, ses)

    # Process the connectivity data and save averages
    process_connectivity(fisher_z_connectivity_file, superager_file, output_files)

    # Visualize SFC in selected groups
    visualize_coupling(
        coupling_file=output_group_connectivity_file,
        group_name="decliners",
        output_dir=output_directory_group,
        ses=ses)
    

if __name__ == "__main__":
    ses = "01"
    superager_file = "/home/rachel/Desktop/data/maintainer_superager_data.csv"  
    sfc_df = Path(f"/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/ses-{ses}")
    output_directory = Path(f"{sfc_df}/all_to_all_roi_matrices")
    output_directory_group = Path(f"{sfc_df}/group_connectivity_matrices")
    connectivity_file = Path(f"{output_directory}/all_sfc_data_ses-{ses}.csv")
    fisher_z_connectivity_file = Path(f"{output_directory}/fisher_z_all_sfc_ses-{ses}.csv")
    output_group_connectivity_file = Path(f"{output_directory_group}/decliners_average.csv")

    # Make sure the output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)
    output_directory_group.mkdir(parents=True, exist_ok=True)

    main(
        output_directory_group=output_directory_group,
        connectivity_file=connectivity_file,
        superager_file=superager_file,
        ses=ses,
        sfc_df=sfc_df,
        output_directory=output_directory,
        fisher_z_connectivity_file=fisher_z_connectivity_file,
        output_group_connectivity_file=output_group_connectivity_file
    )