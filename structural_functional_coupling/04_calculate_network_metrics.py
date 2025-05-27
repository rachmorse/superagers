#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import os
# import re


def extract_network_from_roi_name(roi_name):
    """
    Extract network name from the ROI name in Schaefer atlas with 7Networks format.
    
    Args:
        roi_name (str): ROI name from Schaefer atlas (e.g., "7Networks_LH_Cont_PFCl_1")
        
    Returns:
        str: Network name or "Unknown" if not found
    """
    if not isinstance(roi_name, str):
        return "Unknown"
    
    # For 7Networks format, the network abbreviation is typically the 3rd part after splitting by underscore
    parts = roi_name.split('_')
    if len(parts) >= 3:
        network_abbr = parts[2]
        
        # Map network abbreviations to full names
        network_mapping = {
            'Vis': 'Visual',
            'SomMot': 'Somatomotor',
            'DorsAttn': 'DorsalAttention',
            'SalVentAttn': 'VentralAttention',
            'Limbic': 'Limbic',
            'Cont': 'Frontoparietal',  
            'Default': 'Default'
        }
        
        return network_mapping.get(network_abbr, "Unknown")
    
    return "Unknown"

def identify_subcortical_rois(roi_names):
    """
    Identify ROIs that belong to subcortical regions.
    
    Args:
        roi_names (list): List of ROI names
        
    Returns:
        list: List of ROIs that are subcortical
    """
    return [roi for roi in roi_names if "Subcortical" in roi]


def group_rois_by_network(roi_names):
    """
    Group ROIs by network based on their names.
    
    Args:
        roi_names (list): List of ROI names
        
    Returns:
        dict: Dictionary mapping network name to list of ROI names
    """
    network_to_rois = {}
    
    for roi_name in roi_names:
        network = extract_network_from_roi_name(roi_name)
        if network not in network_to_rois:
            network_to_rois[network] = []
        network_to_rois[network].append(roi_name)
    
    return network_to_rois


def calculate_network_metrics(subject, ses, func_dir, struct_dir, sfc_dir):
    """
    Calculate network connectivity metrics for a single subject.
    
    Args:
        subject (str): Subject ID
        ses (str): Session ID
        func_dir (Path): Directory with functional connectivity matrices
        struct_dir (Path): Directory with structural connectivity matrices
        sfc_dir (Path): Directory with structure-function coupling data

    Returns:
        dict: Dictionary with network metrics for the subject
    """
    metrics = {'subject': subject}
    
    # Load connectivity matrices
    func_file = func_dir / f"{subject}_ses-0{ses}_functional_connectivity_matrix.csv"
    struct_file = struct_dir / f"{subject}_ses-0{ses}_structural_connectivity_matrix.csv"
    sfc_file = sfc_dir / f"{subject}_ses-0{ses}_structure_function_coupling.csv"
    
    try:
        # Load functional connectivity data
        if func_file.exists():
            func_data = pd.read_csv(func_file, index_col=0)
            
            # Apply Fisher z-transform to functional connectivity values
            func_data = np.arctanh(func_data)
            
            # Calculate average FC across all ROIs (excluding diagonal)
            mask = ~np.eye(func_data.shape[0], dtype=bool)
            all_fc = func_data.values[mask].mean()
            metrics[f'func_all_{ses}'] = all_fc

            # Group ROIs by network
            roi_names = func_data.index.tolist()
            network_to_rois = group_rois_by_network(roi_names)

            # Identify subcortical ROIs and hippocampus specifically
            subcortical_rois = identify_subcortical_rois(roi_names)
            hipp_rois = [roi for roi in roi_names if "Subcortical" in roi and ("Left Hippocampus" in roi or "Right Hippocampus" in roi)]

            # Calculate subcortical metrics if any subcortical ROIs exist
            if subcortical_rois:
                # Calculate within-subcortical connectivity
                if len(subcortical_rois) > 1:
                    subcort_submatrix = func_data.loc[subcortical_rois, subcortical_rois]
                    mask = ~np.eye(subcort_submatrix.shape[0], dtype=bool)
                    within_subcort = subcort_submatrix.values[mask].mean()
                    metrics[f'func_within_Subcortical_{ses}'] = within_subcort
                    
                    # Calculate between subcortical and other regions
                    other_rois = [r for r in roi_names if r not in subcortical_rois]
                    if other_rois:
                        between_subcort = func_data.loc[subcortical_rois, other_rois].values.mean()
                        metrics[f'func_between_Subcortical_{ses}'] = between_subcort
                    
                    # Calculate average subcortical connectivity
                    metrics[f'func_all_Subcortical_{ses}'] = func_data.loc[subcortical_rois, :].values.mean()

            # Calculate hippocampus metrics
            if hipp_rois:
                hipp_conn = func_data.loc[hipp_rois, :].values.mean()
                metrics[f'func_Hippocampus_{ses}'] = hipp_conn
            
            # Calculate within-network and between-network functional connectivity
            for network, roi_list in network_to_rois.items():
                if network == "Unknown":
                    continue
                    
                if len(roi_list) > 1:  
                    # Get the submatrix for this network
                    network_submatrix = func_data.loc[roi_list, roi_list]
                    
                    # Calculate within-network connectivity (excluding diagonal)
                    mask = ~np.eye(network_submatrix.shape[0], dtype=bool)
                    within_conn = network_submatrix.values[mask].mean()
                    metrics[f'func_within_{network}_{ses}'] = within_conn
                    
                    # Calculate between-network connectivity
                    other_rois = [r for r in roi_names if r not in roi_list]
                    if other_rois:
                        between_conn = func_data.loc[roi_list, other_rois].values.mean()
                        metrics[f'func_between_{network}_{ses}'] = between_conn
                    else:
                        metrics[f'func_between_{network}_{ses}'] = np.nan

                    # Calculate total network connectivity (within + between)
                    within_matrix = network_submatrix.values
                    between_matrix = func_data.loc[roi_list, other_rois].values if other_rois else np.array([])
                    within_vals = within_matrix.flatten()
                    between_vals = between_matrix.flatten()
                    
                    all_vals = np.concatenate([within_vals, between_vals]) if between_vals.size > 0 else within_vals
                    all_conn = all_vals.mean() if len(all_vals) > 0 else np.nan
                    metrics[f'func_all_{network}_{ses}'] = all_conn


                else:
                    metrics[f'func_within_{network}_{ses}'] = np.nan
                    metrics[f'func_between_{network}_{ses}'] = np.nan
            
        # Load structural connectivity data
        if struct_file.exists():
            struct_data = pd.read_csv(struct_file, index_col=0)

            # Calculate average SC across all ROIs (excluding diagonal)
            mask = ~np.eye(struct_data.shape[0], dtype=bool)
            all_sc = struct_data.values[mask].mean()
            metrics[f'struct_all_{ses}'] = all_sc
            
            # Group ROIs by network
            roi_names = struct_data.index.tolist()
            network_to_rois = group_rois_by_network(roi_names)

            # Identify subcortical ROIs and hippocampus specifically
            subcortical_rois = identify_subcortical_rois(roi_names)
            hipp_rois = [roi for roi in roi_names if "Subcortical" in roi and ("Left Hippocampus" in roi or "Right Hippocampus" in roi)]
            
            # Calculate subcortical metrics if any subcortical ROIs exist
            if subcortical_rois:
                # Calculate within-subcortical connectivity
                if len(subcortical_rois) > 1:
                    subcort_submatrix = struct_data.loc[subcortical_rois, subcortical_rois]
                    mask = ~np.eye(subcort_submatrix.shape[0], dtype=bool)
                    within_subcort = subcort_submatrix.values[mask].mean()
                    metrics[f'struct_within_Subcortical_{ses}'] = within_subcort
                    
                    # Calculate between subcortical and other regions
                    other_rois = [r for r in roi_names if r not in subcortical_rois]
                    if other_rois:
                        between_subcort = struct_data.loc[subcortical_rois, other_rois].values.mean()
                        metrics[f'struct_between_Subcortical_{ses}'] = between_subcort
                    
                    # Calculate average subcortical connectivity
                    metrics[f'struct_all_Subcortical_{ses}'] = struct_data.loc[subcortical_rois, :].values.mean()
            
            # Calculate left hippocampus metrics
            if hipp_rois:
                hipp_rois = struct_data.loc[hipp_rois, :].values.mean()
                metrics[f'struct_Hippocampus_{ses}'] = hipp_rois
            
            # Calculate within-network and between-network structural connectivity
            for network, roi_list in network_to_rois.items():
                if network == "Unknown":
                    continue
                    
                if len(roi_list) > 1:  
                    # Get the submatrix for this network
                    network_submatrix = struct_data.loc[roi_list, roi_list]
                    within_matrix = network_submatrix.values
                    
                    # Calculate within-network connectivity (excluding diagonal)
                    mask = ~np.eye(network_submatrix.shape[0], dtype=bool)
                    within_conn = network_submatrix.values[mask].mean()
                    metrics[f'struct_within_{network}_{ses}'] = within_conn
                    
                    # Calculate between-network connectivity
                    other_rois = [r for r in roi_names if r not in roi_list]
                    if other_rois:
                        between_conn = struct_data.loc[roi_list, other_rois].values.mean()
                        metrics[f'struct_between_{network}_{ses}'] = between_conn
                    else:
                        metrics[f'struct_between_{network}_{ses}'] = np.nan
                        
                    # Calculate total network connectivity (within + between)
                    within_matrix = network_submatrix.values
                    between_matrix = struct_data.loc[roi_list, other_rois].values if other_rois else np.array([])
                    within_vals = within_matrix.flatten()
                    between_vals = between_matrix.flatten()
                    
                    all_vals = np.concatenate([within_vals, between_vals]) if between_vals.size > 0 else within_vals
                    all_conn = all_vals.mean() if len(all_vals) > 0 else np.nan
                    metrics[f'struct_all_{network}_{ses}'] = all_conn

                else:
                    metrics[f'struct_within_{network}_{ses}'] = np.nan
                    metrics[f'struct_between_{network}_{ses}'] = np.nan
        
        # Load SFC data
        if sfc_file.exists():
            sfc_data = pd.read_csv(sfc_file)
            
            # Get column names
            roi_col = sfc_data.columns[0]
            val_col = sfc_data.columns[1] if len(sfc_data.columns) > 1 else None
            
            if val_col:
                # Apply Fisher z-transform to SFC correlation values
                sfc_data['z_value'] = np.arctanh(sfc_data[val_col])
                
                # Calculate average SFC across all ROIs first
                metrics[f'sfc_all_{ses}'] = np.nanmean(sfc_data['z_value'])

                # Group ROIs by network
                roi_names = sfc_data[roi_col].tolist()
                roi_to_network = {roi: extract_network_from_roi_name(roi) for roi in roi_names}
                
                # Identify subcortical ROIs and hippocampus specifically in SFC data
                subcortical_rois = [roi for roi in roi_names if "Subcortical" in roi]
                hipp_rois = [roi for roi in roi_names if "Subcortical" in roi and ("Left Hippocampus" in roi or "Right Hippocampus" in roi)]
                
                # Calculate average SFC for subcortical regions
                if subcortical_rois:
                    subcort_values = sfc_data[sfc_data[roi_col].isin(subcortical_rois)]['z_value'].values
                    if len(subcort_values) > 0:
                        metrics[f'sfc_Subcortical_{ses}'] = np.nanmean(subcort_values)
                    else:
                        metrics[f'sfc_Subcortical_{ses}'] = np.nan
                
                # Calculate average SFC for hippocampus
                if hipp_rois:
                    hipp_rois = sfc_data[sfc_data[roi_col].isin(hipp_rois)]['z_value'].values
                    if len(hipp_rois) > 0:
                        metrics[f'sfc_Hippocampus_{ses}'] = np.nanmean(hipp_rois)
                    else:
                        metrics[f'sfc_Hippocampus_{ses}'] = np.nan
                        
                # Calculate average SFC for each network
                for network in set(roi_to_network.values()):
                    if network == "Unknown":
                        continue
                        
                    network_rois = [roi for roi, net in roi_to_network.items() if net == network]
                    if network_rois:
                        network_values = sfc_data[sfc_data[roi_col].isin(network_rois)]['z_value'].values
                        if len(network_values) > 0:
                            metrics[f'sfc_{network}_{ses}'] = np.nanmean(network_values)
                        else:
                            metrics[f'sfc_{network}_{ses}'] = np.nan
                    else:
                        metrics[f'sfc_{network}_{ses}'] = np.nan
            
    except Exception as e:
        print(f"Error processing {subject}: {e}")
    
    return metrics

def main(ses, func_base_dir, struct_base_dir, sfc_base_dir, output_dir):
    """
    Main function to calculate network metrics for all subjects.
    
    Args:
        ses (str): Session ID
        func_base_dir (Path): Base directory for functional connectivity
        struct_base_dir (Path): Base directory for structural connectivity
        sfc_base_dir (Path): Base directory for structure-function coupling
        output_dir (Path): Directory to save output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define directories for each data type
    func_dir = func_base_dir / f"native_space/ses-0{ses}/individual_connectivity_matrices"
    struct_dir = struct_base_dir / f"ses-0{ses}/individual_connectivity_matrices"
    sfc_dir = sfc_base_dir / f"ses-0{ses}/individual_coupling_matrices"
    
    # Get all subject files
    func_files = list(func_dir.glob("sub-*_ses-*_functional_connectivity_matrix.csv"))
    struct_files = list(struct_dir.glob("sub-*_ses-*_structural_connectivity_matrix.csv"))
    sfc_files = list(sfc_dir.glob("sub-*_ses-*_structure_function_coupling.csv"))

    # Get unique subject IDs
    subject_ids = set()
    for f in func_files + struct_files + sfc_files:
        subject_id = f.name.split('_')[0]
        subject_ids.add(subject_id)
    
    print(f"Found {len(subject_ids)} unique subjects")
    
    # Calculate metrics for each subject
    all_metrics = []
    
    for subject in sorted(subject_ids):
        metrics = calculate_network_metrics(subject, ses, func_dir, struct_dir, sfc_dir)
        all_metrics.append(metrics)
    
    # Convert to DataFrame and save
    metrics_df = pd.DataFrame(all_metrics)
    output_file = os.path.join(output_dir, f"network_connectivity_metrics_ses-0{ses}.csv")
    metrics_df.to_csv(output_file, index=False)
    
    print(f"Saved metrics for {len(metrics_df)} subjects to {output_file}")
    
    # Print a summary of the metrics
    print("\nMetrics summary:")
    for column in metrics_df.columns:
        if column != 'subject':
            mean_val = metrics_df[column].mean()
            std_val = metrics_df[column].std()
            if not np.isnan(mean_val):
                print(f"{column}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    # Define parameters
    sessions = ["1", "2"]
    
    for ses in sessions:
        print("--------------------------")
        print(f"Processing ses-0{ses}")
        print("--------------------------")

        # Base directories
        base_dir = Path("/home/rachel/Desktop/schaefer_analysis")
        func_base_dir = base_dir / "functional_connectivity"
        struct_base_dir = base_dir / "structural_connectivity"
        sfc_base_dir = base_dir / "structure_function_coupling"
        output_dir = sfc_base_dir / f"ses-0{ses}/network_metrics"
        
        # Run the main function
        main(ses, func_base_dir, struct_base_dir, sfc_base_dir, output_dir)