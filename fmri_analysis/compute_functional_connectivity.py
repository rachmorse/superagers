from itertools import combinations
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure


def fisher_transform(correlations: np.ndarray) -> np.ndarray:
    """Apply Fisher z-transformation to the correlation coefficients.

    Args:
        correlations (np.ndarray): Correlation coefficients.

    Returns:
        np.ndarray: Transformed correlation coefficients.
    """
    return np.arctanh(correlations)


def prepare_directories(root_dir: Path, session: str, subdir_types: List[str]):
    """Ensure all necessary directories exist."""
    for subdir in subdir_types:
        path = root_dir / f"ses-{session}" / subdir
        path.mkdir(parents=True, exist_ok=True)


def compute_functional_connectivity(
    subject_id: str, timeseries: np.ndarray, output_dir: Path, ses: str, combined_labels: List[str]
) -> np.ndarray:
    """Compute the connectivity matrix for all ROIs and each network, save them to CSV files.

    Args:
        subject_id (str): Subject ID.
        timeseries (np.ndarray): Timeseries data for the subject.
        output_dir (Path): Directory where the connectivity data will be saved.
        ses (str): Session / timepoint
        timeseries_path (Path): Path to the directory containing the timeseries data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The full connectivity and fisher z-transformed matrices.
    """
    # Extract ROI names and networks using the combined labels
    network_mappings = create_network_mappings(combined_labels)
    
    # Create special mappings for subcortical regions (combining left and right)
    subcortical_structures = {}
    all_subcortical_indices = []  # Track all subcortical indices
    
    # First, identify all subcortical structures and their indices
    for index, label in enumerate(combined_labels):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
            
        if "Subcortical" in label:
            # Add to all subcortical indices list
            all_subcortical_indices.append(index)
            
            # Extract the structure name without "Left" or "Right" prefix
            parts = label.split(":")
            if len(parts) >= 2:
                # Extract structure name and remove "Left" or "Right" prefix
                structure_name = parts[1].strip()
                if "Left" in structure_name:
                    structure_name = structure_name.replace("Left", "").strip()
                elif "Right" in structure_name:
                    structure_name = structure_name.replace("Right", "").strip()
                
                # Add to the dictionary
                if structure_name not in subcortical_structures:
                    subcortical_structures[structure_name] = []
                subcortical_structures[structure_name].append(index)

    # Compute full connectivity matrix
    print("Computing full connectivity matrix...")
    correlation_measure = ConnectivityMeasure(kind="correlation", standardize=False)
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

    # Set the diagonal to 0 to exclude self-connectivity
    np.fill_diagonal(connectivity_matrix, 0)

    # Apply Fisher z-transformation
    fisher_z_matrix = fisher_transform(connectivity_matrix)

    # Define target directories
    all_to_all_dir = output_dir / f"ses-{ses}/all_to_all_roi_matrices"
    within_network_dir = output_dir / f"ses-{ses}/within_network_matrices"
    subcortical_dir = output_dir / f"ses-{ses}/subcortical_matrices"

    # Ensure directories exist
    prepare_directories(output_dir, ses, ["all_to_all_roi_matrices"])
    prepare_directories(output_dir, ses, ["within_network_matrices"]) 
    prepare_directories(output_dir, ses, ["subcortical_matrices"])

    # Save complete connectivity data
    save_connectivity_data(subject_id, "all_to_all_roi", connectivity_matrix, None, combined_labels, all_to_all_dir)
    save_connectivity_data(subject_id, "all_to_all_roi", None, fisher_z_matrix, combined_labels, all_to_all_dir)

    # Process cortical networks (non-subcortical)
    for network, indices in network_mappings.items():
        # Skip subcortical networks to handle them separately
        if network == "Subcortical":
            continue
            
        network_timeseries = timeseries[:, indices]
        network_correlation_matrix = correlation_measure.fit_transform([network_timeseries])[0]
        np.fill_diagonal(network_correlation_matrix, 0)
        network_fisher_z_matrix = fisher_transform(network_correlation_matrix)

        network_labels = [combined_labels[i] for i in indices]

        # Save network connectivity data
        save_connectivity_data(
            subject_id,
            f"{network}_within_network",
            network_correlation_matrix,
            None,
            network_labels,
            within_network_dir,
        )

        save_connectivity_data(
            subject_id,
            f"{network}_within_network",
            None,
            network_fisher_z_matrix,
            network_labels,
            within_network_dir,
        )
    
    # Create an all subcortical 'network' connectivity matrix
    if all_subcortical_indices:
        all_subcortical_timeseries = timeseries[:, all_subcortical_indices]
        all_subcortical_corr_matrix = correlation_measure.fit_transform([all_subcortical_timeseries])[0]
        np.fill_diagonal(all_subcortical_corr_matrix, 0)
        all_subcortical_fisher_z_matrix = fisher_transform(all_subcortical_corr_matrix)
        
        all_subcortical_labels = [combined_labels[i] for i in all_subcortical_indices]
        
        # Save all-subcortical connectivity data
        save_connectivity_data(
            subject_id,
            "all_subcortical_rois",
            all_subcortical_corr_matrix,
            None,
            all_subcortical_labels,
            subcortical_dir,
        )
        
        save_connectivity_data(
            subject_id,
            "all_subcortical_rois",
            None,
            all_subcortical_fisher_z_matrix,
            all_subcortical_labels,
            subcortical_dir,
        )
    
    # Process individual subcortical structures (left and right combined)
    for structure_name, indices in subcortical_structures.items():
        if len(indices) > 0:
            structure_timeseries = timeseries[:, indices]
            structure_correlation_matrix = correlation_measure.fit_transform([structure_timeseries])[0]
            np.fill_diagonal(structure_correlation_matrix, 0)
            structure_fisher_z_matrix = fisher_transform(structure_correlation_matrix)

            structure_labels = [combined_labels[i] for i in indices]

            # Save subcortical structure connectivity data
            save_connectivity_data(
                subject_id,
                f"{structure_name.strip()}_bilateral",
                structure_correlation_matrix,
                None,
                structure_labels,
                subcortical_dir,
            )

            save_connectivity_data(
                subject_id,
                f"{structure_name.strip()}_bilateral",
                None,
                structure_fisher_z_matrix,
                structure_labels,
                subcortical_dir,
            )

    return connectivity_matrix, fisher_z_matrix


def create_network_mappings(labels: List[str]) -> Dict[str, List[int]]:
    """Create a dictionary mapping network names to ROI indices.

    Args:
        labels (List[str]): List of ROI labels.

    Returns:
        Dict[str, List[int]]: Dictionary mapping network names to indices.
    """
    # Initialize network mappings
    network_mappings = {}

    # Extract network names from the labels
    for index, label in enumerate(labels):
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        
        # For Schaefer atlas regions (format: 7Networks_LH_Vis_1)
        if label.startswith("7Networks_"):
            parts = label.split("_")
            if len(parts) >= 3:
                network_name = parts[2]  # Extract network name (e.g., "Vis", "SomMot", etc.)
            else:
                network_name = "Other"
        
        # For subcortical regions (format: Subcortical 202: Left Caudate)
        elif label.startswith("Subcortical"):
            network_name = "Subcortical"
        
        # For any other format
        else:
            # Try to extract from first part before a space or underscore
            parts = label.split(" ")[0].split("_")[0]
            network_name = parts
        
        # Add the index to the network mapping
        if network_name not in network_mappings:
            network_mappings[network_name] = []

        network_mappings[network_name].append(index)

    return network_mappings


def save_connectivity_data(
    subject_id: str,
    label: str,
    matrix: np.ndarray,
    fisher_z_matrix: np.ndarray,
    roi_names: List[str],
    output_dir: Path,
):
    """Save the connectivity data to CSV files.

    Args:
        subject_id (str): Subject ID.
        label (str): Label for the connectivity data.
        matrix (np.ndarray): The connectivity matrix.
        fisher_z_matrix (np.ndarray): The Fisher z-transformed connectivity matrix.
        roi_names (List[str]): List of ROI names.
        output_dir (Path): Directory where the connectivity data will be saved.
    """
    csv_output_path = output_dir / f"{subject_id}_{label}_matrix.csv"
    fisher_z_csv_output_path = output_dir / f"{subject_id}_fisher_z_{label}_matrix.csv"

    def save_csv(dataframe: pd.DataFrame, file_path: Path):
        # Always write a new file for the subject
        dataframe.to_csv(file_path, index_label="id", header=True)

    if matrix is not None:
        columns = [f"{roi1}-{roi2}" for roi1, roi2 in combinations(roi_names, 2)]
        df_all_fc = pd.DataFrame(index=[subject_id], columns=columns)
        upper_tri_indices = np.triu_indices(matrix.shape[0], k=1)
        df_all_fc.loc[subject_id, :] = matrix[upper_tri_indices]
        save_csv(df_all_fc, csv_output_path)

    if fisher_z_matrix is not None:
        columns = [f"{roi1}-{roi2}" for roi1, roi2 in combinations(roi_names, 2)]
        df_all_fc_fisher_z = pd.DataFrame(index=[subject_id], columns=columns)
        upper_tri_indices = np.triu_indices(fisher_z_matrix.shape[0], k=1)
        df_all_fc_fisher_z.loc[subject_id, :] = fisher_z_matrix[upper_tri_indices]
        save_csv(df_all_fc_fisher_z, fisher_z_csv_output_path)


def visualize_fc_data(
    subject_id: str,
    connectivity_matrix: np.ndarray,
    output_directory: Path,
    ses: str,
    is_fisher_z: bool = False,
):
    """Visualize the connectivity matrix.

    Args:
        subject_id (str): Subject ID.
        connectivity_matrix (np.ndarray): The connectivity matrix to be visualized.
        output_directory (Path): Directory where the plot will be saved.
        ses (str): Session / timepoint
        is_fisher_z (bool): Whether the matrix is Fisher Z-transformed.
    """
    # Visualize connectivity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(label="Correlation Coefficient")
    plt.title(f"Connectivity Matrix for Subject {subject_id}")
    plt.xlabel("Regions")
    plt.ylabel("Regions")
    plt.grid(False)
    plt.show()

    # Determine plot path based on Fisher Z transformation
    if is_fisher_z:
        plot_filename = f"{subject_id}_ses-{ses}_fisher_z_all_to_all_roi_matrix.png"
    else:
        plot_filename = f"{subject_id}_ses-{ses}_all_to_all_roi_matrix.png"

    # Create the visualization directory if it doesn't exist
    visualization_dir = output_directory / f"ses-{ses}/visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plot_path = visualization_dir / plot_filename
    plt.savefig(plot_path)
    plt.close()
