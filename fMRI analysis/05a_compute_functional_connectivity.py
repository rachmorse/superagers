import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from typing import List, Optional, Dict
from itertools import combinations


def fisher_transform(correlations: np.ndarray) -> np.ndarray:
    """Apply Fisher z-transformation to the correlation coefficients.

    Args:
        correlations (np.ndarray): Correlation coefficients.

    Returns:
        np.ndarray: Transformed correlation coefficients.
    """
    return np.arctanh(correlations)


def compute_functional_connectivity(
    subject_id: str,
    timeseries: np.ndarray,
    output_dir: Path,
    schaefer_atlas,
    subjects: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute the connectivity matrix for all ROIs and each network, save them to CSV files.

    Args:
        subject_id (str): Subject ID.
        timeseries (np.ndarray): Timeseries data for the subject.
        output_dir (Path): Directory where the connectivity data will be saved.
        roi_names (List[str]): List of ROI names.
        network_mappings (Dict[str, List[int]]): Dict mapping network names to ROI indices.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The full connectivity and fisher z-transformed matrices.
    """
    # Extract ROI names and networks from the Schaeffer atlas
    labels = schaefer_atlas['labels']
    network_mappings = create_network_mappings(labels)

    # Compute full connectivity matrix
    print("Computing full connectivity matrix...")
    correlation_measure = ConnectivityMeasure(kind="correlation", standardize=False)
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

    # Set the diagonal to 0 to exclude self-connectivity
    np.fill_diagonal(connectivity_matrix, 0)

    # Apply Fisher z-transformation
    fisher_z_matrix = fisher_transform(connectivity_matrix)

    # Save full connectivity data
    save_connectivity_data(subject_id, "all", connectivity_matrix, fisher_z_matrix, labels, output_dir)

    # Compute network-specific connectivity matrices
    for network, indices in network_mappings.items():
        network_timeseries = timeseries[:, indices]
        
        network_correlation_matrix = correlation_measure.fit_transform([network_timeseries])[0]
        np.fill_diagonal(network_correlation_matrix, 0)
        network_fisher_z_matrix = fisher_transform(network_correlation_matrix)
        
        network_roi_names = [labels[i] for i in indices]
        
        save_connectivity_data(subject_id, network, network_correlation_matrix, network_fisher_z_matrix, network_roi_names, output_dir)

    return connectivity_matrix, fisher_z_matrix

def create_network_mappings(labels: List[str]) -> Dict[str, List[int]]:
    """Create a dictionary mapping network names to ROI indices.

    Args:
        labels (List[str]): List of ROI labels.
    """
    # Initialize network mappings
    network_mappings = {}
    # Extract network names from the labels
    for index, label in enumerate(labels):
        if isinstance(label, bytes):
            label = label.decode('utf-8')
        parts = label.split('_')
        network_name = parts[1]  
        
        if network_name not in network_mappings:
            network_mappings[network_name] = []

        network_mappings[network_name].append(index)

    return network_mappings

def save_connectivity_data(subject_id: str, label: str, matrix: np.ndarray, fisher_z_matrix: np.ndarray, roi_names: List[str], output_dir: Path):
    """Save the connectivity data to CSV files.

    Args:
        subject_id (str): Subject ID.
        label (str): Label for the connectivity data.
        matrix (np.ndarray): The connectivity matrix.
        fisher_z_matrix (np.ndarray): The Fisher z-transformed connectivity matrix.
        roi_names (List[str]): List of ROI names.
        output_dir (Path): Directory where the connectivity data will be saved.
    """
    # Save connectivity data
    columns = [f"{roi1}-{roi2}" for roi1, roi2 in combinations(roi_names, 2)]

    # Prepare connectivity DataFrame
    df_all_fc = pd.DataFrame(index=[subject_id], columns=columns)
    upper_tri_indices = np.triu_indices(matrix.shape[0], k=1)
    upper_tri_values = matrix[upper_tri_indices]
    df_all_fc.loc[subject_id, :] = upper_tri_values

    # Save connectivity DataFrame for all subjects to CSV
    csv_output_path = output_dir / f"{label}_fc_data.csv"
    df_all_fc.to_csv(csv_output_path, mode="a", header=not csv_output_path.exists(), index_label="SubjectID")

    # Prepare connectivity Fisher z DataFrame
    df_all_fc_fisher_z = pd.DataFrame(index=[subject_id], columns=columns)
    upper_tri_values_fisher_z = fisher_z_matrix[upper_tri_indices]
    df_all_fc_fisher_z.loc[subject_id, :] = upper_tri_values_fisher_z

    # Save connectivity Fisher z DataFrame for all subjects to CSV
    fisher_z_csv_output_path = output_dir / f"{label}_fc_data_fisher_z.csv"
    df_all_fc_fisher_z.to_csv(fisher_z_csv_output_path, mode="a", header=not fisher_z_csv_output_path.exists(), index_label="SubjectID")


def visualize_fc_data(
    subject_id: str,
    connectivity_matrix: np.ndarray,
):
    """Visualize the connectivity matrix.

    Args:
        subject_id (str): Subject ID.
        connectivity_matrix (np.ndarray): The connectivity matrix to be visualized.
    """
    # Visualize connectivity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Correlation Coefficient")
    plt.title(f"Connectivity Matrix for Subject {subject_id}")
    plt.xlabel("Regions")
    plt.ylabel("Regions")
    plt.grid(False)
    plt.show()
