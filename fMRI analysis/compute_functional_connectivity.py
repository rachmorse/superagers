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
        path = root_dir / subdir / f"ses-{session}"
        path.mkdir(parents=True, exist_ok=True)


def compute_functional_connectivity(
    subject_id: str, timeseries: np.ndarray, output_dir: Path, schaefer_atlas, ses: str
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
    labels = schaefer_atlas["labels"]
    network_mappings = create_network_mappings(labels)

    # Compute full connectivity matrix
    print("Computing full connectivity matrix...")
    correlation_measure = ConnectivityMeasure(kind="correlation", standardize=False)
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

    # Set the diagonal to 0 to exclude self-connectivity
    np.fill_diagonal(connectivity_matrix, 0)

    # Apply Fisher z-transformation
    fisher_z_matrix = fisher_transform(connectivity_matrix)

    # Corrected path definition
    base_dir = output_dir / "connectivity_matrices"  # Ensure only once

    # Define target directories
    all_to_all_dir = base_dir / "all_to_all_roi_matrices"
    within_network_dir = base_dir / "within_network_matrices"

    # Ensure directories exist, eliminating redundancy
    prepare_directories(all_to_all_dir, ses, ["no_fisher_z", "fisher_z"])
    prepare_directories(within_network_dir, ses, ["no_fisher_z", "fisher_z"])

    # Save complete connectivity data
    save_connectivity_data(
        subject_id, "all_to_all_roi", connectivity_matrix, None, labels, all_to_all_dir / "no_fisher_z", ses
    )
    save_connectivity_data(
        subject_id, "fisher_z_all_to_all_roi", None, fisher_z_matrix, labels, all_to_all_dir / "fisher_z", ses
    )

    # Compute network-specific connectivity matrices
    for network, indices in network_mappings.items():
        network_timeseries = timeseries[:, indices]
        network_correlation_matrix = correlation_measure.fit_transform([network_timeseries])[0]
        np.fill_diagonal(network_correlation_matrix, 0)
        network_fisher_z_matrix = fisher_transform(network_correlation_matrix)

        network_labels = [labels[i] for i in indices]

        # Save network connectivity data
        save_connectivity_data(
            subject_id,
            f"{network}_within_network",
            network_correlation_matrix,
            None,
            network_labels,
            within_network_dir / "no_fisher_z",
            ses,
        )

        save_connectivity_data(
            subject_id,
            f"fisher_z_{network}_within_network",
            None,
            network_fisher_z_matrix,
            network_labels,
            within_network_dir / "fisher_z",
            ses,
        )

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
            label = label.decode("utf-8")
        parts = label.split("_")
        network_name = parts[2]  # Based on structure '7Networks_LH_Vis_1'

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
    ses: str,
):
    """Save the connectivity data to CSV files.

    Args:
        subject_id (str): Subject ID.
        label (str): Label for the connectivity data.
        matrix (np.ndarray): The connectivity matrix.
        fisher_z_matrix (np.ndarray): The Fisher z-transformed connectivity matrix.
        roi_names (List[str]): List of ROI names.
        output_dir (Path): Directory where the connectivity data will be saved.
        ses (str): Session / timepoint.
    """
    if matrix is not None:
        # Save connectivity data
        columns = [f"{roi1}-{roi2}" for roi1, roi2 in combinations(roi_names, 2)]
        df_all_fc = pd.DataFrame(index=[subject_id], columns=columns)
        upper_tri_indices = np.triu_indices(matrix.shape[0], k=1)
        df_all_fc.loc[subject_id, :] = matrix[upper_tri_indices]

        # Save connectivity DataFrame for all subjects to CSV
        csv_output_path = output_dir / f"ses-{ses}/{label}_matrix.csv"
        df_all_fc.to_csv(csv_output_path, mode="a", header=not csv_output_path.exists(), index_label="SubjectID")

    if fisher_z_matrix is not None:
        # Prepare connectivity Fisher z DataFrame
        columns = [f"{roi1}-{roi2}" for roi1, roi2 in combinations(roi_names, 2)]
        df_all_fc_fisher_z = pd.DataFrame(index=[subject_id], columns=columns)
        upper_tri_indices = np.triu_indices(fisher_z_matrix.shape[0], k=1)
        df_all_fc_fisher_z.loc[subject_id, :] = fisher_z_matrix[upper_tri_indices]

        # Save Fisher z DataFrame to CSV
        fisher_z_csv_output_path = output_dir / f"ses-{ses}/fisher_z_{label}_matrix.csv"
        df_all_fc_fisher_z.to_csv(
            fisher_z_csv_output_path, mode="a", header=not fisher_z_csv_output_path.exists(), index_label="SubjectID"
        )


def visualize_fc_data(
    subject_id: str,
    connectivity_matrix: np.ndarray,
    output_directory: Path,
    ses: str,
):
    """Visualize the connectivity matrix.

    Args:
        subject_id (str): Subject ID.
        connectivity_matrix (np.ndarray): The connectivity matrix to be visualized.
        output_directory (Path): Directory where the plot will be saved.
        ses (str): Session / timepoint
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

    # Save the plot
    # plot_path = output_directory / f"all_to_all_roi_matrices/ses-{ses}/fisher_z_all_to_all_roi_matrix.png"
    plt.savefig(plot_path)
    plt.close()
