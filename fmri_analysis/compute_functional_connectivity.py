"""Functions to compute and save functional connectivity matrices from ROI timeseries."""

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

NETWORK_FIELD = 2  # Index of the network name in Schaefer labels split on "_": 7Networks_LH_Vis_1 -> "Vis"


def fisher_transform(correlations: np.ndarray) -> np.ndarray:
    """Apply Fisher z-transformation to the correlation coefficients.

    Args:
        correlations: Correlation coefficients.

    Returns:
        Transformed correlation coefficients.
    """
    return np.arctanh(correlations)


def prepare_directories(root_dir: Path, session: str, subdir_types: List[str]) -> None:
    """Ensure all necessary directories exist.

    Args:
        root_dir: Root directory for the timeseries data at a specific timepoint.
        session: Timepoint.
        subdir_types: List of subdirectory types to create.
    """
    for subdir in subdir_types:
        path = root_dir / session / subdir
        path.mkdir(parents=True, exist_ok=True)


def compute_functional_connectivity(
    subject_id: str, timeseries: np.ndarray, output_dir: Path, ses: str, combined_labels: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the connectivity matrix for all ROIs and each network, save them to CSV files.

    Args:
        subject_id: Subject ID.
        timeseries: Timeseries data for the subject.
        output_dir: Directory where the connectivity data will be saved.
        ses: Timepoint.
        combined_labels: List of ROI labels, in the same order as the timeseries columns.

    Returns:
        The full connectivity and fisher z-transformed matrices.
    """
    # Extract ROI names and networks using the combined labels
    network_mappings = create_network_mappings(combined_labels)

    # Create special mappings for subcortical regions (combining left and right)
    subcortical_structures = {}
    all_subcortical_indices = []  # Track all subcortical indices

    # First, identify all subcortical structures and their indices
    for index, raw_label in enumerate(combined_labels):
        label = raw_label.decode("utf-8") if isinstance(raw_label, bytes) else raw_label

        if "Subcortical" in label:
            # Add to all subcortical indices list
            all_subcortical_indices.append(index)

            # Extract the structure name without "Left" or "Right" prefix
            if ":" in label:
                # Extract structure name and remove "Left" or "Right" prefix
                structure_name = label.split(":")[1].strip()
                if "Left" in structure_name:
                    structure_name = structure_name.replace("Left", "").strip()
                elif "Right" in structure_name:
                    structure_name = structure_name.replace("Right", "").strip()

                # Add to the dictionary
                if structure_name not in subcortical_structures:
                    subcortical_structures[structure_name] = []
                subcortical_structures[structure_name].append(index)

    # Define target directories
    all_to_all_dir = output_dir / ses / "all_to_all_roi_matrices"
    within_network_dir = output_dir / ses / "within_network_matrices"
    subcortical_dir = output_dir / ses / "subcortical_matrices"

    # Ensure directories exist
    prepare_directories(output_dir, ses, ["all_to_all_roi_matrices", "within_network_matrices", "subcortical_matrices"])

    # Process cortical networks (non-subcortical)
    for network, indices in network_mappings.items():
        # Skip subcortical networks to handle them separately
        if network == "Subcortical":
            continue

        compute_and_save_connectivity(
            subject_id,
            f"{network}_within_network",
            timeseries[:, indices],
            [combined_labels[i] for i in indices],
            within_network_dir,
        )

    # Create an all subcortical 'network' connectivity matrix
    if all_subcortical_indices:
        compute_and_save_connectivity(
            subject_id,
            "all_subcortical_rois",
            timeseries[:, all_subcortical_indices],
            [combined_labels[i] for i in all_subcortical_indices],
            subcortical_dir,
        )

    # Process individual subcortical structures (left and right combined)
    for structure_name, indices in subcortical_structures.items():
        if len(indices) > 0:
            compute_and_save_connectivity(
                subject_id,
                f"{structure_name.strip()}_bilateral",
                timeseries[:, indices],
                [combined_labels[i] for i in indices],
                subcortical_dir,
            )

    # Compute full connectivity matrix
    # Saved last since 02_compute_subject_functional_connectivity.py uses this CSV to mark subjects as done
    print("Computing full connectivity matrix...")
    return compute_and_save_connectivity(subject_id, "all_to_all_roi", timeseries, combined_labels, all_to_all_dir)


def compute_and_save_connectivity(
    subject_id: str, label: str, timeseries: np.ndarray, roi_names: List[str], output_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the connectivity and Fisher z matrices for a set of ROIs and save both to CSV files.

    Args:
        subject_id: Subject ID.
        label: Label for the connectivity data.
        timeseries: Timeseries data for these ROIs.
        roi_names: List of ROI names, in the same order as the timeseries columns.
        output_dir: Directory where the connectivity data will be saved.

    Returns:
        The connectivity and fisher z-transformed matrices.
    """
    correlation_measure = ConnectivityMeasure(kind="correlation", standardize=False)
    connectivity_matrix = correlation_measure.fit_transform([timeseries])[0]

    # Set the diagonal to 0 to exclude self-connectivity
    np.fill_diagonal(connectivity_matrix, 0)

    # Apply Fisher z-transformation
    fisher_z_matrix = fisher_transform(connectivity_matrix)

    # Fisher z saved first so the all_to_all_roi CSV that marks a subject as done is written last
    save_connectivity_data(subject_id, f"fisher_z_{label}", fisher_z_matrix, roi_names, output_dir)
    save_connectivity_data(subject_id, label, connectivity_matrix, roi_names, output_dir)

    return connectivity_matrix, fisher_z_matrix


def create_network_mappings(labels: List[str]) -> Dict[str, List[int]]:
    """Create a dictionary mapping network names to ROI indices.

    Args:
        labels: List of ROI labels.

    Returns:
        Dictionary mapping network names to indices.
    """
    # Initialize network mappings
    network_mappings = {}

    # Extract network names from the labels
    for index, raw_label in enumerate(labels):
        label = raw_label.decode("utf-8") if isinstance(raw_label, bytes) else raw_label

        # For Schaefer atlas regions (format: 7Networks_LH_Vis_1)
        if label.startswith("7Networks_"):
            parts = label.split("_")
            # Extract network name (e.g., "Vis", "SomMot", etc.)
            network_name = parts[NETWORK_FIELD] if len(parts) > NETWORK_FIELD else "Other"

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


def save_csv(dataframe: pd.DataFrame, file_path: Path) -> None:
    """Save a DataFrame to a CSV file.

    Appends without header if the file exists (e.g. subs have already been processed).
    Subjects already in the file are skipped.

    Args:
        dataframe: The DataFrame to save.
        file_path: The path to the CSV file.
    """
    if file_path.exists():
        existing_df = pd.read_csv(file_path, index_col="id")
        dataframe = dataframe[~dataframe.index.isin(existing_df.index)]
        # Append without writing header
        dataframe.to_csv(file_path, mode="a", header=False)
    else:
        # Initial write with header
        dataframe.to_csv(file_path, index_label="id", header=True)


def save_connectivity_data(
    subject_id: str, label: str, matrix: np.ndarray, roi_names: List[str], output_dir: Path
) -> None:
    """Save the connectivity data to a CSV file.

    Args:
        subject_id: Subject ID.
        label: Label for the connectivity data.
        matrix: The connectivity matrix.
        roi_names: List of ROI names.
        output_dir: Directory where the connectivity data will be saved.
    """
    csv_output_path = output_dir / f"{label}_matrix.csv"

    columns = [f"{roi1}-{roi2}" for roi1, roi2 in combinations(roi_names, 2)]
    df_all_fc = pd.DataFrame(index=[subject_id], columns=columns)
    upper_tri_indices = np.triu_indices(matrix.shape[0], k=1)
    df_all_fc.loc[subject_id, :] = matrix[upper_tri_indices]
    save_csv(df_all_fc, csv_output_path)


def visualize_fc_data(
    subject_id: str,
    connectivity_matrix: np.ndarray,
    output_directory: Path,
    ses: str,
    is_fisher_z: bool = False,
) -> None:
    """Visualize the connectivity matrix.

    Args:
        subject_id: Subject ID.
        connectivity_matrix: The connectivity matrix to be visualized.
        output_directory: Directory where the plot will be saved.
        ses: Session / timepoint.
        is_fisher_z: Whether the matrix is Fisher Z-transformed.
    """
    # Visualize connectivity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(connectivity_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(label="Correlation Coefficient")
    plt.title(f"Connectivity Matrix for Subject {subject_id}")
    plt.xlabel("Regions")
    plt.ylabel("Regions")
    plt.grid(False)

    # Determine plot path based on Fisher Z transformation
    if is_fisher_z:
        plot_filename = f"{subject_id}_{ses}_fisher_z_all_to_all_roi_matrix.png"
    else:
        plot_filename = f"{subject_id}_{ses}_all_to_all_roi_matrix.png"

    # Create the visualization directory if it doesn't exist
    visualization_dir = output_directory / ses / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # Save the plot
    plot_path = visualization_dir / plot_filename
    plt.savefig(plot_path)

    # Show after saving so the saved figure is not blank
    plt.show()
    plt.close()
