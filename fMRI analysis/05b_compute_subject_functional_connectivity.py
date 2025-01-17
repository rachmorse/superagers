from pathlib import Path
import pandas as pd
from typing import Optional, Union
import numpy as np
from datetime import datetime
from nilearn import datasets
from compute_functional_connectivity import (
    compute_functional_connectivity,
    compute_one_to_all_connectivity,
    visualize_fc_data,
)


def process_subject_functional(args):
    """
    Processes a single subject: loads pre-extracted timeseries, computes connectivity,
    saves the connectivity matrix, and optionally, visualizes the matrix.

    Args:
        args (tuple): Contains the following:
            subject_id (str): Subject ID.
            output_dir (Path): Path to the directory where output data is saved.
            root_directory (Path): Root directory for the timeseries data.
            subjects (list): List of all subjects to be processed.
            error_log_path (Path): Path to the error log file.
            roi_names (list): List of all ROI names.
            network_mappings (dict): Dict mapping network names to ROI indices.

    Raises:
        FileNotFoundError: If the timeseries file is not found.
        Exception: If an error occurs while loading the timeseries data
    """
    (
        subject_id,
        output_dir,
        root_directory,
        error_log_path,
        subjects,
        roi_names,
        network_mappings,
    ) = args

    timeseries_file = root_directory / f"{subject_id}_timeseries.csv"

    # Load extracted timeseries
    print(f"--- Processing subject: {subject_id} ---")
    print("Reading extracted timeseries...")
    try:
        timeseries = np.loadtxt(timeseries_file, delimiter=",")
        print("Timeseries loaded")
    except FileNotFoundError:
        print(f"Timeseries file not found: {timeseries_file}")
        return
    except Exception as e:
        print(f"Error loading timeseries: {e}")
        with open(error_log_path, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error loading timeseries for subject {subject_id}:\n")
            f.write(f"{str(e)}\n\n")
        return

    if timeseries is None or timeseries.size == 0:
        print(f"No valid timeseries loaded for subject {subject_id}")
        return

    # Compute functional connectivity
    connectivity_matrix, fisher_z_matrix = compute_functional_connectivity(
        subject_id=subject_id,
        timeseries=timeseries,
        output_dir=output_dir,
        roi_names=roi_names,
        network_mappings=network_mappings,
        subjects=subjects,
    )

    # Visualize data if you would like by uncommenting the line below
    # visualize_fc_data(subject_id, connectivity_matrix)

    print(f"Processing completed for subject: {subject_id}")


def main(
    todo_path: Union[str, Path],
    output_dir: Union[str, Path],
    root_directory: Union[str, Path],
):
    """
    Main function to run the script.

    This function reads the pre-extracted timeseries data for each subject,
    computes the functional connectivity matrices for all subjects.

    Args:
        todo_path (Union[str, Path]): Path to the todo file with subject IDs to be processed.
        output_dir (Union[str, Path]): Path where processed data will be output.
        root_directory (Union[str, Path]): Root directory for the timeseries data.
        selected_rois_csv (Path): Path to the selected ROIs CSV.
        roi_column_name (str): Name of the column containing ROI names.
        one_timeseries_index (Optional[Union[int, str]]): Index or name of the ROI to focus on for one-to-all connectivity (optional).

    Raises:
        FileNotFoundError: If the selected ROIs file is not found.
        KeyError: If the specified column name is not found in the selected ROIs file
    """

    output_dir = Path(output_dir)
    root_directory = Path(root_directory)
    todo_path = Path(todo_path)
    error_log_path = output_dir / "error_log.txt"  # Define error log path

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        todo_df = pd.read_csv(todo_path)
        todo = todo_df["todo"].tolist()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print(f"Number of subjects to process: {len(todo)}")

    # Load the Schaefer atlas
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)

    subjects = todo  # Assign subjects list

    args = [
        (
            subject,
            output_dir,
            root_directory,
            subjects,
            error_log_path,
        )
        for subject in todo
    ]

    for arg in args:
        process_subject_functional(arg)


if __name__ == "__main__":
    todo_file = Path("/home/rachel/Desktop/fMRI Analysis/todo.csv")
    root_directory = Path("/home/rachel/Desktop/fMRI Analysis/Schaeffer_200/timeseries")
    output_directory = Path(
        "/home/rachel/Desktop/fMRI Analysis/Schaeffer_200/connectivity_matrices"
    )

    main(
        todo_path=todo_file,
        output_dir=output_directory,
        root_directory=root_directory,
    )
