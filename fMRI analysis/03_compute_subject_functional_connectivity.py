import os
from datetime import datetime
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd
from compute_functional_connectivity import compute_functional_connectivity, visualize_fc_data
# from nilearn import datasets


def process_subject_functional(args):
    """Processes a single subject: loads pre-extracted timeseries, computes connectivity,
    saves the connectivity matrix, and optionally, visualizes the matrix.

    Args:
        args (tuple): Contains the following:
            subject_id (str): Subject ID.
            ses (str): Session or timepoint.
            output_dir (Path): Path to the directory where output data is saved.
            root_directory (Path): Root directory for the timeseries data.
            error_log_path (Path): Path to the error log file.
            schaefer_atlas:

    Raises:
        FileNotFoundError: If the timeseries file is not found.
        Exception: If an error occurs while loading the timeseries data
    """
    subject_id, ses, output_dir, root_directory, error_log_path = args

    timeseries_file = root_directory / f"{subject_id}_ses-{ses}_subcortical_schaefer200_timeseries.csv"

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
            f.write(f"{e!s}\n\n")
        return

    if timeseries is None or timeseries.size == 0:
        print(f"No valid timeseries loaded for subject {subject_id}")
        return
        
    # Compute functional connectivity with the combined atlas
    connectivity_matrix, fisher_z_matrix = compute_functional_connectivity(
        subject_id=subject_id,
        ses=ses,
        timeseries=timeseries,
        output_dir=output_dir,
        timeseries_path=timeseries_path
    )

    # Visualize data if you would like by uncommenting the line below
    # visualize_fc_data(subject_id, fisher_z_matrix, output_dir, ses, True)

    print(f"Processing completed for subject: {subject_id}")


def get_subjects_to_process(root_directory, output_directory, ses):
    """Generate a list of subjects to process based on whether they have
    timeseries data and connectivity data already generated.

    Args:
        root_directory (Path): Path to the root directory containing the timeseries data.
        output_directory (Path): Path to the output directory where connectivity data is saved.
        ses (str): Session / timepoint.
    """
    subjects_to_process = []

    csv_path = output_directory / f"ses-{ses}/all_to_all_roi_matrices/all_to_all_roi_matrix.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        processed_subjects = set(df["id"].unique())
    else:
        print("CSV file not found. No subjects have been processed.")
        processed_subjects = set()

    for filename in os.listdir(root_directory):
        if filename.startswith("sub-") and filename.endswith(f"_ses-{ses}_subcortical_schaefer200_timeseries.csv"):
            subject_id = filename.split("_")[0]
            if subject_id not in processed_subjects:
                subjects_to_process.append(subject_id)

    return subjects_to_process


def main(output_dir: Union[str, Path], root_directory: Union[str, Path], ses: str):
    """Main function to run the script.

    This function reads the pre-extracted timeseries data for each subject,
    computes the functional connectivity matrices for all subjects.

    Args:
        output_dir (Union[str, Path]): Path where processed data will be output.
        root_directory (Union[str, Path]): Root directory for the timeseries data.
        ses (str): Session / timepoint.

    Raises:
        FileNotFoundError: If the selected ROIs file is not found.
        KeyError: If the specified column name is not found in the selected ROIs file
    """
    output_dir = Path(output_dir)
    root_directory = Path(root_directory)
    error_log_path = output_dir / "error_log.txt"  # Define error log path

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        (
            subject_id,
            ses,
            output_dir,
            root_directory,
            error_log_path,
        )
        for subject_id in subjects
    ]

    for arg in args:
        process_subject_functional(arg)


if __name__ == "__main__":
    ses = "02"
    timeseries_path = Path("/home/rachel/Desktop/schaefer_analysis/timeseries_data/native_space")
    root_directory = Path(f"{timeseries_path}/ses-{ses}")
    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")

    # Create the output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    # Generate a list of subjects to process
    subjects = get_subjects_to_process(root_directory, output_directory, ses)
    print(f"Number of subjects to process: {len(subjects)}")

    main(
        output_dir=output_directory,
        root_directory=root_directory,
        ses=ses,
    )
