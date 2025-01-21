from datetime import datetime
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd
from compute_functional_connectivity import compute_functional_connectivity, visualize_fc_data
from nilearn import datasets


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
    subject_id, ses, output_dir, root_directory, error_log_path, schaefer_atlas = args

    timeseries_file = root_directory / f"{subject_id}_ses-{ses}_schaefer200_timeseries.csv"

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

    # Compute functional connectivity
    connectivity_matrix, fisher_z_matrix = compute_functional_connectivity(
        subject_id=subject_id,
        ses=ses,
        timeseries=timeseries,
        output_dir=output_dir,
        schaefer_atlas=schaefer_atlas,
    )

    # Visualize data if you would like by uncommenting the line below
    visualize_fc_data(subject_id, fisher_z_matrix, output_dir, ses, True)

    print(f"Processing completed for subject: {subject_id}")


def main(todo_path: Union[str, Path], output_dir: Union[str, Path], root_directory: Union[str, Path], ses: str):
    """Main function to run the script.

    This function reads the pre-extracted timeseries data for each subject,
    computes the functional connectivity matrices for all subjects.

    Args:
        todo_path (Union[str, Path]): Path to the todo file with subject IDs to be processed.
        output_dir (Union[str, Path]): Path where processed data will be output.
        root_directory (Union[str, Path]): Root directory for the timeseries data.

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

    # Save the Schaefer atlas as a Nifti file
    schaefer_nifti_path = Path("/home/rachel/Desktop/schaefer_analysis/schaefer200_atlas.nii")
    nifti_image = nib.load(schaefer_atlas.maps)
    nib.save(nifti_image, str(schaefer_nifti_path))

    args = [
        (
            subject_id,
            ses,
            output_dir,
            root_directory,
            error_log_path,
            schaefer_atlas,
        )
        for subject_id in todo
    ]

    for arg in args:
        process_subject_functional(arg)


if __name__ == "__main__":
    ses = "02"
    todo_file = Path("/home/rachel/Desktop/schaefer_analysis/todo.csv")
    root_directory = Path(f"/home/rachel/Desktop/schaefer_analysis/timeseries_data/ses-{ses}")
    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/connectivity_matrices")

    # Create the output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    main(
        todo_path=todo_file,
        output_dir=output_directory,
        root_directory=root_directory,
        ses=ses,
    )
