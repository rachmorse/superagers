import os
from datetime import datetime
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import pandas as pd
from compute_functional_connectivity import compute_functional_connectivity, visualize_fc_data


def process_subject_functional(args):
    """Processes a single subject: loads pre-extracted timeseries, computes connectivity,
    saves the connectivity matrix, and optionally, visualizes the matrix.

    Args:
        args (tuple): Contains the following:
            subject_id (str): Subject ID.
            ses (str): Session or timepoint.
            output_dir (Path): Path to the directory where output data is saved.
            root_directory (Path): Root directory for the timeseries data.
            timeseries_path (Path): Path to the directory containing the timeseries data.
            error_log_path (Path): Path to the error log file.
            combined_labels (List[str]): List of ROI labels.

    Raises:
        FileNotFoundError: If the timeseries file is not found.
        Exception: If an error occurs while loading the timeseries data
    """
    subject_id, ses, output_dir, root_directory, timeseries_path, error_log_path, combined_labels = args

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
        ses=ses,
        combined_labels=combined_labels
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


def main(output_dir: Union[str, Path], root_directory: Union[str, Path], timeseries_path: Path, ses: str):
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

    # Read combined labels once
    labels_csv_path = timeseries_path / "combined_labels.csv"
    try:
        combined_labels = pd.read_csv(labels_csv_path, header=None).squeeze().tolist()
    except FileNotFoundError:
        print(f"Error: Labels file not found at {labels_csv_path}")
        return

    args = [
        (
            subject_id,
            ses,
            output_dir,
            root_directory,
            timeseries_path,
            error_log_path,
            combined_labels,
        )
        for subject_id in subjects
    ]

    for arg in args:
        process_subject_functional(arg)

    # Concatenate and Cleanup
    print("Concatenating individual subject matrix files...")
    
    # Define directories to check
    dirs_to_check = [
        output_dir / f"ses-{ses}/all_to_all_roi_matrices",
        output_dir / f"ses-{ses}/within_network_matrices",
        output_dir / f"ses-{ses}/subcortical_matrices"
    ]

    for directory in dirs_to_check:
        if not directory.exists():
            continue
            
        # Group files by suffix using a single pass
        files_by_suffix = {}
        all_files = list(directory.glob("sub-*_matrix.csv"))
        
        for f in all_files:
            # Extract suffix (everything after the first underscore)
            parts = f.name.split("_", 1)
            if len(parts) > 1:
                suffix = parts[1]
                if suffix not in files_by_suffix:
                    files_by_suffix[suffix] = []
                files_by_suffix[suffix].append(f)
        
        # Process each group
        for suffix, matrix_files in files_by_suffix.items():
            if not matrix_files:
                continue
                
            dfs = []
            for mf in matrix_files:
                try:
                    dfs.append(pd.read_csv(mf))
                except Exception as e:
                    print(f"Error reading {mf}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Sort by ID if present
                if 'id' in combined_df.columns:
                    combined_df = combined_df.sort_values('id')
                
                final_output = directory / suffix
                
                # Safe write and delete
                try:
                    combined_df.to_csv(final_output, index=False)
                    print(f"Created {final_output}")
                    
                    # Verify file exists and has content before deleting originals
                    if final_output.exists() and final_output.stat().st_size > 0:
                        for mf in matrix_files:
                            mf.unlink()
                        print(f"Deleted {len(matrix_files)} individual files for {suffix}")
                    else:
                        print(f"Error: Output file {final_output} was not created correctly. Keeping individual files.")
                        
                except Exception as e:
                    print(f"Error writing combined file {final_output}: {e}. Keeping individual files.")


if __name__ == "__main__":
    sessions = ["01", "02"]
    timeseries_path = Path("/home/rachel/Desktop/schaefer_analysis/timeseries_data/native_space")
    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")

    for ses in sessions:
        print("-----------------------")
        print(f"Processing ses-{ses}")
        print("-----------------------")

        # Create the output directory if it does not exist
        output_directory.mkdir(parents=True, exist_ok=True)
        root_directory = Path(f"{timeseries_path}/ses-{ses}")

        # Generate a list of subjects to process
        subjects = get_subjects_to_process(root_directory, output_directory, ses)
        print(f"Number of subjects to process: {len(subjects)}")

        main(
            output_dir=output_directory,
            root_directory=root_directory,
            ses=ses,
            timeseries_path=timeseries_path,
        )
