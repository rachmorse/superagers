import os
from datetime import datetime
from pathlib import Path
from typing import Union
import nibabel as nib
import numpy as np
import pandas as pd
from compute_functional_connectivity import compute_functional_connectivity, visualize_fc_data
from run_fmri_pipeline import ANNOT_DIR, ATLAS, FC_ROOT, N_PARCELS, SESSIONS, SUBJECTS, TIMESERIES_ROOT

# Order matches the subcortical label values in 03_combine_t1_atlases.py (left then right, each 1 to 7)
SUBCORTICAL_STRUCTURES = ["Hippocampus", "Amygdala", "Pallidum", "Putamen", "Caudate", "Accumbens", "Thalamus"]


def write_combined_labels(labels_csv_path: Path):
    """Build the ROI labels in mask order (LH cortical, RH cortical, left subcortical, right subcortical)
    from the Schaefer annot files, check them, and save them to CSV.

    Args:
        labels_csv_path (Path): Path to save the labels CSV.

    Raises:
        ValueError: If the labels do not match the expected atlas layout.
    """
    labels = []
    for hemi in ["lh", "rh"]:
        _, _, names = nib.freesurfer.read_annot(str(ANNOT_DIR / f"{hemi}.Schaefer2018_{N_PARCELS}Parcels_7Networks_order.annot"))
        # Skip index 0 which is the medial wall
        labels += [name.decode("utf-8") for name in names[1:]]

    for side in ["Left", "Right"]:
        for structure in SUBCORTICAL_STRUCTURES:
            labels.append(f"Subcortical {len(labels) + 1}: {side} {structure}")

    # Check the labels line up with the mask values (cortical 1 to N_PARCELS, subcortical N_PARCELS + 1 to N_PARCELS + 14)
    half = N_PARCELS // 2
    if len(labels) != N_PARCELS + 14:
        raise ValueError(f"Expected {N_PARCELS + 14} labels for {ATLAS}, got {len(labels)}")
    if not all(label.startswith("7Networks_LH_") for label in labels[:half]):
        raise ValueError(f"First {half} labels are not all left hemisphere cortical labels")
    if not all(label.startswith("7Networks_RH_") for label in labels[half:N_PARCELS]):
        raise ValueError(f"Labels {half + 1} to {N_PARCELS} are not all right hemisphere cortical labels")
    if labels[N_PARCELS] != f"Subcortical {N_PARCELS + 1}: Left Hippocampus" or labels[-1] != f"Subcortical {N_PARCELS + 14}: Right Thalamus":
        raise ValueError("Subcortical labels are not numbered as expected")

    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    labels_csv_path.write_text("\n".join(labels) + "\n")
    print(f"Saved {len(labels)} labels to {labels_csv_path}")


def process_subject_functional(args):
    """Processes a single subject: loads pre-extracted timeseries, computes connectivity,
    saves the connectivity matrix, and optionally, visualizes the matrix.

    Args:
        args (tuple): Contains the following:
            subject_id (str): Subject ID.
            ses (str): Timepoint.
            output_dir (Path): Path to the directory where output data is saved.
            root_directory (Path): Root directory for the timeseries data at a specific timepoint.
            timeseries_path (Path): Path to the directory containing the timeseries data not at a specific timepoint.
            error_log_path (Path): Path to the error log file.
            combined_labels (List[str]): List of ROI labels.

    Raises:
        FileNotFoundError: If the timeseries file is not found.
        Exception: If an error occurs while loading the timeseries data
    """
    subject_id, ses, output_dir, root_directory, timeseries_path, error_log_path, combined_labels = args

    timeseries_file = root_directory / f"{subject_id}_{ses}_subcortical_{ATLAS}_timeseries.csv"

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

    # Check there is one timeseries column per label so connectivity values are not mislabelled
    if timeseries.shape[1] != len(combined_labels):
        print(f"Skipping {subject_id}: timeseries has {timeseries.shape[1]} ROIs but there are {len(combined_labels)} labels")
        with open(error_log_path, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"ROI count mismatch for subject {subject_id}: {timeseries.shape[1]} timeseries columns, {len(combined_labels)} labels\n\n")
        return

    # Compute functional connectivity with the combined atlas
    try:
        connectivity_matrix, fisher_z_matrix = compute_functional_connectivity(
            subject_id=subject_id,
            timeseries=timeseries,
            output_dir=output_dir,
            combined_labels=combined_labels,
            ses=ses
        )
    except ValueError as e:
        print(f"Error computing connectivity for {subject_id}: {e}")
        with open(error_log_path, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error computing connectivity for subject {subject_id}:\n")
            f.write(f"{e!s}\n\n")
        return
    except Exception as e:
        print(f"Unexpected error for {subject_id}: {e}")
        with open(error_log_path, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unexpected error for subject {subject_id}:\n")
            f.write(f"{e!s}\n\n")
        return

    # Visualize data by uncommenting the line below
    # visualize_fc_data(subject_id, fisher_z_matrix, output_dir, ses, True)

    print(f"Processing completed for subject: {subject_id}")


def get_subjects_to_process(root_directory, output_directory, ses):
    """Generate a list of subjects to process based on whether they have
    timeseries data and connectivity data already generated.

    Args:
        root_directory (Path): Path to the root directory containing the timeseries data.
        output_directory (Path): Path to the output directory where connectivity data is saved.
        ses (str): Session label (e.g. 'ses-01').
    """
    subjects_to_process = []

    if not root_directory.exists():
        print(f"Timeseries directory not found: {root_directory}")
        return subjects_to_process

    csv_path = output_directory / f"{ses}/all_to_all_roi_matrices/all_to_all_roi_matrix.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        processed_subjects = set(df["id"].unique())
    else:
        print("CSV file not found. No subjects have been processed.")
        processed_subjects = set()

    for filename in os.listdir(root_directory):
        if filename.startswith("sub-") and filename.endswith(f"_{ses}_subcortical_{ATLAS}_timeseries.csv"):
            subject_id = filename.split("_")[0]
            if subject_id in SUBJECTS and subject_id not in processed_subjects:
                subjects_to_process.append(subject_id)

    return subjects_to_process


def main(output_dir: Union[str, Path], root_directory: Union[str, Path], timeseries_path: Path, ses: str):
    """This function reads the pre-extracted timeseries data for each subject,
    computes the functional connectivity matrices for all subjects.

    Args:
        output_dir (Union[str, Path]): Path where processed data will be output.
        root_directory (Union[str, Path]): Root directory for the timeseries data at a specific timepoint.
        timeseries_path (Path): Path to the directory containing the timeseries data not at a specific timepoint.
        ses (str): Timepoint.

    Raises:
        FileNotFoundError: If the selected ROIs file is not found.
        KeyError: If the specified column name is not found in the selected ROIs file
    """
    output_dir = Path(output_dir)
    root_directory = Path(root_directory)
    error_log_path = output_dir / "error_log.txt"  # Define error log path

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read combined labels once, generating them from the annot files if they do not exist yet
    labels_csv_path = timeseries_path / "combined_labels.csv"
    if not labels_csv_path.exists():
        write_combined_labels(labels_csv_path)
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


if __name__ == "__main__":
    timeseries_path = TIMESERIES_ROOT
    output_directory = FC_ROOT

    for ses in SESSIONS:
        print("-----------------------")
        print(f"Processing {ses}")
        print("-----------------------")

        # Create the output directory if it does not exist
        output_directory.mkdir(parents=True, exist_ok=True)
        root_directory = timeseries_path / ses

        # Generate a list of subjects to process
        subjects = get_subjects_to_process(root_directory, output_directory, ses)
        print(f"Number of subjects to process: {len(subjects)}")

        main(
            output_dir=output_directory,
            root_directory=root_directory,
            ses=ses,
            timeseries_path=timeseries_path,
        )
