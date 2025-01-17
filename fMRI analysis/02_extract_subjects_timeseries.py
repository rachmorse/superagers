import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from extract_timeseries import extract_timeseries
from nilearn import datasets


def process_subject_extract(args):
    """Processes a single subject: extracts timeseries and saves it.
    Optionally, visualizes the timeseries data.

    Args:
        subject_id (str): Identifier for the subject whose data is processed.
        ses (str): Session identifier for the specific data collection session.
        threshold (float): Threshold value used for data processing, e.g., for filtering.
        bold_template (str): File path template for the BOLD timeseries data.
        atlas_file (str): File path for the atlas used for extracting timeseries.
        output_dir (str): Directory where processed data and any outputs are saved.
        roi_indices (list of int): List of region of interest indices for timeseries extraction.
        error_log_path (str): File path where error logs should be written.

    Raises:
        Exception: If no valid timeseries is extracted for a subject.
    """
    (
        subject_id,
        ses,
        threshold,
        bold_template,
        atlas_file,
        output_dir,
        roi_indices,
        error_log_path,
    ) = args

    bold_path_template = bold_template.format(subject=subject_id, ses=ses, threshold=threshold)

    fmri_file = Path(bold_path_template)

    print(f"--- Processing subject: {subject_id} ---")

    # Process masks and extract timeseries
    timeseries = extract_timeseries(atlas_file, fmri_file, error_log_path)

    if timeseries is None or timeseries.size == 0:
        print(f"No valid timeseries extracted for subject {subject_id}")
        return

    # Ensure the directory exists
    output_subdir = os.path.join(output_dir, f"ses-{ses}")
    os.makedirs(output_subdir, exist_ok=True)

    # Save the extracted timeseries
    timeseries_output_path = output_dir / f"ses-{ses}/{subject_id}_ses-{ses}_schaefer200_timeseries.csv"
    print(f"Saving extracted timeseries to {timeseries_output_path}")
    np.savetxt(timeseries_output_path, timeseries, delimiter=",")

    # Run this line if you want to visualize the data
    # visualize_timeseries(subject_id, timeseries, roi_indices)

    print(f"Processing completed for subject: {subject_id}")


def main(
    ses: str,
    threshold: float,
    todo_path: Union[str, Path],
    error_log_path: Union[str, Path],
    output_dir: Union[str, Path],
    bold_template: str,
    roi_indices: List[int],
    multi: bool = False,
):
    """Main function to run the script.

    This function defines session timepoints, data directories, and processes subjects' timeseries
    data either sequentially or in parallel based on the multi flag.

    Args:
        ses (str): Session (timepoint).
        threshold (float): Threshold value for scrubbing.
        todo_path (Union[str, Path]): Path to the todo file with subject IDs to be processed.
        error_log_path (Union[str, Path]): Path to log the error file.
        output_dir (Union[str, Path]): Path where processed data will be output.
        bold_template (str): Path / template for the location of BOLD data.
        roi_indices (List[int]): ROI indices for timeseries visualization (e.g. add the index for the ROI/s you want to visualize).
        multi (bool): If True, enables parallel processing using multiprocessing. Defaults to False.
    """
    output_dir = Path(output_dir)
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

    # Get the Schaefer atlas
    schaefer_atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=200,  # Number of regions
        yeo_networks=7,  # Number of networks
        resolution_mm=2,
    )
    atlas_file = schaefer_atlas["maps"]

    args = [
        (
            subject,
            ses,
            threshold,
            bold_template,
            atlas_file,
            output_dir,
            roi_indices,
            error_log_path,
        )
        for subject in todo
    ]

    if multi:
        with Pool(6) as pool:
            pool.map(process_subject_extract, args)
    else:
        for arg in args:
            process_subject_extract(arg)


if __name__ == "__main__":
    ses = "02"
    threshold = "0.5"
    todo_file = Path("/home/rachel/Desktop/schaefer_analysis/todo.csv")
    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/timeseries_data")
    root_directory = Path("/home/rachel/Desktop/schaefer_analysis/scrubbed_data")
    error_log_path = output_directory / "error_log.txt"

    # Create the output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    roi_indices = [0]  # ROIs to visualize

    bold_template = os.path.join(
        root_directory,
        "{subject}",
        f"ses-{ses}",
        "MNI_2mm",
        "{subject}_ses-{ses}_run-01_rest_bold_ap_MNI-space_scrubbed_{threshold}.nii.gz",
    )

    main(
        ses=ses,
        threshold=threshold,
        todo_path=todo_file,
        error_log_path=error_log_path,
        output_dir=output_directory,
        bold_template=bold_template,
        roi_indices=roi_indices,
        multi=False,
    )

    # Uncomment this line to enable parallel processing
    # main(ses=ses, threshold=threshold, todo_path=todo_file, masks_root_path=masks_root_path, output_dir=output_directory, bold_template=bold_template, mask_template=mask_template, mask_type=mask_type, roi_indices=roi_indices, multi=True)
