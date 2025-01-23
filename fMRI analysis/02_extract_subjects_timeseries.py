import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import numpy as np
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


def get_subjects_to_process(root_directory, output_directory, ses):
    """Generate a list of subjects to process based on whether they have
    scrubbed data and a timeseries file already generated.

    Args:
        root_directory (Path): Path to the root directory containing the scrubbed data.
        output_directory (Path): Path to the output directory where timeseries data is saved.
        ses (str): Session / timepoint.
    """
    subjects_to_process = []

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(root_directory):
        if not subject_dir.startswith("sub-"):
            continue
        subject = subject_dir

        # Check if the session exists
        scrubbed_data = Path(f"{root_directory}/{subject}/ses-{ses}")
        output_data = Path(f"{output_directory}/ses-{ses}")

        if scrubbed_data.exists() and any(scrubbed_data.iterdir()):
            expected_output_filename = f"{subject}_ses-{ses}_schaefer200_timeseries.csv"
            output_file_path = output_data / expected_output_filename

            if not output_file_path.exists():
                subjects_to_process.append(subject)

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process


def main(
    ses: str,
    threshold: float,
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
        error_log_path (Union[str, Path]): Path to log the error file.
        output_dir (Union[str, Path]): Path where processed data will be output.
        bold_template (str): Path / template for the location of BOLD data.
        roi_indices (List[int]): ROI indices for timeseries visualization (e.g. add the index for the ROI/s you want to visualize).
        multi (bool): If True, enables parallel processing using multiprocessing. Defaults to False.
    """
    output_dir = Path(output_dir)
    error_log_path = output_dir / "error_log.txt"  # Define error log path

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

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
        for subject in subjects
    ]

    if multi:
        with Pool(4) as pool:
            pool.map(process_subject_extract, args)
    else:
        for arg in args:
            process_subject_extract(arg)


if __name__ == "__main__":
    ses = "02"
    threshold = "0.5"
    output_directory = Path("/home/rachel/Desktop/schaefer_analysis/timeseries_data")
    root_directory = Path("/home/rachel/Desktop/schaefer_analysis/scrubbed_data")
    error_log_path = output_directory / "error_log.txt"

    # Create the output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    roi_indices = [0]  # ROIs to visualize

    # Generate a list of subjects to process
    subjects = get_subjects_to_process(root_directory, output_directory, ses)

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
        error_log_path=error_log_path,
        output_dir=output_directory,
        bold_template=bold_template,
        roi_indices=roi_indices,
        # multi=False,
        multi=True, # Uncomment this line to enable parallel processing
    )
