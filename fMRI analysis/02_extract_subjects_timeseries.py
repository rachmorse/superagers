import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import numpy as np
from extract_timeseries import extract_timeseries
from nilearn import datasets
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_harvard_oxford
from nilearn.image import load_img, new_img_like, resample_to_img
import nibabel as nib

####################################################################################
# NEED TO MAKE SURE THIS RUNS< MIGHT BE A BIT MESSY ATM 
####################################################################################

def process_subject_extract(args):
    """Processes a single subject: extracts timeseries and saves it.
    Optionally, visualizes the timeseries data.

    Args:
        subject_id (str): Identifier for the subject whose data is processed.
        ses (str): Session identifier for the specific data collection session.
        threshold (float): Threshold value used for data processing, e.g., for filtering.
        bold_template (str): File path template for the BOLD timeseries data.
        atlas_file (str): File path for the atlases used for extracting timeseries.
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
    timeseries_output_path = output_dir / f"ses-{ses}/{subject_id}_ses-{ses}_subcortical_schaefer200_timeseries.csv"
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
        scrubbed_data = Path(f"{root_directory}/{subject}/ses-{ses}") or Path(f"{root_directory}/{subject}/ses-{ses}/native_T1")
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
    combined_labels_csv: str,
    atlas_file_template: str,
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
        combined_labels_csv (str): Path to the CSV file containing the combined labels.
        atlas_file_template (str): Template string for the atlas file path.
    """
    output_dir = Path(output_dir)
    error_log_path = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the ROI labels from the CSV
    with open(combined_labels_csv, "r", encoding="utf-8") as f:
        combined_labels = [line.strip() for line in f.readlines()]

    # Pass atlas template and labels to the processing function
    args = [
        (
            subject,
            ses,
            threshold,
            bold_template,
            atlas_file_template, 
            output_dir,
            roi_indices,
            error_log_path,
            combined_labels,  # Add labels to the arguments
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
    root = Path("/home/rachel/Desktop/schaefer_analysis/") 
    output_directory = Path(f"{root}/timeseries_data/fsaverage")
    combined_labels_csv = f"{root}/timeseries_data/fsaverage/combined_labels.csv" # Path to labels for each of the ROIs
    root_directories = [
        Path("/home/rachel/Desktop/schaefer_analysis/scrubbed_data"),
        Path("/pool/guttmann/institut/UB/Superagers/MRI/resting-preprocessed")
    ]
    error_log_path = output_directory / "error_log.txt"

    # Create the output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    roi_indices = [0]  # ROIs to visualize

    # Generate a list of subjects to process using multiple root directories
    def get_subjects_from_multiple_roots(root_dirs, output_dir, ses):
        all_subjects = []
        for root_dir in root_dirs:
            subjects = get_subjects_to_process(root_dir, output_dir, ses)
            all_subjects.extend(subjects)
        # Remove duplicates while preserving order
        unique_subjects = []
        for subject in all_subjects:
            if subject not in unique_subjects:
                unique_subjects.append(subject)
        return unique_subjects
    
    subjects = get_subjects_from_multiple_roots(root_directories, output_directory, ses)

    atlas_file_template = f"{root}/fsaverage/ses-{ses}/{{subject}}/bold_space_masks/{{subject}}_ses-{ses}_schaefer200_subcortical14_bold_space.nii.gz"

    # Since data can be in either directory, check both
    def find_bold_file_path(subject, ses, threshold):
        # Try the first root directory
        path1 = os.path.join(
            root_directories[0],
            subject,
            f"ses-{ses}",
            "native_T1",
            f"{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed_{threshold}.nii.gz",
        )
        
        # Try the second root directory
        path2 = os.path.join(
            root_directories[1],
            subject,
            f"ses-{ses}",
            "native_T1",
            f"{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed_{threshold}.nii.gz",
        )
        
        # Return the path that exists
        if os.path.exists(path1):
            return path1
        elif os.path.exists(path2):
            return path2
        
    bold_template = os.path.join(
        "{root_dir}",
        "{subject}",
        f"ses-{ses}",
        "native_T1",
        "{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed_{threshold}.nii.gz",
    )

    # Modify process_subject_extract to handle checking both root directories
    def process_subject_extract_wrapper(args):
        subject, ses, threshold, _, atlas_file, output_dir, roi_indices, error_log_path, combined_labels = args
        
        # Find the correct bold file path
        for root_dir in root_directories:
            bold_path = bold_template.format(
                root_dir=root_dir,
                subject=subject,
                threshold=threshold
            )
            
            if os.path.exists(bold_path):
                # Found the file, use this path
                args_with_correct_path = (
                    subject, ses, threshold, bold_path, atlas_file,
                    output_dir, roi_indices, error_log_path, combined_labels
                )
                return process_subject_extract(args_with_correct_path)
        
        # If file not found in any root directory, log an error
        with open(error_log_path, "a") as f:
            f.write(f"Could not find BOLD file for subject {subject}\n")
        return None

    main(
        ses=ses,
        threshold=threshold,
        error_log_path=error_log_path,
        output_dir=output_directory,
        bold_template=bold_template,
        roi_indices=roi_indices,
        combined_labels_csv=combined_labels_csv,
        atlas_file_template=atlas_file_template,
        subjects=subjects,
        # multi=False,
        multi=True, # Uncomment this line to enable parallel processing
    )
