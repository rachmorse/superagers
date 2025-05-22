import os
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
from extract_timeseries import extract_timeseries, visualize_timeseries

def process_subject_extract(args):
    """Processes a single subject: extracts timeseries and saves it.
    Optionally, visualizes the timeseries data.

    Args:
        subject (str): Identifier for the subject whose data is processed.
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
        subject,
        ses,
        threshold,
        atlas_file,
        output_dir,
        roi_indices,
        error_log_path,
    ) = args

    if cohort == "bbhi":
        bold_template = Path(f"{root_directory}/{subject}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed-interp-05.nii.gz")
        # This is because subjects who did not need any scrubbing do not have a seperate scrubbed file
        if not bold_template.exists():
            bold_template = Path(f"{root_directory}/{subject}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
    else:
        if ses == "01":
            bold_template = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed-interp-05.nii.gz")
            if not bold_template.exists():
                bold_template = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
        else:
            bold_template = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed_0.5.nii.gz")
            if not bold_template.exists():
                bold_template = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")

    print(f"--- Processing subject: {subject} ---")

    bold_path_template = str(bold_template).format(subject=subject, ses=ses)

    if isinstance(atlas_file, Path):
        atlas_path = str(atlas_file).format(subject=subject, ses=ses)
        atlas_file = Path(atlas_path)
    else:
        atlas_file = Path(atlas_file.format(subject=subject, ses=ses))
        
    fmri_file = Path(bold_path_template)

    # Adding a warning filter to catch issues that likely indicate the wrong atlas or BOLD file is being used
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        timeseries = extract_timeseries(atlas_file, fmri_file, error_log_path)

        for captured_warning in w:
            if "After resampling the label image to the data image, the following labels were removed" in str(captured_warning.message):
                print(f"Error in processing {subject}: some ROIs were unable to be identified.")
                return

    # Check if there is no valid timeseries
    if timeseries is None or timeseries.size == 0:
        print(f"No valid timeseries extracted for subject {subject}")
        return

    # Ensure the directory exists
    output_subdir = os.path.join(output_dir, f"ses-{ses}")
    os.makedirs(output_subdir, exist_ok=True)

    # Save the extracted timeseries
    timeseries_output_path = output_dir / f"ses-{ses}/{subject}_ses-{ses}_subcortical_schaefer200_timeseries.csv"
    print(f"Saving extracted timeseries to {timeseries_output_path}")
    np.savetxt(timeseries_output_path, timeseries, delimiter=",")

    # Run this to visualize the data
    # visualize_timeseries(subject_id, timeseries, roi_indices)

    print(f"Processing completed for subject: {subject}")

def exclude_subjects_framewise_displ(subject, ses, root_directory):
    """Check if the subject has excessive motion (>30% of frames exceeding 0.5 mm).

    Args:
        subject (str): Subject ID.
        ses (str): Session identifier.
        root_directory (Path): Path to the root directory containing the data.

    Returns:
        bool: True if the subject should be excluded due to excessive motion, False otherwise.
    """
    fd_file = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/framewise_displ.txt")
    
    if fd_file.exists():
        try:
            fd_values = pd.read_csv(fd_file, header=None).iloc[:, 0]
            fd_values = pd.to_numeric(fd_values, errors='coerce').fillna(0)
            high_motion_percentage = (fd_values > 0.5).mean() * 100
            
            if high_motion_percentage > 30:
                print(f"Excluding {subject} due to excessive motion: {high_motion_percentage:.2f}% of frames > 0.5mm")
                return True
        except Exception as e:
            print(f"Error reading FD file for {subject}: {str(e)}")
    
    return False


def get_subjects_to_process(root_directory, atlas_file_template, output_directory, ses, cohort):
    """Generate a list of subjects to process based on whether they have
    scrubbed data and a timeseries file already generated. Then exclude subjects    
    with excessive motion (>30% of frames exceeding 0.5 mm).

    Args:
        root_directory (Path): Path to the root directory containing the scrubbed data.
        atlas_file_template (Path): Path to the Schaefer atlas mask.
        output_directory (Path): Path to the output directory where timeseries data is saved.
        ses (str): Session / timepoint.
        cohort (str): Cohort name (e.g., 'bbhi', 'superagers').
    """
    subjects_to_process = []
    subjects_excluded_motion = []
    subjects_excluded_no_bold = []
    subjects_without_atlas_file = []
    subjects = []

    # Iterate over all possible subject directories
    if cohort == "bbhi":
        subjects_df = pd.read_csv(subject_csv)

        if "id" in subjects_df.columns:
            subjects = [f"sub-{subject}" for subject in subjects_df["id"].tolist()]
        else:
            print("Error: 'id' column not found in CSV file.")
            subjects = []
    else:
        # Read from directory
        for subject_dir in os.listdir(root_directory):
            # This checks whether they have data for the correct timepoint 
            if subject_dir.startswith("sub-"):
                if ses == "01":
                    session_path = Path(f"{root_directory}/{subject_dir}/ses-01")
                    session_exists = session_path.exists() and session_path.is_dir()
                elif ses == "02":
                    session_path = Path(f"{root_directory}/{subject_dir}/ses-02") 
                    session_exists = session_path.exists() and session_path.is_dir()
                else:
                    session_exists = False
                    
                if not session_exists:
                    continue
                subjects.append(subject_dir)

    # Now process each subject
    for subject in subjects:
        scrubbed_data = None
        
        if cohort == "bbhi":
            # Check for either scrubbed or original data
            scrubbed_data = Path(f"{root_directory}/{subject}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed-interp-05.nii.gz")
            unscrubbed_file = Path(f"{root_directory}/{subject}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
            fd_file = Path(f"{root_directory}/{subject}/native_T1/framewise_displ.txt")

            # Check if either scrubbed or original data exists
            bold_file_exists = scrubbed_data.exists() or unscrubbed_file.exists()

            # Track subjects with no bold file
            if not bold_file_exists:
                subjects_excluded_no_bold.append(subject)
                continue  # Skip this subject

            # If FWD file exists, check motion criteria
            if fd_file.exists() and bold_file_exists:
                if exclude_subjects_framewise_displ(subject, ses, root_directory):
                    subjects_excluded_motion.append(subject)
                    continue 
        else:
            unscrubbed_file = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
            fd_file = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/framewise_displ.txt")
            
            # For BBHI senior cohorts
            if ses == "01":
                scrubbed_data = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed-interp-05.nii.gz") 

                # Check if either scrubbed or original data exists
                bold_file_exists = scrubbed_data.exists() or unscrubbed_file.exists()

                # Track subjects with no bold file
                if not bold_file_exists:
                    subjects_excluded_no_bold.append(subject)
                    continue 

                if fd_file.exists() and bold_file_exists:
                    if exclude_subjects_framewise_displ(subject, ses, root_directory):
                        subjects_excluded_motion.append(subject)
                        continue 
            else:
                scrubbed_data = Path(f"{root_directory}/{subject}/ses-{ses}/native_T1/{subject}_ses-{ses}_run-01_rest_bold_ap_T1-space_scrubbed_0.5.nii.gz")

                # Check if either scrubbed or unscrubbed data exists
                bold_file_exists = scrubbed_data.exists() or unscrubbed_file.exists()

                # Track subjects with no bold file
                if not bold_file_exists:
                    subjects_excluded_no_bold.append(subject)
                    continue 

                if fd_file.exists() and bold_file_exists:
                    if exclude_subjects_framewise_displ(subject, ses, root_directory):
                        subjects_excluded_motion.append(subject)
                        continue 
                else:
                    # If scrubbed data is not found, 
                    print(f"Scrubbed data not found for {subject}. Skipping this subject.")
                    continue
                
        output_data = Path(f"{output_directory}/ses-{ses}")

        # Format the atlas file path for the specific subject
        if isinstance(atlas_file_template, Path):
            atlas_path = str(atlas_file_template).format(subject=subject, ses=ses)
            subject_atlas_file = Path(atlas_path)
        else:
            subject_atlas_file = Path(atlas_file_template.format(subject=subject, ses=ses))

        # Check if either scrubbed or original data exists and the atlas file exists
        bold_file_exists = scrubbed_data.exists() or (unscrubbed_file is not None and unscrubbed_file.exists())
        atlas_file_exists = subject_atlas_file.exists()

        if bold_file_exists:
            if not atlas_file_exists:
                # Track subjects with no atlas file
                print(f"Atlas file not found for {subject}: {subject_atlas_file}")
                subjects_without_atlas_file.append(subject)
                continue
                
            # Now create the output directory and check if output file exists
            output_data.mkdir(parents=True, exist_ok=True)
            expected_output_filename = f"{subject}_ses-{ses}_subcortical_schaefer200_timeseries.csv"
            output_file_path = output_data / expected_output_filename

            if not output_file_path.exists():
                subjects_to_process.append(subject)

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    print(f"Number of subjects excluded due to no bold file: {len(subjects_excluded_no_bold)}")
    print(f"Number of subjects excluded due to excessive motion: {len(subjects_excluded_motion)}")
    print(f"Number of subjects excluded due to no atlas file: {len(subjects_without_atlas_file)}")
    return subjects_to_process

def main(
    ses: str,
    threshold: float,
    error_log_path: Union[str, Path],
    output_dir: Union[str, Path],
    roi_indices: List[int],
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
        roi_indices (List[int]): ROI indices for timeseries visualization (e.g. add the index for the ROI/s you want to visualize).
        multi (bool): If True, enables parallel processing using multiprocessing. Defaults to False.
        atlas_file_template (str): Template string for the atlas file path.
    """
    output_dir = Path(output_dir)
    error_log_path = Path(f"{output_dir}/error_log.txt")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass atlas template and labels to the processing function
    args = [
        (
            subject,
            ses,
            threshold,
            atlas_file_template, 
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
    timepoint = "2"
    threshold = "0.5"
    cohort = "bbhi senior"   
    root = Path("/home/rachel/Desktop/schaefer_analysis") 
    output_directory = Path(f"{root}/timeseries_data/native_space")

    if cohort == "bbhi":
        if timepoint == "1":
            root_directory = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed")
            subject_csv = "/home/rachel/Desktop/data/bbhi_ids_tp1.csv"
        else:
            root_directory = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp2")
            subject_csv = "/home/rachel/Desktop/data/clean_bbhi.csv"
    else:
        subject_csv = None
        root_directory = Path("/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed")
    error_log_path = output_directory / "error_log.txt"

    # Create the output directory if it does not exist
    output_directory.mkdir(parents=True, exist_ok=True)

    roi_indices = [0]  # ROIs to visualize

    atlas_file_template = Path(f"{root}/fsaverage/ses-{ses}/{{subject}}/bold_space_masks/{{subject}}_ses-{ses}_schaefer200_subcortical14_bold_space.nii.gz")
                
    # Generate a list of subjects to process
    subjects = get_subjects_to_process(root_directory, atlas_file_template, output_directory, ses, cohort) 

    # Optionally, manually specify subjects to process
    # subjects = ["sub-4045"]

    # main(
    #     ses=ses,
    #     threshold=threshold,
    #     error_log_path=error_log_path,
    #     output_dir=output_directory,
    #     roi_indices=roi_indices,
    #     atlas_file_template=atlas_file_template,
    #     # multi=False,
    #     multi=True, # Uncomment this line to enable parallel processing
    # )
