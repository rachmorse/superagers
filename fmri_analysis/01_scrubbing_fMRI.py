import json
import os
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.interpolate

# This script is only to run on the BBHI senior tp2 data because the BBHI and BBHI senior tp1 data has already been scrubbed.

def analyze_threshold(data, threshold, total_scans=740, affected_percentage=0.5):
    """Analyzes and validates subjects with high motion using a given threshold.

    This function visualizes the distribution of moved scans to help determine
    appropriate thresholds for scrubbing.

    Args:
        data (pd.DataFrame): DataFrame containing Framewise Displacement (FWD) data.
        threshold (float): The FWD threshold to identify moved scans.
        total_scans (int, optional): Total number of scans per subject. Defaults to 740.
        affected_percentage (float, optional): Percentage of scans that must exceed threshold
            to count a subject as "moved". Defaults to 0.5.
    """
    moved_subjects_count = (((data > threshold).sum(1) / total_scans) > affected_percentage).sum()
    print(
        f"{moved_subjects_count} subjects with more than {affected_percentage * 100}% of scans moved (threshold {threshold})"
    )
    plt.hist((data > threshold).sum(1) / total_scans)
    plt.title(f"Distribution of Percentage of Moved Scans (Threshold: {threshold})")
    plt.xlabel("Percentage")
    plt.ylabel("Number of Subjects")
    plt.show()


def save_json_sidecar(json_file, status, threshold, percent_scrubbed, num_bad_frames, total_frames, max_scrub_percent):
    """Saves the scrubbing status and statistics to a JSON sidecar file.

    Args:
        json_file (str): Path to the output JSON file.
        status (str): The scrubbing status (e.g., 'Scrubbed', 'NoScrubbingNeeded_LowMotion', 'SkippedScrubbing_HighMotion').
        threshold (float): The FWD threshold used.
        percent_scrubbed (float): Percentage of frames scrubbed.
        num_bad_frames (int): Count of frames scrubbed.
        total_frames (int): Total number of frames in the series.
        max_scrub_percent (float): Maximum allowed percentage of scrubbed frames.
    """
    data = {
        "Description": "Scrubbed fMRI images",
        "ExclusionCriteria": f"Subjects with >{max_scrub_percent}% frames exceeding {threshold}mm FD were excluded.",
        "ScrubbingStatus": status,
        "MotionThreshold": threshold,
        "PercentageFramesScrubbed": round(percent_scrubbed, 2),
        "NumFramesScrubbed": int(num_bad_frames),
        "TotalFrames": int(total_frames)
    }
    
    try:
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved scrub status JSON to {json_file}")
    except Exception as e:
        print(f"Error saving JSON file {json_file}: {e}")


def scrub(subject, bold_file, fwd_data, scrubbed_file, remote_scrubbed_file, json_file, threshold=0.5, method="interpolate", max_scrub_percent=30):
    """Scrubs BOLD fMRI images by either removing or interpolating scans based on FWD threshold.

    Args:
        subject (str): Subject ID used to process the BOLD and FWD files.
        bold_file (str): Path to the BOLD image file (NIfTI format).
        fwd_data (array-like): Framewise Displacement data.
        scrubbed_file (str): Path to save the scrubbed BOLD image file (NIfTI format).
        json_file (str): Path to save the JSON sidecar file.
        threshold (float, optional): Threshold for FWD above which scans are considered moved. Defaults to 0.5 FWD.
        method (str, optional): Method for handling moved scans. Either 'cut' or 'interpolate'. Defaults to 'interpolate'.
        max_scrub_percent (float, optional): Maximum percentage of frames that can be scrubbed before the subject is skipped. Defaults to 30%.

    Returns:
        bool: True if scrubbing was performed, False if subject was skipped due to too many frames exceeding threshold.
    """
    # Check if no frames exceed threshold, skip loading BOLD and scrubbing
    fwd_check = np.array(fwd_data)
    total_frames_est = len(fwd_check) + 1 

    if not np.any(np.nan_to_num(fwd_check) >= threshold):
        print(f"No frames exceed the threshold for subject {subject}. Skipping scrubbing.")
        save_json_sidecar(json_file, "NoScrubbingNeeded_LowMotion", threshold, 0.0, 0, total_frames_est, max_scrub_percent)
        return False

    # Then check if too many frames exceed threshold, skip loading BOLD and scrubbing
    num_bad_frames = np.sum(np.nan_to_num(fwd_check) >= threshold)
    est_scrub_percent = (num_bad_frames * 100) / total_frames_est
    
    if est_scrub_percent > max_scrub_percent:
        print(f"WARNING: More than {max_scrub_percent}% of timepoints would be scrubbed for subject {subject}. Skipping scrubbing.")
        save_json_sidecar(json_file, "SkippedScrubbing_HighMotion", threshold, est_scrub_percent, num_bad_frames, total_frames_est, max_scrub_percent)
        return False

    # Check for existing scrubbed files (local or remote) to avoid re-scrubbing.
    # If found, make sure the JSON sidecar exists 
    file_exists = False
    source_loc = ""
    
    if os.path.exists(scrubbed_file):
        file_exists = True
        source_loc = "local"
    elif remote_scrubbed_file and os.path.exists(remote_scrubbed_file):
        file_exists = True
        source_loc = "remote"
        
    if file_exists:
        if not os.path.exists(json_file):
            print(f"Scrubbed file exists ({source_loc}) for {subject}, regenerating JSON sidecar only.")
            
            # Use fixed total frames (740) for efficiency to avoid loading BOLD header
            total_frames = 740 
            fwd = np.array(fwd_data)

            # Check for array alignment
            if len(fwd) == total_frames:
                pass 
            elif len(fwd) == total_frames - 1:
                fwd = np.insert(fwd, 0, 0)
            else:
                raise ValueError(f"Length mismatch: FWD has {len(fwd)} points, expected {total_frames} volumes.")

            # Identify volumes with excessive motion
            all_tps = np.arange(total_frames)
            incorrect_tps = all_tps[fwd >= threshold]
            
            # Calculate percentage of volumes that would be scrubbed
            scrub_percent = (len(incorrect_tps) * 100) / total_frames
            
            # Save JSON
            save_json_sidecar(json_file, "Scrubbed", threshold, scrub_percent, len(incorrect_tps), total_frames, max_scrub_percent)
            print(f"JSON sidecar regenerated for {subject}")
            return True
        else:
            print(f"Scrubbed file ({source_loc}) and JSON already exist for {subject}. Skipping re-scrubbing.")
            return True

    print(f"Scrubbing BOLD image for subject: {subject}")

    # Load BOLD image data
    print("Loading BOLD image from file:", bold_file)
    bold = nib.load(bold_file)
    bold_data = bold.get_fdata()
    bold_affine = bold.affine

    # Load FWD data
    fwd = np.array(fwd_data)

    # Check for array alignment
    if len(fwd) == bold_data.shape[3]:
        pass # Lengths match, no adjustment needed
    elif len(fwd) == bold_data.shape[3] - 1:
        # FWD is one shorter than BOLD, prepend 0 to align. 
        # Since this code compares fwd < threshold, prepending 0 means the first volume is not scrubbed.
        fwd = np.insert(fwd, 0, 0)
    else:
        raise ValueError(f"Length mismatch: FWD has {len(fwd)} points, BOLD has {bold_data.shape[3]} volumes.")

    # Identify volumes with excessive motion
    all_tps = np.arange(bold_data.shape[3])
    correct_tps = all_tps[fwd < threshold]
    incorrect_tps = all_tps[fwd >= threshold]
    
    # Calculate percentage of volumes that would be scrubbed
    scrub_percent = (len(incorrect_tps) * 100) / bold_data.shape[3]
    
    print(
        f"{len(incorrect_tps)} out of {bold_data.shape[3]} scans ({round(scrub_percent, 2)}%) exceed the motion threshold (FWD > {threshold})."
    )
    
    # Extract only the correct timepoints for interpolation
    correct_bold = bold_data[:, :, :, correct_tps]

    # Start scrubbing based on the method
    if method == "cut":
        # If the method is 'cut', remove the timepoints with excessive motion. Save as scrubbed_data.
        scrubbed_data = correct_bold
        print(f"Removing {len(incorrect_tps)} scans due to excessive motion.")
    elif method == "interpolate":
        # If the method is 'interpolate', replace the timepoints with excessive motion through interpolation or extrapolation.
        scrubbed_data = bold_data.copy()
        # Check if the first or last timepoints are incorrect because if they are, they should be extrapolated
        if 1 not in incorrect_tps and bold_data.shape[3] - 1 not in incorrect_tps:
            print(f"Interpolating {len(incorrect_tps)} scans due to excessive motion.")
            # Perform interpolation when neither the first nor last timepoints are incorrect
            interpolator = scipy.interpolate.interp1d(correct_tps, correct_bold, axis=3, fill_value="extrapolate")
            scrubbed_data[:, :, :, incorrect_tps] = interpolator(incorrect_tps)
            print("No scans require extrapolation.")
        else:
            extrap_idx = []
            intrap_idx = list(incorrect_tps)
            extrap_text = []
            i = 1
            # Perform left extrapolation for incorrect first timepoint
            while i in incorrect_tps:
                extrap_idx.append(i)
                intrap_idx.remove(i)
                extrap_text.append("left")
                i += 1
            i = 1
            # Perform right extrapolation for incorrect last timepoint
            while bold_data.shape[3] - i in incorrect_tps:
                extrap_idx.append(bold_data.shape[3] - i)
                intrap_idx.remove(bold_data.shape[3] - i)
                extrap_text.append("right")
                i += 1

            print(f"Interpolating {len(intrap_idx)} scans due to excessive motion.")
            interpolator = scipy.interpolate.interp1d(correct_tps, correct_bold, axis=3)
            scrubbed_data[:, :, :, intrap_idx] = interpolator(intrap_idx)

            if extrap_text:
                print(
                    f"Extrapolating {len(extrap_idx)} scans in the {', '.join(extrap_text)} direction(s) due to motion."
                )
                extrapolator = scipy.interpolate.interp1d(correct_tps, correct_bold, fill_value="extrapolate", axis=3)
                scrubbed_data[:, :, :, extrap_idx] = extrapolator(extrap_idx)

    else:
        print("Unknown method specified; returning the original BOLD file.")
        scrubbed_data = bold_data  # Unchanged; return original BOLD data

    scrubbed_image = nib.Nifti1Image(scrubbed_data, bold_affine)
    os.makedirs(os.path.dirname(scrubbed_file), exist_ok=True)
    nib.save(scrubbed_image, scrubbed_file)

    # Save JSON for Performed case
    save_json_sidecar(json_file, "Scrubbed", threshold, scrub_percent, len(incorrect_tps), bold_data.shape[3], max_scrub_percent)

    print(f"Scrubbing complete for subject: {subject}")
    return True


def process_subject(
    subject,
    ses,
    root,
    threshold,
    output_data,
    error_log,
    bold_pattern,
    scrubbed_pattern,
    subject_dir_pattern,
    fwd_data,
):
    """Processes a single subject by scrubbing the BOLD fMRI images based on the FWD.
    Saves the scrubbed BOLD file for the subject and logs errors if any occur.

    Args:
        subject (str): Subject ID used to process the BOLD and FWD files.
        ses (str): Session (timepoint).
        root (str): Root directory path.
        threshold (float): Threshold value for scrubbing.
        output_data (str): Output directory root.
        error_log (str): Path to error log.
        bold_pattern (str): Filename pattern for input BOLD.
        scrubbed_pattern (str): Filename pattern for output BOLD.
        subject_dir_pattern (str): Directory pattern (e.g., ses-X/native_T1).
        fwd_data (list): FWD values.

    Returns:
        bool: True if processing was successful (even if skipped), False if error.
    """
    try:
        bold_file = bold_pattern.format(subject=subject, ses=ses)
        scrubbed_file = scrubbed_pattern.format(subject=subject, ses=ses, threshold=threshold, output_data=output_data)
        
        # Define remote path 
        remote_pattern_str = scrubbed_pattern.replace("{output_data}", str(root))
        remote_scrubbed_file = remote_pattern_str.format(subject=subject, ses=ses, threshold=threshold)

        # Derive JSON filename for the sidecar
        if scrubbed_file.endswith(".nii.gz"):
            json_file = scrubbed_file.replace(".nii.gz", ".json")
        elif scrubbed_file.endswith(".nii"):
            json_file = scrubbed_file.replace(".nii", ".json")
        else:
             json_file = scrubbed_file + ".json"

        print(f"Processing subject: {subject}")

        if fwd_data is None:
            raise ValueError(f"No FWD data available for subject {subject}")

        scrub(
            subject,    
            bold_file,
            fwd_data,
            scrubbed_file,
            remote_scrubbed_file,
            json_file,
            threshold=threshold,
            method="interpolate",
        )
        return True

    except Exception as e:
        print(f"Error processing subject {subject}: {e}")
        with open(error_log, "a") as f:
            f.write(f"Error processing subject {subject}: {e}\n")
        return False


def main(
    ses,
    root,
    output_data,
    threshold,
    bold_pattern,
    scrubbed_pattern,
    subject_dir_pattern,
    multi=False,
):
    """Main function to run this script. This function performs the following steps:

    1. Defines session, threshold, and directories for data input and output.
    2. Iterates over all subjects in the root directory to concatenate their 
        `framewise_displ.txt` files into a single DataFrame.
    3. Saves the concatenated DataFrame to `all_fwd.csv`.
    4. Optionally, visualizes data thresholds using the `analyze_threshold` function.
    5. Generates a list of subjects to be processed and saves it as `todo.csv`.
    6. Scrubs the BOLD images by either serial or parallel processing of subjects.
    7. Saves the scrubbed BOLD images to the output directory.
    8. Logs errors to `scrubbing_errors.txt`.

    Args:
        ses (str): Session identifier (e.g., "01", "02").
        root (str): Root directory path where subject data is located.
        output_data (Path): Path to the directory where scrubbed data will be saved.
        threshold (float): The Framewise Displacement (FWD) threshold for scrubbing.
        bold_pattern (str): A format string for the input BOLD file path,
            e.g., "{root}/{subject}/ses-{ses}/.../{subject}_ses-{ses}_..._bold.nii.gz".
        scrubbed_pattern (str): A format string for the output scrubbed BOLD file path,
            e.g., "{output_data}/{subject}/ses-{ses}/.../{subject}_ses-{ses}_..._scrubbed_{threshold}.nii.gz".
        subject_dir_pattern (str): A format string for the subject's session-specific
            directory containing FWD data, e.g., "ses-{ses}/native_T1".
        multi (bool, optional): If True, use multiprocessing for parallel execution.
            Defaults to False.
    """
    folder_name = subject_dir_pattern.split('/')[-1]
    summary_dir = os.path.join(output_data, folder_name)
    os.makedirs(summary_dir, exist_ok=True)

    error_log = os.path.join(summary_dir, "scrubbing_errors.txt")
    all_fwd_path = os.path.join(summary_dir, "all_fwd.csv")

    # Concatenate all framewise_displ.txt files (per subject) into a single DataFrame
    fwd_list = []
    subject_fwd_map = {}

    # Determine subjects list 
    potential_subjects = os.listdir(root)
    subjects = []

    # Filter subjects based on whether they have the required session directory and collect FWD data 
    for subject in potential_subjects:
        if not subject.startswith("sub-"):
            continue
            
        # Check if the specified session directory exists
        if ses == "01":
            session_path = Path(f"{root}/{subject}/ses-01")
            session_exists = session_path.exists() and session_path.is_dir()
        elif ses == "02":
            session_path = Path(f"{root}/{subject}/ses-02") 
            session_exists = session_path.exists() and session_path.is_dir()
        else:
            session_exists = False
            
        if not session_exists:
            continue

        # FWD data collection
        subject_dir = os.path.join(root, subject, subject_dir_pattern)
        fwd_file = os.path.join(subject_dir, "framewise_displ.txt")

        if os.path.exists(fwd_file):
            try:
                # Read the framewise_displ.txt file
                fwd_data_df = pd.read_csv(fwd_file)

                # Convert each participant's column of data into a list to make it a single row of data instead
                fwd_series = pd.Series(fwd_data_df["FramewiseDisplacement"].tolist(), name=subject)
                fwd_row_df = fwd_series.to_frame().T

                # Append the row DataFrame to the list
                fwd_list.append(fwd_row_df)
                
                # Store for processing later
                subject_fwd_map[subject] = fwd_data_df["FramewiseDisplacement"].tolist()
            except Exception as e:
                 print(f"Error reading FWD for {subject}: {e}")
        else:
            print(f"No framewise_displ.txt found for {subject}")

        # Check if scrubbed data AND json sidecar already exists in either location
        threshold_str = str(threshold)
        local_scrubbed_file = Path(scrubbed_pattern.format(subject=subject, ses=ses, threshold=threshold, output_data=output_data))
        
        # Determine JSON path logic (replicating process_subject logic)
        if str(local_scrubbed_file).endswith(".nii.gz"):
            local_json_file = Path(str(local_scrubbed_file).replace(".nii.gz", ".json"))
        elif str(local_scrubbed_file).endswith(".nii"):
             local_json_file = Path(str(local_scrubbed_file).replace(".nii", ".json"))
        else:
             local_json_file = Path(str(local_scrubbed_file) + ".json")

        remote_pattern_str = scrubbed_pattern.replace("{output_data}", str(root))
        remote_scrubbed_file = Path(remote_pattern_str.format(subject=subject, ses=ses, threshold=threshold))
        
        # Determine processing needs:
        # 1. Full scrub if no scrubbed data exists (local or remote).
        # 2. JSON regeneration if scrubbed data exists but sidecar is missing.
        should_process = False
        
        local_exists = local_scrubbed_file.exists()
        remote_exists = remote_scrubbed_file.exists()
        json_exists = local_json_file.exists()
        
        if not local_exists and not remote_exists:
             should_process = True # Full scrub needed
        elif local_exists and not json_exists:
             print(f"Subject {subject} - Local NIfTI exists, but JSON missing. Adding to process list.")
             should_process = True
        elif remote_exists and not json_exists:
             print(f"Subject {subject} - Remote NIfTI exists, but JSON missing. Adding to process list.")
             should_process = True

        if should_process:
            subjects.append(subject)
        else:
            pass

    print(f"Total subjects needing processing: {len(subjects)}")

    # Save the concatenated DataFrame to all_fwd.csv
    if fwd_list:
        all_fwd_df = pd.concat(fwd_list)
        all_fwd_df.to_csv(all_fwd_path, index=True, header=False)
        print(f"all_fwd.csv has been created at {all_fwd_path}")
    else:
        print("No FWD data found to concatenate.")

    # Visualize what different thresholds would look like in the data
    # analyze_threshold(all_fwd_df, 0.2)
    # analyze_threshold(all_fwd_df, 0.5)

    # Print subjects to be processed
    print(f"Subjects to be processed: {subjects}")

    # Parallel processing
    if multi:
        with Pool(3) as pool:
            pool.starmap(
                process_subject,
                [
                    (
                        subject,
                        ses,
                        root,
                        threshold,
                        output_data,
                        error_log,
                        bold_pattern,
                        scrubbed_pattern,
                        subject_dir_pattern,
                        subject_fwd_map.get(subject)
                    )
                    for subject in subjects
                ],
            )
    else:
        for subject in subjects:
            process_subject(
                subject,
                ses,
                root,
                threshold,
                output_data,
                error_log,
                bold_pattern,
                scrubbed_pattern,
                subject_dir_pattern,
                subject_fwd_map.get(subject)
            )

    print("Processing complete.")


if __name__ == "__main__":
    # Change to your paths and settings
    threshold = 0.5
    ses = "02"
    # root = "/pool/guttmann/institut/UB/Superagers/MRI/resting_preproc_fs6-recon"
    root = "/home/rachel/Desktop/preprocessing-updated_reconall/bbhi/resting_preprocessed"
    output_data = Path("/home/rachel/Desktop/schaefer_analysis/scrubbed_data")

    # Create the output directory if it does not exist
    output_data.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "folder_name": "native_T1",
            "file_suffix": "T1-space"
        },
        {
            "folder_name": "MNI_2mm",
            "file_suffix": "MNI-space"
        }
    ]

    for config in configs:
        folder_name = config["folder_name"]
        file_suffix = config["file_suffix"]
        
        subject_dir_pattern = f"ses-{ses}/{folder_name}"
        
        print(f"\n--- Processing {folder_name} ---")

        # Define file patterns
        bold_pattern = os.path.join(
            root,
            "{subject}",
            subject_dir_pattern,
            f"{{subject}}_ses-{{ses}}_run-01_rest_bold_ap_{file_suffix}.nii.gz",
        )
        scrubbed_pattern = os.path.join(
            "{output_data}",
            "{subject}",
            f"ses-{ses}",
            folder_name,
            f"{{subject}}_ses-{{ses}}_run-01_rest_bold_ap_{file_suffix}_scrubbed_{{threshold}}.nii.gz",
        )

        main(
            ses,
            root,
            output_data,
            threshold,
            bold_pattern,
            scrubbed_pattern,
            subject_dir_pattern,
            multi=True,
        )
