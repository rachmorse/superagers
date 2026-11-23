"""Extract ROI timeseries from each subject's preprocessed resting-state BOLD image using their native space atlas."""

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from extract_timeseries import extract_timeseries, visualize_timeseries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    BOLD_MASK,
    COHORTS,
    FRAMEWISE_DISPL,
    MASK_ROOT,
    RESTING_BOLD,
    SESSIONS,
    SUBJECTS,
    TIMESERIES_FILE,
    TIMESERIES_ROOT,
    cohort_from_id,
    resting_native_dir,
)

N_JOBS = 3  # Subjects extracted in parallel
FD_THRESHOLD_MM = 0.5  # Frames with framewise displacement (FD) above this many mm are high motion and get scrubbed
MAX_HIGH_MOTION_PERCENT = 30  # Subjects with more than this percentage of frames above FD_THRESHOLD_MM are excluded
VISUALIZE = False  # Set to True to plot the timeseries of VISUALIZE_ROIS for each subject
VISUALIZE_ROIS = [0]  # Timeseries columns to plot, one per ROI in combined_labels order (0 = first ROI)


def get_atlas_file(subject: str, ses: str) -> Path:
    """Get the path to a subject's atlas in native BOLD space.

    Args:
        subject: Subject ID.
        ses: Session label (e.g. 'ses-01').

    Returns:
        Path to the atlas file.
    """
    return MASK_ROOT / ses / subject / BOLD_MASK.format(subject=subject, ses=ses)


def read_framewise_displ(subject: str, fd_file: Path) -> Optional[pd.Series]:
    """Read a subject's FD values.

    Args:
        subject: Subject ID.
        fd_file: Path to the FD file.

    Returns:
        FD per frame, or None if the file is missing or unreadable,
        since motion cannot be checked without it.
    """
    if not fd_file.exists():
        print(f"FD file not found for {subject} at {fd_file}. Skipping this subject.")
        return None
    try:
        fd_values = pd.read_csv(fd_file, header=None).iloc[:, 0]
        return pd.to_numeric(fd_values, errors="coerce").fillna(0)
    except Exception as e:
        print(f"Error reading FD file for {subject}: {e!s}. Skipping this subject.")
        return None


def exclude_subjects_framewise_displ(subject: str, fd_values: pd.Series) -> bool:
    """Check if the subject has excessive motion (>30% of frames exceeding 0.5 mm).

    Args:
        subject: Subject ID.
        fd_values: FD per frame.

    Returns:
        True if the subject should be excluded due to excessive motion, False otherwise.
    """
    high_motion_percentage = (fd_values > FD_THRESHOLD_MM).mean() * 100
    if high_motion_percentage > MAX_HIGH_MOTION_PERCENT:
        print(f"Excluding {subject} due to excessive motion: {high_motion_percentage:.2f}% of frames > 0.5mm")
        return True
    return False


def select_bold_file(subject: str, ses: str, native_dir: Path, needs_scrubbing: bool) -> Optional[Path]:
    """Select the BOLD image to extract the timeseries from.

    Subjects with any frame above the FD threshold must have exactly one scrubbed image.
    Subjects with none were not scrubbed, so the original image is used.

    Args:
        subject: Subject ID.
        ses: Session label (e.g. 'ses-01').
        native_dir: Folder holding the subject's T1-space BOLD images.
        needs_scrubbing: Whether any frame is above the FD threshold.

    Returns:
        Path to the BOLD image, or None if the expected image is missing or there is more than one scrubbed image.
    """
    original_file = native_dir / RESTING_BOLD.format(subject=subject, ses=ses)
    prefix = original_file.name.removesuffix(".nii.gz")

    if not needs_scrubbing:
        if original_file.exists():
            return original_file
        print(f"Original BOLD file not found for {subject}: {original_file}. Skipping this subject.")
        return None

    # Naming differs by cohort (e.g. scrubbed-interp-05, scrubbed_0.5), so match any scrubbed suffix
    scrubbed_files = sorted(native_dir.glob(f"{prefix}_scrubbed*.nii.gz"))
    if len(scrubbed_files) == 1:
        return scrubbed_files[0]
    print(
        f"{subject} has frames above {FD_THRESHOLD_MM} mm but {len(scrubbed_files)} scrubbed BOLD files "
        f"(expected 1): {[f.name for f in scrubbed_files]}. Skipping this subject."
    )
    return None


def process_subject_extract(subject: str, ses: str, bold_file: Path, atlas_file: Path, output_dir: Path) -> None:
    """Processes a single subject. Extracts timeseries and saves it.

    Optionally, visualizes the timeseries data.

    Args:
        subject: Subject ID.
        ses: Session label (e.g. 'ses-01').
        bold_file: Preprocessed BOLD image to extract the timeseries from.
        atlas_file: File path for the atlases used for extracting timeseries.
        output_dir: Directory where processed data and any outputs are saved.
    """
    print(f"--- Processing subject: {subject} ---")
    error_log_path = output_dir / "error_log.txt"

    # Adding a warning filter to catch issues that likely indicate the wrong atlas or BOLD file is being used
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", UserWarning)
        timeseries = extract_timeseries(atlas_file, bold_file, error_log_path)

        for captured_warning in w:
            if "After resampling the label image to the data image, the following labels were removed" in str(
                captured_warning.message
            ):
                print(f"Error in processing {subject}: some ROIs were unable to be identified.")
                return

    # Check if there is no valid timeseries
    if timeseries is None or timeseries.size == 0:
        print(f"No valid timeseries extracted for subject {subject}")
        return

    # Ensure the directory exists
    output_subdir = output_dir / ses
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Save the extracted timeseries
    timeseries_output_path = output_subdir / TIMESERIES_FILE.format(subject=subject, ses=ses)
    print(f"Saving extracted timeseries to {timeseries_output_path}")
    np.savetxt(timeseries_output_path, timeseries, delimiter=",")

    # Visualize data if VISUALIZE is set to True at the top of the script
    if VISUALIZE:
        visualize_timeseries(subject, timeseries, VISUALIZE_ROIS)

    print(f"Processing completed for subject: {subject}")


def get_subjects_to_process(output_directory: Path, ses: str, cohort: str) -> Dict[str, Path]:
    """Generate a list of subjects to process, with the BOLD image to use for each.

    Subjects are included if they have BOLD data, an FD file and an atlas file, and no
    timeseries file already generated. Subjects with excessive motion (>30% of frames
    exceeding 0.5 mm) are excluded.

    Args:
        output_directory: Path to the output directory where timeseries data is saved.
        ses: Session label (e.g. 'ses-01').
        cohort: Cohort name (e.g., 'bbhi', 'bbhi senior').

    Returns:
        Subject IDs to process, mapped to the BOLD image to extract from.
    """
    subjects_to_process = {}
    subjects_excluded_motion = []
    subjects_excluded_no_bold = []
    subjects_excluded_no_fd = []
    subjects_excluded_bold_selection = []
    subjects_excluded_no_atlas_file = []

    for subject in SUBJECTS:
        # Only keep subjects that belong to the current cohort by ID convention
        try:
            subject_cohort = cohort_from_id(subject)
        except ValueError:
            print(f"Skipping {subject}: could not parse numeric ID")
            continue
        if subject_cohort != cohort:
            continue

        native_dir = resting_native_dir(subject, ses, cohort)

        # Track subjects with no bold file (scrubbed or original)
        bold_prefix = RESTING_BOLD.format(subject=subject, ses=ses).removesuffix(".nii.gz")
        if not any(native_dir.glob(f"{bold_prefix}*.nii.gz")):
            subjects_excluded_no_bold.append(subject)
            continue

        # Motion cannot be checked without an FD file
        fd_values = read_framewise_displ(subject, native_dir / FRAMEWISE_DISPL)
        if fd_values is None:
            subjects_excluded_no_fd.append(subject)
            continue

        if exclude_subjects_framewise_displ(subject, fd_values):
            subjects_excluded_motion.append(subject)
            continue

        bold_file = select_bold_file(subject, ses, native_dir, needs_scrubbing=(fd_values > FD_THRESHOLD_MM).any())
        if bold_file is None:
            subjects_excluded_bold_selection.append(subject)
            continue

        # Track subjects with no atlas file
        atlas_file = get_atlas_file(subject, ses)
        if not atlas_file.exists():
            print(f"Atlas file not found for {subject}: {atlas_file}")
            subjects_excluded_no_atlas_file.append(subject)
            continue

        output_file_path = output_directory / ses / TIMESERIES_FILE.format(subject=subject, ses=ses)
        if not output_file_path.exists():
            subjects_to_process[subject] = bold_file

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    print(f"Number of subjects excluded due to no bold file: {len(subjects_excluded_no_bold)}")
    print(f"Number of subjects excluded due to no FD file: {len(subjects_excluded_no_fd)}")
    print(f"Number of subjects excluded due to excessive motion: {len(subjects_excluded_motion)}")
    print(f"Number of subjects excluded due to missing or extra scrubbed file: {len(subjects_excluded_bold_selection)}")
    print(f"Number of subjects excluded due to no atlas file: {len(subjects_excluded_no_atlas_file)}")
    return subjects_to_process


def main(ses: str, subjects: Dict[str, Path], output_dir: Path, multi: bool = False) -> None:
    """Extracts the timeseries for each subject, either sequentially or in parallel based on the multi flag.

    Args:
        ses: Session label (e.g. 'ses-01').
        subjects: Subject IDs to process, mapped to the BOLD image to extract from.
        output_dir: Path where processed data will be output.
        multi: If True, enables parallel processing using multiprocessing. Defaults to False.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (subject, ses, bold_file, get_atlas_file(subject, ses), output_dir) for subject, bold_file in subjects.items()
    ]

    if multi:
        with Pool(N_JOBS) as pool:
            pool.starmap(process_subject_extract, jobs)
    else:
        for job in jobs:
            process_subject_extract(*job)

    print("Done.")


if __name__ == "__main__":
    for ses in SESSIONS:
        for cohort in COHORTS:
            print("-------------------------")
            print(f"Processing {cohort} {ses}")
            print("-------------------------")

            # Generate a list of subjects to process
            subjects = get_subjects_to_process(TIMESERIES_ROOT, ses, cohort)

            main(ses=ses, subjects=subjects, output_dir=TIMESERIES_ROOT, multi=True)
