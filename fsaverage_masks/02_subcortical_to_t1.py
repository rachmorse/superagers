#!/usr/bin/env python3
"""Extract the 14 subcortical regions from each subject's recon-all aseg into native T1 space."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    COHORTS,
    CORTICAL_T1_MASK,
    FSL_DIR,
    SESSIONS,
    SHARED_MASK_ROOT,
    SUBCORTICAL_ASEG_CODES,
    SUBCORTICAL_T1_MASK,
    SUBJECTS,
    cohort_from_id,
    reconall_subject_dir,
)
from pipeline_utils import record_summary, run_step

# Files written next to the subcortical masks while building each hemisphere, removed afterwards
ASEG_SPACE_MASK = "{subject}_{side}_subcortical14_asegspace.nii.gz"  # Before resampling to T1 space
TMP_MGZ = "tmp.mgz"
TMP_NII = "tmp.nii.gz"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_subjects_to_process(output_folder: Path, cohort: str) -> Tuple[List[str], List[str]]:
    """Generate a list of subjects to process.

    Subjects are chosen based on whether they have fsaverage to t1 process done for the specified timepoint.

    Args:
        output_folder: Path to the directory fsaverage to t1 results
        cohort: Cohort name (e.g., 'bbhi', 'bbhi senior').

    Returns:
        List of subject IDs to process, and list of subject IDs already processed.
    """
    subjects_to_process = []
    already_processed = []

    # Iterate over all possible subject directories
    for subject in os.listdir(output_folder):
        if not subject.startswith("sub-") or subject not in SUBJECTS or cohort_from_id(subject) != cohort:
            continue

        # Check if the required directory exists and hasn't been processed yet
        subject_folder = output_folder / subject

        t1_left_path = subject_folder / CORTICAL_T1_MASK.format(subject=subject, hemi="lh")
        t1_right_path = subject_folder / CORTICAL_T1_MASK.format(subject=subject, hemi="rh")
        left_subcort_output = subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side="left")
        right_subcort_output = subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side="right")

        # Only queue subjects that have both cortical masks and are missing at least one subcortical mask
        if t1_left_path.exists() and t1_right_path.exists():
            if not left_subcort_output.exists() or not right_subcort_output.exists():
                subjects_to_process.append(subject)
            else:
                already_processed.append(subject)

    return subjects_to_process, already_processed


def process_subcortical_regions(
    aseg_file: Path, subject: str, reference_file: Path, subject_folder: Path, region: str = "left"
) -> bool:
    """Process subcortical regions and create labeled volume.

    Args:
        aseg_file: Path to the aseg.mgz file.
        subject: Subject ID.
        reference_file: Path to the reference file.
        subject_folder: The subject's output folder for this session.
        region: 'left' or 'right' to specify which hemisphere to process.

    Returns:
        True if processing completed successfully, False otherwise.
    """
    hemi = "left" if region.lower() == "left" else "right"
    final_output = subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side=hemi)
    output_folder_sub = final_output.parent
    aseg_space_output = output_folder_sub / ASEG_SPACE_MASK.format(subject=subject, side=hemi)
    tmp_mgz = output_folder_sub / TMP_MGZ
    tmp_nii = output_folder_sub / TMP_NII

    try:
        if not aseg_file.exists():
            logger.error(f"aseg file not found: {aseg_file}")
            return False
        if not reference_file.exists():
            logger.error(f"Reference file not found: {reference_file}")
            return False

        output_folder_sub.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {region} subcortical regions")

        # Set up FSL environment variables inside this function
        # Automatically pulls 6.0.4 for consistency with fMRI analysis
        os.environ["FSLDIR"] = str(FSL_DIR)
        fsl_bin = f"{FSL_DIR}/bin"
        if fsl_bin not in os.environ["PATH"].split(":"):
            os.environ["PATH"] = f"{fsl_bin}:{os.environ['PATH']}"
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

        # Add this path as it has trouble finding fslmaths otherwise
        fslmaths_bin = str(FSL_DIR / "bin" / "fslmaths")

        # Process each subcortical aseg code for this hemisphere, in mask value order (1 to 7)
        for i, codes in enumerate(SUBCORTICAL_ASEG_CODES.values(), 1):
            lab = codes[0 if hemi == "left" else 1]
            logger.info(f"Processing label {lab} (value {i})")

            # Step 1: Extract the regions from aseg, giving selected region a value of 1
            cmd_binarize = [
                "mri_binarize",
                "--i", str(aseg_file),
                "--match", str(lab),
                "--o", str(tmp_mgz),
            ]  # fmt: skip
            run_step(cmd_binarize, "in mri_binarize")
            logger.debug("Binarization completed")

            # Step 2: Convert the binarized file to NIfTI format
            cmd_convert = [
                "mri_convert",
                "--in_type", "mgz",
                "--out_type", "nii",
                str(tmp_mgz),
                str(tmp_nii),
            ]  # fmt: skip
            run_step(cmd_convert, "in mri_convert")
            logger.debug("Conversion to NIfTI completed")

            # Step 3: Create the output volume with appropriate labeling
            # First unbinarize to set each ROI to its corresponding value
            if i == 1:  # First label initializes new volume
                cmd_init = [
                    fslmaths_bin,
                    str(tmp_nii),
                    "-mul", str(i),
                    str(aseg_space_output),
                ]  # fmt: skip
                run_step(cmd_init, "initializing output file")
                logger.debug(f"Initialized output file for {region} hemisphere")
            else:
                # Multiply by the value (i)
                cmd_mul = [
                    fslmaths_bin,
                    str(tmp_nii),
                    "-mul", str(i),
                    str(tmp_nii),
                ]  # fmt: skip
                run_step(cmd_mul, "in multiplication step")

                # Add to the existing file
                cmd_add = [
                    fslmaths_bin,
                    str(aseg_space_output),
                    "-add", str(tmp_nii),
                    str(aseg_space_output),
                ]  # fmt: skip
                run_step(cmd_add, "in addition step")

        # Resample to reference (native T1) space using nearest-neighbor interpolation
        cmd_resample = [
            "mri_vol2vol",
            "--mov", str(aseg_space_output),
            "--targ", str(reference_file),
            "--o", str(final_output),
            "--interp", "nearest",
            "--regheader",
        ]  # fmt: skip
        run_step(cmd_resample, "resampling subcortical mask to reference space")

        return True

    except subprocess.CalledProcessError:
        # Already logged by run_step
        return False
    except Exception as e:
        logger.exception(f"Error processing {subject}'s subcortical regions: {e!s}")
        return False
    finally:
        cleanup_temp_files([aseg_space_output, tmp_mgz, tmp_nii])


def cleanup_temp_files(temp_files: List[Path]) -> None:
    """Clean up temporary files.

    Args:
        temp_files: Temporary files to remove if they exist.
    """
    try:
        for file in temp_files:
            if file.exists():
                logger.debug(f"Removing temporary file: {file}")
                file.unlink()
    except Exception as e:
        logger.warning(f"Error during cleanup: {e!s}")


def main() -> None:
    """Extract the subcortical regions in native T1 space for each cohort and session."""
    # Set up logging level
    logger.setLevel(logging.DEBUG)

    print("-----------------------Running 02_subcortical_to_t1.py-----------------------")

    for cohort in COHORTS:
        for ses in SESSIONS:
            print("-------------------------")
            print(f"Processing {cohort} {ses}")
            print("-------------------------")

            output_folder = SHARED_MASK_ROOT / ses

            # Determine subjects to process
            subjects, already_processed = get_subjects_to_process(output_folder, cohort)

            print(f"Number of subjects already processed: {len(already_processed)}")

            if not subjects:
                logger.info("No subjects found that need processing.")
                continue

            # Initialize lists to track successful and failed subjects
            successful_subjects = []
            failed_subjects = []

            print(f"Number of subjects to process: {len(subjects)}")

            for subject in subjects:
                # Recon-all paths (folder naming differs by cohort)
                recon_mri_dir = reconall_subject_dir(subject, ses, cohort) / "mri"
                aseg_file = recon_mri_dir / "aseg.mgz"
                reference_file = recon_mri_dir / "T1.mgz"

                logger.info(f"Processing subject {subject}...")

                # Process left and right subcortical regions
                left_success = process_subcortical_regions(
                    aseg_file, subject, reference_file, output_folder / subject, region="left"
                )
                right_success = process_subcortical_regions(
                    aseg_file, subject, reference_file, output_folder / subject, region="right"
                )

                if left_success and right_success:
                    print(f"Successfully processed {subject}")
                    successful_subjects.append(subject)
                else:
                    print(f"Failed to process {subject}")
                    failed_subjects.append(subject)

            print(f"\n{len(successful_subjects)}/{len(subjects)} subjects processed successfully")

            if failed_subjects:
                print("Failed subjects:")
                for subject in failed_subjects:
                    print(f"  - {subject}")

            record_summary(__file__, cohort, ses, len(successful_subjects), failed_subjects)


if __name__ == "__main__":
    main()
