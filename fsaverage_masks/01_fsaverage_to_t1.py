#!/usr/bin/env python3
"""Transform the fsaverage Schaefer atlas to each subject's native T1 space."""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ANNOT_DIR,
    ATLAS,
    COHORTS,
    CORTICAL_T1_MASK,
    MASK_ROOT,
    N_PARCELS,
    RECONALL_DIRS,
    SESSIONS,
    SUBJECTS,
    SUPERAGER_CSV,
    cohort_from_id,
)
from pipeline_utils import record_summary

N_JOBS = 6  # Subjects processed in parallel

# Schaefer atlas on fsaverage (annotation files from the Schaefer GitHub)
SCHAEFER_LH_ANNOT = ANNOT_DIR / f"lh.Schaefer2018_{N_PARCELS}Parcels_7Networks_order.annot"
SCHAEFER_RH_ANNOT = ANNOT_DIR / f"rh.Schaefer2018_{N_PARCELS}Parcels_7Networks_order.annot"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_subjects_to_process(
    reconall_dir: Path, out_dir: Path, ses: str, cohort: str
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Generate a list of subjects to process.

    Subjects are chosen based on whether they have recon-all done for the specified timepoint
    and have neuropsych data.

    Args:
        reconall_dir: Path to the directory containing the recon-all results.
        out_dir: Path to the output directory where the results will be saved.
        ses: Session (e.g., ses-01).
        cohort: Cohort name (e.g., 'bbhi', 'bbhi senior').

    Returns:
        List of (subject ID, recon-all directory name) pairs to process, and list of subject IDs already processed.
    """
    subjects_to_process = []
    already_processed = []

    # Read the CSV file to filter valid IDs based on neuropsych data (e.g., whether they have the data to be classified as superager)
    valid_ids = None
    df = pd.read_csv(SUPERAGER_CSV)
    required_superager_col = "superager_tp2" if ses == "ses-02" else "superager_tp1"

    if required_superager_col not in df.columns:
        logger.warning(
            "Column %s not found in superager.csv; cannot ensure participants have required data.",
            required_superager_col,
        )
    else:
        valid_ids = set(df[df[required_superager_col].notna()]["id"].astype(str).tolist())

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(reconall_dir):
        # Check if it's a subject directory with session (sub-xxx_ses-yy format, optionally _run-xx as with BBHI)
        if not subject_dir.startswith("sub-"):
            continue

        # Normalize naming (handles sub-xxx_ses-yy and sub-xxx_ses-yy_run-zz), skipping other sessions
        subject_id, _, dir_session = subject_dir.partition("_ses-")
        # Remove optional run information
        if f"ses-{dir_session.split('_run-')[0]}" != ses:
            continue

        # Filter subjects based on cohort
        try:
            if cohort_from_id(subject_id) != cohort:
                continue
        except ValueError:
            logger.warning(f"Could not parse subject number from {subject_id}, skipping cohort check.")

        # If sub is not in valid IDs, skip
        if valid_ids is not None and subject_id.replace("sub-", "") not in valid_ids:
            continue

        if subject_id not in SUBJECTS:
            continue

        # Check if the required directory exists and hasn't been processed yet
        subject_recon_dir = reconall_dir / subject_dir
        output_lh_path = out_dir / subject_id / CORTICAL_T1_MASK.format(subject=subject_id, hemi="lh")
        output_rh_path = out_dir / subject_id / CORTICAL_T1_MASK.format(subject=subject_id, hemi="rh")

        # Reprocess if either hemisphere is missing (e.g., interrupted run)
        if subject_recon_dir.exists() and not (output_lh_path.exists() and output_rh_path.exists()):
            subjects_to_process.append((subject_id, subject_dir))
        elif output_lh_path.exists() and output_rh_path.exists():
            already_processed.append(subject_id)
        elif not subject_recon_dir.exists():
            logger.warning(f"Subject {subject_id} does not have recon-all.")

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    print(f"Number of subjects already processed: {len(already_processed)}")

    return subjects_to_process, already_processed


def process_subject(subject_dir: str, subject_id: str, reconall_dir: Path, subject_folder: Path) -> bool:
    """Process a single subject to transform Schaefer atlas to native T1 space.

    Args:
        subject_dir: Subject directory (format: sub-xxx_ses-0y).
        subject_id: Subject ID.
        reconall_dir: Directory containing recon-all output.
        subject_folder: The subject's output folder for this session.

    Returns:
        True if processing completed successfully, False otherwise.
    """
    output_lh_mask = subject_folder / CORTICAL_T1_MASK.format(subject=subject_id, hemi="lh")
    output_rh_mask = subject_folder / CORTICAL_T1_MASK.format(subject=subject_id, hemi="rh")
    output_folder = output_lh_mask.parent

    try:
        # Create the output directory if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing subject: {subject_id} (directory: {subject_dir})")
        logger.info(f"{'=' * 50}\n")

        # Check the Schaefer annotation files exist
        if not SCHAEFER_LH_ANNOT.exists() or not SCHAEFER_RH_ANNOT.exists():
            logger.error(f"Could not find the local annotation files at {SCHAEFER_LH_ANNOT} and {SCHAEFER_RH_ANNOT}")
            logger.error("Please download these files manually and place them in the fsaverage_masks directory")
            return False

        # Set up paths for output annotations
        lh_annotation = output_folder / SCHAEFER_LH_ANNOT.name
        rh_annotation = output_folder / SCHAEFER_RH_ANNOT.name

        # Set SUBJECTS_DIR per subject because reused parallel workers keep the environment of the first cohort
        custom_env = os.environ.copy()
        custom_env["SUBJECTS_DIR"] = str(reconall_dir)

        # Map annotations from fsaverage to subject-specfic surface space
        logger.info(f"Mapping {ATLAS} atlas annotations from fsaverage to subject's surface")

        # Left hemisphere
        cmd_map_left = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject_dir,
            "--hemi", "lh",
            "--sval-annot", str(SCHAEFER_LH_ANNOT),
            "--tval", str(lh_annotation),
        ]  # fmt: skip

        # Right hemisphere
        cmd_map_right = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject_dir,
            "--hemi", "rh",
            "--sval-annot", str(SCHAEFER_RH_ANNOT),
            "--tval", str(rh_annotation),
        ]  # fmt: skip

        # Then transform the subject-specific surface annotations to volumetric T1 space
        # Left hemisphere
        subject_t1 = reconall_dir / subject_dir / "mri" / "T1.mgz"

        cmd_lh_label2vol = [
            "mri_label2vol",
            "--annot", str(lh_annotation),
            "--temp", str(subject_t1),
            "--identity",
            "--fillthresh", "0.3",
            "--proj", "frac", "0", "1", "0.1",
            "--subject", subject_dir,
            "--o", str(output_lh_mask),
            "--hemi", "lh",
        ]  # fmt: skip

        # Right hemisphere
        cmd_rh_label2vol = [
            "mri_label2vol",
            "--annot", str(rh_annotation),
            "--temp", str(subject_t1),
            "--identity",
            "--fillthresh", "0.3",
            "--proj", "frac", "0", "1", "0.1",
            "--subject", subject_dir,
            "--o", str(output_rh_mask),
            "--hemi", "rh",
        ]  # fmt: skip

        # Run each step in order, stopping at the first failure (surface mapping output is not printed)
        steps = [
            (cmd_map_left, "mapping left hemisphere", subprocess.PIPE),
            (cmd_map_right, "mapping right hemisphere", subprocess.PIPE),
            (cmd_lh_label2vol, "mapping left hemisphere to volume", None),
            (cmd_rh_label2vol, "mapping right hemisphere to volume", None),
        ]
        for cmd, description, stdout in steps:
            try:
                subprocess.run(cmd, env=custom_env, check=True, stderr=subprocess.PIPE, stdout=stdout, text=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error {description}: {e.stderr}")
                return False

        logger.info(f"Successfully processed subject {subject_id}")
        return True

    except Exception as e:
        logger.exception(f"Error processing subject {subject_id}: {e!s}")
        return False


def main() -> None:
    """Transform the Schaefer atlas to native T1 space for each cohort and session."""
    # Set up logging level
    logger.setLevel(logging.DEBUG)

    print("-----------------------Running 01_fsaverage_to_t1.py-----------------------")

    for cohort in COHORTS:
        for session in SESSIONS:
            print("-------------------------")
            print(f"Processing {cohort} {session}")
            print("-------------------------")

            output_folder = MASK_ROOT / session

            reconall_dir = RECONALL_DIRS[cohort]

            # Determine subjects to process
            subject_data, _ = get_subjects_to_process(reconall_dir, output_folder, session, cohort)

            if not subject_data:
                logger.info("No subjects found that need processing.")
                continue

            subjects = [s[0] for s in subject_data]
            logger.info(f"Will process {len(subjects)} subjects: {', '.join(subjects)}")

            # Process each subject
            results = Parallel(n_jobs=N_JOBS)(
                delayed(process_subject)(subject_dir, subject_id, reconall_dir, output_folder / subject_id)
                for subject_id, subject_dir in subject_data
            )

            successful = sum(results)

            logger.info("\nProcessing summary:")
            logger.info(f"Total subjects: {len(subject_data)}")
            logger.info(f"Successfully processed: {successful}")
            logger.info(f"Failed: {len(subject_data) - successful}")

            failed_subjects = [subject_id for (subject_id, _), ok in zip(subject_data, results) if not ok]
            record_summary(__file__, cohort, session, successful, failed_subjects)


if __name__ == "__main__":
    main()
