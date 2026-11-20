#!/usr/bin/env python3
"""Combine each subject's cortical Schaefer and subcortical masks into one atlas in native T1 space."""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ATLAS,
    COHORTS,
    COMBINED_T1_MASK,
    CORTICAL_T1_MASK,
    MASK_ROOT,
    N_PARCELS,
    SESSIONS,
    SHARED_MASK_ROOT,
    SUBCORTICAL_T1_MASK,
    SUBJECTS,
    cohort_from_id,
)
from pipeline_utils import record_summary

OVERLAP_LOG = "overlap_values_percentages.txt"  # Written next to each subject's combined atlas
INCOMPLETE_LOG = MASK_ROOT / f"{ATLAS}_incomplete.txt"  # Subjects whose combined atlas is missing regions

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def get_subjects_to_process(output_folder: Path, ses: str, cohort: str) -> Tuple[List[str], List[str]]:
    """Generate a list of subjects to process.

    Subjects are chosen based on whether they have t1 cortical and subcortical atlases for the specified timepoint.

    Args:
        output_folder: Path to the directory t1 results
        ses: Session ID (format: ses-01).
        cohort: Cohort to keep (e.g., 'bbhi', 'bbhi senior').

    Returns:
        List of subject IDs to process, and list of subject IDs already processed.
    """
    subjects_to_process = []
    already_processed = []

    # Iterate over all possible subject directories
    for subject in os.listdir(output_folder):
        if not subject.startswith("sub-") or subject not in SUBJECTS:
            continue

        # Filter by cohort before any checks so warnings print under the right cohort
        if cohort_from_id(subject) != cohort:
            continue

        # Check if the required directory exists and hasn't been processed yet
        subject_folder = output_folder / subject
        shared_subject_folder = SHARED_MASK_ROOT / ses / subject

        left_t1_file_path = subject_folder / CORTICAL_T1_MASK.format(subject=subject, hemi="lh")
        right_t1_file_path = subject_folder / CORTICAL_T1_MASK.format(subject=subject, hemi="rh")
        subcort_left_file_path = shared_subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side="left")
        subcort_right_file_path = shared_subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side="right")
        output_file_path = subject_folder / COMBINED_T1_MASK.format(subject=subject, ses=ses)

        has_cortical = (left_t1_file_path.exists(), right_t1_file_path.exists())
        has_subcortical = (subcort_left_file_path.exists(), subcort_right_file_path.exists())

        if all(has_cortical) and all(has_subcortical) and not output_file_path.exists():
            subjects_to_process.append(subject)
        elif output_file_path.exists():
            already_processed.append(subject)
        elif any(has_cortical) and not all(has_cortical):
            print(f"Subject {subject} missing one cortical atlas file")
        elif any(has_subcortical) and not all(has_subcortical):
            print(f"Subject {subject} missing one subcortical atlas file")

    return subjects_to_process, already_processed


def load_matching_volumes(paths: Dict[str, Path], subject: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load volumes and check they all share the shape and affine of the first one.

    Args:
        paths: Volume name (e.g. "left cortical") to file path, the first is the reference.
        subject: Subject ID, for the error message.

    Returns:
        The volume data by name, and the reference affine.

    Raises:
        ValueError: If a volume's shape or affine does not match the reference.
    """
    images = {name: nib.load(path) for name, path in paths.items()}
    reference_img = next(iter(images.values()))
    for name, img in images.items():
        if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine):
            raise ValueError(
                f"{subject}: {name} volume shape/affine mismatch prevents combining "
                f"(expected {reference_img.shape}, got {img.shape})"
            )
    return {name: img.get_fdata() for name, img in images.items()}, reference_img.affine


def process_subject(output_folder: Path, subject: str, ses: str) -> bool:
    """Process a single subject to create a combined Schaefer atlas in native T1 space.

    Args:
        output_folder: Path to the directory containing subject data.
        subject: Subject ID to process.
        ses: Session ID.

    Returns:
        True if processing completed successfully, False otherwise.
    """
    try:
        # Define the subject folders and the combined atlas output
        subject_folder = output_folder / subject
        shared_subject_folder = SHARED_MASK_ROOT / ses / subject
        combined_output = subject_folder / COMBINED_T1_MASK.format(subject=subject, ses=ses)
        combined_output.parent.mkdir(parents=True, exist_ok=True)

        # Load the individual brain atlas components, checking they share the same shape and affine
        volumes, reference_affine = load_matching_volumes(
            {
                "left cortical": subject_folder / CORTICAL_T1_MASK.format(subject=subject, hemi="lh"),
                "right cortical": subject_folder / CORTICAL_T1_MASK.format(subject=subject, hemi="rh"),
                "left subcortical": shared_subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side="left"),
                "right subcortical": shared_subject_folder / SUBCORTICAL_T1_MASK.format(subject=subject, side="right"),
            },
            subject,
        )
        left_cortical = volumes["left cortical"]
        right_cortical = volumes["right cortical"]
        left_subcortical = volumes["left subcortical"]
        right_subcortical = volumes["right subcortical"]
        reference_shape = left_cortical.shape

        # Adjust values to create non-overlapping ranges for each region (value ranges below are for Schaefer 200)
        # cortical: values from 1 to 100 (keeping background as 0)
        left_cortical_adj = np.where(left_cortical != 0, left_cortical, 0)
        # cortical: values from 101 to 200 (keeping background as 0)
        right_cortical_adj = np.where(right_cortical != 0, right_cortical + N_PARCELS // 2, 0)
        # ls: values from 201 to 207 (keeping background as 0)
        left_subcortical_adj = np.where(left_subcortical != 0, left_subcortical + N_PARCELS, 0)
        # rs: values from 208 to 214 (keeping background as 0)
        right_subcortical_adj = np.where(right_subcortical != 0, right_subcortical + N_PARCELS + 7, 0)

        # Get the affine transformation matrix from the original image
        affine = reference_affine  # uses left cortical as reference as they all have same affine

        # Create a blank matrix to store the final result with the same shape as the left cortical atlas
        result = np.zeros_like(left_cortical)

        # Create a stacked matrix for comparison
        stacked = np.stack(
            [left_cortical_adj, right_cortical_adj, left_subcortical_adj, right_subcortical_adj], axis=-1
        )

        # Identify positions with overlap (more than one non-zero value)
        overlap_mask = np.sum(stacked != 0, axis=-1) > 1
        print(f"Subject {subject} has {np.sum(overlap_mask)} overlapping voxels out of {np.prod(reference_shape)}.")

        # Check if there are any overlapping voxels
        if not np.any(overlap_mask):
            logger.info(f"No overlapping voxels found for subject {subject}.")

        # Extract overlap values
        overlap_values = stacked[overlap_mask]
        overlap_values = np.unique(overlap_values, axis=0, return_counts=True)

        # Open file in write mode so each run overwrites previous logs
        with open(combined_output.parent / OVERLAP_LOG, "w") as file:
            for idx, voxel_values in enumerate(overlap_values[0], start=1):
                counts = {
                    "left_cortical": (voxel_values[0], left_cortical_adj),
                    "right_cortical": (voxel_values[1], right_cortical_adj),
                    "left_subcortical": (voxel_values[2], left_subcortical_adj),
                    "right_subcortical": (voxel_values[3], right_subcortical_adj),
                }
                n_values_overlaped = overlap_values[1][idx - 1]
                percentages = {
                    region: (n_values_overlaped * 100 / np.sum(arr == value)) if value != 0 else 0.0
                    for region, (value, arr) in counts.items()
                }

                file.write(
                    f"{subject}, {idx}, {n_values_overlaped}, "
                    f"{counts['left_cortical'][0]}, {counts['right_cortical'][0]}, "
                    f"{counts['left_subcortical'][0]}, {counts['right_subcortical'][0]}, "
                    f"{percentages['left_cortical']}, {percentages['right_cortical']}, "
                    f"{percentages['left_subcortical']}, {percentages['right_subcortical']}\n"
                )

                # Select the region with the highest overlap percentage
                labels = [
                    counts["left_cortical"][0],
                    counts["right_cortical"][0],
                    counts["left_subcortical"][0],
                    counts["right_subcortical"][0],
                ]
                max_perc_idx = int(
                    np.argmax([
                        percentages["left_cortical"],
                        percentages["right_cortical"],
                        percentages["left_subcortical"],
                        percentages["right_subcortical"],
                    ])
                )

                result[
                    (overlap_mask)
                    & (stacked[:, :, :, 0] == counts["left_cortical"][0])
                    & (stacked[:, :, :, 1] == counts["right_cortical"][0])
                    & (stacked[:, :, :, 2] == counts["left_subcortical"][0])
                    & (stacked[:, :, :, 3] == counts["right_subcortical"][0])
                ] = labels[max_perc_idx]

        # For non-overlapping areas, sum the values since only one will be non-zero. This preserves the original labels.
        result[~overlap_mask] = np.sum(stacked[~overlap_mask], axis=-1)

        # Check that all 214 areas are preserved (414 for Schaefer 400)
        n_regions = N_PARCELS + 14
        unique_values = np.unique(result)
        if unique_values.shape[0] != n_regions + 1:
            logger.warning(
                f"Subject {subject} has incomplete atlas: {unique_values.shape[0] - 1} regions instead of {n_regions}"
            )
            # Log incomplete Schaefer atlas information
            with open(INCOMPLETE_LOG, "a") as f:
                f.write(f"{subject} {unique_values.shape[0] - 1}\n")

        # Save the combined atlas
        nib.save(nib.Nifti1Image(result, affine), combined_output)

        logger.info(f"Successfully processed subject {subject}")
        return True

    except Exception as e:
        logger.exception(f"Error processing subject {subject}: {e!s}")
        return False


def main() -> None:
    """Combine the cortical and subcortical masks in native T1 space for each cohort and session."""
    # Set up logging level
    logger.setLevel(logging.INFO)

    print("-----------------------Running 03_combine_t1_atlases.py-----------------------")

    for cohort in COHORTS:
        for ses in SESSIONS:
            print("-------------------------")
            print(f"Processing {cohort} {ses}")
            print("-------------------------")

            output_folder = MASK_ROOT / ses

            # Get subjects to process
            subjects, already_processed = get_subjects_to_process(output_folder, ses, cohort)

            print(f"Number of subjects already processed: {len(already_processed)}")
            print(f"Number of subjects to process: {len(subjects)}")

            if not subjects:
                logger.info("No subjects found that need processing.")
                continue

            # Create tracking lists
            successful_subjects = []
            failed_subjects = []

            # Process each subject
            for i, subject in enumerate(subjects, 1):
                logger.info(f"Processing subject {i}/{len(subjects)}: {subject}")

                success = process_subject(output_folder, subject, ses)

                if success:
                    print(f"Successfully processed {subject}")
                    successful_subjects.append(subject)
                else:
                    print(f"Failed to process {subject}")
                    failed_subjects.append(subject)

            # Print summary
            print(f"\n{len(successful_subjects)}/{len(subjects)} subjects processed successfully")

            if failed_subjects:
                print("Failed subjects:")
                for subject in failed_subjects:
                    print(f"  - {subject}")

            record_summary(__file__, cohort, ses, len(successful_subjects), failed_subjects)


if __name__ == "__main__":
    main()
