#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import nibabel as nib
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_subjects_to_process(output_folder, ses):
    """Generate a list of subjects to process based on whether they have
    t1 cortical and subcortical atlases for the specified timepoint.

    Args:
        output_folder (Path): Path to the directory t1 results
        ses (str): Session ID (format: ses-01).

    Returns:
        list: List of subject IDs to process.
    """
    subjects_to_process = []
    already_processed = []

    # Iterate over all possible subject directories
    for subject in os.listdir(output_folder):
        if not subject.startswith("sub-"):
            continue
        
        # Check if the required directory exists and hasn't been processed yet  
        subject_t1_folder = output_folder / subject / "t1_masks"
        subject_subcort_folder = output_folder / subject / "subcortical_t1_masks"

        left_t1_file_path = subject_t1_folder / f"{subject}_schaefer_volumetric_t1_lh.nii.gz"
        right_t1_file_path = subject_t1_folder / f"{subject}_schaefer_volumetric_t1_rh.nii.gz"
        subcort_left_file_path = subject_subcort_folder / f"{subject}_left_subcortical14_t1.nii.gz"
        subcort_right_file_path = subject_subcort_folder / f"{subject}_right_subcortical14_t1.nii.gz"
        output_file_path = subject_subcort_folder / f"{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz"

        if left_t1_file_path.exists() and right_t1_file_path.exists() and subcort_left_file_path.exists() and subcort_right_file_path.exists() and not output_file_path.exists():
            subjects_to_process.append(subject)
        elif output_file_path.exists():
            already_processed.append(subject)
        elif left_t1_file_path.exists() and not right_t1_file_path.exists() or not left_t1_file_path.exists() and right_t1_file_path.exists():
            print(f"Subject {subject} missing one cortical atlas file")
        elif subcort_left_file_path.exists() and not subcort_right_file_path.exists() or not subcort_left_file_path.exists() and subcort_right_file_path.exists():
            print(f"Subject {subject} missing one subcortical atlas file")

    return subjects_to_process, already_processed


def process_subject(output_folder, subject, ses):
    """Process a single subject to create a combined Schaefer atlas in native T1 space.
    
    Args:
        output_folder (Path): Path to the directory containing subject data.
        subject (str): Subject ID to process.
        ses (str): Session ID.
    
    Returns:
        bool: True if processing completed successfully, False otherwise.
    """
    try:                        
        # Define the subject folder
        subject_t1_folder = output_folder / subject / "t1_masks"
        subject_subcort_folder = output_folder / subject / "subcortical_t1_masks"
        
        # Load the individual brain atlas components with correct paths
        right_cortical_path = subject_t1_folder / f"{subject}_schaefer_volumetric_t1_rh.nii.gz"
        left_cortical_path = subject_t1_folder / f"{subject}_schaefer_volumetric_t1_lh.nii.gz"
        left_subcortical_path = subject_subcort_folder / f"{subject}_left_subcortical14_t1.nii.gz"
        right_subcortical_path = subject_subcort_folder / f"{subject}_right_subcortical14_t1.nii.gz"

        left_cortical_img = nib.load(left_cortical_path)
        right_cortical_img = nib.load(right_cortical_path)
        left_subcortical_img = nib.load(left_subcortical_path)
        right_subcortical_img = nib.load(right_subcortical_path)

        left_cortical = left_cortical_img.get_fdata()
        right_cortical = right_cortical_img.get_fdata()
        left_subcortical = left_subcortical_img.get_fdata()
        right_subcortical = right_subcortical_img.get_fdata()

        # Ensure all volumes share the same shape and affine before combining
        reference_shape = left_cortical.shape
        reference_affine = left_cortical_img.affine
        for img, label in [
            (right_cortical_img, "right cortical"),
            (left_subcortical_img, "left subcortical"),
            (right_subcortical_img, "right subcortical"),
        ]:
            if img.shape != reference_shape or not np.allclose(img.affine, reference_affine):
                raise ValueError(
                    f"{subject}: {label} volume shape/affine mismatch prevents combining "
                    f"(expected {reference_shape}, got {img.shape})"
                )
        
        # Adjust values to create non-overlapping ranges for each region
        left_cortical_adj = np.where(left_cortical != 0, left_cortical, 0)                    # cortical: values from 1 to 100 (keeping background as 0)
        right_cortical_adj = np.where(right_cortical != 0, right_cortical + 100, 0)           # cortical: values from 101 to 200 (keeping background as 0)
        left_subcortical_adj = np.where(left_subcortical != 0, left_subcortical + 200, 0)     # ls: values from 201 to 207 (keeping background as 0)
        right_subcortical_adj = np.where(right_subcortical != 0, right_subcortical + 207, 0)  # rs: values from 208 to 214 (keeping background as 0)
        
        # Get the affine transformation matrix from the original image
        affine = reference_affine # uses left cortical as reference as they all have same affine
        
        # Create a blank matrix to store the final result with the same shape as the left cortical atlas
        result = np.zeros_like(left_cortical)
        
        # Create a stacked matrix for comparison 
        stacked = np.stack([left_cortical_adj, right_cortical_adj, left_subcortical_adj, right_subcortical_adj], axis=-1)
        
        # Identify positions with overlap (more than one non-zero value)
        overlap_mask = np.sum(stacked != 0, axis=-1) > 1
        print(f"Subject {subject} has {np.sum(overlap_mask)} overlapping voxels out of {np.prod(reference_shape)}.")

        # Check if there are any overlapping voxels
        if not np.any(overlap_mask):
            logger.info(f"No overlapping voxels found for subject {subject}.")
        
        # Extract overlap values
        overlap_values = stacked[overlap_mask]
        overlap_values = np.unique(overlap_values, axis=0, return_counts=True)
        
        output_file = subject_subcort_folder / "overlap_values_percentages.txt"
        
        # Open file in write mode so each run overwrites previous logs
        with open(output_file, "w") as file:
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
                    counts['left_cortical'][0],
                    counts['right_cortical'][0],
                    counts['left_subcortical'][0],
                    counts['right_subcortical'][0],
                ]
                max_perc_idx = int(
                    np.argmax(
                        [
                            percentages["left_cortical"],
                            percentages["right_cortical"],
                            percentages["left_subcortical"],
                            percentages["right_subcortical"],
                        ]
                    )
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
        
        # Check that all 214 areas are preserved
        unique_values = np.unique(result)
        if unique_values.shape[0] != 215:  # 214 regions + background (0)
            logger.warning(f"Subject {subject} has incomplete atlas: {unique_values.shape[0]-1} regions instead of 214")
            # Log incomplete Schaefer atlas information
            with open("schaefer_incomplete.txt", "a") as f:
                f.write(f"{subject} {unique_values.shape[0]-1}\n")
        
        # Save the combined atlas
        nib.save(nib.Nifti1Image(result, affine), Path(f"{subject_subcort_folder}/{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz"))
        
        logger.info(f"Successfully processed subject {subject}")
        return True
        
    except Exception as e:
        logger.exception(f"Error processing subject {subject}: {str(e)}")
        return False


def main():
    # Define paths
    cohorts = ["bbhi", "bbhi senior"]
    sessions = ["ses-01", "ses-02"]
    
    # Set up logging level
    logger.setLevel(logging.INFO)

    print("-----------------------Running 03_combine_t1_altases.py-----------------------")
        
    for cohort in cohorts:
        for ses in sessions:
            print("-------------------------")
            print(f"Processing {cohort} {ses}")
            print("-------------------------")

            output_folder = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")

            # Get subjects to process
            subjects, already_processed = get_subjects_to_process(output_folder, ses)

            # Filter subjects by cohort ID
            if cohort == "bbhi":
                # Keep only subjects whose numeric ID is >= 5000
                subjects = [subject for subject in subjects if int(subject.split("-")[1]) > 5000]
                already_processed = [subject for subject in already_processed if int(subject.split("-")[1]) > 5000]
            else:  # cohort == "bbhi senior"
                subjects = [subject for subject in subjects if int(subject.split("-")[1]) < 5000]
                already_processed = [subject for subject in already_processed if int(subject.split("-")[1]) < 5000]
            
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


if __name__ == "__main__":
    main()
