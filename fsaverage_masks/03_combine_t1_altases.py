#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script to combine atlas components (left/right subcortical regions and 
left/right Schaefer cortical regions) into a single atlas in T1 space.
"""

import os
import nibabel as nib
import numpy as np
import logging
from datetime import datetime, timezone
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

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process


def process_subject(output_folder, subject, ses):
    """Process a single subject to create a combined Schaefer atlas.
    
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

        # Load the individual brain atlas components with correct paths
        left_cortical = nib.load(left_cortical_path).get_fdata()
        right_cortical = nib.load(right_cortical_path).get_fdata()
        left_subcortical = nib.load(left_subcortical_path).get_fdata()
        right_subcortical = nib.load(right_subcortical_path).get_fdata()
        
        # Adjust values to create non-overlapping ranges for each region
        left_cortical_adj = np.where(left_cortical != 0, left_cortical, 0)                    # cortical: values from 1 to 100 (keeping 0 as 0)
        right_cortical_adj = np.where(right_cortical != 0, right_cortical + 100, 0)           # cortical: values from 101 to 200 (keeping 0 as 0)
        left_subcortical_adj = np.where(left_subcortical != 0, left_subcortical + 200, 0)     # ls: values from 201 to 207 (keeping 0 as 0)
        right_subcortical_adj = np.where(right_subcortical != 0, right_subcortical + 207, 0)  # rs: values from 208 to 214 (keeping 0 as 0)
        
        # Get the affine transformation matrix from the original image
        affine = nib.load(left_cortical_path).affine
        
        # Create a matrix to store the final result
        result = np.zeros_like(left_cortical)
        
        # Create a stacked matrix for comparison
        stacked = np.stack([left_cortical_adj, right_cortical_adj, left_subcortical_adj, right_subcortical_adj], axis=-1)
        
        # Identify positions with overlap (more than one non-zero value)
        overlap_mask = np.sum(stacked != 0, axis=-1) > 1

        # Check if there are any overlapping voxels
        if not np.any(overlap_mask):
            logger.info(f"No overlapping voxels found for subject {subject}.")
        
        # Extract overlap values
        overlap_values = stacked[overlap_mask]
        overlap_values = np.unique(overlap_values, axis=0, return_counts=True)
        
        output_file = subject_subcort_folder / "overlap_values_percentages.txt"
        
        # Open file in append mode
        with open(output_file, "a") as file:
            # Iterate through each overlap instance
            for idx, voxel_values in enumerate(overlap_values[0], start=1):
                left_cortical_value = voxel_values[0]
                right_cortical_value = voxel_values[1]  
                left_subcortical_value = voxel_values[2]
                right_subcortical_value = voxel_values[3]
                n_values_overlaped = overlap_values[1][idx-1]
                left_cortical_perc = 0.
                right_cortical_perc = 0.
                left_subcortical_perc = 0. 
                right_subcortical_perc = 0.
                # Calculate percentage of overlap for each region
                if left_cortical_value != 0: left_cortical_perc = n_values_overlaped*100/np.sum(left_cortical_adj==left_cortical_value)
                if right_cortical_value != 0: right_cortical_perc = n_values_overlaped*100/np.sum(right_cortical_adj==right_cortical_value)
                if left_subcortical_value != 0: left_subcortical_perc = n_values_overlaped*100/np.sum(left_subcortical_adj==left_subcortical_value)
                if right_subcortical_value != 0: right_subcortical_perc = n_values_overlaped*100/np.sum(right_subcortical_adj==right_subcortical_value)

                # Write overlap information to file
                new_line = f"{subject}, {idx}, {n_values_overlaped}, {left_cortical_value}, {right_cortical_value}, {left_subcortical_value}, {right_subcortical_value}, {left_cortical_perc}, {right_cortical_perc}, {left_subcortical_perc}, {right_subcortical_perc}\n"
                file.write(new_line)

        # For voxels with overlap, keep the value from the area with the highest percentage of affected voxels
        for idx, voxel_values in enumerate(overlap_values[0], start=1):
            left_cortical_value = voxel_values[0]
            right_cortical_value = voxel_values[1]
            left_subcortical_value = voxel_values[2]
            right_subcortical_value = voxel_values[3]
            n_values_overlaped = overlap_values[1][idx-1]
            left_cortical_perc = 0.
            right_cortical_perc = 0.
            left_subcortical_perc = 0. 
            right_subcortical_perc = 0.
            # Calculate percentage of overlap for each region
            if left_cortical_value != 0: left_cortical_perc = n_values_overlaped*100/np.sum(left_cortical_adj==left_cortical_value)
            if right_cortical_value != 0: right_cortical_perc = n_values_overlaped*100/np.sum(right_cortical_adj==right_cortical_value)
            if left_subcortical_value != 0: left_subcortical_perc = n_values_overlaped*100/np.sum(left_subcortical_adj==left_subcortical_value)
            if right_subcortical_value != 0: right_subcortical_perc = n_values_overlaped*100/np.sum(right_subcortical_adj==right_subcortical_value)
            
            # Find the region with the highest overlap percentage
            max_perc_idx = np.argmax(np.array([left_cortical_perc, right_cortical_perc, left_subcortical_perc, right_subcortical_perc]))
            
            # Assign values based on the region with highest percentage
            result[(overlap_mask) &
                      (stacked[:,:,:,0]==left_cortical_value) &
                      (stacked[:,:,:,1]==right_cortical_value) &
                      (stacked[:,:,:,2]==left_subcortical_value) &
                      (stacked[:,:,:,3]==right_subcortical_value)] = np.array([left_cortical_value, right_cortical_value, left_subcortical_value, right_subcortical_value])[max_perc_idx]

        # For non-overlapping areas, simply sum the values (only one will be non-zero)
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
    ses = "ses-01"
    output_folder = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")
    
    # Set up logging level
    logger.setLevel(logging.INFO)
    
    # Log execution info
    start_time = datetime.now(timezone.utc)
    logger.info(f"Script executed by: {os.environ.get('USER', 'rachel')}")
    logger.info(f"Date and time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Get subjects to process
    subjects = get_subjects_to_process(output_folder, ses)
    
    if not subjects:
        logger.info("No subjects found that need processing.")
        return
    
    # Create tracking lists
    successful_subjects = []
    failed_subjects = []
    
    # Process each subject
    for i, subject in enumerate(subjects, 1):
        logger.info(f"Processing subject {i}/{len(subjects)}: {subject}")
        
        success = process_subject(output_folder, subject, ses)
        
        if success:
            successful_subjects.append(subject)
        else:
            failed_subjects.append(subject)
    
    # Print summary
    print(f"\n{len(successful_subjects)}/{len(subjects)} subjects processed successfully")
    
    if failed_subjects:
        print("Failed subjects:")
        for subject in failed_subjects:
            print(f"  - {subject}")

if __name__ == "__main__":
    main()