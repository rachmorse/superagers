#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script to extract subcortical regions from FreeSurfer's aseg.mgz file
and create labeled volumes for left and right subcortical structures.
"""

import os
import subprocess
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

def get_subjects_to_process(output_folder):
    """Generate a list of subjects to process based on whether they have
    fsaverage to t1 done for the specified timepoint.

    Args:
        output_folder (Path): Path to the directory fsaverage to t1 results

    Returns:
        list: List of subject IDs to process.
    """
    subjects_to_process = []

    # Iterate over all possible subject directories
    for subject in os.listdir(output_folder):
        if not subject.startswith("sub-"):
            continue
        
        # Check if the required directory exists and hasn't been processed yet  
        subject_folder = output_folder / subject
      
        t1_file_path = subject_folder / f"{subject}_schaefer_volumetric_t1.nii.gz"
        output_file_path = subject_folder / f"{subject}_left_subcortical14_t1.nii.gz"

        if t1_file_path.exists() and not output_file_path.exists():
            subjects_to_process.append(subject)

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process


def process_subcortical_regions(aseg_file, subject, reference_file, output_folder_sub, region='left'):
    """Process subcortical regions and create labeled volume.
    
    Args:
        aseg_file (str): Path to the aseg.mgz file.
        subject (str): Subject ID.
        reference_file (str): Path to the reference file.
        output_folder_sub (str): Output directory for results.
        region (str): 'left' or 'right' to specify which hemisphere to process.
    
    Returns:
        bool: True if processing completed successfully, False otherwise.
    """
    try:
        # Define subcortical labels and their corresponding values
        # 17 L_Hippocampus -> 1, 18 L_Amygdala -> 2, 13 L_Pallidum -> 3,
        # 12 L_Putamen -> 4, 11 L_Caudate -> 5, 26 L_Accumbens -> 6, 10 L_Thalamus -> 7
        if region.lower() == 'left':
            subcortical_labels = "17 18 13 12 11 26 10"
            output_file = f"{output_folder_sub}/{subject}_left_subcortical14_t1.nii.gz"
        else:
        # 53 R_Hippocampus -> 1, 54 R_Amygdala -> 2, 52 R_Pallidum -> 3,
        # 51 R_Putamen -> 4, 50 R_Caudate -> 5, 58 R_Accumbens -> 6, 49 R_Thalamus -> 7
            subcortical_labels = "53 54 52 51 50 58 49"
            output_file = f"{output_folder_sub}/{subject}_right_subcortical14_t1.nii.gz"
            
        logger.info(f"Processing {region} subcortical regions")
        
        # Set up FSL environment variables inside this function
        os.environ["FSLDIR"] = "/home/rachel/fsl"
        os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:{os.environ['PATH']}"
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        
        # Use absolute paths for commands
        fslmaths_bin = "/home/rachel/fsl/bin/fslmaths"

        # Process each label
        labels_list = subcortical_labels.split()
        for i, lab in enumerate(labels_list, 1):
            lab = int(lab)
            logger.info(f"Processing label {lab} (value {i})")
            
            # Step 1: Extract the region from aseg
            cmd_binarize = [
                'mri_binarize',
                '--i', aseg_file,
                '--match', str(lab),
                '--o', f'{output_folder_sub}/tmp.mgz'
            ]
            try:
                subprocess.run(cmd_binarize, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                logger.debug("Binarization completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error in mri_binarize: {e.stderr}")
                return False
                
            # Step 2: Convert to NIfTI format
            cmd_convert = [
                'mri_convert',
                '--in_type', 'mgz',
                '--out_type', 'nii',
                f'{output_folder_sub}/tmp.mgz',
                f'{output_folder_sub}/tmp.nii.gz'
            ]
            try:
                subprocess.run(cmd_convert, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                logger.debug("Conversion to NIfTI completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error in mri_convert: {e.stderr}")
                return False
            
            # Step 3: Create or update the output file
            if lab == int(labels_list[0]):  # First label
                cmd_init = [
                    fslmaths_bin,
                    f'{output_folder_sub}/tmp.nii.gz',
                    '-mul', str(i),
                    output_file
                ]
                try:
                    subprocess.run(cmd_init, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    logger.debug(f"Initialized output file for {region} hemisphere")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error initializing output file: {e}")
                    return False
            else:
                # Multiply by the value (i)
                cmd_mul = [
                    fslmaths_bin,
                    f'{output_folder_sub}/tmp.nii.gz',
                    '-mul', str(i),
                    f'{output_folder_sub}/tmp.nii.gz'
                ]
                try:
                    subprocess.run(cmd_mul, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error in multiplication step: {e}")
                    return False
                
                # Add to the existing file
                cmd_add = [
                    fslmaths_bin,
                    output_file,
                    '-add', f'{output_folder_sub}/tmp.nii.gz',
                    output_file
                ]
                try:
                    subprocess.run(cmd_add, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error in addition step: {e}")
                    return False
                
        logger.info(f"Successfully processed {subject}")
        return True
        
    except Exception as e:
        logger.exception(f"Error processing {subject}'s subcortical regions: {str(e)}")
        return False

def cleanup_temp_files(output_folder_sub):
    """Clean up temporary files.
    
    Args:
        output_folder_sub (str): Directory containing temporary files.
    """
    try:
        temp_files = [
            f'{output_folder_sub}/tmp.nii.gz',
            f'{output_folder_sub}/tmp.mgz'
        ]
        
        for file in temp_files:
            if os.path.exists(file):
                logger.debug(f"Removing temporary file: {file}")
                os.remove(file)
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

def main():
    # Set up logging level based on verbose flag
    logger.setLevel(logging.DEBUG)

    ses = "ses-02"
    output_folder = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")

    # # Determine subjects to process
    # subject_list = get_subjects_to_process(output_folder)
    # print(f"Number of subjects to process: {len(subject_list)}")
    # print(subject_list)
    
    # if not subject_list:
    #     logger.info("No subjects found that need processing.")
    #     return
        
    # subjects = subject_list

    # Uncomment this line to run the script with one subject
    subjects = ["sub-4064"]
   
    # Log execution info
    logger.info(f"Script executed by: {os.environ.get('USER', 'unknown')}")
    logger.info(f"Date and time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")

    for subject in subjects:
        output_folder_sub = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}/{subject}")

        # BBHI senior
        aseg_file = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/freesurfer-reconall/{subject}_{ses}/mri/aseg.mgz")
        reference_file = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject}/{ses}/native_T1/{subject}_{ses}_run-01_rest_sbref_ap_T1-space.nii.gz")

        # BBHI 
        # aseg_file = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall/{subject_id}_{ses}/mri/aseg.mgz")
        # reference_file = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed/{subject_id}/native_T1/{subject_id}_{ses}_run-01_rest_sbref_ap_T1-space.nii.gz")

        successful_subjects = []
        failed_subjects = []

        logger.info(f"Processing subject {subject}...")
        # Process left subcortical regions
        left_success = process_subcortical_regions(
            aseg_file,
            subject,
            reference_file,
            output_folder_sub,
            region='left'
        )
        
        # Process right subcortical regions
        right_success = process_subcortical_regions(
            aseg_file,
            subject,
            reference_file,
            output_folder_sub,
            region='right'
        )
    
        # Clean up temporary files
        cleanup_temp_files(output_folder_sub)

        if left_success and right_success:
            successful_subjects.append(subject)
        else:
            failed_subjects.append(subject)

    print(f"\n{len(successful_subjects)}/{len(subjects)} subjects processed successfully")
    
    if failed_subjects:
        print("Failed subjects:")
        for subject in failed_subjects:
            print(f"  - {subject}")

if __name__ == "__main__":
    main()