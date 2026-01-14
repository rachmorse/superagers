#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def get_subjects_to_process(output_folder, reconall_folder):
    """Generate a list of subjects to process based on whether they have
    fsaverage to t1 process done for the specified timepoint.

    Args:
        output_folder (Path): Path to the directory fsaverage to t1 results
        reconall_folder (Path): Path to the directory containing FreeSurfer recon-all outputs

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
        subject_folder = output_folder / subject
      
        t1_left_path = subject_folder / f"t1_masks/{subject}_schaefer_volumetric_t1_lh.nii.gz"
        t1_right_path = subject_folder / f"t1_masks/{subject}_schaefer_volumetric_t1_rh.nii.gz"
        left_subcort_output = subject_folder / f"subcortical_t1_masks/{subject}_left_subcortical14_t1.nii.gz"
        right_subcort_output = subject_folder / f"subcortical_t1_masks/{subject}_right_subcortical14_t1.nii.gz"

        # Only queue subjects that have both cortical masks and are missing at least one subcortical mask
        if t1_left_path.exists() and t1_right_path.exists() and reconall_folder.exists():
            if not left_subcort_output.exists() or not right_subcort_output.exists():
                subjects_to_process.append(subject)
            else:
                already_processed.append(subject)

    return subjects_to_process, already_processed


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
        output_folder_sub = Path(output_folder_sub)
        aseg_file = Path(aseg_file)
        reference_file = Path(reference_file)

        if not aseg_file.exists():
            logger.error(f"aseg file not found: {aseg_file}")
            return False
        if not reference_file.exists():
            logger.error(f"Reference file not found: {reference_file}")
            return False

        # Define subcortical labels and their corresponding values
        # 17 L_Hippocampus -> 1, 18 L_Amygdala -> 2, 13 L_Pallidum -> 3,
        # 12 L_Putamen -> 4, 11 L_Caudate -> 5, 26 L_Accumbens -> 6, 10 L_Thalamus -> 7
        if region.lower() == 'left':
            subcortical_labels = "17 18 13 12 11 26 10"
            aseg_space_output = output_folder_sub / f"{subject}_left_subcortical14_asegspace.nii.gz"
            final_output = output_folder_sub / f"{subject}_left_subcortical14_t1.nii.gz"
        else:
        # 53 R_Hippocampus -> 1, 54 R_Amygdala -> 2, 52 R_Pallidum -> 3,
        # 51 R_Putamen -> 4, 50 R_Caudate -> 5, 58 R_Accumbens -> 6, 49 R_Thalamus -> 7
            subcortical_labels = "53 54 52 51 50 58 49"
            aseg_space_output = output_folder_sub / f"{subject}_right_subcortical14_asegspace.nii.gz"
            final_output = output_folder_sub / f"{subject}_right_subcortical14_t1.nii.gz"
            
        logger.info(f"Processing {region} subcortical regions")

        # Set up FSL environment variables inside this function
        # Automatically pulls 6.0.4 for consistency with fMRI analysis
        os.environ["FSLDIR"] = "/vol/software/fsl_6_0_4"
        os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:{os.environ['PATH']}"
        os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

        # Add this path as it has trouble finding fslmaths otherwise
        fslmaths_bin = "/vol/software/fsl_6_0_4/bin/fslmaths"

        # Process each label
        labels_list = subcortical_labels.split()
        for i, lab in enumerate(labels_list, 1):
            lab = int(lab)
            logger.info(f"Processing label {lab} (value {i})")
            
            # Step 1: Extract the regions from aseg, giving selected region a value of 1
            cmd_binarize = [
                'mri_binarize',
                '--i', str(aseg_file),
                '--match', str(lab),
                '--o', str(output_folder_sub / 'tmp.mgz')
            ]
            try:
                subprocess.run(cmd_binarize, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                logger.debug("Binarization completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error in mri_binarize: {e.stderr}")
                return False
                
            # Step 2: Convert the binarized file to NIfTI format
            cmd_convert = [
                'mri_convert',
                '--in_type', 'mgz',
                '--out_type', 'nii',
                str(output_folder_sub / 'tmp.mgz'),
                str(output_folder_sub / 'tmp.nii.gz')
            ]
            try:
                subprocess.run(cmd_convert, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                logger.debug("Conversion to NIfTI completed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error in mri_convert: {e.stderr}")
                return False
            
            # Step 3: Create the output volume with appropriate labeling
            # First unbinarize to set each ROI to its corresponding value
            if i == 1:  # First label initializes new volume
                cmd_init = [
                    fslmaths_bin,
                    str(output_folder_sub / 'tmp.nii.gz'),
                    '-mul', str(i),
                    str(aseg_space_output)
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
                    str(output_folder_sub / 'tmp.nii.gz'),
                    '-mul', str(i),
                    str(output_folder_sub / 'tmp.nii.gz')
                ]
                try:
                    subprocess.run(cmd_mul, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error in multiplication step: {e}")
                    return False
                
                # Add to the existing file
                cmd_add = [
                    fslmaths_bin,
                    str(aseg_space_output),
                    '-add', str(output_folder_sub / 'tmp.nii.gz'),
                    str(aseg_space_output)
                ]
                try:
                    subprocess.run(cmd_add, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error in addition step: {e}")
                    return False
                
        # Resample to reference (native T1) space using nearest-neighbor interpolation
        cmd_resample = [
            'mri_vol2vol',
            '--mov', str(aseg_space_output),
            '--targ', str(reference_file),
            '--o', str(final_output),
            '--interp', 'nearest',
            '--regheader'
        ]
        try:
            subprocess.run(cmd_resample, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error resampling subcortical mask to reference space: {e.stderr}")
            return False
        finally:
            try:
                if aseg_space_output.exists():
                    aseg_space_output.unlink()
            except OSError as cleanup_error:
                logger.debug(f"Could not remove temporary file {aseg_space_output}: {cleanup_error}")
        
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
    cohorts = ["bbhi", "bbhi senior"]
    sessions = ["ses-01", "ses-02"]   

    print("-----------------------Running 02_subcortical_to_t1.py-----------------------")

    for cohort in cohorts:
        for ses in sessions:
            print("-------------------------")
            print(f"Processing {cohort} {ses}")
            print("-------------------------")

            output_folder = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")

            if cohort == "bbhi":
                reconall_folder = Path('/pool/guttmann/institut/BBHI/MRI/derivatives/reconall_fs6')
            else:  # cohort == "bbhi senior"
                reconall_folder = Path('/pool/guttmann/institut/UB/Superagers/MRI/derivatives/reconall_fs6')

            # Determine subjects to process
            subject_list, already_processed = get_subjects_to_process(output_folder, reconall_folder)
                        
            # Filter subjects by cohort before processing so printed count statements are accurate
            if cohort == "bbhi":
                subjects = [s for s in subject_list if int(s.split('-')[1]) > 5000]
                already_processed = [s for s in already_processed if int(s.split('-')[1]) > 5000]
            else:
                subjects = [s for s in subject_list if int(s.split('-')[1]) < 5000]
                already_processed = [s for s in already_processed if int(s.split('-')[1]) < 5000]

            print(f"Number of subjects already processed subjects: {len(already_processed)}")
            
            if not subjects:
                logger.info("No subjects found that need processing.")
                continue

            # Uncomment this line to run the script with one subject
            # subjects = ["sub-1014"]
        
            # Initialize lists to track successful and failed subjects
            successful_subjects = []
            failed_subjects = []

            print(f"Number of subjects to process: {len(subjects)}")

            for subject in subjects:
                output_folder_sub = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}/{subject}/subcortical_t1_masks")

                # Create the output directory if it doesn't exist
                os.makedirs(output_folder_sub, exist_ok=True)

                # Set paths based on cohort
                if cohort == "bbhi":
                    # BBHI paths
                    aseg_file = Path(f"{reconall_folder}/{subject}_{ses}_run-01/mri/aseg.mgz")
                    reference_file = Path(f"{reconall_folder}/{subject}_{ses}_run-01/mri/T1.mgz")
                else:  # cohort == "bbhi senior"
                    aseg_file = Path(f"{reconall_folder}/{subject}_{ses}/mri/aseg.mgz")
                    reference_file = Path(f"{reconall_folder}/{subject}_{ses}/mri/T1.mgz")

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


if __name__ == "__main__":
    main()
