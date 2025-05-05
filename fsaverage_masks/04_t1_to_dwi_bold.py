#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import Normalize12, NewSegment
from nipype.interfaces.spm.preprocess import ApplyDeformations 
import nipype.interfaces.spm.utils as spmu

def get_subjects_to_process(dwi_bbhi_dir, dwi_bbhi_senior_dir, out_dir, ses, cohort):
    """Generate a list of subjects to process based on whether they have
    T1 Schaefer / subcortical mask created and DWI data and do not already 
    have a DWI space mask.

    Args:
        dwi_bbhi_dir (Path): Path to the directory containing the DWI data for BBHI cohort.
        dwi_bbhi_senior_dir (Path): Path to the directory containing the DWI data for BBHI senior cohort.
        out_dir (Path): Path to the output directory where the native space masks are saved.
        ses (str): Timepoint (format ses-01).
        cohort (str): Cohort identifier (e.g., "bbhi" or "bbhi senior").
    """
    subjects_to_process = []
    already_processed = []

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(out_dir):
        if not subject_dir.startswith("sub-"):
            continue
        subject = subject_dir

        # Set paths based on cohort and timepoint (as they vary)
        if cohort == "bbhi":
            dwi_root_dir = Path(f"{dwi_bbhi_dir}/{subject}_{ses}")
        else:  # cohort == "bbhi senior"
            if ses == "ses-01":
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}")
            else:  # ses == "ses-02"
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}_{ses}")

        # Check if the required files exist
        schaefer_subcort_atlas = Path(f"{out_dir}/{subject}/subcortical_t1_masks/{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz")
        eddy_corrected = Path(f"{dwi_root_dir}/eddy_corrected_data.nii.gz")
        dwi_mask_output = Path(f"{out_dir}/{subject}/dwi_space_masks/{subject}_{ses}_schaefer200_subcortical14_dwi_space.nii.gz")

        if schaefer_subcort_atlas.exists() and eddy_corrected.exists() and not dwi_mask_output.exists():
            subjects_to_process.append(subject)
        elif dwi_mask_output.exists():
            already_processed.append(subject)

    return subjects_to_process, already_processed


def extract_b0(input_path, output_path):
    """Extract the b0 volume from the eddy-corrected data.
    
    Args:
        input_path (Path): Path to the eddy-corrected data.
        output_path (Path): Path where to save the extracted b0 volume.
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Using FSL's fslroi to extract the first volume (b0)
    cmd = f"fslroi {input_path} {output_path} 0 1"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Extracted b0 volume to {output_path}")


def transform_t1w_to_dwi(t1w_mask, b0_ref, output_path, out_native_masks):
    """Transform the T1w space mask to native space using the transformation matrix.
    
    Args:
        t1w_mask (Path): Path to the T1w space mask.
        b0_ref (Path): Path to the b0 reference image.
        output_path (Path): Path to save the native space mask.
        out_native_masks (Path): Output directory to where the T1 mask is.
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define the transformation matrix path
    transform_mat = out_native_masks / "T1w_to_b0.mat"  

    # Step 1: Generate transformation matrix directly
    subprocess.run(f"flirt -in {t1w_mask} -ref {b0_ref} -omat {transform_mat}", shell=True, check=True)

    # Step 2: Apply transformation to mask
    subprocess.run(f"flirt -in {t1w_mask} -ref {b0_ref} -applyxfm -init {transform_mat} -out {output_path} -paddingsize 0.0 -interp nearestneighbour", shell=True, check=True)

    print(f"Transformed T1w mask to DWI space: {output_path}")
    

def transform_t1w_to_bold(t1w_mask, bold_path, out_bold_masks, output_path_bold):
    """Transform the T1w space mask to native space using the transformation matrix.
    
    Args:
        t1w_mask (Path): Path to the T1w space mask.
        bold_path (Path): Path to the BOLD image.
        out_bold_masks (Path): Output directory to where the matrix and mask are saved.
        output_path_bold (Path): Path to save the BOLD space mask.
    """
    # Create output directory if it doesn't exist
    output_path_bold.parent.mkdir(parents=True, exist_ok=True)

    # Check if the BOLD file exists
    if not bold_path.exists():
        print(f"ERROR: BOLD file not found: {bold_path}")
        raise FileNotFoundError(f"BOLD file not found: {bold_path}")
    
    # Extract the first volume from the 4D BOLD dataset
    bold_ref_path = out_bold_masks / "bold_reference.nii.gz"
    subprocess.run(f"fslroi {bold_path} {bold_ref_path} 0 1", shell=True, check=True)

    # Verify the reference file was created
    if not bold_ref_path.exists():
        print(f"ERROR: Failed to create BOLD reference: {bold_ref_path}")
        raise FileNotFoundError(f"BOLD reference not created: {bold_ref_path}")
    
    # Define the transformation matrix path
    transform_mat = out_bold_masks / "T1w_to_bold.mat"  

    # Step 1: Generate transformation matrix using the single-volume reference
    subprocess.run(f"flirt -in {t1w_mask} -ref {bold_ref_path} -omat {transform_mat}", 
                   shell=True, check=True)

    # Verify the transformation matrix was created
    if not transform_mat.exists():
        print(f"ERROR: Failed to create transformation matrix: {transform_mat}")
        raise FileNotFoundError(f"Transformation matrix not created: {transform_mat}")
    
    # Step 2: Apply transformation to mask
    subprocess.run(f"flirt -in {t1w_mask} -ref {bold_ref_path} -applyxfm -init {transform_mat} "
                   f"-out {output_path_bold} -paddingsize 0.0 -interp nearestneighbour", 
                   shell=True, check=True)
    
    # Verify the output file was created
    if not output_path_bold.exists():
        print(f"ERROR: Failed to create BOLD space mask: {output_path_bold}")
        raise FileNotFoundError(f"BOLD space mask not created: {output_path_bold}")

    print(f"Transformed T1w mask to BOLD space: {output_path_bold}")


def process_subject(subject, dwi_root_dir, bold_root_dir, out_dir, ses, cohort):
    """
    Process a single subject's DWI data.
    
    Args:
        subject: The subject ID
        dwi_root_dir: Root directory for DWI data
        bold_root_dir: Root directory for BOLD data
        out_dir: Output directory
        ses: Session identifier
        cohort: Cohort identifier (e.g., "bbhi" or "bbhi senior")
        
    Returns:
        subject ID if processing was successful, None otherwise
    """
    out_subject_dir = out_dir / subject
    print(f"\nProcessing {subject}...")
    
    # Define output directories
    out_b0 = out_subject_dir / f"b0"
    # out_t1_masks =  out_subject_dir / f"t1w_masks"
    out_native_masks =  out_subject_dir / f"dwi_space_masks"
    out_bold_masks =  out_subject_dir / f"bold_space_masks"
    
    # Define paths for this subject
    eddy_corrected = Path(f"{dwi_root_dir}/eddy_corrected_data.nii.gz")
    
    # Define output paths and file names
    b0_output = out_b0 / f"{subject}_{ses}_b0.nii.gz"
    t1w_mask = out_subject_dir / f"subcortical_t1_masks/{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz"
    output_path_dwi = out_native_masks / f"{subject}_{ses}_schaefer200_subcortical14_dwi_space.nii.gz"
    output_path_bold = out_bold_masks / f"{subject}_{ses}_schaefer200_subcortical14_bold_space.nii.gz"

    # Define BOLD paths
    if cohort == "bbhi":
        bold_path = bold_root_dir / f"{subject}/native_T1/{subject}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz"
    else:  # cohort == "bbhi senior"
        bold_path = bold_root_dir / f"{subject}/{ses}/native_T1/{subject}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz"
    
    try:
        # Step 1: Extract b0 from eddy corrected data
        if not b0_output.exists():
            extract_b0(eddy_corrected, b0_output)

        # Step 2: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_dwi(t1w_mask, b0_output, output_path_dwi, out_native_masks)

        # Step 3: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_bold(t1w_mask, bold_path, out_bold_masks, output_path_bold)

        print(f"Successfully created native space mask for {subject}")

        # Step 4: Clean up individual subject's intermediate files
        if out_b0.exists():
            shutil.rmtree(out_b0)
        
        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None

def main():
    """
    Main function to process transform the Schaefer/
    Harvard-Oxford MNI mask to native space.
    """
    
    # Set up parameters
    timepoint = "1"
    ses = "ses-01"
    cohort = "bbhi"

    sys.stdout.flush() 
    print("-----------------------Running 04_t1_to_dwi_bold.py-----------------------")

    out_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")

    # Set up FSL so it runs correctly in this script
    os.environ["FSLDIR"] = "/home/rachel/fsl"
    os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
    subprocess.run(["bash", "-c", "source /home/rachel/fsl/etc/fslconf/fsl.sh"], check=True)

    # Set FSL to output compressed NIFTI files
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    dwi_bbhi_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp{timepoint}")
    dwi_bbhi_senior_dir = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/DTIFIT_TP{timepoint}")

    # Generate the list of subjects to process
    subjects, already_processed = get_subjects_to_process(dwi_bbhi_dir, dwi_bbhi_senior_dir, out_dir, ses, cohort)

    # Uncomment the following line to process one subject
    # subjects = ["sub-1014"] 
    
    # Process each subject
    results = []
    print(f"Number of subjects to process: {len(subjects)}")
    print(f"Already processed subjects: {len(already_processed)}")
    sys.stdout.flush() 

    # Check if there are subjects to process
    if not subjects:
        print("No subjects found that need processing.")
        return []

    for subject in subjects:

        # Set paths based on cohort
        if cohort == "bbhi":
            # BBHI paths
            dwi_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp{timepoint}/{subject}_{ses}")
            if timepoint == "2":
                bold_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp{timepoint}")
            else:  # timepoint == "1"
                bold_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed")
        else:  # cohort == "bbhi senior"
            bold_root_dir = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed")
            if ses == "ses-01":
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}")
            else:  # ses == "ses-02"
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}_{ses}")
        
        result = process_subject(subject, dwi_root_dir, bold_root_dir, out_dir, ses, cohort)

        if result:
            print(f"Successfully processed {subject}")
            results.append(result)
        else:
            print(f"Failed to process {subject}")
            
    print(f"Successfully processed {len(results)} subjects")
    print(f"Failed to process {len(subjects) - len(results)} subjects")
    print(f"Failed subjects: {set(subjects) - set(results)}")
    return results

if __name__ == "__main__":
    main()