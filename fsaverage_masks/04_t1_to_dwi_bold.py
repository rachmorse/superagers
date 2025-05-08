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

def get_subjects_to_process(dwi_bbhi_dir, dwi_bbhi_senior_dir, out_dir, ses, timepoint, cohort):
    """Generate a list of subjects to process based on whether they have
    T1 Schaefer / subcortical mask created and DWI data and do not already 
    have a DWI space mask.

    Args:
        dwi_bbhi_dir (Path): Path to the directory containing the DWI data for BBHI cohort.
        dwi_bbhi_senior_dir (Path): Path to the directory containing the DWI data for BBHI senior cohort.
        out_dir (Path): Path to the output directory where the native space masks are saved.
        ses (str): Timepoint (format ses-01).
        timepoint (str): Timepoint (format 1 or 2).
        cohort (str): Cohort identifier (e.g., "bbhi" or "bbhi senior").
    """
    already_processed_dwi = []
    already_processed_bold = []
    already_processed = []
    missing_files = []
    subjects_for_dwi = []
    subjects_for_bold = []

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(out_dir):
        if not subject_dir.startswith("sub-"):
            continue
        subject = subject_dir

        # If cohort is BBHI, filter subject ids to only include those > 6000
        if cohort == "bbhi" and int(subject.split("-")[1]) < 6000:
            continue  
        if cohort == "bbhi senior" and int(subject.split("-")[1]) > 6000:
            continue

        # Set paths based on cohort and timepoint (as they vary)
        if cohort == "bbhi":
            dwi_root_dir = Path(f"{dwi_bbhi_dir}/{subject}")
            if timepoint == "2":
                bold_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp{timepoint}/{subject}/native_T1")
            else:  # timepoint == "1"
                bold_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed/{subject}/native_T1")
        else:  # cohort == "bbhi senior"
            bold_root_dir = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed/{subject}/{ses}/native_T1")
            if ses == "ses-01":
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}")
            else:  # ses == "ses-02"
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}_{ses}")

        # Check if the required files exist
        schaefer_subcort_atlas = Path(f"{out_dir}/{subject}/subcortical_t1_masks/{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz")
        bold_data = Path(f"{bold_root_dir}/{subject}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz")
        eddy_corrected = Path(f"{dwi_root_dir}/eddy_corrected_data.nii.gz")
        dwi_mask_output = Path(f"{out_dir}/{subject}/dwi_space_masks/{subject}_{ses}_schaefer200_subcortical14_dwi_space.nii.gz")
        bold_mask_output = Path(f"{out_dir}/{subject}/bold_space_masks/{subject}_{ses}_schaefer200_subcortical14_bold_space.nii.gz")

        # Check files separately to diagnose issues
        has_atlas = schaefer_subcort_atlas.exists()
        has_eddy = eddy_corrected.exists()
        has_bold = bold_data.exists()
        has_dwi_mask = dwi_mask_output.exists()
        has_bold_mask = bold_mask_output.exists()   

        # Check if we need to process DWI and/or BOLD
        needs_dwi_processing = has_atlas and has_eddy and not has_dwi_mask
        needs_bold_processing = has_atlas and has_bold and not has_bold_mask

        # Categorize subjects
        if needs_dwi_processing:
            subjects_for_dwi.append(subject)      
        if needs_bold_processing:
            subjects_for_bold.append(subject)
        if has_dwi_mask:
            already_processed_dwi.append(subject)
        if has_bold_mask:
            already_processed_bold.append(subject)
        if has_dwi_mask and has_bold_mask:
            already_processed.append(subject)

        if not has_atlas:
            missing_files.append(subject)
            print(f"Subject {subject} missing atlas file")
        elif not has_eddy or not has_bold:
            missing_files.append(subject)
            print(f"Subject {subject} missing both eddy and BOLD data")
        elif not has_eddy:
            print(f"Subject {subject} missing eddy data")
        elif not has_bold:
            print(f"Subject {subject} missing BOLD data")

    print(f"Subjects already with DWI mask: {len(already_processed_dwi)}")
    print(f"Subjects already with BOLD mask: {len(already_processed_bold)}")
    print(f"Subjects already fully processed: {len(already_processed)}")
    print(f"Subjects missing required files: {len(missing_files)}")     

    return subjects_for_dwi, subjects_for_bold, already_processed


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


def process_subject_dwi(subject, dwi_root_dir, out_dir, ses, cohort):
    """
    Process a single subject's DWI data.
    
    Args:
        subject: The subject ID
        dwi_root_dir: Root directory for DWI data
        out_dir: Output directory
        ses: Session identifier
        cohort: Cohort identifier (e.g., "bbhi" or "bbhi senior")
        
    Returns:
        subject ID if processing was successful, None otherwise
    """
    out_subject_dir = out_dir / subject
    print(f"\nProcessing DWI for {subject}...")
    
    # Define output directories
    out_b0 = out_subject_dir / f"b0"
    out_native_masks =  out_subject_dir / f"dwi_space_masks"
    
    # Define paths for this subject
    eddy_corrected = Path(f"{dwi_root_dir}/eddy_corrected_data.nii.gz")
    
    # Define output paths and file names
    b0_output = out_b0 / f"{subject}_{ses}_b0.nii.gz"
    t1w_mask = out_subject_dir / f"subcortical_t1_masks/{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz"
    output_path_dwi = out_native_masks / f"{subject}_{ses}_schaefer200_subcortical14_dwi_space.nii.gz"

    try:
        # Step 1: Extract b0 from eddy corrected data
        if not b0_output.exists():
            extract_b0(eddy_corrected, b0_output)

        # Step 2: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_dwi(t1w_mask, b0_output, output_path_dwi, out_native_masks)

        print(f"Successfully created native space DWI mask for {subject}")

        # Step 4: Clean up individual subject's intermediate files
        if out_b0.exists():
            shutil.rmtree(out_b0)
        
        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None

def process_subject_bold(subject, bold_root_dir, out_dir, ses, cohort):
    """
    Process a single subject's DWI data.
    
    Args:
        subject: The subject ID
        bold_root_dir: Root directory for BOLD data
        out_dir: Output directory
        ses: Session identifier
        cohort: Cohort identifier (e.g., "bbhi" or "bbhi senior")
        
    Returns:
        subject ID if processing was successful, None otherwise
    """
    out_subject_dir = out_dir / subject
    print(f"\nProcessing BOLD for {subject}...")
    
    # Define output directories
    out_bold_masks =  out_subject_dir / f"bold_space_masks"
    t1w_mask = out_subject_dir / f"subcortical_t1_masks/{subject}_{ses}_schaefer200_subcortical14_t1_space.nii.gz"
    output_path_bold = out_bold_masks / f"{subject}_{ses}_schaefer200_subcortical14_bold_space.nii.gz"

    # Define BOLD paths
    if cohort == "bbhi":
        bold_path = bold_root_dir / f"{subject}/native_T1/{subject}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz"
    else:  # cohort == "bbhi senior"
        bold_path = bold_root_dir / f"{subject}/{ses}/native_T1/{subject}_{ses}_run-01_rest_bold_ap_T1-space.nii.gz"

    try:
        # Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_bold(t1w_mask, bold_path, out_bold_masks, output_path_bold)

        print(f"Successfully created native space BOLD mask for {subject}")

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
    subjects_for_dwi, subjects_for_bold, already_processed = get_subjects_to_process(dwi_bbhi_dir, dwi_bbhi_senior_dir, out_dir, ses, timepoint, cohort)

    # Uncomment the following line to process one subject
    # subjects_for_dwi = ["sub-1014"] 
    # subjects_for_bold = ["sub-1014"] 
    
    # Process each subject
    result_dwi = []
    result_bold = []

    print(f"Subjects needing DWI processing: {len(subjects_for_dwi)}")
    print(f"Subjects needing BOLD processing: {len(subjects_for_bold)}")
    print(f"Already processed subjects: {len(already_processed)}")
    # sys.stdout.flush() 

    # Check if there are subjects to process
    if not subjects_for_dwi and not subjects_for_bold:
        print("No subjects found that need processing.")
        return []

    for subject in subjects_for_dwi:

        # Set paths based on cohort to run DWI 
        if cohort == "bbhi":
            dwi_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_tp{timepoint}/{subject}")
        else:  # cohort == "bbhi senior"
            if ses == "ses-01":
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}")
            else:  # ses == "ses-02"
                dwi_root_dir = Path(f"{dwi_bbhi_senior_dir}/{subject}_{ses}")
        
        subject_result = process_subject_dwi(subject, dwi_root_dir, out_dir, ses, cohort)

        if subject_result:
            print(f"Successfully processed DWI for {subject}")
            result_dwi.append(subject)
        else:
            print(f"Failed to process DWI for {subject}")

    # Then run BOLD processing
    for subject in subjects_for_bold:
        if cohort == "bbhi":
            if timepoint == "2":
                bold_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed_tp{timepoint}")
            else:  # timepoint == "1"
                bold_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/fMRI-preprocessed")
        else:  # cohort == "bbhi senior"
            bold_root_dir = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/resting_preprocessed")
        
        subject_result = process_subject_bold(subject, bold_root_dir, out_dir, ses, cohort)

        if subject_result:
            print(f"Successfully processed BOLD for {subject}")
            result_bold.append(subject)
        else:
            print(f"Failed to process BOLD for {subject}")
            
    # Summary statistics
    print(f"Successfully processed DWI for {len(result_dwi)} subjects")
    print(f"Successfully processed BOLD for {len(result_bold)} subjects")
    print(f"Total subjects with successful processing: {len(set(result_dwi + result_bold))}")

    # Calculate failures
    failed_dwi = [s for s in subjects_for_dwi if s not in result_dwi]
    failed_bold = [s for s in subjects_for_bold if s not in result_bold]

    print(f"Failed to process DWI for {len(failed_dwi)} subjects")
    print(f"Failed to process BOLD for {len(failed_bold)} subjects")

    if failed_dwi:
        print(f"Failed DWI subjects: {failed_dwi}")
    if failed_bold:
        print(f"Failed BOLD subjects: {failed_bold}")

    # Return both result lists
    return result_dwi, result_bold

if __name__ == "__main__":
    main()