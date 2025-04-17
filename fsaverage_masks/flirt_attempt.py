#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np
import shutil
import gzip
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import Normalize12, NewSegment
from nipype.interfaces.spm.preprocess import ApplyDeformations 
import nipype.interfaces.spm.utils as spmu

def get_subjects_to_process(root_directory, out_dir, ses):
    """Generate a list of subjects to process based on whether they have
    T1w_brain.nii.gz and have their timeseries data and do not already 
    have a native space mask.

    Args:
        root_directory (Path): Path to the root directory containing the DWI data.
        out_dir (Path): Path to the output directory where the native space masks are saved.
        ses (str): Timepoint (format ses-01).
    """
    subjects_to_process = []

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(root_directory):
        if not subject_dir.startswith("sub-"):
            continue
        subject = subject_dir

        # Check if the required files exist
        t1w_brain = Path(f"{root_directory}/{subject}/T1w_brain.nii.gz")
        native_mask_output = Path(f"{out_dir}/{ses}/native_space_masks/{subject}_{ses}_schaefer_oxford_native_space_mask.nii.gz")
        timeseries_data = Path(f"/home/rachel/Desktop/schaefer_analysis/timeseries_data/{ses}/{subject}_{ses}_schaefer200_timeseries.csv")

        if t1w_brain.exists() and timeseries_data.exists() and not native_mask_output.exists():
            subjects_to_process.append(subject)

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process


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


def transform_t1w_to_native(t1w_mask, t1w_to_native_matrix, b0_ref, output_path):
    """Transform the T1w space mask to native space using the transformation matrix.
    
    Args:
        t1w_mask (Path): Path to the T1w space mask.
        t1w_to_native_matrix (Path): Path to the T1w to native space transformation matrix.
        b0_ref (Path): Path to the b0 reference image.
        output_path (Path): Path to save the native space mask.
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use FSL's flirt to apply the transformation matrix
    cmd = f"flirt -in {t1w_mask} -ref {b0_ref} -out {output_path} -applyxfm -init {t1w_to_native_matrix} -interp nearestneighbour"
    subprocess.run(cmd, shell=True, check=True)
    print(f"Transformed T1w mask to native space: {output_path}")


def process_subject(subject, dwi_root_dir, anat_dir, mni_mask, out_dir, ses, subject_id):
    """
    Process a single subject's DWI data.
    
    Args:
        subject: The subject ID
        dwi_root_dir: Root directory for DWI data
        anat_dir: Directory containing anatomical data
        mni_mask: Path to the MNI mask
        out_dir: Output directory
        ses: Session identifier
        
    Returns:
        subject ID if processing was successful, None otherwise
    """
    subject_dir = dwi_root_dir / subject
    print(f"\nProcessing {subject}...")
    
    # Define output directories
    out_b0 = out_dir / f"{ses}/b0"
    out_t1_masks = Path(f"{out_dir}/{ses}/t1w_masks")
    out_native_masks = Path(f"{out_dir}/{ses}/native_space_masks")
    
    # Define paths for this subject
    eddy_corrected = subject_dir / "eddy_corrected_data.nii.gz"
    t1w_brain = anat_dir / f"{subject}/{ses}/anat/{subject}_{ses}_run-01_T1w.nii.gz"
    t1w_to_native_matrix = subject_dir / "T1w2SBdMRI"
    
    # Define output paths and file names
    b0_output = out_b0 / f"{subject}_{ses}_b0.nii.gz"
    t1w_mask_output = out_t1_masks / f"{subject}_{ses}_schaefer_oxford_t1w_space_mask.nii.gz"
    mni_mask_reoriented = out_t1_masks / f"{subject}_{ses}_reoriented_schaefer_oxford_mni_mask.nii.gz"
    transforms_dir = out_t1_masks / "transforms"
    transform_mni_t1 = transforms_dir / f"{subject}_{ses}_mni_to_t1.mat"
    transform_t1_mni = transforms_dir / f"{subject}_{ses}_t1_to_mni.mat"
    native_mask_output = out_native_masks / f"{subject}_{ses}_schaefer_oxford_native_space_mask.nii.gz"
    
    # Check if required files exist
    if not t1w_brain.exists():
        print(f"WARNING: T1w_brain.nii.gz not found for {subject}, skipping.")
        return None
        
    if not t1w_to_native_matrix.exists():
        print(f"WARNING: T1w2SBdMRI not found for {subject}, skipping.")
        return None
    
    try:
        # Step 1: Extract b0 from eddy corrected data
        if not b0_output.exists():
            extract_b0(eddy_corrected, b0_output)
            
        # Step 3: Transform T1w mask to native space using the transformation matrix
        # transform_t1w_to_native(t1w_mask_output, t1w_to_native_matrix, b0_output, native_mask_output)

        print(f"Successfully created native space mask for {subject}")

        # Step 4: Clean up individual subject's intermediate files
        # if out_b0.exists():
        #     shutil.rmtree(out_b0)
        # if out_t1_masks.exists():
        #     shutil.rmtree(out_t1_masks)
        
        return subject
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        return None

def main():
    """
    Main function to process transform the Schaefer/
    Harvard-Oxford MNI mask to native space.
    """
    
    # Set timepoint
    timepoint = "tp1"
    ses = "ses-01"

    # Set up paths
    dwi_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_{timepoint}")
    mni_mask = Path("/home/rachel/Desktop/schaefer_analysis/timeseries_data/combined_schaefer_harvard_subcortical_atlas.nii.gz")
    anat_dir = Path("/pool/guttmann/institut/BBHI/MRI/BIDS")
    out_dir = Path("/home/rachel/Desktop/schaefer_analysis/structural_masks")

    # Set up FSL so it runs correctly in this script
    os.environ["FSLDIR"] = "/home/rachel/fsl"
    os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
    subprocess.run(["bash", "-c", "source /home/rachel/fsl/etc/fslconf/fsl.sh"], check=True)

    # Set FSL to output compressed NIFTI files
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    # Configure MATLAB and SPM so they run correctly in this script
    mlab_cmd = "/usr/local/bin/matlab -nodesktop -nosplash"
    spm.SPMCommand.set_mlab_paths(paths="/home/rachel/spm12", matlab_cmd=mlab_cmd)
    
    # Generate the list of subjects to process
    # subjects = get_subjects_to_process(dwi_root_dir, out_dir, ses)

    # Process problematic subjects again manually
    subjects = ["sub-120927"]
    # "sub-46808", 
                    # "sub-120927", "sub-101848", "sub-154095", "sub-139350", 
                    # "sub-134038", "sub-153265", "sub-141692", "sub-178055", "sub-182146", 
                    # "sub-116054", "sub-163261"]    
    
    # Process each subject
    results = []
    for subject in subjects:
        result = process_subject(subject, dwi_root_dir, anat_dir, mni_mask, out_dir, ses, subject)
        if result:
            results.append(result)
            
    print(f"Successfully processed {len(results)} subjects")
    return results

if __name__ == "__main__":
    main()