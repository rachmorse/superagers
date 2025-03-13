#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np

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
        native_mask_output = Path(f"{out_dir}/{subject}/native_space_masks/schaefer_native_space_mask.nii.gz")
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

def transform_mni_to_t1w(mni_mask, t1w_brain, output_path, transform_mni_t1, transform_t1_mni):
    """
    Transform the MNI space mask to T1w space.

    Args:
        mni_mask (Path): Path to the MNI space mask.
        t1w_brain (Path): Path to the T1w brain image.
        output_path (Path): Path to save the transformed mask.
        transform_mni_t1 (Path): Path to save the MNI to T1w transformation matrix.
        transform_t1_mni (Path): Path to save the T1w to MNI transformation matrix.
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Path to standard MNI template in FSL
    fsl_dir = os.environ.get('FSLDIR', '/usr/local/fsl')
    mni_template = Path(fsl_dir) / 'data/standard/MNI152_T1_2mm_brain.nii.gz'
    
    if not mni_template.exists():
        raise FileNotFoundError(f"MNI template not found at {mni_template}. Please check your FSL installation.")
    
    # Create temporary directory for transforms in a location with guaranteed write access
    tmp_dir = Path("/tmp/fsl_transforms")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    tmp_t1_to_mni = tmp_dir / "tmp_t1_to_mni.mat"
    tmp_mni_to_t1 = tmp_dir / "tmp_mni_to_t1.mat"
    
    # Run FLIRT to register T1 to MNI
    print(f"Registering T1w brain to MNI space...")
    cmd1 = f"flirt -in {t1w_brain} -ref {mni_template} -omat {tmp_t1_to_mni} -dof 12"
    subprocess.run(cmd1, shell=True, check=True)
    
    # Now, invert the matrix to get MNI->T1 transform
    cmd_convert = f"convert_xfm -omat {tmp_mni_to_t1} -inverse {tmp_t1_to_mni}"
    subprocess.run(cmd_convert, shell=True, check=True)
    
    # Ensure the final transform directory exists with proper permissions
    transform_dir = transform_mni_t1.parent
    transform_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(transform_dir, 0o755)  # rwxr-xr-x permissions
    
    # Copy the temporary transforms to the final locations
    try:
        # Read the transform files as binary
        with open(tmp_t1_to_mni, 'rb') as f:
            t1_to_mni_data = f.read()
        with open(tmp_mni_to_t1, 'rb') as f:
            mni_to_t1_data = f.read()
            
        # Write to the final locations
        with open(transform_t1_mni, 'wb') as f:
            f.write(t1_to_mni_data)
        with open(transform_mni_t1, 'wb') as f:
            f.write(mni_to_t1_data)
            
        print(f"Created MNI to T1 transformation matrix: {transform_mni_t1}")
    except Exception as e:
        print(f"Error copying transform matrices: {e}")
        # Use the temporary transforms as fallback
        transform_t1_mni = tmp_t1_to_mni
        transform_mni_t1 = tmp_mni_to_t1
    
    # Apply the transformation matrix to the MNI mask
    print(f"Transforming MNI mask to T1w space...")
    cmd2 = (
        f"flirt -in {mni_mask} -ref {t1w_brain} "
        f"-out {output_path} -applyxfm -init {transform_mni_t1} -interp nearestneighbour"
    )
    subprocess.run(cmd2, shell=True, check=True)
    
    print(f"Transformed MNI mask to T1w space: {output_path}")
    
    return {
        "transform_matrix": transform_mni_t1,
        "transformed_mask": output_path
    }

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

def main():
    # Set timepoint
    timepoint = "tp1"
    ses = "ses-01"

    # Set up paths
    dwi_root_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/DWI_dtifit_{timepoint}")
    mni_mask = Path("/home/rachel/Desktop/schaefer_analysis/timeseries_data/combined_schaefer_harvard_subcortical_atlas.nii.gz")
    out_dir = Path("/home/rachel/Desktop/schaefer_analysis/dwi_analysis")
    anat_dir = Path("/pool/guttmann/institut/BBHI/MRI/BIDS")

    # Set up FSL so it runs correctly in this script
    os.environ["FSLDIR"] = "/home/rachel/fsl"
    os.environ["PATH"] = f"{os.environ['FSLDIR']}/bin:" + os.environ["PATH"]
    subprocess.run(["bash", "-c", "source /home/rachel/fsl/etc/fslconf/fsl.sh"], check=True)

    # Set FSL to output compressed NIFTI files
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    
    # Generate the list of subjects to process
    # subjects = get_subjects_to_process(dwi_root_dir, out_dir, ses)

    # Run a test on a single subject
    subjects = ["sub-55772"]
    
    # Process each subject
    for subject in subjects:
        subject_dir = dwi_root_dir / subject
        print(f"\nProcessing {subject}...")
        
        # Define paths for this subject
        eddy_corrected = subject_dir / "eddy_corrected_data.nii.gz"
        t1w_brain = anat_dir / f"{subject}/{ses}/anat/{subject}_{ses}_run-01_T1w.nii.gz"
        t1w_to_native_matrix = subject_dir / "T1w2SBdMRI"
        
        # Define output paths
        b0_output = out_dir / f"{ses}/b0/{subject}_{ses}_b0.nii.gz"
        t1w_mask_output = out_dir / f"{ses}/t1w_masks/{subject}_{ses}_schaefer_t1w_space_mask.nii.gz"
        transform_mni_t1 = out_dir / f"{ses}/transforms/{subject}_{ses}_mni_to_t1.mat"
        transform_t1_mni = out_dir / f"{ses}/transforms/{subject}_{ses}_t1_to_mni.mat"
        native_mask_output = out_dir / f"{ses}/native_space_masks/{subject}_{ses}_schaefer_native_space_mask.nii.gz"
        
        # Check if required files exist
        if not t1w_brain.exists():
            print(f"WARNING: T1w_brain.nii.gz not found for {subject}, skipping.")
            continue
            
        if not t1w_to_native_matrix.exists():
            print(f"WARNING: T1w2SBdMRI not found for {subject}, skipping.")
            continue
        
        # Step 1: Extract b0 from eddy corrected data if needed
        if not b0_output.exists():
            extract_b0(eddy_corrected, b0_output)
        
        # Step 2: Transform MNI mask to T1w space
        transform_mni_to_t1w(mni_mask, t1w_brain, t1w_mask_output, transform_mni_t1, transform_t1_mni)
        
        # Step 3: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_native(t1w_mask_output, t1w_to_native_matrix, b0_output, native_mask_output)
        
        print(f"Successfully created native space mask for {subject}")

if __name__ == "__main__":
    main()