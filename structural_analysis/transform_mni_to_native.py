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

##########################################################
# NOTE TO SELF THAT THE FLIRT AND SPM METHODS DO NOT WORK
# FIRST TRY TO GET FLIRT TO WORK if it is not working. FLIRT SAYS THAT ITS WORKING BUT THE ORIENTATIONS ARE DIFFERENT IN FSLEYES unclear what the implecations are
# THEN TRY TO GET SPM TO WORK. SPM IS NOT RUNNING AT ALL.
##########################################################

def transform_mni_to_t1w(mni_mask, t1w_brain, output_file, transform_mni_t1, transform_t1_mni, out_t1_masks):
    """
    Transform the MNI space mask to T1w space with proper resampling.
    """
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Path to standard MNI template in FSL
    fsl_dir = os.environ.get('FSLDIR', '/usr/local/fsl')
    # Create a 0.8mm version from the 1mm template if needed
    mni_template_original = Path(fsl_dir) / 'data/standard/MNI152_T1_1mm_brain.nii.gz'
    mni_template_08mm = Path(tmp_dir) / 'MNI152_T1_0.8mm_brain.nii.gz'

    # Resample to 0.8mm resolution
    cmd_resample = f"flirt -in {mni_template_original} -ref {mni_template_original} -out {mni_template_08mm} -applyisoxfm 0.8"
    subprocess.run(cmd_resample, shell=True, check=True)

    # Then use this resampled template
    mni_template = mni_template_08mm
    
    # Create temporary directory for transforms 
    tmp_dir = Path(out_t1_masks / "transforms")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 2: Run FLIRT to register downsampled T1 to MNI
    cmd1 = f"flirt -in {t1w_brain} -ref {mni_template} -omat {transform_t1_mni} -dof 12"
    subprocess.run(cmd1, shell=True, check=True)
    
    # STEP 3: Invert the matrix to get MNI->T1 transform
    cmd_convert = f"convert_xfm -omat {transform_mni_t1} -inverse {transform_t1_mni}"
    subprocess.run(cmd_convert, shell=True, check=True)
    
    # STEP 4: Apply the transformation to the MNI mask
    # Option A: Transform to high-res T1w space directly (may cause interpolation artifacts)
    cmd2 = (
        f"flirt -in {mni_mask} -ref {t1w_brain} "
        f"-out {output_file} -applyxfm -init {transform_mni_t1} -interp nearestneighbour"
    )
    subprocess.run(cmd2, shell=True, check=True)
    
    print(f"Transformed MNI Schaefer/Harvard-Oxford mask to T1w space: {output_file}")
    
    return {
        "transform_matrix": transform_mni_t1,
        "transformed_mask": output_file
    }

# def transform_mni_to_t1w(mni_mask, t1w_brain, output_file, out_t1_masks, subject, ses):
#     """
#     Transform the MNI space mask to T1w space using SPM.

#     Args:
#         mni_mask (Path): Path to the MNI space mask.
#         t1w_brain (Path): Path to the T1w brain image.
#         output_file (Path): Path to save the transformed mask.
#         out_t1_masks (Path): Path to the output directory.
#         subject (str): Subject ID.
#         ses (str): Timepoint.
#     """        
#     # Create output directory if it doesn't exist
#     output_file.parent.mkdir(parents=True, exist_ok=True)
    
#     # Create temporary directory for intermediate files
#     tmp_dir = Path(out_t1_masks / "tmp")
#     tmp_dir.mkdir(parents=True, exist_ok=True)
    
#     # Function to unzip a gzipped file using Python's gzip module
#     def ungzip_file(gzipped_file, output_file):
#         try:
#             with gzip.open(gzipped_file, 'rb') as f_in:
#                 with open(output_file, 'wb') as f_out:
#                     f_out.write(f_in.read())
#             return True
#         except Exception as e:
#             print(f"Error unzipping file {gzipped_file}: {e}")
#             return False
    
#     # Unzip files for SPM (SPM requires uncompressed NIfTI files)
#     # Copy and unzip t1w_brain
#     t1w_copy = tmp_dir / t1w_brain.name
#     t1w_unzipped = Path(str(t1w_copy).replace('.gz', ''))
    
#     try:
#         shutil.copy(t1w_brain, t1w_copy)
#         if not ungzip_file(t1w_copy, t1w_unzipped):
#             raise Exception(f"Failed to unzip {t1w_copy}")
#     except Exception as e:
#         print(f"Error with T1w brain: {e}")
#         # Try direct unzipping without copying first
#         if not ungzip_file(t1w_brain, t1w_unzipped):
#             raise Exception(f"Could not process T1w brain file {t1w_brain}")
    
#     # Copy and unzip mni_mask
#     mni_mask_copy = tmp_dir / mni_mask.name
#     mni_mask_unzipped = Path(str(mni_mask_copy).replace('.gz', ''))
    
#     try:
#         shutil.copy(mni_mask, mni_mask_copy)
#         if not ungzip_file(mni_mask_copy, mni_mask_unzipped):
#             raise Exception(f"Failed to unzip {mni_mask_copy}")
#     except Exception as e:
#         print(f"Error with MNI mask: {e}")
#         # Try direct unzipping without copying first
#         if not ungzip_file(mni_mask, mni_mask_unzipped):
#             raise Exception(f"Could not process MNI mask file {mni_mask}")
    
#     # Verify files exist
#     if not t1w_unzipped.exists():
#         raise FileNotFoundError(f"Unzipped T1w file does not exist: {t1w_unzipped}")
#     if not mni_mask_unzipped.exists():
#         raise FileNotFoundError(f"Unzipped MNI mask does not exist: {mni_mask_unzipped}")
        
#     # Use SPM's Normalize12 and ApplyDeformations for a simpler approach
#     import nipype.interfaces.spm as spm
    
#     print(f"Setting up normalization from T1w to MNI space to invert...")
    
#     # 1. Setup normalization to estimate the deformation field
#     normalize = spm.Normalize12()
#     normalize.inputs.image_to_align = str(t1w_unzipped)
#     normalize.inputs.jobtype = 'est'  # Just estimate the deformation
    
#     print("Running normalization to compute deformation fields...")
#     norm_result = normalize.run()
    
#     # Get the deformation field (T1w -> MNI)
#     forward_def_field = norm_result.outputs.deformation_field
    
#     # 2. Invert the deformation field
#     print("Inverting the deformation field...")
#     invert_def = spmu.ApplyInverseDeformation()
#     invert_def.inputs.in_files = str(mni_mask_unzipped)
#     invert_def.inputs.deformation = forward_def_field
#     invert_def.inputs.target = str(t1w_unzipped) 
#     invert_result = invert_def.run()
    
#     # Get the inverted deformation field (MNI -> T1w)
#     inverse_def_field = invert_result.outputs.inverted_deformation
    
#     print(f"Applying inverse deformation to MNI mask...")
    
#     # 3. Apply the inverted deformation field to the MNI mask
#     apply_def = spm.preprocess.ApplyDeformations()
#     apply_def.inputs.in_files = str(mni_mask_unzipped)
#     apply_def.inputs.deformation_field = inverse_def_field
#     apply_def.inputs.reference_volume = str(t1w_unzipped)
#     apply_def.inputs.interp = 0  # Nearest neighbor interpolation for masks
    
#     print("Running ApplyDeformations to transform MNI mask to T1w space...")
#     apply_result = apply_def.run()
    
#     # Get the transformed mask
#     transformed_mask = Path(apply_result.outputs.out_files[0])
    
#     if transformed_mask.exists():
#         # Compress the result and move to the final output location
#         gzipped_mask = Path(f"{transformed_mask}.gz")
        
#         # Use Python's gzip to compress
#         try:
#             with open(transformed_mask, 'rb') as f_in:
#                 with gzip.open(gzipped_mask, 'wb') as f_out:
#                     f_out.write(f_in.read())
#         except Exception as e:
#             print(f"Error compressing output file: {e}")
#             # Try using command line gzip as fallback
#             try:
#                 subprocess.run(["gzip", "-f", str(transformed_mask)], check=True)
#             except Exception as e2:
#                 print(f"Command-line gzip also failed: {e2}")
        
#         # Move the gzipped mask to the final output location
#         if gzipped_mask.exists():
#             shutil.copy(gzipped_mask, output_file)
#             print(f"Transformed MNI mask to T1w space: {output_file}")
#         else:
#             print(f"WARNING: Expected gzipped mask at {gzipped_mask} was not found!")
#     else:
#         print("Transformation failed!")
    
#     # Clean up temporary files
#     try:
#         shutil.rmtree(tmp_dir)
#     except Exception as e:
#         print(f"Warning: Could not remove temp directory: {e}")
    
#     return {
#         "transformed_mask": output_file
#     }
    


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


def process_subject(subject, dwi_root_dir, anat_dir, mni_mask, out_dir, ses):
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
        
        # Step 2: Transform MNI mask to T1w space
        transform_mni_to_t1w(mni_mask, t1w_brain, t1w_mask_output, transform_mni_t1, transform_t1_mni, out_t1_masks)
        # transform_mni_to_t1w(mni_mask, t1w_brain, t1w_mask_output, out_t1_masks, subject, ses)

        # Step 3: Transform T1w mask to native space using the transformation matrix
        transform_t1w_to_native(t1w_mask_output, t1w_to_native_matrix, b0_output, native_mask_output)

        print(f"Successfully created native space mask for {subject}")

        # Step 4: Clean up individual subject's intermediate files
        if out_b0.exists():
            shutil.rmtree(out_b0)
        if out_t1_masks.exists():
            shutil.rmtree(out_t1_masks)
        
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
    subjects = ["sub-46808"]
    # subjects = ["sub-139895"]
                    # "sub-120927", "sub-101848", "sub-154095", "sub-139350", 
                    # "sub-134038", "sub-153265", "sub-141692", "sub-178055", "sub-182146", 
                    # "sub-116054", "sub-163261"]    
    
    # Process each subject
    results = []
    for subject in subjects:
        result = process_subject(subject, dwi_root_dir, anat_dir, mni_mask, out_dir, ses)
        if result:
            results.append(result)
            
    print(f"Successfully processed {len(results)} subjects")
    return results

if __name__ == "__main__":
    main()