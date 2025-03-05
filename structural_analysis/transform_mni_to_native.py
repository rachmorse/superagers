import os
import numpy as np
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.align import imaffine, transforms
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric
import datetime
import shutil
import subprocess

def extract_b0_from_data(dwi_path, bval_path, output_path, b0_threshold=50):
    """Extract and average b0 volumes from DWI data"""
    print("Extracting b0 images...")
    # Load the bvals to identify b0 volumes
    bvals, _ = read_bvals_bvecs(bval_path, None)
    b0_indices = np.where(bvals < b0_threshold)[0]
    
    print(f"Found {len(b0_indices)} b0 volumes at indices: {b0_indices}")
    
    if len(b0_indices) == 0:
        raise ValueError("No b0 volumes found in the data!")
    
    # Load the DWI data
    dwi_img = nib.load(dwi_path)
    dwi_data = dwi_img.get_fdata()
    
    # Extract all b0 volumes
    b0_vols = dwi_data[:,:,:,b0_indices]
    
    # Average the b0 volumes to increase SNR
    b0_avg = np.mean(b0_vols, axis=3)
    
    # Create and save the averaged b0 image
    b0_img = nib.Nifti1Image(b0_avg, dwi_img.affine, dwi_img.header)
    nib.save(b0_img, output_path)
    
    print(f"Averaged b0 image saved to: {output_path}")
    return output_path

def create_acquisition_parameters(ap_json_path, pa_json_path, output_path):
    """Create an acquisition parameters file for topup based on JSON files"""
    print("Creating acquisition parameters file for topup...")
    
    # Default parameters if JSON parsing fails
    ap_readout_time = 0.05  # Default value
    pa_readout_time = 0.05  # Default value
    
    # Try to extract from JSON files if available
    try:
        import json
        
        # Get AP parameters
        with open(ap_json_path, 'r') as f:
            ap_data = json.load(f)
            
        # Get PA parameters
        with open(pa_json_path, 'r') as f:
            pa_data = json.load(f)
            
        # Extract readout time and PE direction
        # Different scanners/sequences might store this differently, try common fields
        readout_fields = ['TotalReadoutTime', 'EffectiveEchoSpacing', 'EchoSpacing']
        
        for field in readout_fields:
            if field in ap_data:
                ap_readout_time = ap_data[field]
                break
                
        for field in readout_fields:
            if field in pa_data:
                pa_readout_time = pa_data[field]
                break
                
        # If we found EchoSpacing, need to convert to total readout time
        if field == 'EffectiveEchoSpacing' or field == 'EchoSpacing':
            # Need PE matrix size
            if 'AcquisitionMatrixPE' in ap_data:
                ap_readout_time *= (ap_data['AcquisitionMatrixPE'] - 1)
                
            if 'AcquisitionMatrixPE' in pa_data:
                pa_readout_time *= (pa_data['AcquisitionMatrixPE'] - 1)
    
    except Exception as e:
        print(f"Warning: Could not extract parameters from JSON: {e}")
        print("Using default parameters instead")
    
    # Create the acquisition parameters file for topup
    # Format: [x y z readout_time]
    # For AP: typically [0 1 0 <readout_time>]
    # For PA: typically [0 -1 0 <readout_time>]
    with open(output_path, 'w') as f:
        f.write(f"0 1 0 {ap_readout_time}\n")
        f.write(f"0 -1 0 {pa_readout_time}\n")
    
    print(f"Acquisition parameters file created: {output_path}")
    return output_path

def run_topup(ap_b0_path, pa_b0_path, acq_params_path, work_dir):
    """
    Run FSL's topup to correct susceptibility distortions
    
    Parameters
    ----------
    ap_b0_path : str
        Path to AP b0 image
    pa_b0_path : str
        Path to PA b0 image
    acq_params_path : str
        Path to acquisition parameters file
    work_dir : str
        Working directory
    
    Returns
    -------
    str
        Path to the corrected b0 image
    """
    print("Running FSL's topup for susceptibility distortion correction...")
    
    # Create paths for outputs
    merged_b0_path = os.path.join(work_dir, "merged_b0.nii.gz")
    topup_out_prefix = os.path.join(work_dir, "topup_results")
    corrected_b0_path = os.path.join(work_dir, "corrected_b0.nii.gz")
    
    # Merge the AP and PA b0 images
    print("Merging AP and PA b0 images...")
    ap_img = nib.load(ap_b0_path)
    pa_img = nib.load(pa_b0_path)
    
    merged_data = np.concatenate([
        ap_img.get_fdata()[..., np.newaxis],
        pa_img.get_fdata()[..., np.newaxis]
    ], axis=3)
    
    merged_img = nib.Nifti1Image(merged_data, ap_img.affine)
    nib.save(merged_img, merged_b0_path)
    
    # Run topup
    print("Running topup (this may take a while)...")
    topup_cmd = [
        "topup",
        "--imain=" + merged_b0_path,
        "--datain=" + acq_params_path,
        "--config=b02b0.cnf",  # Default config file
        "--out=" + topup_out_prefix,
        "--iout=" + corrected_b0_path,
        "--verbose"
    ]
    
    try:
        subprocess.run(topup_cmd, check=True)
        print("Topup completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running topup: {e}")
        print("Falling back to simple averaging method...")
        
        # If topup fails, fall back to simple averaging
        combined_data = (ap_img.get_fdata() + pa_img.get_fdata()) / 2.0
        combined_img = nib.Nifti1Image(combined_data, ap_img.affine)
        nib.save(combined_img, corrected_b0_path)
    except FileNotFoundError:
        print("Topup not found! Is FSL installed and in your PATH?")
        print("Falling back to simple averaging method...")
        
        # If topup is not installed, fall back to simple averaging
        combined_data = (ap_img.get_fdata() + pa_img.get_fdata()) / 2.0
        combined_img = nib.Nifti1Image(combined_data, ap_img.affine)
        nib.save(combined_img, corrected_b0_path)
    
    return corrected_b0_path

def transform_mask_to_dwi_native(mask_path, reference_dwi_path, output_path, intermediate_path=None):
    """
    Transform a mask from MNI space to native DWI space
    
    Parameters
    ----------
    mask_path : str
        Path to the mask in MNI space
    reference_dwi_path : str
        Path to reference image in native DWI space (e.g., combined b0)
    output_path : str
        Path to save transformed mask
    intermediate_path : str, optional
        Path to save intermediate files for QC
    """
    print("\nTransforming mask to DWI native space...")
    
    # Create directory for intermediate files if needed
    if intermediate_path:
        os.makedirs(intermediate_path, exist_ok=True)
    
    # Load images
    mask_img = nib.load(mask_path)
    ref_img = nib.load(reference_dwi_path)
    
    mask_data = mask_img.get_fdata()
    ref_data = ref_img.get_fdata()
    
    print("Computing affine registration...")
    # Initialize identity transform
    affine_map = imaffine.AffineMap(
        np.eye(4),
        ref_data.shape, ref_img.affine,
        mask_data.shape, mask_img.affine
    )
    
    # Set up affine registration with mutual information (good for cross-modal registration)
    affine_reg = imaffine.AffineRegistration(
        metric=imaffine.MutualInformationMetric(32),
        level_iters=[10000, 1000, 100],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1]
    )
    
    # Use center of mass for initial alignment
    transform = transforms.TranslationTransform3D()
    center_of_mass = imaffine.transform_centers_of_mass(
        ref_data, ref_img.affine,
        mask_data, mask_img.affine
    )
    
    # Optimize rigid transformation first
    rigid_transform = affine_reg.optimize(
        ref_data, mask_data, 
        transforms.RigidTransform3D(), None,
        center_of_mass.affine, affine_map.domain_grid_shape
    )
    
    if intermediate_path:
        # Save rigid-aligned result for QC
        rigid_aligned = rigid_transform.transform(mask_data)
        rigid_img = nib.Nifti1Image(rigid_aligned, ref_img.affine)
        nib.save(rigid_img, os.path.join(intermediate_path, "mask_rigid_aligned.nii.gz"))
    
    # Optimize affine transformation
    print("Refining with affine registration...")
    affine_transform = affine_reg.optimize(
        ref_data, mask_data,
        transforms.AffineTransform3D(), None,
        rigid_transform.affine, affine_map.domain_grid_shape
    )
    
    # Apply affine transformation to mask
    affine_aligned = affine_transform.transform(mask_data)
    
    if intermediate_path:
        # Save affine-aligned result for QC
        affine_img = nib.Nifti1Image(affine_aligned, ref_img.affine)
        nib.save(affine_img, os.path.join(intermediate_path, "mask_affine_aligned.nii.gz"))
    
    # For better accuracy, use non-linear registration
    print("Computing non-linear registration (this may take a while)...")
    metric = CCMetric(3)  # Cross-correlation metric works well for this purpose
    
    # Configure the non-linear registration
    sdr = SymmetricDiffeomorphicRegistration(
        metric, [100, 50, 25]  # Number of iterations at each level
    )
    
    # Perform non-linear registration
    mapping = sdr.optimize(ref_data, affine_aligned)
    
    # Apply both transformations (affine + non-linear)
    transformed_data = mapping.transform(affine_aligned)
    
    if intermediate_path:
        # Save the displacement field for QC
        disp_field = mapping.get_forward_field()
        disp_img = nib.Nifti1Image(disp_field, ref_img.affine)
        nib.save(disp_img, os.path.join(intermediate_path, "displacement_field.nii.gz"))
    
    # Since this is a combined atlas with multiple ROIs, preserve the ROI IDs
    # Round to nearest integer instead of simple thresholding
    final_data = np.round(transformed_data).astype(np.int16)
    
    # Create and save final transformed mask
    final_img = nib.Nifti1Image(final_data, ref_img.affine)
    nib.save(final_img, output_path)
    
    print(f"Transformed mask saved to: {output_path}")
    return output_path

def cleanup_intermediate_files(files_to_remove, directories_to_remove=None):
    """
    Clean up intermediate files and directories
    
    Parameters
    ----------
    files_to_remove : list
        List of file paths to remove
    directories_to_remove : list, optional
        List of directory paths to remove
    """
    print("\nCleaning up intermediate files...")
    
    # Remove individual files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
    
    # Remove directories
    if directories_to_remove:
        for dir_path in directories_to_remove:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                except Exception as e:
                    print(f"Error removing directory {dir_path}: {e}")

def main():
    """Main workflow function"""
    # Timestamp for logs
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting mask transformation process at {timestamp} UTC")
    print(f"Current user: rachmorse")
    print(f"Current date and time: 2025-03-05 15:59:08 UTC")
    
    # Set paths
    work_dir = "/path/to/work_directory"  # Change to your work directory
    os.makedirs(work_dir, exist_ok=True)
    
    # Input files
    mask_path = "/path/to/schaefer_harvard_mask_mni.nii.gz"  # Your MNI space mask
    ap_dwi_path = "/path/to/dwi_dir_ap.nii.gz"  # AP direction DWI
    pa_dwi_path = "/path/to/dwi_dir_pa.nii.gz"  # PA direction DWI
    ap_bval_path = "/path/to/dwi_dir_ap.bval"  # AP bval file
    pa_bval_path = "/path/to/dwi_dir_pa.bval"  # PA bval file
    ap_json_path = "/path/to/dwi_dir_ap.json"  # AP JSON sidecar
    pa_json_path = "/path/to/dwi_dir_pa.json"  # PA JSON sidecar
    
    # Output files
    ap_b0_path = os.path.join(work_dir, "ap_b0.nii.gz")
    pa_b0_path = os.path.join(work_dir, "pa_b0.nii.gz")
    acq_params_path = os.path.join(work_dir, "acq_params.txt")
    merged_b0_path = os.path.join(work_dir, "merged_b0.nii.gz")
    corrected_b0_path = os.path.join(work_dir, "corrected_b0.nii.gz")
    transformed_mask_path = os.path.join(work_dir, "mask_in_dwi_space.nii.gz")
    qc_dir = os.path.join(work_dir, "qc")
    
    try:
        # Step 1: Extract b0 images from AP and PA DWI data
        print("\nStep 1: Extracting b0 images from DWI data...")
        extract_b0_from_data(ap_dwi_path, ap_bval_path, ap_b0_path)
        extract_b0_from_data(pa_dwi_path, pa_bval_path, pa_b0_path)
        
        # Step 2: Create acquisition parameters file for topup
        print("\nStep 2: Creating acquisition parameters for topup...")
        create_acquisition_parameters(ap_json_path, pa_json_path, acq_params_path)
        
        # Step 3: Run topup for distortion correction
        print("\nStep 3: Running topup for distortion correction...")
        corrected_b0_path = run_topup(ap_b0_path, pa_b0_path, acq_params_path, work_dir)
        
        # Step 4: Transform the mask from MNI to DWI native space
        print("\nStep 4: Transforming mask from MNI to DWI space...")
        transform_mask_to_dwi_native(
            mask_path,
            corrected_b0_path,  # Use the distortion-corrected b0 as reference
            transformed_mask_path,
            qc_dir
        )
        
        print("\nTransformation complete!")
        print(f"Transformed mask: {transformed_mask_path}")
        
        # Clean up intermediate files to save disk space
        # List all intermediate files to remove
        files_to_remove = [
            ap_b0_path,
            pa_b0_path,
            merged_b0_path,
            acq_params_path,
            corrected_b0_path,
            os.path.join(work_dir, "topup_results_fieldcoef.nii.gz"),
            os.path.join(work_dir, "topup_results_movpar.txt")
        ]
        
        # Delete QC directory if not needed for review
        directories_to_remove = [qc_dir]
        
        # Perform cleanup
        cleanup_intermediate_files(files_to_remove, directories_to_remove)
        
        print("\nNext steps:")
        print("1. Verify the transformed mask by overlaying it on the DWI data")
        print("2. Use the transformed mask with your colleague's tractography results")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        # Keep intermediate files in case of error for debugging
        print("Intermediate files were not deleted due to error.")
    
    # Final timestamp
    end_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nProcess completed at {end_timestamp} UTC")

if __name__ == "__main__":
    main()
