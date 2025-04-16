#!/usr/bin/env python3
import os
import subprocess
import datetime

# Current Date and Time information
print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-04-16 09:02:40")
print(f"Current User's Login: rachmorse")

# Load modules (in Python, we'd need to ensure these are in PATH or use full paths)
# These module load commands don't have direct Python equivalents
# subprocess.run(['module', 'load', 'fsl/6.0.4'])
# subprocess.run(['module', 'load', 'freesurfer/freesurfer-7.1'])
# Instead, you might need to modify PATH or use full paths to binaries

# Define paths
processed_dir_root = "/pool/guttmann/institut/BBHI/MRI/processed_data"
working_dir_root = "/home/mariacabello/working_dir/multimodal/atlases/glasser374/all_subjects"
freesurfer_root = "/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall"
MATLAB_PATH = "/usr/local/bin/matlab"
ses = "ses-01"

# Count subjects
# Getting number of directories in working_dir_root
result = subprocess.run(['ls', working_dir_root], capture_output=True, text=True)
n = len(result.stdout.strip().split('\n'))

# Initialize counter
i = 0

# Read subjects from file
with open('/home/mariacabello/working_dir/multimodal/todo.txt', 'r') as f:
    subjects = f.read().strip().split('\n')

for s in subjects:
    i += 1
    print(f"Working on {s}... {i} out of {n}")

    wd = f"{working_dir_root}/{s}/{ses}/"
    
    # Copy T1
    subprocess.run(['mrconvert', '-force', f"{freesurfer_root}/{s}_{ses}/mri/T1.mgz", f"{wd}/fs2_T1w.nii"])

    # Atlas resampling 2 BOLD-T1
    print("      Structural 2 diff. coregistration (1/5)")
    if not os.path.isfile(f"{wd}/glasser374_BOLDspace.nii.gz"):
        # Copy bold
        subprocess.run(['cp', f"{processed_dir_root}/fMRI-preprocessed/{s}/native_T1/{s}_{ses}_run-01_rest_sbref_ap_T1-space.nii.gz", f"{wd}/fs2_boldSbref.nii.gz"])

        subprocess.run(['flirt', '-in', f"{wd}/glasser374_T1.nii.gz", '-ref', f"{wd}/fs2_boldSbref.nii.gz", '-omat', 'transform.mat'])
        subprocess.run(['flirt', '-in', f"{wd}/glasser374_T1.nii.gz", '-applyxfm', '-init', 'transform.mat', 
                        '-out', f"{wd}/glasser374_BOLDspace.nii.gz", '-paddingsize', '0.0', 
                        '-interp', 'nearestneighbour', '-ref', f"{wd}/fs2_boldSbref.nii.gz"])

    # Structural 2 DWI coregistration
    print("      Structural 2 diff. coregistration (1/5)")
    if not os.path.isfile(f"{wd}/glasser374_DWIspace.nii"):
        # Create dwi b0
        subprocess.run(['fslroi', f"{processed_dir_root}/DWI_dtifit_tp1/{s}/eddy_corrected_data.nii.gz", 
                       f"{wd}/fs2_b0.nii.gz", '0', '1'])
        subprocess.run(['gunzip', '-f', '-d', f"{wd}/fs2_b0.nii.gz"])

        # Copy atlas
        subprocess.run(['cp', f"{wd}/glasser374_T1.nii.gz", f"{wd}/glasser374_DWIspace.nii.gz"])
        subprocess.run(['gunzip', '-f', f"{wd}/glasser374_DWIspace.nii.gz"])

        # Note: git_dir is not defined in the original script
        # You might need to define it or modify this call
        git_dir = ""  # Add the appropriate path here
        matlab_command = f"addpath('{git_dir}/dependences'); spm_coregister_parcellation('{wd}/fs2_b0.nii', '{wd}/fs2_T1w.nii', '{wd}/glasser374_DWIspace.nii'); exit;"
        subprocess.run([MATLAB_PATH, '-nodisplay', '-nosplash', '-nodesktop', '-r', matlab_command])