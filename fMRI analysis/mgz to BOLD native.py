
# To run this file, run this in terminal because FreeSurfer needs to be called first:
# source /home/rachel/freesurfer/freesurfer/SetUpFreeSurfer.sh
# /home/rachel/Desktop/fMRI\ Analysis/venv/bin/python /home/rachel/Desktop/fMRI\ Analysis/DK76\ atlas\ to\ native\ space.py

import os
import subprocess

# Function to source the FreeSurfer environment
def source_freesurfer():
    """
    This function locates the FreeSurfer setup script (`SetUpFreeSurfer.sh`) in the FreeSurfer home directory,
    executes it in a new shell, and updates the current environment with any variables defined in the script.

    Note:
        This function uses `/bin/bash` to source the script and capture the environment variables.

    Raises:
        Exception: If the FreeSurfer setup script cannot be found or executed.
    """
    freesurfer_home = "/home/rachel/freesurfer/freesurfer"
    setup_script = os.path.join(freesurfer_home, 'SetUpFreeSurfer.sh')
    command = f"source {setup_script} && env"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, executable="/bin/bash")
    
    # Update the environment variables
    for line in process.stdout:
        key, _, value = line.decode().partition("=")
        os.environ[key.strip()] = value.strip()
    process.communicate()

# Source FreeSurfer environment
source_freesurfer()

# Main function for script
def main():
    """
    Main function to create a DK atlas in the BOLD image space for each subject.

    This function performs the following tasks:

    1. Defines and updates necessary paths for input and output directories.
    2. Reads list of subject IDs to process from a specified file (todo_file).
    3. Ensures the output directory exists.
    4. For each subject, constructs and executes the `mri_vol2vol` command to register the DK altas segmentation (that is in T1 from reconall process) to the subject BOLD image.
    5. Checks for the existence of necessary files before processing each subject.
    6. Handles any errors that occur during the execution of the command.

    Notes:
        - Paths such as `freesurfer_home`, `freesurfer_folder`, `bids_folder`, `output_folder`, and `todo_file` should be updated to reflect the correct locations on your system.
        - The session timepoint `ses` is set to "02" by default; modify as needed.
        - The `mri_vol2vol` command uses Nearest Neighbor interpolation and performs registration using the header.

    Outputs:
        - Native space DK atlas BOLD images per subject saved in the specified `output_folder` with filenames in the format `{subject_id}_DK76_BOLD-nativespace.nii.gz`.
        - Console messages indicating the progress and status of each subject's processing.

    Raises:
        SystemExit: If the todo_file does not exist.
        Exception: If any subprocess command fails or if required files for a subject are missing.
    """
    
    # Define paths and update as needed
    freesurfer_folder = "/home/rachel/Desktop/fMRI Analysis/subjects/freesurfer-reconall"
    bids_folder = "/home/rachel/Desktop/fMRI Analysis/subjects/BIDS"
    output_folder = "/home/rachel/Desktop/fMRI Analysis/DK76"
    todo_file = "/home/rachel/Desktop/fMRI Analysis/todo.csv"
    ses = "01"

    # Read subject IDs from the file
    if not os.path.isfile(todo_file):
        print(f"Todo file {todo_file} does not exist. Exiting.")
        exit(1)

    with open(todo_file, 'r') as file:
        subject_ids = file.read().splitlines()

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each subject
    for subject_id in subject_ids:
        print(f"Processing {subject_id}...")
        
        # Define file paths
        mov_file = os.path.join(freesurfer_folder, subject_id, "mri", "aparc.DKTatlas+aseg.mgz")
        targ_file = os.path.join(bids_folder, subject_id, f"ses-{ses}", "func", f"{subject_id}_ses-{ses}_run-01_rest_bold_ap.nii.gz")
        output_file = os.path.join(output_folder, f"{subject_id}_DK76_BOLD-nativespace.nii.gz")

        # Check if the files exist before processing
        if not os.path.isfile(mov_file):
            print(f"Mov file {mov_file} does not exist. Skipping {subject_id}.")
            continue

        if not os.path.isfile(targ_file):
            print(f"Targ file {targ_file} does not exist. Skipping {subject_id}.")
            continue

        # Construct the mri_vol2vol command
        cmd = [
            "mri_vol2vol",
            "--mov", mov_file,
            "--targ", targ_file,
            "--o", output_file,
            "--regheader",
            "--interp", "nearest"
        ]

        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully processed {subject_id}. Output saved to {output_file}.")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {subject_id}: {e}")

if __name__ == '__main__':
    main()

# import os
# import nibabel as nib
# import numpy as np
# from scipy.ndimage import affine_transform
# from nipy.algorithms.registration import Affine
# from nipy.algorithms.registration.histogram_registration import HistogramRegistration
# from pathlib import Path
# from typing import Union

# def register_and_transform(moving_image_path: str,
#                            fixed_image_path: str,
#                            output_image_path: str,
#                            interpolation: str = "nearest"):
#     """
#     Registers and applies transformation from moving_image to fixed_image using Nipy and Scipy.

#     Parameters:
#     moving_image_path (str): Path to the moving image (e.g., atlas).
#     fixed_image_path (str): Path to the fixed image (e.g., subject's BOLD image).
#     output_image_path (str): Path to save the output transformed image.
#     interpolation (str): Interpolation method to use (default is "nearest").
#     """
#     try:
#         # Load images using nibabel
#         print(f"Loading moving image from {moving_image_path}")
#         moving_img = nib.load(moving_image_path)
#         print(f"Loading fixed image from {fixed_image_path}")
#         fixed_img = nib.load(fixed_image_path)

#         # Get data and affine matrices
#         moving_data = moving_img.get_fdata()
#         fixed_data = fixed_img.get_fdata()
#         moving_affine = moving_img.affine
#         fixed_affine = fixed_img.affine

#         print("Performing affine registration using nipy")
#         # Perform affine registration using nipy
#         reg = HistogramRegistration(moving_data, fixed_data, similarity='cr')
#         affreg = Affine()
#         transformation_matrix = reg.optimize(affine=affreg.param)
#         print(f"Transformation matrix:\n{transformation_matrix}")

#         print("Applying transformation using scipy")
#         # Apply transformation using scipy
#         transformed_data = affine_transform(moving_data, transformation_matrix[:3, :3], offset=transformation_matrix[:3, 3], order=0 if interpolation == "nearest" else 1)

#         print(f"Saving transformed image to {output_image_path}")
#         # Save the transformed data as a new NIfTI image
#         transformed_img = nib.Nifti1Image(transformed_data, fixed_affine)
#         nib.save(transformed_img, output_image_path)
#         print("Transformation and saving completed successfully")
#     except Exception as e:
#         print(f"Error in register_and_transform: {e}")
#         raise

# def main(freesurfer_folder: Union[str, Path],
#          bids_folder: Union[str, Path],
#          output_folder: Union[str, Path],
#          todo_file: Union[str, Path],
#          session: str,
#          mov_file_template: str,
#          targ_file_template: str,
#          output_file_template: str):
#     """
#     Main function to create a DK atlas in the BOLD image space for each subject.

#     This function performs the following tasks:
#     1. Defines and updates necessary paths for input and output directories.
#     2. Reads list of subject IDs to process from a specified file (todo_file).
#     3. Ensures the output directory exists.
#     4. For each subject:
#         a. Constructs paths for the DK atlas and BOLD image.
#         b. Registers the DK atlas to the subject's BOLD image using affine transformation.
#         c. Handles any errors that occur during the process.

#     Outputs:
#         - Native space DK atlas BOLD images per subject saved in the specified `output_folder`.

#     Raises:
#         SystemExit: If the todo_file does not exist.
#         Exception: If any registration or transformation step fails or if required files are missing.
#     """
#     try:
#         # Convert paths to Path objects
#         freesurfer_folder = Path(freesurfer_folder)
#         bids_folder = Path(bids_folder)
#         output_folder = Path(output_folder)
#         todo_file = Path(todo_file)

#         # Read subject IDs from the file
#         if not todo_file.is_file():
#             print(f"Todo file {todo_file} does not exist. Exiting.")
#             exit(1)

#         with todo_file.open('r') as file:
#             subject_ids = file.read().splitlines()

#         # Ensure the output directory exists
#         os.makedirs(output_folder, exist_ok=True)

#         # Process each subject
#         for subject_id in subject_ids:
#             print(f"Processing {subject_id}...")

#             # Define file paths
#             mov_file = mov_file_template.format(subject_id=subject_id)
#             targ_file = targ_file_template.format(subject_id=subject_id, ses=session)
#             output_file = output_file_template.format(subject_id=subject_id)

#             # Check if the files exist before processing
#             if not os.path.isfile(mov_file):
#                 print(f"Mov file {mov_file} does not exist. Skipping {subject_id}.")
#                 continue

#             if not os.path.isfile(targ_file):
#                 print(f"Targ file {targ_file} does not exist. Skipping {subject_id}.")
#                 continue

#             # Register and transform DK atlas to BOLD space
#             try:
#                 register_and_transform(mov_file, targ_file, output_file, interpolation="nearest")
#                 print(f"Successfully processed {subject_id}. Output saved to {output_file}.")
#             except Exception as e:
#                 print(f"Error processing {subject_id}: {e}")
#     except Exception as e:
#         print(f"Error in main: {e}")
#         raise

# if __name__ == '__main__':
#     # Change to your paths
#     freesurfer_folder = Path("/home/rachel/Desktop/fMRI Analysis/subjects/freesurfer-reconall")
#     bids_folder = Path("/home/rachel/Desktop/fMRI Analysis/subjects/BIDS")
#     output_folder = Path("/home/rachel/Desktop/fMRI Analysis/DK76")
#     todo_file = Path("/home/rachel/Desktop/fMRI Analysis/todo.csv")
#     session = "01"

#     # Define file templates
#     mov_file_template = str(freesurfer_folder / "{subject_id}" / "mri" / "aparc.DKTatlas+aseg.mgz")
#     targ_file_template = str(bids_folder / "{subject_id}" / "ses-{ses}" / "func" / "{subject_id}_ses-{ses}_run-01_rest_bold_ap.nii.gz")
#     output_file_template = str(output_folder / "{subject_id}_DK76_BOLD-nativespace_TEST.nii.gz")

#     main(freesurfer_folder=freesurfer_folder,
#          bids_folder=bids_folder,
#          output_folder=output_folder,
#          todo_file=todo_file,
#          session=session,
#          mov_file_template=mov_file_template,
#          targ_file_template=targ_file_template,
#          output_file_template=output_file_template)