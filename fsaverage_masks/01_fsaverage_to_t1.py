#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script to transform the Schaefer 200 atlas from fsaverage (surface) to native BoldStandard volume
of a given subject in SUBJECTS_DIR
"""

import os
import numpy as np
import subprocess
import logging
from datetime import datetime, timezone
import nibabel as nib
from pathlib import Path        

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_subjects_to_process(reconall_dir, out_dir, ses):
    """Generate a list of subjects to process based on whether they have
    recon-all done for the specified timepoint.

    Args:
        reconall_dir (Path): Path to the directory containing the recon-all results.
        out_dir (Path): Path to the output directory where the results will be saved.
        ses (str): Session (e.g., ses-01).
        
    Returns:
        list: List of subject IDs to process.
    """
    subjects_to_process = []

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(reconall_dir):
        # Check if it's a subject directory with session (sub-xxx_ses-yy format)
        if not subject_dir.startswith("sub-"):
            continue
            
        # Extract subject ID without session
        if "_ses-" in subject_dir:
            # Format is sub-xxx_ses-yy
            subject_id = subject_dir.split("_ses-")[0]
            dir_session = subject_dir.split("_ses-")[1]
            
            # Check if this directory matches the requested session
            if f"ses-{dir_session}" != ses and f"{dir_session}" != ses:
                continue
        else:
            # Format is just sub-xxx
            subject_id = subject_dir
        
        # Check if the required directory exists and hasn't been processed yet
        subject_recon_dir = reconall_dir / subject_dir
        output_subject_dir = out_dir / subject_id
        
        output_file_path = output_subject_dir / f"{subject_id}_schaefer200_aparc+aseg.mgz"
        
        if subject_recon_dir.exists() and not output_file_path.exists():
            subjects_to_process.append((subject_id, subject_dir))

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process


def process_subject(subject_dir, subject_id, reconall_dir, output_folder):
    """Process a single subject to transform Schaefer atlas to native space.
    
    Args:
        subject_dir (str): Subject directory (format: sub-xxx_ses-0y).
        subject_id (str): Subject ID.
        reconall_dir (str): Directory containing recon-all results.
        output_folder (str): Output directory.
    """
    try:
        # Create the output directory if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Set up annotation files - these are the Schaefer 200 atlas files from GitHub
        schaefer_fsaverage_left = Path('/home/rachel/Desktop/superagers/fsaverage_masks/lh.Schaefer2018_200Parcels_7Networks_order.annot')
        schaefer_fsaverage_right = Path('/home/rachel/Desktop/superagers/fsaverage_masks/rh.Schaefer2018_200Parcels_7Networks_order.annot')        
        
        # Check if they exist
        if not schaefer_fsaverage_left.exists() or not schaefer_fsaverage_right.exists():
            logger.error(f"Could not find the local annotation files at {schaefer_fsaverage_left} and {schaefer_fsaverage_right}")
            logger.error("Please download these files manually and place them in the fsaverage_masks directory")
            return False
            
        logger.info(f"Processing subject: {subject_id}")

        lh_annotation = output_folder / f"lh.Schaefer2018_200Parcels_7Networks_order.annot"
        rh_annotation = output_folder / f"rh.Schaefer2018_200Parcels_7Networks_order.annot"
        
        # 2. Map annotations from fsaverage to subject
        logger.info("Mapping Schaefer 200 atlas annotations from fsaverage to subject's surface")
        
        # Left hemisphere
        cmd_map_left = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject_dir,
            "--hemi", "lh",
            "--sval-annot", str(schaefer_fsaverage_left),
            "--tval", str(lh_annotation)
        ]
        try:
            subprocess.run(cmd_map_left, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error mapping left hemisphere: {e.stderr}")
            return False

        # Right hemisphere
        cmd_map_right = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject_dir,
            "--hemi", "rh",
            "--sval-annot", str(schaefer_fsaverage_right),
            "--tval", str(rh_annotation)
        ]
        try:
            subprocess.run(cmd_map_right, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error mapping right hemisphere: {e.stderr}")
            return False

        # 3. Create volume in native T1 space 

        # Create a custom environment for FreeSurfer to be able to take the results from the previous step
        subjects_dir = os.environ.get('SUBJECTS_DIR', str(output_folder))
        custom_env = os.environ.copy()
        custom_env["SUBJECTS_DIR"] = subjects_dir
  
        # Left hemisphere
        cmd_lh_label2vol = [
            "mri_label2vol",
            "--annot", str(lh_annotation),
            "--temp", f"{reconall_dir}/{subject_dir}/mri/T1.mgz",
            "--identity",
            "--fillthresh", "0.3",
            "--proj", "frac", "0", "1", "0.1",
            "--subject", subject_dir,
            "--o", f"{output_folder}/{subject_id}_schaefer_volumetric_t1_lh.nii.gz",
            "--hemi", "lh"
        ]
        try:
            subprocess.run(cmd_lh_label2vol, env=custom_env, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error mapping left hemisphere to volume: {e.stderr}")
            return False

        # Right hemisphere 
        cmd_rh_label2vol = [
            "mri_label2vol",
            "--annot", str(rh_annotation),
            "--temp", f"{reconall_dir}/{subject_dir}/mri/T1.mgz",
            "--identity",
            "--fillthresh", "0.3",
            "--proj", "frac", "0", "1", "0.1",
            "--subject", subject_dir,
            "--o", f"{output_folder}/{subject_id}_schaefer_volumetric_t1_rh.nii.gz",
            "--hemi", "rh"
        ]
        try:
            subprocess.run(cmd_rh_label2vol, env=custom_env, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error mapping right hemisphere to volume: {e.stderr}")
            return False

        logger.info(f"Successfully processed subject {subject_id}")
        return True  
       
    except Exception as e:  
        logger.exception(f"Error processing subject {subject_id}: {str(e)}")
        return False

def main():
    # Set up logging level based on verbose flag
    logger.setLevel(logging.DEBUG)

    # Set up parameters
    cohort = "bbhi senior" 
    session = 'ses-01'
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    logger.info(f"Processing for session: {session}")

    # Set paths based on cohort
    if cohort == "bbhi":
        # BBHI paths
        reconall_dir = Path('/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall')
    else:  # cohort == "bbhi senior"
        reconall_dir = Path('/pool/guttmann/institut/UB/Superagers/MRI/freesurfer-reconall')

    # Log execution info
    logger.info(f"Script executed by: {os.environ.get('USER', 'rachel')}")
    logger.info(f"Date and time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")    
    
    # Set environment variables
    os.environ['SUBJECTS_DIR'] = str(reconall_dir)
    
    # Determine subjects to process
    subject_data = get_subjects_to_process(reconall_dir, output_folder, session)
    
    if not subject_data:
        logger.info("No subjects found that need processing.")
        return
        
    subjects = [s[0] for s in subject_data]
    logger.info(f"Will process {len(subjects)} subjects: {', '.join(subjects)}")
        
    # Find the corresponding directory in reconall_dir
    subject_dir = None
    expected_dir = f"{subject_id}_ses-{session.replace('ses-', '')}"
    
    if os.path.exists(reconall_dir / expected_dir):
        subject_dir = expected_dir
    else:
        logger.error(f"Could not find expected directory {expected_dir} for subject {subject_id} in {reconall_dir}")
        return
    
    # Uncomment this code to process a single subject 
    # Format subject_data = [(subject_id, subject_dir)]
    # subject_data = [("sub-1014", "sub-1014_ses-01")]
    
    # Process each subject
    successful = 0
    for subject_id, subject_dir in subject_data:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing subject: {subject_id} (directory: {subject_dir})")
        logger.info(f"{'='*50}\n")

        # Set up paths
        output_folder = Path(f'/home/rachel/Desktop/schaefer_analysis/fsaverage/{session}/{subject_id}/t1_masks')
        
        if process_subject(
            subject_dir,  
            subject_id,   
            reconall_dir,
            output_folder
        ):
            successful += 1
    
    logger.info(f"\nProcessing summary:")
    logger.info(f"Total subjects: {len(subject_data)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {len(subject_data) - successful}")

if __name__ == "__main__":
    main()