#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import subprocess
import logging
from pathlib import Path
from joblib import Parallel, delayed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_subjects_to_process(reconall_dir, out_dir, ses):
    """Generate a list of subjects to process based on whether they have
    recon-all done for the specified timepoint and have neuropsych data.

    Args:
        reconall_dir (Path): Path to the directory containing the recon-all results.
        out_dir (Path): Path to the output directory where the results will be saved.
        ses (str): Session (e.g., ses-01).
        
    Returns:
        list: List of subject IDs to process.
        list: List of subject IDs already processed.
    """
    subjects_to_process = []
    already_processed = []

    # Read the CSV file to filter valid IDs based on neuropsych data (e.g., whether they have the data to be classified as superager)
    valid_ids = None
    df = pd.read_csv('/home/rachel/Desktop/data/superager.csv')
    required_superager_col = 'superager_tp2' if ses == 'ses-02' else 'superager_tp1'

    if required_superager_col not in df.columns:
        logger.warning(
            "Column %s not found in superager.csv; cannot ensure participants have required data.",
            required_superager_col,
        )
    else:
        valid_ids = set(
            df[df[required_superager_col].notna()]['id'].astype(str).tolist()
        )

    # Iterate over all possible subject directories
    for subject_dir in os.listdir(reconall_dir):
        # Check if it's a subject directory with session (sub-xxx_ses-yy format, optionally _run-xx as with BBHI)
        if not subject_dir.startswith("sub-"):
            continue
            
        # Normalize naming (handles sub-xxx_ses-yy and sub-xxx_ses-yy_run-zz)
        parts = subject_dir.split("_ses-")
        subject_id = parts[0]
        dir_session = None

        if len(parts) > 1:
            dir_session = parts[1]
            # Remove optional run information
            dir_session_core = dir_session.split("_run-")[0]
            if f"ses-{dir_session_core}" != ses and dir_session_core != ses:
                continue
        elif ses:
            # If session requested but not present in name, skip
            continue

        out_dir_t1 = out_dir / subject_id / "t1_masks"

        # If sub is not in valid IDs, skip
        if valid_ids is not None and subject_id.replace("sub-", "") not in valid_ids:
            continue
        
        # Check if the required directory exists and hasn't been processed yet
        subject_recon_dir = reconall_dir / subject_dir
        output_lh_path = out_dir_t1 / f"{subject_id}_schaefer_volumetric_t1_lh.nii.gz"
        output_rh_path = out_dir_t1 / f"{subject_id}_schaefer_volumetric_t1_rh.nii.gz"

        if subject_recon_dir.exists() and not output_lh_path.exists() and not output_rh_path.exists():
            subjects_to_process.append((subject_id, subject_dir))
        elif output_lh_path.exists() and output_rh_path.exists():
            already_processed.append(subject_id)

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process, already_processed   


def process_subject(subject_dir, subject_id, reconall_dir, output_folder):
    """Process a single subject to transform Schaefer atlas to native T1 space.
    
    Args:
        subject_dir (str): Subject directory (format: sub-xxx_ses-0y).
        subject_id (str): Subject ID.
        reconall_dir (str): Directory containing recon-all output.
        output_folder (str): Output directory.
    """
    try:
        # Create the output directory if it doesn't exist
        output_folder.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing subject: {subject_id} (directory: {subject_dir})")
        logger.info(f"{'='*50}\n")
        
        # Set up annotation files - these are the Schaefer 200 atlas files from GitHub
        schaefer_fsaverage_left = Path('/home/rachel/Desktop/superagers/fsaverage_masks/lh.Schaefer2018_200Parcels_7Networks_order.annot')
        schaefer_fsaverage_right = Path('/home/rachel/Desktop/superagers/fsaverage_masks/rh.Schaefer2018_200Parcels_7Networks_order.annot')        
        
        # Check if they exist
        if not schaefer_fsaverage_left.exists() or not schaefer_fsaverage_right.exists():
            logger.error(f"Could not find the local annotation files at {schaefer_fsaverage_left} and {schaefer_fsaverage_right}")
            logger.error("Please download these files manually and place them in the fsaverage_masks directory")
            return False
            
        # Set up paths for output annotations
        lh_annotation = output_folder / f"lh.Schaefer2018_200Parcels_7Networks_order.annot"
        rh_annotation = output_folder / f"rh.Schaefer2018_200Parcels_7Networks_order.annot"
        
        # Map annotations from fsaverage to subject-specfic surface space
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

        # Then transform the subject-specific surface annotations to volumetric T1 space

        # Create a custom environment for FreeSurfer; default to recon-all dir if SUBJECTS_DIR unset
        custom_env = os.environ.copy()
        custom_env["SUBJECTS_DIR"] = custom_env.get("SUBJECTS_DIR", str(reconall_dir))
  
        # Left hemisphere
        subject_t1 = reconall_dir / subject_dir / "mri" / "T1.mgz"

        cmd_lh_label2vol = [
            "mri_label2vol",
            "--annot", str(lh_annotation),
            "--temp", str(subject_t1),
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
            "--temp", str(subject_t1),
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

    print("-----------------------Running 01_fsaverage_to_t1.py-----------------------")

    # Set up parameters
    cohorts = ["bbhi", "bbhi senior"]
    sessions = ['ses-01', 'ses-02']

    for cohort in cohorts:
        for session in sessions:
            print("-------------------------")
            print(f"Processing {cohort} {session}")
            print("-------------------------")

            if not session.startswith('ses-'):
                session = f'ses-{session}'

            output_folder = Path(f'/home/rachel/Desktop/schaefer_analysis/fsaverage/{session}')

            # Set paths based on cohort
            if cohort == "bbhi":
                # BBHI paths
                reconall_dir = Path('/pool/guttmann/institut/BBHI/MRI/derivatives/freesurfer-reconall')
            else:  # cohort == "bbhi senior"
                reconall_dir = Path('/pool/guttmann/institut/UB/Superagers/MRI/derivatives/freesurfer-reconall')

            # Set environment variables
            os.environ['SUBJECTS_DIR'] = str(reconall_dir)
            
            # Determine subjects to process
            subject_data, already_processed = get_subjects_to_process(reconall_dir, output_folder, session)
            print(f"Number of subjects already processed: {len(already_processed)}")

            if not subject_data:
                logger.info("No subjects found that need processing.")
                continue
                
            subjects = [s[0] for s in subject_data]
            logger.info(f"Will process {len(subjects)} subjects: {', '.join(subjects)}")
            
            # Uncomment this code to process a single subject 
            # Format subject_data = [(subject_id, subject_dir)]
            # subject_data = [("sub-1014", "sub-1014_ses-01")]
            
            # Process each subject
            results = Parallel(n_jobs=10)(
                delayed(process_subject)(
                    subject_dir,  
                    subject_id,   
                    reconall_dir,
                    Path(f'/home/rachel/Desktop/schaefer_analysis/fsaverage/{session}/{subject_id}/t1_masks')
                ) for subject_id, subject_dir in subject_data
            )
            
            successful = sum(results)
            
            logger.info(f"\nProcessing summary:")
            logger.info(f"Total subjects: {len(subject_data)}")
            logger.info(f"Successfully processed: {successful}")
            logger.info(f"Failed: {len(subject_data) - successful}")

if __name__ == "__main__":
    main()
