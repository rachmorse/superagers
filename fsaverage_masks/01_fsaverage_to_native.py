#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python script to transform the Schaefer 200 atlas from fsaverage (surface) to native BoldStandard volume
of a given subject in SUBJECTS_DIR
"""

import os
import argparse
import subprocess
from datetime import datetime
import nibabel as nib
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Transform Schaefer 200 atlas from fsaverage to native BOLD volume')
    
    # Define arguments without default values for subjects_dir and output_dir
    parser.add_argument('--subjects_dir', 
                        help='Directory with the results of recon-all')
    parser.add_argument('--subject_id', 
                        help='Subject ID (e.g., sub-XXXXX). If not provided, will process all eligible subjects')
    parser.add_argument('--left_annotation', 
                        help='Path to left annotation file for Schaefer 200')
    parser.add_argument('--right_annotation', 
                        help='Path to right annotation file for Schaefer 200')
    parser.add_argument('--output_dir', 
                        help='Output directory')
    parser.add_argument('--reconall_dir', 
                        help='Root directory containing subject data')
    parser.add_argument('--session', 
                        help='Session identifier (e.g., ses-01)')
    parser.add_argument('--process_all', action='store_true',
                        help='Process all eligible subjects')
    
    return parser.parse_args()

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
        
        if subject_recon_dir.exists() and not (output_subject_dir / f"{subject_id}_schaefer200_aparc+aseg.mgz").exists():
            subjects_to_process.append(subject_id)

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process

def process_subject(subject_id, reconall_dir, output_folder, fs_home, left_annotation=None, right_annotation=None):
    """Process a single subject to transform Schaefer atlas to native space.
    
    Args:
        subject_id (str): Subject ID to process.
        reconall_dir (Path): Directory with the results of recon-all.
        output_folder (Path): Output directory for the results.
        fs_home (Path): FreeSurfer home directory.
        left_annotation (Path, optional): Path to left annotation file.
        right_annotation (Path, optional): Path to right annotation file.
    
    Returns:
        bool: True if processing completed successfully, False otherwise.
    """
    try:
        # Get FreeSurfer's subjects directory
        freesurfer_subjects_dir = fs_home / "subjects"

        # Construct paths to standard Schaefer atlas annotation files
        schaefer_fsaverage_left = Path(left_annotation) if left_annotation else freesurfer_subjects_dir / "fsaverage" / "label" / "lh.Schaefer2018_200Parcels_7Networks_order.annot"
        schaefer_fsaverage_right = Path(right_annotation) if right_annotation else freesurfer_subjects_dir / "fsaverage" / "label" / "rh.Schaefer2018_200Parcels_7Networks_order.annot"

        # Verify files exist
        if not schaefer_fsaverage_left.exists() or not schaefer_fsaverage_right.exists():
            print(f"Warning: Standard Schaefer atlas files not found for subject {subject_id}")
            return False
        
        # Create the output directory if it doesn't exist
        subject_output_dir = output_folder / subject_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing subject: {subject_id}")
        
        # Run commands using subprocess
        
        # 1. Generate paths for outputting the surface annotations
        lh_annotation = subject_output_dir / f"lh.{subject_id}_Schaefer2018_200Parcels_7Networks_order.annot"
        rh_annotation = subject_output_dir / f"rh.{subject_id}_Schaefer2018_200Parcels_7Networks_order.annot"
        
        # 2. Map annotations from fsaverage to subject
        print("Mapping Schaefer 200 atlas annotations from fsaverage to subject's surface")
        
        # Left hemisphere
        cmd_map_left = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject_id,
            "--hemi", "lh",
            "--sval-annot", str(schaefer_fsaverage_left),
            "--tval", str(lh_annotation)
        ]
        subprocess.run(cmd_map_left, check=True)
        
        # Right hemisphere
        cmd_map_right = [
            "mri_surf2surf",
            "--srcsubject", "fsaverage",
            "--trgsubject", subject_id,
            "--hemi", "rh",
            "--sval-annot", str(schaefer_fsaverage_right),
            "--tval", str(rh_annotation)
        ]
        subprocess.run(cmd_map_right, check=True)
        
        # 3. Create volume in native space
        print("Creating volume in native space for Schaefer parcels")
        
        output_volume = subject_output_dir / f"{subject_id}_schaefer200_aparc+aseg.mgz"
        
        cmd_vol = [
            "mri_aparc2aseg",
            "--s", subject_id,
            "--o", str(output_volume),
            "--annot", "Schaefer2018_200Parcels_7Networks_order"
        ]
        subprocess.run(cmd_vol, check=True)
        
        print(f"Processing completed successfully for subject: {subject_id}")
        print(f"Output volume created at: {output_volume}")
        return True
    
    except Exception as e:
        print(f"Error processing subject {subject_id}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set default values in main if not provided via arguments
    output_folder = Path(args.output_dir or '/home/rachmorse/data/schaefer')
    reconall_dir = Path(args.reconall_dir or '/pool/guttmann/institut/UB/Superagers/MRI/freesurfer-reconall')
    
    # Handle session identifier
    session = args.session or 'ses-01'
    if not session.startswith('ses-'):
        session = f'ses-{session}'
    print(f"Processing for session: {session}")
    
    # Get FreeSurfer's home directory from environment variable
    fs_home = Path(os.environ.get('FREESURFER_HOME', '/home/rachel/freesurfer/freesurfer'))
    
    # Log execution info
    print(f"Script executed by: rachmorse")
    print(f"Date and time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Set environment variables
    os.environ['SUBJECTS_DIR'] = str(reconall_dir)
    
    # Determine subjects to process
    if args.process_all or not args.subject_id:
        # Get all subjects that need processing for the specified session
        subjects = get_subjects_to_process(reconall_dir, output_folder, session)
        
        if not subjects:
            print("No subjects found that need processing.")
            return
            
        print(f"Will process {len(subjects)} subjects: {', '.join(subjects)}")
    else:
        # Process a single subject
        subjects = [args.subject_id]
    
    # Process each subject
    successful = 0
    for subject_id in subjects:
        print(f"\n{'='*50}")
        print(f"Processing subject: {subject_id}")
        print(f"{'='*50}\n")
        
        # Determine the correct subject directory in reconall based on session
        subject_dir_in_reconall = None
        for dir_name in os.listdir(reconall_dir):
            if dir_name == subject_id or dir_name.startswith(f"{subject_id}_ses-") or dir_name.startswith(f"{subject_id}_ses{session}"):
                subject_dir_in_reconall = dir_name
                break
                
        if subject_dir_in_reconall is None:
            print(f"Could not find subject directory for {subject_id} in {reconall_dir}")
            continue
        
        # Set the SUBJECTS_DIR environment variable to include session information
        os.environ['SUBJECTS_DIR'] = str(reconall_dir)
        
        if process_subject(
            subject_dir_in_reconall,  # Pass the full directory name with session
            reconall_dir,                 # Pass the reconall directory as subjects_dir
            output_folder, 
            fs_home, 
            args.left_annotation, 
            args.right_annotation
        ):
            successful += 1
    
    print(f"\nProcessing summary:")
    print(f"Total subjects: {len(subjects)}")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {len(subjects) - successful}")

if __name__ == "__main__":
    main()