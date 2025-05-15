import os
import subprocess
import numpy as np
import pandas as pd
import logging
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Import functions from functional connectivity script
sys.path.append('/home/rachel/Desktop/superagers/fMRI analysis')
from compute_functional_connectivity import (
    prepare_directories,
    create_network_mappings,
)


def setup_logging(output_dir):
    """Setup basic logging to file and console"""
    log_file = output_dir / f"structural_connectivity_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


def get_subjects_to_process(tractogram_dir, mask_dir, output_dir, ses):
    """Generate a list of subjects to process based on whether they 
    have a native space mask and the tractogram file but don't already have an entry in the output CSV.

    Args:
        tractogram_dir (Path): Path to the directory containing the tractogram files.
        mask_dir (Path): Path to the directory containing the native space masks.
        output_dir (Path): Path to the output directory where matrices will be saved.
        ses (str): Timepoint (format 01 or 02).

    Returns:
        list: List of subject IDs to process.
    """
    subjects_to_process = []
    
    # Get all subjects with tractogram files
    tractogram_subjects = []
    for file in os.listdir(tractogram_dir):
        if file.endswith("_dwi_tractogram_1M_SIFT.tck"):
            subject_id = file.split("_")[0]
            tractogram_subjects.append(subject_id)
    
    # Define output CSV paths
    all_to_all_csv = output_dir / f"ses-{ses}/all_to_all_roi_matrices/all_to_all_roi_matrix.csv"
    
    # Load existing subjects from CSV if it exists
    existing_subjects = set()
    if all_to_all_csv.exists():
        try:
            existing_df = pd.read_csv(all_to_all_csv, index_col="Unnamed: 0")
            existing_subjects = set(existing_df.index)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
    
    # Iterate through subjects with tractograms
    for subject in tractogram_subjects:
        # Check if native space mask exists
        mask_file = Path(f"{mask_dir}/ses-{ses}/{subject}/dwi_space_masks/{subject}_ses-{ses}_schaefer200_subcortical14_dwi_space.nii.gz")
        # Check if subject already processed (in the CSV)
        if mask_file.exists() and subject not in existing_subjects:
            subjects_to_process.append(subject)
    
    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process

def save_structural_connectivity(subject_id, label, matrix, roi_names, output_dir):
    """
    Save structural connectivity matrix without normalization.
    Only saves to the group CSV file, not individual subject files.
    
    Args:
        subject_id (str): Subject ID
        label (str): Label for the matrix (e.g., 'all_to_all_roi')
        matrix (np.ndarray): Connectivity matrix
        roi_names (list): List of ROI names
        output_dir (Path): Directory to save the output file
    """
    try:
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare the group file path
        all_subjects_file = output_dir / f"{label}_matrix.csv"
        
        # Flatten the matrix, skip diagonal elements
        flat_data = []
        column_names = []
        
        for i in range(len(roi_names)):
            for j in range(i+1, len(roi_names)):  # Start from i+1 to get upper triangle only
                if i != j:  # Skip diagonal
                    flat_data.append(matrix[i, j])
                    column_names.append(f"{roi_names[i]}-{roi_names[j]}")
        
        # Create new row for this subject
        subject_df = pd.DataFrame([flat_data], index=[subject_id], columns=column_names)
        
        if all_subjects_file.exists():
            # Read existing file
            all_subjects_df = pd.read_csv(all_subjects_file, index_col=0)
            
            # Append new subject
            updated_df = pd.concat([all_subjects_df, subject_df])
            updated_df.to_csv(all_subjects_file)
        else:
            # Create new file with this subject
            subject_df.to_csv(all_subjects_file)    

        return True
    
    except Exception as e:
            logging.error(f"{subject_id}: Error saving connectivity data: {str(e)}")
            return False

def visualize_sc_data(subject_id, connectivity_matrix, output_directory, ses, cmap='RdBu_r'):
    """
    Visualize structural connectivity matrices using the same colormap as functional connectivity.
    
    Args:
        subject_id (str): Subject ID for labeling the figure
        connectivity_matrix (np.ndarray): The structural connectivity matrix
        output_directory (Path): Directory to save the visualization
        ses (str): Session identifier (e.g., "01" or "02")
        cmap (str): Colormap to use, default 'RdBu_r' to match FC visualization
    """
    # Set up the figure
    plt.figure(figsize=(10, 8))
    
    # For structural connectivity, use absolute max for symmetric color scaling
    vmax = np.max(connectivity_matrix)
    
    # Create the heatmap
    im = plt.imshow(connectivity_matrix, cmap=cmap, vmin=0, vmax=vmax)
    
    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Number of Streamlines')
    
    # Set title and labels
    plt.title(f'Structural Connectivity: {subject_id} (ses-{ses})')
    plt.xlabel('ROI Index')
    plt.ylabel('ROI Index')
    
    # Ensure output directory exists
    vis_dir = Path(output_directory) / f"ses-{ses}/visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    output_file = vis_dir / f"{subject_id}_structural_connectivity.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_file}")
    

def generate_structural_connectivity(subject, tractogram_dir, mask_dir, output_dir, ses, labels_csv_path, run_visualization=True):
    """Generate a structural connectivity matrix using MRTrix tck2connectome.
    focusing on roi-to-roi connectivity.
    
    Args:
        subject (str): Subject ID.
        tractogram_dir (Path): Directory containing tractogram files.
        mask_dir (Path): Directory containing native space masks.
        output_dir (Path): Directory to save output files.
        ses (str): Session / timepoint (format 01 or 02).
        labels_csv_path (str): Path to the CSV file containing ROI labels.
        run_visualization (bool): Whether to visualize the matrix after generation.
    
    Returns:
        np.ndarray: The generated connectivity matrix.
    """
    tractogram_file = tractogram_dir / f"{subject}_dwi_tractogram_1M_SIFT.tck"
    mask_file = Path(f"{mask_dir}/ses-{ses}/{subject}/dwi_space_masks/{subject}_ses-{ses}_schaefer200_subcortical14_dwi_space.nii.gz")
    
    # Create output directories
    prepare_directories(output_dir, ses, ["all_to_all_roi_matrices", "within_network_matrices", "subcortical_matrices", "visualization"])
    
    # Create a temporary file for the matrix
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_matrix_file = Path(temp_dir) / f"{subject}_ses-{ses}_temp_matrix.csv"
    
    # Run tck2connectome to generate the connectivity matrix
    cmd = [
        "tck2connectome", 
        str(tractogram_file), 
        str(mask_file), 
        str(temp_matrix_file),
        "-force",
        "-zero_diagonal",  # Set diagonal elements to zero
        "-symmetric"      # Ensure matrix is symmetric
    ]

    try:
        print(f"Processing {subject} for ses-{ses}...")
        subprocess.run(cmd, check=True)
        print(f"Successfully created connectivity matrix for {subject}")
        
        # Read the generated connectivity matrix
        connectivity_matrix = np.loadtxt(temp_matrix_file, delimiter=',')

        # Read labels from CSV (same as in functional connectivity)
        combined_labels = pd.read_csv(labels_csv_path, header=None).squeeze().tolist()
        
        # Save the connectivity data to CSV
        all_to_all_dir = output_dir / f"ses-{ses}/all_to_all_roi_matrices"
        save_structural_connectivity(
            subject_id=subject,
            label="all_to_all_roi",
            matrix=connectivity_matrix,
            roi_names=combined_labels,
            output_dir=all_to_all_dir
        )
        
        # Process network-specific matrices
        process_network_matrices(subject, connectivity_matrix, combined_labels, output_dir, ses)
        
        # To run visualization
        if run_visualization:
            visualize_sc_data(
                subject_id=subject,
                connectivity_matrix=connectivity_matrix,
                output_directory=output_dir,
                ses=ses
            )
        
        # Remove temporary file
        if temp_matrix_file.exists():
            os.remove(temp_matrix_file)
        
        return connectivity_matrix
    
    except subprocess.CalledProcessError as e:
        print(f"Error processing {subject}: {e}")
        if temp_matrix_file.exists():
            os.remove(temp_matrix_file)
        return None


def process_network_matrices(subject_id, full_matrix, combined_labels, output_dir, ses):
    """Process and save network-specific connectivity matrices.
    
    Args:
        subject_id (str): Subject ID.
        full_matrix (np.ndarray): Full connectivity matrix.
        combined_labels (list): List of region labels.
        output_dir (Path): Output directory.
        ses (str): Session / timepoint.
    """
    # Create network mappings
    network_mappings = create_network_mappings(combined_labels)
    
    # Process each network
    for network, indices in network_mappings.items():
        # Skip subcortical 'network' to handle separately
        if network == "Subcortical":
            continue
        
        # Extract network submatrix
        network_matrix = full_matrix[np.ix_(indices, indices)]
        network_labels = [combined_labels[i] for i in indices]
        
        # Save network connectivity data
        network_dir = output_dir / f"ses-{ses}/within_network_matrices"
        save_structural_connectivity(
            subject_id=subject_id,
            label=f"{network}_within_network",
            matrix=network_matrix,
            roi_names=network_labels,
            output_dir=network_dir
        )
    
    # Process subcortical matrices if available
    if "Subcortical" in network_mappings:
        subcortical_indices = network_mappings["Subcortical"]
        
        # All subcortical ROIs
        subcortical_matrix = full_matrix[np.ix_(subcortical_indices, subcortical_indices)]
        subcortical_labels = [combined_labels[i] for i in subcortical_indices]
        
        # Save all subcortical connectivity data
        subcortical_dir = output_dir / f"ses-{ses}/subcortical_matrices"
        save_structural_connectivity(
            subject_id=subject_id,
            label="all_subcortical_rois",
            matrix=subcortical_matrix,
            roi_names=subcortical_labels,
            output_dir=subcortical_dir
        )
        
        # Process bilateral subcortical structures
        subcortical_structures = {}
        
        # Group subcortical regions by structure
        for idx in subcortical_indices:
            label = combined_labels[idx]
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            
            # Extract structure name without "Left" or "Right" prefix
            if ":" in label:
                structure_name = label.split(":")[1].strip()
                if "Left" in structure_name:
                    structure_name = structure_name.replace("Left", "").strip()
                elif "Right" in structure_name:
                    structure_name = structure_name.replace("Right", "").strip()
                
                if structure_name not in subcortical_structures:
                    subcortical_structures[structure_name] = []
                subcortical_structures[structure_name].append(idx)
        
        # Process each bilateral structure
        for structure_name, indices in subcortical_structures.items():
            if len(indices) > 0:
                structure_matrix = full_matrix[np.ix_(indices, indices)]
                structure_labels = [combined_labels[i] for i in indices]
                
                # Save structure connectivity data
                save_structural_connectivity(
                    subject_id=subject_id,
                    label=f"{structure_name.strip()}_bilateral",
                    matrix=structure_matrix,
                    roi_names=structure_labels,
                    output_dir=subcortical_dir
                )


def main():
    """Main function to process structural connectivity for subjects."""
    # Set parameters
    ses = "02"  
    cohort = "bbhi"
    mask_dir = Path("/home/rachel/Desktop/schaefer_analysis/fsaverage")
    output_dir = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    labels_csv_path = "/home/rachel/Desktop/schaefer_analysis/timeseries_data/native_space/combined_labels.csv"

    if cohort == "bbhi":
        if ses == "01":
            tractogram_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_MSMTCSD_TP1")
        else:  # ses-02
            tractogram_dir = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_MSMTCSD_TP2")
    else:
        if ses == "01":
            tractogram_dir = Path("/pool/guttmann/institut/UB/Superagers/MRI/tracto_MSMTCSD_TP1")
        else:  
            tractogram_dir = Path("/pool/guttmann/institut/UB/Superagers/MRI/tracto_MSMTCSD_TP2")

    # Setup MRTrix 
    os.environ["PATH"] = f"/home/rachel/miniconda3/bin:{os.environ['PATH']}"
    
    # Create output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_directories(output_dir, ses, ["all_to_all_roi_matrices", "within_network_matrices", "subcortical_matrices", "visualization"])

    # Setup logging
    setup_logging(output_dir)
    
    # Get subjects to process
    subjects = get_subjects_to_process(tractogram_dir, mask_dir, output_dir, ses)
    
    # For testing with a single subject
    # subjects = ["sub-1191"]

    # Track both failed and successful subjects
    successful_subjects = []
    failed_subjects = []
    
    # Process each subject
    for subject in subjects:
        logging.info(f"Processing {subject} for ses-{ses}...")
        
        try:
            # Assuming generate_structural_connectivity returns the connectivity matrix if successful, or None if it fails
            result = generate_structural_connectivity(
                subject=subject,
                tractogram_dir=tractogram_dir,
                mask_dir=mask_dir,
                output_dir=output_dir,
                ses=ses, 
                labels_csv_path=labels_csv_path,
                run_visualization=True
            )
            
            # Check if processing was successful
            if result is not None:
                successful_subjects.append(subject)
                logging.info(f"Successfully processed {subject}")
            else:
                failed_subjects.append(subject)
                logging.error(f"Failed to process {subject} - generate_structural_connectivity returned None")
                
        except Exception as e:
            logging.error(f"{subject}: Error during processing: {str(e)}")
            failed_subjects.append(subject)
    
    # Log summary
    total_subjects = len(subjects)
    successful_count = len(successful_subjects)
    failed_count = len(failed_subjects)
    
    logging.info(f"Processing complete: {successful_count}/{total_subjects} successful, {failed_count}/{total_subjects} failed")
    if failed_subjects:
        logging.info(f"Failed subjects: {', '.join(failed_subjects)}")

if __name__ == "__main__":
    main()
