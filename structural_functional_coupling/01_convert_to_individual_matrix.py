import pandas as pd
import numpy as np
import os
from pathlib import Path
import re

# Function to convert the data 
def create_individual_matrices(input_file, output_dir, ses, is_functional=False, is_fisher_z=False):
    """Convert a CSV file of connectivity matrices into individual matrices for each subject.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the individual matrices.
        ses (str): Session identifier (e.g., "ses-01", "ses-02").
        is_functional (bool): If True, process functional connectivity; otherwise, structural.
        is_fisher_z (bool): If True, indicates that the functional connectivity values are Fisher Z-transformed.
    """
    # Read the input csv file
    df = pd.read_csv(input_file)
    num_rois = 214

    # Get the subject IDs 
    subjects = df.iloc[:, 0].tolist()

    # Extract ROI labels from column names
    column_names = df.columns[1:]
    
    # Get a unique set of ROI labels from the connection columns
    roi_labels = set()
    for col in column_names:
        # Extract both ROI names from column name
        matches = re.findall(r'([^-]+)-([^-]+)', col)
        if matches:
            roi1, roi2 = matches[0]
            roi_labels.add(roi1.strip())
            roi_labels.add(roi2.strip())
    
    # Convert to sorted list
    roi_labels = sorted(list(roi_labels))
    
    # Verify it has the expected number of ROIs
    if len(roi_labels) != num_rois:
        print(f"Warning: Expected {num_rois} ROIs, but found {len(roi_labels)}.")
        print("Using detected ROI labels, but results may be incorrect.")
        # If it can't extract labels, create generic ones
        if len(roi_labels) == 0:
            roi_labels = [f"ROI_{i+1}" for i in range(num_rois)]
            print("Created generic ROI labels.")
    
    # Create a mapping from ROI pairs to matrix indices
    roi_to_index = {roi: i for i, roi in enumerate(roi_labels)}
    
    # Expected number of connections for 214 ROIs (upper triangle only)
    expected_connections = (num_rois * (num_rois - 1)) // 2
    
    # Verify the data has the expected number of columns
    actual_connections = len(column_names)
    if actual_connections != expected_connections:
        print(f"Warning: Expected {expected_connections} connections for {num_rois} ROIs, but found {actual_connections}.")
        print("Continuing with processing, but results may be incorrect.")
        
    # Process each subject
    for i, subject in enumerate(subjects):
        # Create an empty connectivity matrix
        conn_matrix = np.zeros((num_rois, num_rois))
            
        # Get the connectivity values for this subject
        row_data = df.iloc[i]
        
        # Fill the connectivity matrix based on column names
        for j, col in enumerate(column_names):
            matches = re.findall(r'([^-]+)-([^-]+)', col)
            if matches:
                roi1, roi2 = matches[0]
                roi1 = roi1.strip()
                roi2 = roi2.strip()
                
                # Get the indices for these ROIs
                idx1 = roi_to_index.get(roi1)
                idx2 = roi_to_index.get(roi2)
                
                if idx1 is not None and idx2 is not None:
                    value = row_data[col]
                    conn_matrix[idx1, idx2] = value
                    conn_matrix[idx2, idx1] = value  # Make it symmetric
        
        # Zero out the diagonal
        np.fill_diagonal(conn_matrix, 0)

        # Normalize the structural connectivity matrix
        if not is_functional:

            # Apply Gaussian normalization
            mu = np.mean(conn_matrix)
            sigma = np.std(conn_matrix)

            if sigma > 0:
                # Z-score (mean=0, std=1)
                conn_matrix = (conn_matrix - mu) / sigma
                
                # Rescale to mean = 0.5, std = 0.1
                conn_matrix = conn_matrix * 0.1 + 0.5

        # Convert to DataFrame with original ROI labels
        conn_df = pd.DataFrame(conn_matrix, index=roi_labels, columns=roi_labels)

        # Save to CSV
        if is_functional:
            if is_fisher_z:
                output_file = output_dir / f"{subject}_{ses}_functional_connectivity_matrix_fisher_z.csv"
            else:
                output_file = output_dir / f"{subject}_{ses}_functional_connectivity_matrix.csv"
        else:
            output_file = output_dir / f"{subject}_{ses}_structural_connectivity_matrix_normalized.csv"
        conn_df.to_csv(output_file)

def main():
    # Paths 
    sessions = ["ses-01", "ses-02"]

    for ses in sessions:
        print("--------------------------")
        print(f"Processing session: {ses}")
        print("--------------------------")

        structural_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/structural_connectivity/{ses}")
        functional_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space/{ses}")
        structural_matrix = structural_dir / "all_to_all_roi_matrices" / "all_to_all_roi_matrix.csv"
        is_fisher_z = True  # Set to True for getting fisher z values
        if is_fisher_z:
            func_matrix = functional_dir / "all_to_all_roi_matrices" / "fisher_z_all_to_all_roi_matrix.csv"
        else:
            func_matrix = functional_dir / "all_to_all_roi_matrices" / "all_to_all_roi_matrix.csv"

        # Output directories
        struct_output_dir = structural_dir / "individual_connectivity_matrices"
        func_output_dir = functional_dir / "individual_connectivity_matrices"

        # Create output directories if they don't exist
        os.makedirs(struct_output_dir, exist_ok=True)
        os.makedirs(func_output_dir, exist_ok=True)
        
        # Process structural connectivity matrices
        print("Processing structural connectivity matrices...")
        create_individual_matrices(structural_matrix, struct_output_dir, ses, is_functional=False, is_fisher_z=False)

        # Process functional connectivity matrices
        print("Processing functional connectivity matrices...")
        create_individual_matrices(func_matrix, func_output_dir, ses, is_functional=True, is_fisher_z=is_fisher_z)

        print("Conversion complete!")

if __name__ == "__main__":
    main()