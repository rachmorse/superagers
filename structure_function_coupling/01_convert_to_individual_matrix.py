import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from run_sfc_pipeline import FC_ROOT, IS_FISHER_Z, N_ROIS, SESSIONS, STRUCT_ROOT, SUBJECTS

# Function to convert the data
def create_individual_matrices(input_file, output_dir, ses, is_functional=False, is_fisher_z=False):
    """Convert a CSV file of connectivity matrices into individual matrices for each subject.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the individual matrices.
        ses (str): Session (e.g., "ses-01", "ses-02").
        is_functional (bool): If True, process functional connectivity; otherwise, structural.
        is_fisher_z (bool): If True, indicates that the functional connectivity values are Fisher Z-transformed.
    """
    # Read the input csv file
    df = pd.read_csv(input_file)
    df = df[df.iloc[:, 0].isin(SUBJECTS)].reset_index(drop=True)
    num_rois = N_ROIS

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
    
    # Expected number of connections (upper triangle only)
    expected_connections = (num_rois * (num_rois - 1)) // 2
    
    # Verify the data has the expected number of columns
    actual_connections = len(column_names)
    if actual_connections != expected_connections:
        print(f"Warning: Expected {expected_connections} connections for {num_rois} ROIs, but found {actual_connections}.")
        print("Continuing with processing, but results may be incorrect.")
        
    # Process each subject
    print(f"Processing {len(subjects)} subjects...")
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

        # Convert to DataFrame with original ROI labels
        conn_df = pd.DataFrame(conn_matrix, index=roi_labels, columns=roi_labels)

        # Save to CSV
        if is_functional:
            if is_fisher_z:
                output_file = output_dir / f"{subject}_{ses}_functional_connectivity_matrix_fisher_z.csv"
            else:
                output_file = output_dir / f"{subject}_{ses}_functional_connectivity_matrix.csv"
        else:
            output_file = output_dir / f"{subject}_{ses}_structural_connectivity_matrix.csv"
        conn_df.to_csv(output_file)

def main():
    for ses in SESSIONS:
        print("--------------------------")
        print(f"Processing session: {ses}")
        print("--------------------------")

        structural_dir = STRUCT_ROOT / ses
        structural_matrix = structural_dir / "all_to_all_roi_matrices" / "all_to_all_roi_matrix.csv"
        struct_output_dir = structural_dir / "individual_connectivity_matrices"
        os.makedirs(struct_output_dir, exist_ok=True)

        functional_dir = FC_ROOT / ses
        func_filename = "fisher_z_all_to_all_roi_matrix.csv" if IS_FISHER_Z else "all_to_all_roi_matrix.csv"
        func_matrix = functional_dir / "all_to_all_roi_matrices" / func_filename
        func_output_dir = functional_dir / "individual_connectivity_matrices"
        os.makedirs(func_output_dir, exist_ok=True)

        # Process structural connectivity matrices
        print("Processing structural connectivity matrices...")
        create_individual_matrices(structural_matrix, struct_output_dir, ses, is_functional=False, is_fisher_z=False)

        # Process functional connectivity matrices
        print("Processing functional connectivity matrices...")
        create_individual_matrices(func_matrix, func_output_dir, ses, is_functional=True, is_fisher_z=IS_FISHER_Z)

        print("Conversion complete!")

if __name__ == "__main__":
    main()