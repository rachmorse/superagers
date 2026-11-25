"""Split the group SC and Fisher z FC matrices (one row per subject) into one matrix per subject."""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ALL_TO_ALL_DIR,
    ALL_TO_ALL_LABEL,
    FC_MATRIX_FISHER_Z,
    FC_ROOT,
    FISHER_Z_PREFIX,
    INDIVIDUAL_MATRIX_DIR,
    MATRIX_CSV,
    N_ROIS,
    SC_MATRIX,
    SC_ROOT,
    SESSIONS,
    SUBJECTS,
)


def split_roi_pair(column: str) -> Optional[Tuple[str, str]]:
    """Get the two ROI names from a connection column name (e.g. "ROI1-ROI2").

    Args:
        column: Connection column name.

    Returns:
        The two ROI names, or None if the column is not a connection.
    """
    matches = re.findall(r"([^-]+)-([^-]+)", column)
    if not matches:
        return None
    roi1, roi2 = matches[0]
    return roi1.strip(), roi2.strip()


def create_individual_matrices(input_file: Path, output_dir: Path, ses: str, output_name: str) -> None:
    """Convert a CSV file of connectivity matrices into individual matrices for each subject.

    Args:
        input_file: Path to the input CSV file.
        output_dir: Directory to save the individual matrices.
        ses: Session (e.g., "ses-01", "ses-02").
        output_name: File name template for each subject's matrix (SC_MATRIX or FC_MATRIX_FISHER_Z).
    """
    # Read the input csv file
    df = pd.read_csv(input_file)
    df = df[df.iloc[:, 0].isin(SUBJECTS)].reset_index(drop=True)

    # Get the subject IDs
    subjects = df.iloc[:, 0].tolist()

    # Extract ROI labels from column names
    column_names = df.columns[1:]
    roi_pairs = {col: split_roi_pair(col) for col in column_names}

    # Get a unique set of ROI labels from the connection columns, as a sorted list
    roi_labels = sorted({roi for pair in roi_pairs.values() if pair for roi in pair})

    # Verify it has the expected number of ROIs
    if len(roi_labels) != N_ROIS:
        print(f"Warning: Expected {N_ROIS} ROIs, but found {len(roi_labels)}.")
        print("Using detected ROI labels, but results may be incorrect.")
        # If it can't extract labels, create generic ones
        if len(roi_labels) == 0:
            roi_labels = [f"ROI_{i + 1}" for i in range(N_ROIS)]
            print("Created generic ROI labels.")

    # Create a mapping from ROI pairs to matrix indices
    roi_to_index = {roi: i for i, roi in enumerate(roi_labels)}

    # Expected number of connections (upper triangle only)
    expected_connections = (N_ROIS * (N_ROIS - 1)) // 2

    # Verify the data has the expected number of columns
    actual_connections = len(column_names)
    if actual_connections != expected_connections:
        print(
            f"Warning: Expected {expected_connections} connections for {N_ROIS} ROIs, but found {actual_connections}."
        )
        print("Continuing with processing, but results may be incorrect.")

    # Process each subject
    print(f"Processing {len(subjects)} subjects...")
    for i, subject in enumerate(subjects):
        # Create an empty connectivity matrix
        conn_matrix = np.zeros((N_ROIS, N_ROIS))

        # Get the connectivity values for this subject
        row_data = df.iloc[i]

        # Fill the connectivity matrix based on column names
        for col, pair in roi_pairs.items():
            if pair:
                # Get the indices for these ROIs
                idx1 = roi_to_index.get(pair[0])
                idx2 = roi_to_index.get(pair[1])

                if idx1 is not None and idx2 is not None:
                    value = row_data[col]
                    conn_matrix[idx1, idx2] = value
                    conn_matrix[idx2, idx1] = value  # Make it symmetric

        # Zero out the diagonal
        np.fill_diagonal(conn_matrix, 0)

        # Convert to DataFrame with original ROI labels and save to CSV
        conn_df = pd.DataFrame(conn_matrix, index=roi_labels, columns=roi_labels)
        conn_df.to_csv(output_dir / output_name.format(subject=subject, ses=ses))


def main() -> None:
    """Split the group SC and Fisher z FC matrices into individual matrices for each session."""
    sc_group_file = MATRIX_CSV.format(label=ALL_TO_ALL_LABEL)
    fc_group_file = MATRIX_CSV.format(label=FISHER_Z_PREFIX + ALL_TO_ALL_LABEL)

    for ses in SESSIONS:
        print("--------------------------")
        print(f"Processing session: {ses}")
        print("--------------------------")

        structural_dir = SC_ROOT / ses
        struct_output_dir = structural_dir / INDIVIDUAL_MATRIX_DIR
        struct_output_dir.mkdir(parents=True, exist_ok=True)

        functional_dir = FC_ROOT / ses
        func_output_dir = functional_dir / INDIVIDUAL_MATRIX_DIR
        func_output_dir.mkdir(parents=True, exist_ok=True)

        # Process structural connectivity matrices
        print("Processing structural connectivity matrices...")
        create_individual_matrices(structural_dir / ALL_TO_ALL_DIR / sc_group_file, struct_output_dir, ses, SC_MATRIX)

        # Process functional connectivity matrices
        print("Processing functional connectivity matrices...")
        create_individual_matrices(
            functional_dir / ALL_TO_ALL_DIR / fc_group_file, func_output_dir, ses, FC_MATRIX_FISHER_Z
        )

        print("Conversion complete!")


if __name__ == "__main__":
    main()
