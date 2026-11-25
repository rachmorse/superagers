"""Flatten each subject's FC and SC matrices to one mean connectivity value per ROI for the classifiers."""

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FC_FLAT,
    FC_MATRIX_FISHER_Z,
    FC_ROOTS,
    FLAT_DIR,
    INDIVIDUAL_MATRIX_DIR,
    ML_ATLAS,
    SC_FLAT,
    SC_MATRIX,
    SC_ROOTS,
    SESSIONS,
    SFC_FILE,
    SFC_INDIVIDUAL_DIR,
    SFC_ROOTS,
    SUBJECTS,
)


def get_subjects_to_process(sfc_dir: Path, ses: str) -> List[str]:
    """Generate a list of subjects to process.

    Args:
        sfc_dir: Path to the directory with the coupling results.
        ses: Session ID (format: ses-01).

    Returns:
        Subjects in the final cohort with coupling results for the session, sorted.
    """
    return sorted(subject for subject in SUBJECTS if (sfc_dir / SFC_FILE.format(subject=subject, ses=ses)).exists())


def flatten_connectivity_csv(matrix_csv: Path, measure_col: str = "pearson_rho") -> pd.DataFrame:
    """Reads an NxN connectivity CSV (214x214) and flattens it into a long DataFrame.

    Takes the row-wise mean to match the format of SFC data which takes the rho
    of each ROI connection in the row, weighing them the same.

    Args:
        matrix_csv: Path to the NxN CSV.
        measure_col: Name for the connectivity measure (e.g. "sc_value", "fc_value").

    Returns:
        DataFrame with columns ["ROI_name", measure_col].

    Raises:
        ValueError: If the matrix is not square or its row and column ROI labels differ.
    """
    # Read the NxN matrix
    df_matrix = pd.read_csv(matrix_csv, header=0, index_col=0)
    row_rois = df_matrix.index.astype(str).tolist()
    col_rois = df_matrix.columns.astype(str).tolist()

    # Validate structure to avoid silently computing invalid summaries.
    if df_matrix.shape[0] != df_matrix.shape[1]:
        raise ValueError(f"Connectivity matrix must be square, got shape {df_matrix.shape} for {matrix_csv}.")
    if row_rois != col_rois:
        raise ValueError(
            f"Row/column ROI labels do not match in {matrix_csv}. "
            f"First row labels: {row_rois[:5]} | first column labels: {col_rois[:5]}"
        )

    roi_list = df_matrix.index.tolist()  # ROI names

    # Exclude self-connections from the row mean by removing diagonal entries.
    n = df_matrix.shape[0]
    arr = df_matrix.to_numpy(dtype=float, copy=True)
    arr[np.arange(n), np.arange(n)] = np.nan
    avg_conn = pd.DataFrame(arr, index=df_matrix.index).mean(axis=1, skipna=True).values

    # Build the output DataFrame
    return pd.DataFrame({"ROI_name": roi_list, measure_col: avg_conn})


def main() -> None:
    """Flatten the FC and SC matrices of every subject with coupling results at either session."""
    # Get the list of subjects to process
    sfc_root = SFC_ROOTS[ML_ATLAS]
    subjects_tp1 = get_subjects_to_process(sfc_root / "ses-01" / SFC_INDIVIDUAL_DIR, "ses-01")
    subjects_tp2 = get_subjects_to_process(sfc_root / "ses-02" / SFC_INDIVIDUAL_DIR, "ses-02")
    print(f"Subjects at tp1: {len(subjects_tp1)}")
    print(f"Subjects at tp2: {len(subjects_tp2)}")
    subjects = sorted(set(subjects_tp1) | set(subjects_tp2))

    # Connectivity type -> (root folder, matrix file name, flat file name)
    connectivity = {
        "FC": (FC_ROOTS[ML_ATLAS], FC_MATRIX_FISHER_Z, FC_FLAT),
        "SC": (SC_ROOTS[ML_ATLAS], SC_MATRIX, SC_FLAT),
    }
    found = dict.fromkeys(connectivity, 0)
    missing = dict.fromkeys(connectivity, 0)

    # Flatten the NxN FC and SC matrices to per-ROI mean connectivity vectors
    n_total = len(subjects) * len(SESSIONS)
    for i, (sub, ses) in enumerate(((s, ses) for s in subjects for ses in SESSIONS), start=1):
        for conn, (root, matrix_name, flat_name) in connectivity.items():
            ses_path = root / ses / INDIVIDUAL_MATRIX_DIR
            matrix_csv = ses_path / matrix_name.format(subject=sub, ses=ses)
            if matrix_csv.is_file():
                flat = flatten_connectivity_csv(matrix_csv, measure_col="pearson_rho")
                output_dir = ses_path / FLAT_DIR
                output_dir.mkdir(parents=True, exist_ok=True)
                flat.to_csv(output_dir / flat_name.format(subject=sub, ses=ses), index=False)
                found[conn] += 1
            else:
                missing[conn] += 1

        if i % 25 == 0 or i == n_total:
            print(f"Progress: {i}/{n_total} subject-sessions processed")

    print(f"\nFC: {found['FC']} written, {missing['FC']} missing")
    print(f"SC: {found['SC']} written, {missing['SC']} missing")


if __name__ == "__main__":
    main()
