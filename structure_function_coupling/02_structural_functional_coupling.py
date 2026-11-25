"""Compute regional structure-function coupling from each subject's SC and Fisher z FC matrices."""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FC_MATRIX_FISHER_Z,
    FC_ROOT,
    INDIVIDUAL_MATRIX_DIR,
    SC_MATRIX,
    SC_ROOT,
    SESSIONS,
    SFC_FILE,
    SFC_FILE_FISHER_Z,
    SFC_INDIVIDUAL_DIR,
    SFC_ROOT,
    SUBJECTS,
)

SAVE_RAW = True  # Save Pearson's rho
SAVE_FISHER_Z = True  # Also save the Fisher z-transformed rho

Coupling = Dict[str, Union[str, np.ndarray, List[str]]]


def get_subjects_to_process(
    output_folder: Path, ses: str, functional_dir: Path, structural_dir: Path, require_fisher_z: bool = False
) -> Tuple[List[str], List[str]]:
    """Generate a list of subjects to process.

    Subjects are processed if they have a structural and functional matrix and do not have
    coupling for the specified timepoint.

    Args:
        output_folder: Path to the directory coupling results
        ses: Session ID (format: ses-01).
        functional_dir: Path to the directory containing functional connectivity matrices.
        structural_dir: Path to the directory containing structural connectivity matrices.
        require_fisher_z: If True, only consider subjects that are missing the Fisher z-transformed coupling file as
            needing processing. If False, only consider subjects missing the raw coupling file.

    Returns:
        List of subject IDs to process, list of already processed subject IDs.
    """
    subjects_to_process = []
    already_processed = []

    for subject in SUBJECTS:
        func_conn_path = functional_dir / FC_MATRIX_FISHER_Z.format(subject=subject, ses=ses)
        struct_conn_path = structural_dir / SC_MATRIX.format(subject=subject, ses=ses)
        output_file_path = output_folder / SFC_FILE.format(subject=subject, ses=ses)
        output_file_fz_path = output_folder / SFC_FILE_FISHER_Z.format(subject=subject, ses=ses)

        needs_processing = not output_file_path.exists()
        if require_fisher_z:
            needs_processing = needs_processing or not output_file_fz_path.exists()

        if func_conn_path.exists() and struct_conn_path.exists() and needs_processing:
            subjects_to_process.append(subject)
        elif output_file_path.exists() and (not require_fisher_z or output_file_fz_path.exists()):
            already_processed.append(subject)
        elif struct_conn_path.exists() and not func_conn_path.exists():
            print(f"Functional connectivity matrix not found for {subject} {ses} likely due to scrubbing exclusion.")
        elif not struct_conn_path.exists():
            print(f"Structural connectivity matrix not found for {subject} {ses}.")

    return subjects_to_process, already_processed


def calculate_structure_function_coupling(
    structural_dir: Path, functional_dir: Path, subject: str, ses: str
) -> Coupling:
    """Calculate structure-function coupling for each ROI.

    Args:
        structural_dir: Directory containing structural connectivity matrices
        functional_dir: Directory containing functional connectivity matrices
        subject: Subject identifier
        ses: Timepoint

    Returns:
        Dictionary containing coupling results

    Raises:
        ValueError: If the structural and functional matrices have different dimensions.
    """
    # Load connectivity matrices
    func_conn_df = pd.read_csv(functional_dir / FC_MATRIX_FISHER_Z.format(subject=subject, ses=ses))
    struct_conn_df = pd.read_csv(structural_dir / SC_MATRIX.format(subject=subject, ses=ses))

    # Extract ROI names from the first column (excluding the header row)
    roi_names = func_conn_df.iloc[:, 0].tolist()

    # Drop the first row and column to get the numeric data only
    func_conn_matrix = func_conn_df.iloc[:, 1:].to_numpy()
    struct_conn_matrix = struct_conn_df.iloc[:, 1:].to_numpy()

    # Check matrix dimensions match
    if func_conn_matrix.shape != struct_conn_matrix.shape:
        raise ValueError(
            f"Matrix dimensions don't match: functional {func_conn_matrix.shape} vs structural {struct_conn_matrix.shape}"
        )

    n_rois = len(roi_names)

    # Initialize coupling data structures
    coupling = {"subject": subject, "ses": ses, "rho": np.zeros((n_rois, 1)), "roi_names": roi_names}

    # Calculate structure-function coupling for each ROI
    for m in range(n_rois):
        # Get structural and functional connectivity for this region
        x = struct_conn_matrix[:, m]
        y = func_conn_matrix[:, m]

        # Find indices where both are non-zero
        idx_both_non_zero = (x != 0) & (y != 0)
        idx_both_non_zero[m] = False  # Set diagonal to False

        x_filtered = x[idx_both_non_zero]
        y_filtered = y[idx_both_non_zero]

        if len(x_filtered) == 0 or len(y_filtered) == 0:
            coupling["rho"][m, 0] = np.nan
        else:
            coupling["rho"][m, 0] = stats.pearsonr(x_filtered, y_filtered)[0]

    return coupling


def save_coupling_results(
    coupling_dict: Coupling, output_path: Path, save_raw: bool = True, save_fisher_z: bool = False
) -> None:
    """Save coupling results to CSV files.

    Args:
        coupling_dict: The coupling dictionary returned by calculate_structure_function_coupling
        output_path: Directory to save results
        save_raw: Whether to save the raw Pearson's rho values
        save_fisher_z: Whether to save the Fisher z-transformed values
    """
    output_path.mkdir(parents=True, exist_ok=True)

    subject = coupling_dict["subject"]
    ses = coupling_dict["ses"]
    rho_raw = coupling_dict["rho"].flatten()

    # Raw and Fisher z rho are saved the same way, each with ROI names
    outputs = [(save_raw, rho_raw, SFC_FILE), (save_fisher_z, np.arctanh(rho_raw), SFC_FILE_FISHER_Z)]
    for save, rho, file_name in outputs:
        if save:
            results_df = pd.DataFrame({"ROI_name": coupling_dict["roi_names"], "pearson_rho": rho})
            output_file = output_path / file_name.format(subject=subject, ses=ses)
            results_df.to_csv(output_file, index=False)
            print(f"Completed processing for {subject}: {output_file}")


def main() -> None:
    """Compute coupling for every subject that still needs it, for each session."""
    for ses in SESSIONS:
        print("--------------------------")
        print(f"Processing {ses} ...")
        print("--------------------------")

        structural_dir = SC_ROOT / ses / INDIVIDUAL_MATRIX_DIR
        functional_dir = FC_ROOT / ses / INDIVIDUAL_MATRIX_DIR
        output_dir = SFC_ROOT / ses / SFC_INDIVIDUAL_DIR

        # Make sure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        subjects_to_process, already_processed = get_subjects_to_process(
            output_dir,
            ses,
            functional_dir,
            structural_dir,
            require_fisher_z=SAVE_FISHER_Z,
        )
        print(f"Number of subjects already processed: {len(already_processed)}")
        print(f"Number of subjects to process: {len(subjects_to_process)}")

        for subject in subjects_to_process:
            # Calculate structure-function coupling for the specified ses
            results = calculate_structure_function_coupling(structural_dir, functional_dir, subject, ses)

            save_coupling_results(results, output_dir, save_raw=SAVE_RAW, save_fisher_z=SAVE_FISHER_Z)


if __name__ == "__main__":
    main()
