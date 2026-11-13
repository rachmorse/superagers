"""Generate SIFT2-weighted structural connectivity matrices with MRtrix tck2connectome.

For each subject and session, builds an ROI-to-ROI connectome from the tractogram and
native space atlas mask, then appends the all-to-all, within-network and subcortical
matrices to group CSV files.
"""

import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import functions from functional connectivity script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fmri_analysis"))
from compute_functional_connectivity import (
    create_network_mappings,
    prepare_directories,
)

ATLAS: Literal["schaefer200", "schaefer400"] = "schaefer400"
TEST_SUBJECT: Optional[str] = None  # e.g. "sub-1014" to run only that subject
SUBJECTS_FILE = Path("/home/rachel/Desktop/data/final_cohort_ids.txt")  # final longitudinal cohort
SUBJECTS: List[str] = [TEST_SUBJECT] if TEST_SUBJECT else SUBJECTS_FILE.read_text().split()
COHORTS: List[Literal["bbhi", "bbhi senior"]] = ["bbhi", "bbhi senior"]
SESSIONS: List[Literal["ses-01", "ses-02"]] = ["ses-01", "ses-02"]

TRACTOGRAM_DIRS: Dict[str, Path] = {
    "bbhi": Path("/pool/guttmann/institut/BBHI/MRI/processed_data/tracto_SIFT2"),
    "bbhi senior": Path("/pool/guttmann/institut/UB/Superagers/MRI/tracto_SIFT2"),
}
MRTRIX_BIN = Path("/home/rachel/miniconda3/bin")
N_THREADS = 2  # MRtrix uses all cores otherwise

# Unsuffixed output folders are the original run, each atlas writes to its own suffixed folders
# Schaefer 200 masks are the exception and live in fsaverage
ROOT = Path("/home/rachel/Desktop/schaefer_analysis")
MASK_ROOT = ROOT / ("fsaverage" if ATLAS == "schaefer200" else f"fsaverage_{ATLAS}")
OUTPUT_ROOT = ROOT / f"structural_connectivity_{ATLAS}"
LABELS_CSV = ROOT / f"timeseries_data_{ATLAS}" / "native_space" / "combined_labels.csv"


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up basic logging to file and console.

    Args:
        output_dir: Path to the output directory where the log file will be saved.

    Returns:
        Logger object for logging messages.
    """
    log_file = output_dir / f"structural_connectivity_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def get_subjects_to_process(tractogram_dir: Path, mask_dir: Path, output_dir: Path, ses: str) -> List[str]:
    """Generate a list of subjects to process.

    Subjects are chosen based on whether they have a native space mask and the tractogram file
    but don't already have an entry in the output CSV.

    Args:
        tractogram_dir: Path to the directory containing the tractogram files.
        mask_dir: Path to the directory containing the native space masks.
        output_dir: Path to the output directory where matrices will be saved.
        ses: Timepoint (format ses-01 or ses-02).

    Returns:
        List of subject IDs to process.
    """
    subjects_to_process = []
    tractogram_subjects = []

    # Find the subs with tractograms
    for sub_dir in tractogram_dir.glob("sub-*"):
        if not sub_dir.is_dir() or "_ses-" not in sub_dir.name:
            continue
        sub, session = sub_dir.name.split("_", 1)  # e.g., sub-1283, ses-02

        fname = f"{sub}_{session}_dwi_tractogram_10M.tck"
        full = sub_dir / fname

        weights_name = f"{sub}_{session}_dwi_tractogram_10M_SIFT2_weights.txt"
        weights_full = sub_dir / weights_name

        if session != ses:
            continue
        if full.is_file() and weights_full.is_file():
            tractogram_subjects.append(sub_dir.name)
        elif not full.is_file():
            print(f"Tractogram not found for {sub_dir.name}")
        else:
            print(f"SIFT2 weights not found for {sub_dir.name}")

    # Define output CSV paths
    all_to_all_csv = output_dir / ses / "all_to_all_roi_matrices" / "all_to_all_roi_matrix.csv"

    # Load existing subjects from CSV if it exists
    existing_subjects = set()
    if all_to_all_csv.exists():
        print("Output CSV exists. Checking for already processed subjects...")
        try:
            existing_df = pd.read_csv(all_to_all_csv, index_col="Unnamed: 0")
            existing_subjects = set(existing_df.index)
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    # Iterate through subjects with tractograms
    for subject in tractogram_subjects:
        sub, session = subject.split("_")

        # Check if native space mask exists
        mask_file = Path(f"{mask_dir}/{session}/{sub}/dwi_space_masks/{subject}_{ATLAS}_subcortical14_dwi_space.nii.gz")
        if sub not in SUBJECTS:
            continue
        # Check if subject is already processed (in the CSV)
        if mask_file.exists() and sub not in existing_subjects and session == ses:
            subjects_to_process.append(sub)
        elif not mask_file.exists() and session == ses:
            print(f"Native space mask not found for {subject}")

    print(f"Number of subjects to process: {len(subjects_to_process)}")
    return subjects_to_process


def save_structural_connectivity(
    subject_id: str, label: str, matrix: np.ndarray, roi_names: List[str], output_dir: Path
) -> None:
    """Save structural connectivity matrix without normalization.

    Only saves to the group CSV file, not individual subject files. Checks for a partial
    run and replaces the subject's row if needed.

    Args:
        subject_id: Subject ID.
        label: Label for the matrix (e.g., 'all_to_all_roi').
        matrix: Connectivity matrix.
        roi_names: List of ROI names.
        output_dir: Directory to save the output file.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the group file path
    all_subjects_file = output_dir / f"{label}_matrix.csv"

    # Flatten the matrix, skip diagonal elements
    flat_data = []
    column_names = []

    for i in range(len(roi_names)):
        for j in range(i + 1, len(roi_names)):  # Start from i+1 to get upper triangle only
            if i != j:  # Skip diagonal
                flat_data.append(matrix[i, j])
                column_names.append(f"{roi_names[i]}-{roi_names[j]}")

    # Create new row for this subject
    subject_df = pd.DataFrame([flat_data], index=[subject_id], columns=column_names)

    if all_subjects_file.exists():
        # Read existing file and check for a partial run
        all_subjects_df = pd.read_csv(all_subjects_file, index_col=0)
        all_subjects_df = all_subjects_df.drop(index=subject_id, errors="ignore")

        # Append new subject
        updated_df = pd.concat([all_subjects_df, subject_df])
        updated_df.to_csv(all_subjects_file)
    else:
        # Create new file with this subject
        subject_df.to_csv(all_subjects_file)


def visualize_sc_data(
    subject_id: str, connectivity_matrix: np.ndarray, output_directory: Path, ses: str, cmap: str = "RdBu_r"
) -> None:
    """Visualize structural connectivity matrices using the same colormap as functional connectivity.

    Args:
        subject_id: Subject ID for labeling the figure.
        connectivity_matrix: The structural connectivity matrix.
        output_directory: Directory to save the visualization.
        ses: Session identifier (e.g., "ses-01" or "ses-02").
        cmap: Colormap to use, default 'RdBu_r' to match FC visualization.
    """
    # Set up the figure
    plt.figure(figsize=(10, 8))

    # For structural connectivity, scale colors from 0 to the max value
    vmax = np.max(connectivity_matrix)

    # Create the heatmap
    im = plt.imshow(connectivity_matrix, cmap=cmap, vmin=0, vmax=vmax)

    # Add a colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Normalized Number of Streamlines")

    # Set title and labels
    plt.title(f"Structural Connectivity: {subject_id} ({ses})")
    plt.xlabel("ROI Index")
    plt.ylabel("ROI Index")

    # Ensure output directory exists
    vis_dir = Path(output_directory) / ses / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Save the figure
    output_file = vis_dir / f"{subject_id}_{ses}_structural_connectivity.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved visualization to {output_file}")


def generate_structural_connectivity(
    *,
    subject: str,
    tractogram_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    ses: str,
    labels_csv_path: Path,
    run_visualization: bool = True,
) -> Optional[np.ndarray]:
    """Generate a structural connectivity matrix using MRtrix tck2connectome, focusing on ROI-to-ROI connectivity.

    Args:
        subject: Subject ID.
        tractogram_dir: Directory containing tractogram files.
        mask_dir: Directory containing native space masks.
        output_dir: Directory to save output files.
        ses: Session / timepoint (format ses-01 or ses-02).
        labels_csv_path: Path to the CSV file containing ROI labels.
        run_visualization: Whether to visualize the matrix after generation.

    Returns:
        The generated connectivity matrix, or None if tck2connectome failed or flagged poor registration.
    """
    tractogram_file = tractogram_dir / f"{subject}_{ses}/{subject}_{ses}_dwi_tractogram_10M.tck"
    weights_file = tractogram_dir / f"{subject}_{ses}/{subject}_{ses}_dwi_tractogram_10M_SIFT2_weights.txt"

    print(f"Tractogram file: {tractogram_file}")
    print(f"Weights file: {weights_file}")

    mask_file = Path(
        f"{mask_dir}/{ses}/{subject}/dwi_space_masks/{subject}_{ses}_{ATLAS}_subcortical14_dwi_space.nii.gz"
    )

    # Create output directories
    prepare_directories(
        output_dir, ses, ["all_to_all_roi_matrices", "within_network_matrices", "subcortical_matrices", "visualization"]
    )

    # Create a temporary file for the matrix
    temp_dir = tempfile.gettempdir()
    temp_matrix_file = Path(temp_dir) / f"{subject}_{ses}_temp_matrix.csv"

    # Create a temporary file for the output
    temp_output_file = Path(temp_dir) / f"{subject}_{ses}_temp_output.txt"

    # Create the out_assignments file path
    # This allows future use of connectome2tck
    out_assignments_file = Path(output_dir) / f"out_assignments/{subject}_{ses}_out_assignments.txt"
    out_assignments_file.parent.mkdir(parents=True, exist_ok=True)

    # Run tck2connectome to generate the connectivity matrix
    cmd = [
        "tck2connectome",
        str(tractogram_file),
        str(mask_file),
        str(temp_matrix_file),
        "-out_assignments", str(out_assignments_file),  # Save assignments
        "-tck_weights_in", str(weights_file),  # File for SIFT2 weights
        "-force",  # Overwrite existing files
        "-zero_diagonal",  # Set diagonal elements to zero
        "-symmetric",  # Ensure matrix is symmetric
        "-scale_invnodevol",  # Scale by the inverse size of each node that the streamlines connect to
        "-nthreads", str(N_THREADS),
    ]  # fmt: skip

    try:
        print(f"Processing {subject} for {ses}...")

        # Run command and capture both stdout and stderr
        with open(temp_output_file, "w") as output_file:
            subprocess.run(cmd, stdout=output_file, stderr=subprocess.STDOUT, text=True, check=True)

        # Check for the warning message about poor registration
        with open(temp_output_file) as output_file:
            output = output_file.read()

        if "tck2connectome: [WARNING] (This may indicate a poor registration)" in output:
            print(f"WARNING: Poor registration detected for {subject}. Skipping this subject.")
            return None

        # If no poor registration warning, proceed normally
        print(f"Successfully created connectivity matrix for {subject}")

        # Read the generated connectivity matrix
        connectivity_matrix = np.loadtxt(temp_matrix_file, delimiter=",")

        # Read labels from CSV (same as in functional connectivity)
        combined_labels = pd.read_csv(labels_csv_path, header=None).squeeze().tolist()

        # Process network-specific matrices
        process_network_matrices(subject, connectivity_matrix, combined_labels, output_dir, ses)

        # Save the connectivity data to CSV
        # Saved last since get_subjects_to_process uses this CSV to mark subjects as done
        all_to_all_dir = output_dir / ses / "all_to_all_roi_matrices"
        save_structural_connectivity(
            subject_id=subject,
            label="all_to_all_roi",
            matrix=connectivity_matrix,
            roi_names=combined_labels,
            output_dir=all_to_all_dir,
        )

        # To run visualization
        if run_visualization:
            visualize_sc_data(
                subject_id=subject, connectivity_matrix=connectivity_matrix, output_directory=output_dir, ses=ses
            )

        return connectivity_matrix

    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing {subject}: {e}")
        logging.error(temp_output_file.read_text())  # tck2connectome's own error message
        return None

    finally:
        # Clean up temporary files
        temp_matrix_file.unlink(missing_ok=True)
        temp_output_file.unlink(missing_ok=True)


def process_network_matrices(
    subject_id: str, full_matrix: np.ndarray, combined_labels: List[str], output_dir: Path, ses: str
) -> None:
    """Process and save network-specific connectivity matrices.

    Args:
        subject_id: Subject ID.
        full_matrix: Full connectivity matrix.
        combined_labels: List of region labels.
        output_dir: Output directory.
        ses: Session / timepoint.
    """
    # Create network mappings
    network_mappings = create_network_mappings(combined_labels)

    # Process each network
    for network, indices in network_mappings.items():
        # Skip subcortical ROIs to handle separately
        if network == "Subcortical":
            continue

        # Extract network submatrix
        network_matrix = full_matrix[np.ix_(indices, indices)]
        network_labels = [combined_labels[i] for i in indices]

        # Save network connectivity data
        network_dir = output_dir / ses / "within_network_matrices"
        save_structural_connectivity(
            subject_id=subject_id,
            label=f"{network}_within_network",
            matrix=network_matrix,
            roi_names=network_labels,
            output_dir=network_dir,
        )

    # Process subcortical matrices if available
    if "Subcortical" in network_mappings:
        subcortical_indices = network_mappings["Subcortical"]

        # All subcortical ROIs
        subcortical_matrix = full_matrix[np.ix_(subcortical_indices, subcortical_indices)]
        subcortical_labels = [combined_labels[i] for i in subcortical_indices]

        # Save all subcortical connectivity data
        subcortical_dir = output_dir / ses / "subcortical_matrices"
        save_structural_connectivity(
            subject_id=subject_id,
            label="all_subcortical_rois",
            matrix=subcortical_matrix,
            roi_names=subcortical_labels,
            output_dir=subcortical_dir,
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
                    output_dir=subcortical_dir,
                )


def main() -> None:
    """Main function to process structural connectivity for subjects."""
    # Create output directory if it does not exist
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(OUTPUT_ROOT)

    # Setup MRTrix
    os.environ["PATH"] = f"{MRTRIX_BIN}:{os.environ['PATH']}"

    # Track both failed and successful subjects
    successful_subjects = []
    failed_subjects = []
    all_subjects = []

    for cohort in COHORTS:
        for ses in SESSIONS:
            print("------------------------------")
            print(f"Processing {cohort} {ses}...")
            print("------------------------------")

            tractogram_dir = TRACTOGRAM_DIRS[cohort]

            # Create output directories if they do not exist
            prepare_directories(
                OUTPUT_ROOT,
                ses,
                ["all_to_all_roi_matrices", "within_network_matrices", "subcortical_matrices", "visualization"],
            )

            # Get subjects to process
            subjects = get_subjects_to_process(tractogram_dir, MASK_ROOT, OUTPUT_ROOT, ses)
            all_subjects.extend(subjects)

            # For testing with a single subject
            # subjects = ["sub-1191"]

            # Process each subject
            for subject in subjects:
                logging.info(f"Processing {subject} for {ses}...")

                try:
                    result = generate_structural_connectivity(
                        subject=subject,
                        tractogram_dir=tractogram_dir,
                        mask_dir=MASK_ROOT,
                        output_dir=OUTPUT_ROOT,
                        ses=ses,
                        labels_csv_path=LABELS_CSV,
                        run_visualization=False,
                    )

                    # Check if processing was successful
                    if result is not None:
                        successful_subjects.append(subject)
                        logging.info(f"Successfully processed {subject}")
                    else:
                        failed_subjects.append(subject)
                        logging.error(f"Failed to process {subject} - generate_structural_connectivity returned None")

                except Exception as e:
                    logging.error(f"{subject}: Error during processing: {e!s}")
                    failed_subjects.append(subject)

    # Log summary
    total_subjects = len(all_subjects)
    successful_count = len(successful_subjects)
    failed_count = len(failed_subjects)

    print("------------------------------")
    print("Summary")
    print("------------------------------")
    logging.info(
        f"Processing complete: {successful_count}/{total_subjects} successful, {failed_count}/{total_subjects} failed"
    )
    if failed_subjects:
        logging.info(f"Failed subjects: {', '.join(failed_subjects)}")


if __name__ == "__main__":
    main()
