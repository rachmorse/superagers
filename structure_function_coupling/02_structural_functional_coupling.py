import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import seaborn as sns
import nibabel as nib
from nilearn import plotting, surface
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_surf_fsaverage
import matplotlib.pyplot as plt


def get_subjects_to_process(output_folder, ses, mask_dir, functional_dir, structural_dir, require_fisher_z=False):
    """Generate a list of subjects to process based on whether they have
    a structural and functional matrix and do not have coupling for the
    specified timepoint.

    Args:
        output_folder (Path): Path to the directory coupling results
        ses (str): Session ID (format: ses-01).
        functional_dir (Path): Path to the directory containing functional connectivity matrices.
        structural_dir (Path): Path to the directory containing structural connectivity matrices.
        mask_dir (Path): Path to the directory containing the mask files to use to check which subjects to process.
        require_fisher_z (bool): If True, only consider subjects that are missing the Fisher z-transformed coupling file as needing processing. 
            If False, only consider subjects missing the raw coupling file.

    Returns:
        list: List of subject IDs to process.
    """
    subjects_to_process = []
    already_processed = []

    # Iterate over all possible subject directories
    for subject in mask_dir.iterdir():
        subject = str(subject.name)
        if not subject.startswith("sub-"):
            continue
        
        # Check if the required directory exists and hasn't been processed yet  
        func_conn_path = functional_dir / f"{subject}_{ses}_functional_connectivity_matrix_fisher_z.csv"
        struct_conn_path = structural_dir / f"{subject}_{ses}_structural_connectivity_matrix.csv"
        output_file_path = output_folder / f"{subject}_{ses}_structure_function_coupling.csv"
        output_file_fz_path = output_folder / f"{subject}_{ses}_structure_function_coupling_fisher_z.csv"

        needs_processing = not output_file_path.exists()
        if require_fisher_z:
            needs_processing = needs_processing or not output_file_fz_path.exists()

        if func_conn_path.exists() and struct_conn_path.exists() and needs_processing:
            subjects_to_process.append(subject)
        elif output_file_path.exists() and (not require_fisher_z or output_file_fz_path.exists()):
            already_processed.append(subject)
        elif struct_conn_path.exists() and not func_conn_path.exists():
            print(f"Functional connectivity matrix not found for {subject} {ses} likely due to scrubbing exclusion.")

    return subjects_to_process, already_processed


def calculate_structure_function_coupling(structural_dir, functional_dir, subject, ses):
    """Calculate structure-function coupling for each ROI
    
    Args:
        subject (str): Subject identifier
        ses (str): Timepoint 
        structural_dir (str or Path): Directory containing structural connectivity matrices
        functional_dir (str or Path): Directory containing functional connectivity matrices
    
    Returns:
        dict: Dictionary containing coupling results
    """
    # Define paths using specific directory structure
    func_conn_path = functional_dir / f"{subject}_{ses}_functional_connectivity_matrix_fisher_z.csv"
    struct_conn_path = structural_dir / f"{subject}_{ses}_structural_connectivity_matrix.csv"

    # Load connectivity matrices
    func_conn_df = pd.read_csv(func_conn_path)
    struct_conn_df = pd.read_csv(struct_conn_path)

    # Extract ROI names from the first column (excluding the header row)
    roi_names = func_conn_df.iloc[:, 0].tolist()
    
    # Drop the first row and column to get the numeric data only
    func_conn_matrix = func_conn_df.iloc[:, 1:].to_numpy()
    struct_conn_matrix = struct_conn_df.iloc[:, 1:].to_numpy()

    # Check matrix dimensions match
    if func_conn_matrix.shape != struct_conn_matrix.shape:
        raise ValueError(f"Matrix dimensions don't match: "
                       f"functional {func_conn_matrix.shape} vs structural {struct_conn_matrix.shape}")
    
    n_rois = len(roi_names)
    
    # Initialize coupling data structures
    coupling = {
        "subject": subject,
        "ses": ses,
        "rho": np.zeros((n_rois, 1)),
        "roi_names": roi_names
    }
    
    # Calculate structure-function coupling for each ROI
    for m in range(n_rois):
        # Get structural and functional connectivity for this region
        X = struct_conn_matrix[:, m]
        Y = func_conn_matrix[:, m]
        
        # Find indices where both are non-zero
        idx_both_non_zero = (X != 0) & (Y != 0)
        idx_both_non_zero[m] = False  # Set diagonal to False
        
        X_filtered = X[idx_both_non_zero]
        Y_filtered = Y[idx_both_non_zero]
        
        if len(X_filtered) == 0 or len(Y_filtered) == 0:
            coupling["rho"][m,0] = np.nan
        else:
            rho = stats.pearsonr(X_filtered, Y_filtered)[0]
            coupling["rho"][m,0] = rho
    
    return coupling


def save_coupling_results(coupling_dict, output_path, save_raw=True, save_fisher_z=False):
    """Save coupling results to CSV files
    
    Args:
        coupling_dict (dict): The coupling dictionary returned by calculate_structure_function_coupling
        output_path (str or Path): Directory to save results
        save_raw (bool): Whether to save the raw Pearson's rho values
        save_fisher_z (bool): Whether to save the Fisher z-transformed values 
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subject = coupling_dict["subject"]
    ses = coupling_dict["ses"]
    
    # Combine ROI index and rho into a single DataFrame
    rho_raw = coupling_dict["rho"].flatten()

    output_file = None
    if save_raw:
        results_df = pd.DataFrame({
            'ROI_name': coupling_dict["roi_names"],
            'pearson_rho': coupling_dict["rho"].flatten(),
        })
        output_file = output_path / f"{subject}_{ses}_structure_function_coupling.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Completed processing for {subject}: {output_file}")

    if save_fisher_z:
        rho_fz = np.arctanh(rho_raw)
        results_df_fz = pd.DataFrame({
            'ROI_name': coupling_dict["roi_names"],
            'pearson_rho': rho_fz,
        })
        output_file_fz = output_path / f"{subject}_{ses}_structure_function_coupling_fisher_z.csv"
        results_df_fz.to_csv(output_file_fz, index=False)
        print(f"Completed processing for {subject}: {output_file_fz}")

    return output_file


def main():
    # Define the directories using Path
    sessions = ["ses-01", "ses-02"]
    save_raw = True
    save_fisher_z = True  # When True, saves *_fisher_z.csv alongside raw

    for ses in sessions:
        print("--------------------------")
        print(f"Processing {ses} ...")
        print("--------------------------")

        structural_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/structural_connectivity/{ses}/individual_connectivity_matrices")
        functional_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space/{ses}/individual_connectivity_matrices")
        output_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/structure_function_coupling/{ses}/individual_coupling_matrices")
        mask_dir = Path(f"/home/rachel/Desktop/schaefer_analysis/fsaverage/{ses}")
        
        # Make sure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        subjects_to_process, already_processed = get_subjects_to_process(
            output_dir,
            ses,
            mask_dir,
            functional_dir,
            structural_dir,
            require_fisher_z=save_fisher_z,
        )
        print(f"Number of subjects already processed: {len(already_processed)}")
        print(f"Number of subjects to process: {len(subjects_to_process)}")
        
        # Uncomment to process manual list
        # subjects_to_process = ["sub-134084"]

        for subject in subjects_to_process:
            # Calculate structure-function coupling for the specified ses
            results = calculate_structure_function_coupling(structural_dir, functional_dir, subject, ses)

            save_coupling_results(results, output_dir, save_raw=save_raw, save_fisher_z=save_fisher_z)
        
        
if __name__ == "__main__":
    main()
