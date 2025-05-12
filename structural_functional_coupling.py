import numpy as np
import pandas as pd
from scipy import stats
import glob
from pathlib import Path

def load_connectivity_matrix(filepath):
    """Load a connectivity matrix from a CSV file.
    
    Args:
        filepath (str or Path): Path to the CSV file containing the connectivity matrix.
        
    Returns:
        numpy.ndarray or None: The loaded connectivity matrix or None if loading failed.
    """
    try:
        return pd.read_csv(filepath, header=None).values
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def calculate_structure_function_coupling(structural_dir, functional_dir, output_dir, ses):
    """Calculate the structure-function coupling for each brain region.
    
    Args:
        structural_dir (str or Path): Directory containing structural connectivity matrices.
        functional_dir (str or Path): Directory containing functional connectivity matrices.
        output_dir (str or Path): Directory to save the results.
        ses (str): Session identifier (e.g., "ses-02").
        
    Returns:
        dict: Dictionary containing the coupling results for the session.
    """
    # Convert input paths to Path objects
    structural_dir = Path(structural_dir)
    functional_dir = Path(functional_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"Processing session: {ses}")
    
    # Find structural connectivity matrix
    structural_files = list(Path(structural_dir / ses).glob("*.csv"))
    if not structural_files:
        print(f"No structural connectivity files found for session {ses}")
        return None
    
    # Load structural connectivity matrix
    str_conn_matrix = load_connectivity_matrix(structural_files[0])
    if str_conn_matrix is None:
        return None
    
    # Find functional connectivity matrix
    func_path = functional_dir / ses / "all_to_all_roi_matrices" / "all_to_all_roi matrix.csv"
    if not func_path.exists():
        print(f"Functional connectivity file not found: {func_path}")
        return None
    
    # Load functional connectivity matrix
    func_conn_matrix = load_connectivity_matrix(func_path)
    if func_conn_matrix is None:
        return None
        
    # Replace negative functional connectivity values with 0
    func_conn_matrix[func_conn_matrix < 0] = 0
    
    # Initialize arrays for storing coupling metrics
    num_regions = str_conn_matrix.shape[0]
    coupling_rho = np.zeros(num_regions)
    coupling_pvalue = np.zeros(num_regions)
    
    # Calculate structure-function coupling for each brain region
    for m in range(num_regions):
        # Define structural and functional connectivity for region m
        X = str_conn_matrix[:, m]
        Y = func_conn_matrix[:, m]
        
        # Select elements where both structural and functional connectivities are non-zero
        idx_both_nonzero = (X != 0) & (Y != 0)
        idx_both_nonzero[m] = False  # Set diagonal to False to exclude it
        
        X_nonzero = X[idx_both_nonzero]
        Y_nonzero = Y[idx_both_nonzero]
        
        # Calculate coupling using Pearson correlation
        if len(X_nonzero) == 0 or len(Y_nonzero) == 0:
            coupling_rho[m] = np.nan
            coupling_pvalue[m] = np.nan
        else:
            corr_result = stats.pearsonr(X_nonzero, Y_nonzero)
            coupling_rho[m] = corr_result[0]
            coupling_pvalue[m] = corr_result[1]
    
    # Store results for this session
    results[ses] = {
        'coupling_rho': coupling_rho,
        'coupling_pvalue': coupling_pvalue
    }
    
    # Save results for this session - include {ses} in the output directory
    ses_output_dir = output_dir / f"{ses}"
    ses_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV - include {ses} in the filename
    results_df = pd.DataFrame({
        'region_index': np.arange(num_regions),
        'pearson_rho': coupling_rho,
        'pearson_pvalue': coupling_pvalue
    })
    results_df.to_csv(ses_output_dir / f"structure_function_coupling_{ses}.csv", index=False)
    
    # Save as NPY for easy loading in Python - include {ses} in the filename
    np.savez(
        ses_output_dir / f"structure_function_coupling_{ses}.npz",
        coupling_rho=coupling_rho,
        coupling_pvalue=coupling_pvalue
    )
    
    return results

if __name__ == "__main__":
    # Define the directories using Path
    structural_dir = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    functional_dir = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    output_dir = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    
    # Specify the session you want to process
    ses = "ses-02"
    
    # Calculate structure-function coupling for the specified session
    results = calculate_structure_function_coupling(structural_dir, functional_dir, output_dir, ses)
    
    if results:
        print(f"Structure-function coupling analysis completed for {ses}.")
    else:
        print(f"Structure-function coupling analysis failed for {ses}.")