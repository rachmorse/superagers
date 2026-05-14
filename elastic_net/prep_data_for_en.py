import pandas as pd
import numpy as np
from pathlib import Path
import os, re

def get_subjects_to_process(output_folder, ses, age_dir):
    """Generate a list of subjects to process. Also, filters for any subs with
    <1.5 years follow-up time.

    Args:
        output_folder (Path): Path to the directory coupling results
        ses (str): Session ID (format: ses-01).
        age_dir (Path): Path to the directory containing the age data CSV.

    Returns:
        list: List of subject IDs to process.
    """
    # Merge in the age data
    age_data = pd.read_csv(age_dir / "superager.csv")
    age_data.columns = [re.sub(r"^w(\d)_(.*)", r"\2_\1", col) for col in age_data.columns] # Rename the columns
    age_data['id'] = 'sub-' + age_data['id'].astype(str) # Add 'sub-' to the id
    age_data_filt = age_data[['id', 'age_1', 'age_2']].copy() # Keep only the relevant columns

    # Apply follow-up filter only when both ages are present:
    has_both_ages = age_data_filt[['age_1', 'age_2']].notna().all(axis=1)
    fu_time = age_data_filt['age_2'] - age_data_filt['age_1']
    keep_mask = (~has_both_ages) | (fu_time > 1.5)

    # Create a set of valid ids under the conditional follow-up rule
    valid_ids = set(age_data_filt.loc[keep_mask, 'id'].astype(str))

    # Subset ids to only those in valid_ids
    subjects = set()
    for fname in os.listdir(output_folder):
        if not fname.startswith("sub-") or not fname.endswith(f"{ses}_structure_function_coupling.csv"):
            continue

        # Extract subject ID 
        subject = fname.split("_")[0]

        sfc_path = output_folder / fname
        if sfc_path.exists() and subject in valid_ids:
            subjects.add(subject)

    return sorted(subjects)


def flatten_connectivity_csv(matrix_csv, measure_col="pearson_rho"):
    """Reads an NxN connectivity CSV (214×214) and flattens it into a long DataFrame
    by taking the row-wise mean to match the format of SFC data which takes the rho
    of each ROI connection in the row, weighing them the same. 

    Args:
        matrix_csv (str or Path): Path to the NxN CSV.
        measure_col (str): Name for the connectivity measure (e.g. "sc_value", "fc_value").

    Returns:
        pd.DataFrame with columns ["ROI_name", measure_col].
    """
    # Read the NxN matrix 
    df_matrix = pd.read_csv(matrix_csv, header=0, index_col=0)
    row_rois = df_matrix.index.astype(str).tolist()
    col_rois = df_matrix.columns.astype(str).tolist()

    # Validate structure to avoid silently computing invalid summaries.
    if df_matrix.shape[0] != df_matrix.shape[1]:
        raise ValueError(
            f"Connectivity matrix must be square, got shape {df_matrix.shape} for {matrix_csv}."
        )
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
    df_out = pd.DataFrame({
        "ROI_name": roi_list,
        measure_col: avg_conn
    })

    return df_out



def main():
    sessions = ["ses-01", "ses-02"]
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    age_dir = Path("/home/rachel/Desktop/data")

    # Get the list of subjects to process
    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01", age_dir)
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02", age_dir)
    print(f"Subjects at tp1: {len(subjects_tp1)}")
    print(f"Subjects at tp2: {len(subjects_tp2)}")
    subjects = sorted(set(subjects_tp1) | set(subjects_tp2))

    # Flatten the NxN FC and SC matrices to per-ROI mean connectivity vectors
    for sub in subjects:
        for ses in sessions:
            ses_path_fc = fc_root_path / ses / "individual_connectivity_matrices"
            ses_path_sc = sc_root_path / ses / "individual_connectivity_matrices"

            # Functional connectivity
            fc_csv = ses_path_fc / f"{sub}_{ses}_functional_connectivity_matrix_fisher_z.csv"
            if fc_csv.is_file():
                fc_flat = flatten_connectivity_csv(fc_csv, measure_col="pearson_rho")
                fc_output_dir = ses_path_fc / "grouped_rois"
                fc_output_dir.mkdir(parents=True, exist_ok=True)
                fc_flat.to_csv(fc_output_dir / f"{sub}_{ses}_functional_connectivity_flat.csv", index=False)

            # Structural connectivity
            sc_csv = ses_path_sc / f"{sub}_{ses}_structural_connectivity_matrix.csv"
            if sc_csv.is_file():
                sc_flat = flatten_connectivity_csv(sc_csv, measure_col="pearson_rho")
                sc_output_dir = ses_path_sc / "grouped_rois"
                sc_output_dir.mkdir(parents=True, exist_ok=True)
                sc_flat.to_csv(sc_output_dir / f"{sub}_{ses}_structural_connectivity_flat.csv", index=False)
            else:
                print(f"Missing SC CSV at path: {sc_csv}")


if __name__ == "__main__":
    main()
