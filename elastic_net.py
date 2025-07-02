import numpy as np
from sklearn.linear_model import ElasticNetCV
import pandas as pd
from pathlib import Path
import os
import re

# Note this uses internal cross-validation, not an external holdout, meaning model metrics may be optimistic. 
# eg R^2 scores are based on the same data used to fit the model, not a separate test set.

def get_subjects_to_process(output_folder, ses):
    """Generate a list of subjects to process.
    Args:
        output_folder (Path): Path to the directory coupling results
        ses (str): Session ID (format: ses-01).

    Returns:
        list: List of subject IDs to process.
    """
    subjects = []

    # Iterate over all files in the folder
    for fname in os.listdir(output_folder):
        if not fname.startswith("sub-") or not fname.endswith(f"{ses}_structure_function_coupling.csv"):
            continue

        # Extract subject ID (everything before the first underscore)
        subject = fname.split("_")[0]

        sfc_path = output_folder / fname
        if sfc_path.exists():
            subjects.append(subject)

    return subjects

def save_grouped_roi_averages(csv_path, output_path):
    """
    Reads a subject's SFC CSV, groups by ROI prefix, 
    averages pearson_rho, and saves a new CSV.

    Args:
        csv_path (str or Path): Path to each subs SFC CSV file.
        output_path (str or Path): Path to save the grouped averages CSV.
    """
    df = pd.read_csv(csv_path)

    # Extract the group prefix (e.g. PFCv)
    df['ROI_group'] = df['ROI_name'].apply(lambda x: re.sub(r'_\d+$', '', x))

    # Group by ROI_group and take the average
    grouped = df.groupby('ROI_group', as_index=False)['pearson_rho'].mean()

    # Rename column back to ROI_name for output
    grouped = grouped.rename(columns={'ROI_group': 'ROI_name'})

    # Save to output_path
    grouped.to_csv(output_path, index=False)

def fit_and_report_en(X, y, group_name):
    """
    Fit an Elastic Net model and report the R^2 score and coefficients.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target variable.
        group_name (str): Name of the group for reporting.
    """
    en = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-2, 1, 10),
        cv=5,
        random_state=42
    )
    en.fit(X, y)
    print(f"Group {group_name} -> R^2: {en.score(X, y):.3f}")
    print("Coefficients:", en.coef_)
    return en


def main():
    sessions = ["ses-01", "ses-02"] 
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    memory_data = pd.read_csv("/home/rachel/Desktop/data/clean_data_all.csv")

    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01")
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02")
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # Make the grouped averages for each subject's SFC CSV
    for sub in subjects:
        for ses in sessions:
            ses_path = root_path / ses / "individual_coupling_matrices"
            csv_path = ses_path / f"{sub}_{ses}_structure_function_coupling.csv"
            output_path = ses_path / f"{sub}_{ses}_structure_function_coupling_grouped.csv"
            if csv_path.is_file():
                save_grouped_roi_averages(csv_path, output_path)

    # Create the memory_dict from the DataFrame
    memory_dict = {
        f"{row['id']}": {"ses-01": row['memory_1'], "ses-02": row['memory_2']}
        for _, row in memory_data.iterrows()
    }

    # Initialize lists to hold feature rows and outcome values
    X_t1 = [] # An N×p array of features at Time 1 for N subjects, p features
    X_t2 = [] 
    y_t1 = [] # Corresponding memory outcome measures
    y_t2 = [] 

    # Prepare the data 
    for sub in subjects:
        ses_01_path = root_path / "ses-01" / "individual_coupling_matrices"
        csv_t1 = ses_01_path / f"{sub}_ses-01_structure_function_coupling_grouped.csv"
        if csv_t1.is_file():
            df_t1 = pd.read_csv(csv_t1)
            sfc_t1_flat = pd.to_numeric(df_t1['pearson_rho'].values, errors='coerce') # Use 'pearson_rho' column which is where SFC values are
            X_t1.append(sfc_t1_flat)
            if sub in memory_dict:
                y_t1.append(memory_dict[sub]["ses-01"])
        else:
            print(f"File not found: {csv_t1}")

        ses_02_path = root_path / "ses-02" / "individual_coupling_matrices"
        csv_t2 = ses_02_path / f"{sub}_ses-02_structure_function_coupling_grouped.csv"
        if csv_t2.is_file():
            df_t2 = pd.read_csv(csv_t2)
            sfc_t2_flat = pd.to_numeric(df_t2['pearson_rho'].values, errors='coerce')
            X_t2.append(sfc_t2_flat)
            if sub in memory_dict:
                y_t2.append(memory_dict[sub]["ses-02"])
        else:
            print(f"File not found: {csv_t2}")

        if np.any(np.isnan(sfc_t1_flat)) or np.any(np.isnan(sfc_t2_flat)):
            print(f"Warning: NaNs found in features for {sub}")

    # Convert lists to numpy arrays
    X_t1 = np.array(X_t1)  # shape: [N_subjects, p_features]
    X_t2 = np.array(X_t2)
    y_t1 = np.array(y_t1)
    y_t2 = np.array(y_t2)

    print("X_t1 shape:", X_t1.shape)
    print("X_t2 shape:", X_t2.shape)
    print("y_t1 shape:", y_t1.shape)
    print("y_t2 shape:", y_t2.shape)

    # Build age arrays for the included subjects
    age_1 = []
    age_2 = []
    for sub in subjects:
        row = memory_data[memory_data['id'] == sub]
        if not row.empty:
            age_1.append(row.iloc[0]['age_1'])
            age_2.append(row.iloc[0]['age_2'])
        else:
            age_1.append(np.nan)
            age_2.append(np.nan)

    age_1 = np.array(age_1)
    age_2 = np.array(age_2)
    age_diff = age_2 - age_1

    # Only keep subjects with valid age data
    valid = ~np.isnan(age_diff) & (age_diff != 0)
    X_t1 = X_t1[valid]
    X_t2 = X_t2[valid]
    y_t1 = y_t1[valid]
    y_t2 = y_t2[valid]
    age_diff = age_diff[valid]

    # Compute slopes
    X_slope = (X_t2 - X_t1) / age_diff[:, None]
    y_slope = (y_t2 - y_t1) / age_diff

    # ----------------------------------------------------
    # 1) Fit an Elastic Net on Time 1 data
    # ----------------------------------------------------
    en_t1 = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],   # Example l1-ratio grid
        alphas=np.logspace(-2, 1, 10),
        cv=5,
        random_state=42
    )
    en_t1.fit(X_t1, y_t1)

    # Extract coefficients
    coefs_t1 = en_t1.coef_

    # Determine which features are nonzero
    selected_t1 = np.flatnonzero(coefs_t1)

    # ----------------------------------------------------
    # 2) Fit an Elastic Net on Time 2 data
    # ----------------------------------------------------
    en_t2 = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-2, 1, 10),
        cv=5,
        random_state=42
    )
    en_t2.fit(X_t2, y_t2)

    coefs_t2 = en_t2.coef_
    selected_t2 = np.flatnonzero(coefs_t2)

    # ----------------------------------------------------
    # 3) Fit an Elastic Net on slope/change data
    # ----------------------------------------------------
    en_slope = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-2, 1, 10),
        cv=5,
        random_state=42
    )
    en_slope.fit(X_slope, y_slope)

    coefs_slope = en_slope.coef_
    selected_slope = np.flatnonzero(coefs_slope)

    # ----------------------------------------------------
    # 4) Combine selected features:
    # ----------------------------------------------------
    common_features = set(selected_t2).intersection(set(selected_slope)) # Only between tp2 and slope as tp1 had none
    all_features    = set(selected_t1).union(set(selected_t2), set(selected_slope))

    # Create a boolean mask for all p features 
    p = X_t1.shape[1]  

    # Create a mask for tp2 features
    final_mask_tp2= np.zeros(p, dtype=bool)
    for f in selected_t2:
        final_mask_tp2[f] = True

    # And for slope features
    final_mask_slope = np.zeros(p, dtype=bool)
    for f in selected_slope:
        final_mask_slope[f] = True

    # Map feature indices back to ROI names 
    roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling_grouped.csv"
    roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    # Print feature names for selected features
    print("Features selected at Time 1 only (names):", [roi_names[int(f)] for f in selected_t1])
    print("Features selected at Time 2 only (names):", [roi_names[int(f)] for f in selected_t2])
    print("Features selected for slope (names):", [roi_names[int(f)] for f in selected_slope])
    print("Common features (names):", [roi_names[int(f)] for f in sorted(common_features)])

    # Gather group labels from memory_data["superager"] for each subject
    group_labels = []
    for sub in subjects:
        row = memory_data[memory_data["id"] == sub]
        if not row.empty:
            group_labels.append(row.iloc[0]["superager"])  # 0 or 1
        else:
            # If subject not found, append NaN or handle error
            group_labels.append(np.nan)

    group_labels = np.array(group_labels)
    group_labels = group_labels[valid]

    # Subset Time 2 and slope data to final features only
    X_t2_final = X_t2[:, final_mask_tp2]      # shape = [N, number_of_selected_features]
    X_slope_final = X_slope[:, final_mask_slope]

    # Corresponding memory outcomes
    y_t2_final = y_t2
    y_slope_final = y_slope

    # Create masks for each group
    groupA_mask = (group_labels == 0)
    groupB_mask = (group_labels == 1)

    # 1) Fit separate models on Time 2 data
    print("\n--- Elastic Net on Time 2 data, separate by group ---")
    X_A_t2 = X_t2_final[groupA_mask]
    X_B_t2 = X_t2_final[groupB_mask]
    y_A_t2 = y_t2_final[groupA_mask]
    y_B_t2 = y_t2_final[groupB_mask]

    enA_t2 = fit_and_report_en(X_A_t2, y_A_t2, group_name="A (Time 2)")
    enB_t2 = fit_and_report_en(X_B_t2, y_B_t2, group_name="B (Time 2)")

    # 2) Fit separate models on slope data
    print("\n--- Elastic Net on Slope data, separate by group ---")
    X_A_slope = X_slope_final[groupA_mask]
    X_B_slope = X_slope_final[groupB_mask]
    y_A_slope = y_slope_final[groupA_mask]
    y_B_slope = y_slope_final[groupB_mask]

    enA_slope = fit_and_report_en(X_A_slope, y_A_slope, group_name="A (Slope)")
    enB_slope = fit_and_report_en(X_B_slope, y_B_slope, group_name="B (Slope)")

    # Print coefficients for each selected feature set by group
    final_feature_indices_slope = np.where(final_mask_slope)[0]
    final_feature_indices_tp2 = np.where(final_mask_tp2)[0]

    print("\n--- Coefficients for final selected features (Time 2) ---")
    for idx, orig_idx in enumerate(final_feature_indices_tp2):
        print(f"{roi_names[orig_idx]}: Group A (non-SA) (Time 2): {enA_t2.coef_[idx]:.4f}, Group B (SA) (Time 2): {enB_t2.coef_[idx]:.4f}")


    print("\n--- Coefficients for final selected features (Slope) ---")
    for idx, orig_idx in enumerate(final_feature_indices_slope):
        print(f"{roi_names[orig_idx]}: Group A (non-SA) (Slope): {enA_slope.coef_[idx]:.4f}, Group B (SA) (Slope): {enB_slope.coef_[idx]:.4f}")

if __name__ == "__main__":
    main()