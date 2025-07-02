import numpy as np
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np
from pathlib import Path
import os

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

def main():
    sessions = ["ses-01", "ses-02"] 
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    memory_data = pd.read_csv("/home/rachel/Desktop/data/clean_data_all.csv")

    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01")
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02")
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # Create the memory_dict from the DataFrame
    memory_dict = {
        f"{row['id']}": {"ses-01": row['memory_1'], "ses-02": row['memory_2']}
        for _, row in memory_data.iterrows()
    }

    # Initialize lists to hold feature rows and outcome values
    X_t1 = [] # an N×p array of features at Time 1 for N subjects, p features
    X_t2 = [] 
    y_t1 = [] # corresponding memory outcome measures
    y_t2 = [] 

    for sub in subjects:
        ses_01_path = root_path / "ses-01" / "individual_coupling_matrices"
        csv_t1 = ses_01_path / f"{sub}_ses-01_structure_function_coupling.csv"
        if csv_t1.is_file():
            df_t1 = pd.read_csv(csv_t1)
            sfc_t1_flat = pd.to_numeric(df_t1['pearson_rho'].values, errors='coerce') # Use 'pearson_rho' column which is where SFC values are
            X_t1.append(sfc_t1_flat)
            if sub in memory_dict:
                y_t1.append(memory_dict[sub]["ses-01"])
        else:
            print(f"File not found: {csv_t1}")

        ses_02_path = root_path / "ses-02" / "individual_coupling_matrices"
        csv_t2 = ses_02_path / f"{sub}_ses-02_structure_function_coupling.csv"
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

    # Only keep subjects with valid age difference
    valid = ~np.isnan(age_diff) & (age_diff != 0)
    X_t1 = X_t1[valid]
    X_t2 = X_t2[valid]
    y_t1 = y_t1[valid]
    y_t2 = y_t2[valid]
    age_diff = age_diff[valid]

    # Compute slopes
    X_slope = (X_t2 - X_t1) / age_diff[:, None]
    y_slope = (y_t2 - y_t1) / age_diff

    print("X_slope shape:", X_slope.shape)
    print("y_slope shape:", y_slope.shape)

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
    # (X_slope might represent Time2 - Time1 features)
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
    # Either intersection (common to all) or union (any).
    # ----------------------------------------------------
    common_features = set(selected_t2).intersection(set(selected_slope), set(selected_slope))
    all_features    = set(selected_t1).union(set(selected_t2), set(selected_slope))

    # Create a boolean mask for all p features (True if feature is in the final chosen set)
    p = X_t1.shape[1]  # or set p to the number of features
    final_mask = np.zeros(p, dtype=bool)
    for f in common_features:
        final_mask[f] = True

    # --- Map feature indices back to ROI names ---
    roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling.csv"
    roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    # Print feature names for selected features
    print("Features selected at Time 1 only (names):", [roi_names[int(f)] for f in selected_t1])
    print("Features selected at Time 2 only (names):", [roi_names[int(f)] for f in selected_t2])
    print("Features selected for slope (names):", [roi_names[int(f)] for f in selected_slope])
    print("Common features (names):", [roi_names[int(f)] for f in sorted(common_features)])

    # -----------------------------------------------------------------------------
    # Gather group labels from memory_data["maintainer"] for each subject
    # -----------------------------------------------------------------------------
    group_labels = []
    for sub in subjects:
        row = memory_data[memory_data["id"] == sub]
        if not row.empty:
            group_labels.append(row.iloc[0]["maintainer"])  # 0 or 1
        else:
            # If subject not found, append NaN or handle error
            group_labels.append(np.nan)

    group_labels = np.array(group_labels)

    # Filter the same "valid" subjects we used earlier
    group_labels = group_labels[valid]

    # -----------------------------------------------------------------------------
    # Subset Time 2 and slope data to final features only
    # -----------------------------------------------------------------------------
    X_t2_final = X_t2[:, final_mask]      # shape = [N, number_of_selected_features]
    X_slope_final = X_slope[:, final_mask]

    # Corresponding memory outcomes
    y_t2_final = y_t2
    y_slope_final = y_slope

    # -----------------------------------------------------------------------------
    # Create masks for each group
    # -----------------------------------------------------------------------------
    groupA_mask = (group_labels == 0)
    groupB_mask = (group_labels == 1)

    # -----------------------------------------------------------------------------
    # Function to run Elastic Net and print results
    # -----------------------------------------------------------------------------
    def fit_and_report_en(X, y, group_name):
        from sklearn.linear_model import ElasticNetCV

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

    # -----------------------------------------------------------------------------
    # 1) Fit separate models on Time 2 data
    # -----------------------------------------------------------------------------
    print("\n--- Elastic Net on Time 2 data, separate by group ---")
    X_A_t2 = X_t2_final[groupA_mask]
    X_B_t2 = X_t2_final[groupB_mask]
    y_A_t2 = y_t2_final[groupA_mask]
    y_B_t2 = y_t2_final[groupB_mask]

    enA_t2 = fit_and_report_en(X_A_t2, y_A_t2, group_name="A (Time 2)")
    enB_t2 = fit_and_report_en(X_B_t2, y_B_t2, group_name="B (Time 2)")

    # -----------------------------------------------------------------------------
    # 2) Fit separate models on slope data
    # -----------------------------------------------------------------------------
    print("\n--- Elastic Net on Slope data, separate by group ---")
    X_A_slope = X_slope_final[groupA_mask]
    X_B_slope = X_slope_final[groupB_mask]
    y_A_slope = y_slope_final[groupA_mask]
    y_B_slope = y_slope_final[groupB_mask]

    enA_slope = fit_and_report_en(X_A_slope, y_A_slope, group_name="A (Slope)")
    enB_slope = fit_and_report_en(X_B_slope, y_B_slope, group_name="B (Slope)")

    # Example: Identify Nonzero Coefficients for Group A

    # Get coefficients from the fitted models
    a_t2_coefs = enA_t2.coef_         # From Time 2 model for Group A
    a_slope_coefs = enA_slope.coef_   # From Slope model for Group A

    # Get indices where coefficients are nonzero
    nonzero_a_t2 = [i for i, val in enumerate(a_t2_coefs) if val != 0]
    nonzero_a_slope = [i for i, val in enumerate(a_slope_coefs) if val != 0]

    # Intersect them
    common_indices_a = set(nonzero_a_t2).intersection(nonzero_a_slope)

    # Map indices -> ROI names (roi_names should be already defined)
    common_roi_names_a = [roi_names[i] for i in sorted(common_indices_a)]
    print("Common features (nonzero) for Group A in Time 2 & Slope:")
    print(common_roi_names_a)

    # Get coefficients from the fitted models
    b_t2_coefs = enB_t2.coef_         # From Time 2 model for Group b
    b_slope_coefs = enB_slope.coef_   # From Slope model for Group b

    # Get indices where coefficients are nonzero
    nonzero_b_t2 = [i for i, val in enumerate(b_t2_coefs) if val != 0]
    nonzero_b_slope = [i for i, val in enumerate(b_slope_coefs) if val != 0]

    # Intersect them
    common_indices_b = set(nonzero_b_t2).intersection(nonzero_b_slope)

    # Map indices -> ROI names (roi_names should be already defined)
    common_roi_names_b = [roi_names[i] for i in sorted(common_indices_b)]
    print("Common features (nonzero) for Group B in Time 2 & Slope:")
    print(common_roi_names_b)

    # Figure out which features are common to both groups
    common_features = set(common_indices_a).intersection(common_indices_b)
    common_roi_names = [roi_names[i] for i in sorted(common_features)]
    print("Common features (nonzero) for both groups in Time 2 & Slope:")
    print(common_roi_names)

    # Figure out which features are unique to each group
    unique_a = set(common_indices_a).difference(common_indices_b)
    unique_b = set(common_indices_b).difference(common_indices_a)
    unique_roi_names_a = [roi_names[i] for i in sorted(unique_a)]
    unique_roi_names_b = [roi_names[i] for i in sorted(unique_b)]
    print("Unique features for Group A in Time 2 & Slope:")
    print(unique_roi_names_a)
    print("Unique features for Group B in Time 2 & Slope:")
    print(unique_roi_names_b)

    # Print feature name and coefficient for unique features in Group A (Time 2)
    print("Unique features for Group A in Time 2 & Slope (name, coef):")
    for i in sorted(unique_a):
        print(f"{roi_names[i]}: {enA_t2.coef_[i]}")

    # Print feature name and coefficient for unique features in Group B (Time 2)
    print("Unique features for Group B in Time 2 & Slope (name, coef):")
    for i in sorted(unique_b):
        print(f"{roi_names[i]}: {enB_t2.coef_[i]}")


if __name__ == "__main__":
    main()