import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path
import os
import re

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

def bootstrap_stability_enet(
        X, y, roi_names,
        baseline_X=None,               # e.g. memory_1 + demographics
        stab_threshold=0.75,           # define "stable"
        n_boot=500,
        random_state=42,
        l1_grid=(0.1, 0.5, 0.9),
        alpha_grid=np.logspace(-2, 1, 10)):
    """
    Bootstraps subjects, refits ElasticNetCV, returns selection probs + prints
    extra diagnostics.

    Parameters
    ----------
    baseline_X : ndarray or None
        If provided, will compute 5-fold CV R² for:
            baseline only        (model_A)
            baseline + X         (model_B)
        and report ΔR² = B – A
    """
    rng = np.random.default_rng(random_state)
    p   = X.shape[1]
    counts  = np.zeros(p, dtype=int)
    coefs   = np.zeros((n_boot, p))

    for b in range(n_boot):
        idx = rng.integers(X.shape[0], size=X.shape[0])  # subject bootstrap
        X_b, y_b = X[idx], y[idx]

        en = ElasticNetCV(
            l1_ratio=l1_grid,
            alphas=alpha_grid,
            cv=5,
            random_state=rng.integers(1e9)
        ).fit(X_b, y_b)

        counts += (en.coef_ != 0)
        coefs[b] = en.coef_

    # ---------- selection probability ----------
    stab = counts / n_boot
    stab_series = pd.Series(stab, index=roi_names).sort_values(ascending=False)

    # ---------- extra metrics ----------
    stable_mask = stab_series >= stab_threshold
    n_stable    = stable_mask.sum()
    med_abs_coef = (np.median(np.abs(coefs[:, stable_mask]).mean(axis=0))
                    if n_stable else np.nan)
    
    print(f"\nStability-selection summary (threshold ≥ {stab_threshold:.2f})")
    print(f"  • Stable ROIs      : {n_stable} / {p}")
    if n_stable:
        print(f"  • Median |coef|    : {med_abs_coef:.4f}")

    # ---------- optional incremental R² ----------
    if baseline_X is not None:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_base, r2_full = [], []

        for train, test in kf.split(X):
            # baseline-only
            y_mean = y[train].mean()            # trivial predictor
            r2_base.append(r2_score(y[test], np.full_like(y[test], y_mean)))

            # baseline + ROIs via winning hyper-params from full data
            X_tr  = np.hstack([baseline_X[train], X[train]])
            X_te  = np.hstack([baseline_X[test],  X[test]])
            en    = ElasticNetCV(l1_ratio=l1_grid,
                                 alphas=alpha_grid,
                                 cv=5,
                                 random_state=rng.integers(1e9)).fit(X_tr, y[train])
            r2_full.append(r2_score(y[test], en.predict(X_te)))

        r2_base = np.mean(r2_base)
        r2_full = np.mean(r2_full)
        print(f"  • CV R² demographics   : {r2_base:.3f}")
        print(f"  • CV R² demographics+R : {r2_full:.3f}")
        print(f"  • Δ R²             : {r2_full - r2_base:+.3f}")

    return stab_series

def make_long_df(X_t1, X_t2, y_t1, y_t2, subjects, superager_vec, stable_rois):
    rows = []
    for i, sid in enumerate(subjects):
        for tp, (Xrow, yval) in enumerate([(X_t1[i], y_t1[i]),
                                           (X_t2[i], y_t2[i])]):
            row = {"id":  sid,
                   "time": tp,                 # 0 / 1
                   "y":    yval,
                   "Group": superager_vec[i]}  # 0 CTL, 1 SA
            # keep only the 8 stable ROI columns
            row.update({col: Xrow[j] for j, col in enumerate(stable_rois)})
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    connectivity_type = "SFC"  # Options: "SFC", "FC", "SC"
    sessions = ["ses-01", "ses-02"] 
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
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
        if connectivity_type == "SFC":
            ses_01_path = root_path / "ses-01" / "individual_coupling_matrices"
            csv_t1 = ses_01_path / f"{sub}_ses-01_structure_function_coupling_grouped.csv"
        elif connectivity_type == "FC":
            csv_t1 = fc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_functional_connectivity_grouped.csv"
        else:
            csv_t1 = sc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_structural_connectivity_grouped.csv"
        if csv_t1.is_file():
            df_t1 = pd.read_csv(csv_t1)
            sfc_t1_flat = pd.to_numeric(df_t1['pearson_rho'].values, errors='coerce') # Use 'pearson_rho' column which is where SFC values are
            X_t1.append(sfc_t1_flat)
            if sub in memory_dict:
                y_t1.append(memory_dict[sub]["ses-01"])
        else:
            print(f"File not found: {csv_t1}")

        if connectivity_type == "SFC":
            ses_02_path = root_path / "ses-02" / "individual_coupling_matrices"
            csv_t2 = ses_02_path / f"{sub}_ses-02_structure_function_coupling_grouped.csv"
        elif connectivity_type == "FC":
            csv_t2 = fc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_functional_connectivity_grouped.csv"
        else:
            csv_t2 = sc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_structural_connectivity_grouped.csv"  

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

    # After you have your final subject list and have applied 'valid'
    final_subjects = np.array(subjects)[valid]  # subjects in the same order as X_slope

    # Build a boolean mask for superagers for only these subjects
    superager_mask = memory_data.set_index("id").loc[final_subjects, "superager"].values == 1
    non_superager_mask = memory_data.set_index("id").loc[final_subjects, "superager"].values == 0

    # Compute slopes
    X_slope = (X_t2 - X_t1) / age_diff[:, None]
    y_slope = (y_t2 - y_t1) / age_diff

    # Filter to superagers
    X_slope_SA = X_slope[superager_mask]
    y_slope_SA = y_slope[superager_mask]

    # Filter to non-superagers 
    X_slope_non_SA = X_slope[non_superager_mask]
    y_slope_non_SA = y_slope[non_superager_mask]

    # Scale the data
    sc_SA   = StandardScaler().fit(X_slope_SA)
    X_slope_SA   = sc_SA.transform(X_slope_SA)

    sc_non_SA   = StandardScaler().fit(X_slope_non_SA)
    X_slope_non_SA   = sc_non_SA.transform(X_slope_non_SA)

    # Map feature indices back to ROI names 
    roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling_grouped.csv"
    roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    # Convert sex and cohort to numeric
    memory_data["sex"] = memory_data["sex"].map({"male": 0, "female": 1})


    print(f"Running bootstrap stability analysis for {connectivity_type}...")

    demographic_cols = memory_data.set_index("id").loc[subjects, ["memory_1", "age_1", "sex", "YoE"]].values

    stab = bootstrap_stability_enet(
            X_slope, y_slope,
            roi_names=roi_names,
            baseline_X=demographic_cols,
            stab_threshold=0.75,
            n_boot=1000,
            random_state=42)

    print("\nStable ROIs (≥ 75 %):")
    print(stab[stab >= 0.75].apply(lambda x: f"{x:.2%}"))

    # Save the stable ROI names
    stable_rois = stab[stab >= 0.75].index.tolist()


    ########################################################
    # I do not think this is running corrently because 
    # when run with maintainers I get no relationship between
    # group and memory which of course does not make sense.
    ########################################################

    # 1) Define the outcome variable (column name in df_long)
    outcome = "y"

    # 2) Build the long DataFrame (make sure superager_vec is defined for each subject)
    superager_vec = memory_data.set_index("id").loc[final_subjects, "superager"].values
    df_long = make_long_df(X_t1, X_t2, y_t1, y_t2, subjects, superager_vec, stable_rois)

    # 3) Scale only the stable ROI columns
    sc = StandardScaler()
    df_long[stable_rois] = sc.fit_transform(df_long[stable_rois])

    def sanitise_cols(df):
        rename_map = {}
        for c in df.columns:
            safe = re.sub('[^0-9a-zA-Z_]', '_', c)
            if re.match('^[0-9]', safe):      # can’t start with digit
                safe = f'X_{safe}'
            rename_map[c] = safe
        return df.rename(columns=rename_map), rename_map

    df_long, name_map = sanitise_cols(df_long)

    stable_rois_safe = [name_map[roi] for roi in stable_rois]   # update list

    # 4) Fit a mixed-effects model with main + interaction terms
    import statsmodels.formula.api as smf

    roi_terms = " + ".join(stable_rois_safe)
    int_terms = " + ".join([f"Group:{roi}" for roi in stable_rois_safe])
    formula   = f"{outcome} ~ Group + time + {roi_terms} + {int_terms}"

    md = smf.mixedlm(formula, data=df_long, groups="id")  # random intercept
    fit = md.fit(reml=False)
    print(fit.summary())

    # 5) Run multiple-comparison correction on interaction p-values
    from statsmodels.stats.multitest import multipletests

    print("Correcting p-values for interaction terms...")
    p_int = fit.pvalues[[f"Group:{roi}" for roi in stable_rois_safe]]
    rej, qvals, _, _ = multipletests(p_int, method="fdr_bh")
    print(pd.Series(qvals, index=p_int.index).sort_values())

    # # Subset your feature matrices to only those columns
    # X_slope_SA_stable = X_slope_SA[:, stable_indices]
    # X_slope_non_SA_stable = X_slope_non_SA[:, stable_indices]
    
    # # Run the group analyses with only stable ROIs
    # print(f"Running bootstrap stability analysis for {connectivity_type} with superagers...")

    # stab_SA = bootstrap_stability_enet(
    #         X_slope_SA_stable, y_slope_SA,
    #         roi_names=stable_rois,
    #         baseline_X=demographic_cols,
    #         stab_threshold=0.75,
    #         n_boot=1000,
    #         random_state=42)

    # print("\nStable ROIs (≥ 75 %):")
    # print(stab_SA[stab_SA >= 0.75].apply(lambda x: f"{x:.2%}"))

    # print(f"Running bootstrap stability analysis for {connectivity_type} with non-superagers...")

    # stab_non_SA = bootstrap_stability_enet(
    #     X_slope_non_SA_stable, y_slope_non_SA,
    #     roi_names=stable_rois,
    #     baseline_X=demographic_cols,
    #     stab_threshold=0.75,
    #     n_boot=1000,
    #     random_state=42)

    # print("\nStable ROIs (≥ 75 %):")
    # print(stab_non_SA[stab_non_SA >= 0.75].apply(lambda x: f"{x:.2%}"))

    # # ----------------------------------------------------
    # # 1) Fit an Elastic Net on Time 1 data
    # # ----------------------------------------------------
    # en_t1 = ElasticNetCV(
    #     l1_ratio=[0.1, 0.5, 0.9],   # Example l1-ratio grid
    #     alphas=np.logspace(-2, 1, 10),
    #     cv=5,
    #     random_state=42
    # )
    # en_t1.fit(X_t1, y_t1)

    # # Extract coefficients
    # coefs_t1 = en_t1.coef_

    # # Determine which features are nonzero
    # selected_t1 = np.flatnonzero(coefs_t1)

    # # ----------------------------------------------------
    # # 2) Fit an Elastic Net on Time 2 data
    # # ----------------------------------------------------
    # en_t2 = ElasticNetCV(
    #     l1_ratio=[0.1, 0.5, 0.9],
    #     alphas=np.logspace(-2, 1, 10),
    #     cv=5,
    #     random_state=42
    # )
    # en_t2.fit(X_t2, y_t2)

    # coefs_t2 = en_t2.coef_
    # selected_t2 = np.flatnonzero(coefs_t2)

    # # ----------------------------------------------------
    # # 3) Fit an Elastic Net on slope/change data
    # # ----------------------------------------------------
    # en_slope = ElasticNetCV(
    #     l1_ratio=[0.1, 0.5, 0.9],
    #     alphas=np.logspace(-2, 1, 10),
    #     cv=5,
    #     random_state=42
    # )
    # en_slope.fit(X_slope, y_slope)

    # coefs_slope = en_slope.coef_
    # selected_slope = np.flatnonzero(coefs_slope)

    # # ----------------------------------------------------
    # # 4) Combine selected features:
    # # ----------------------------------------------------
    # common_features = set(selected_t2).intersection(set(selected_slope)) # Only between tp2 and slope as tp1 had none
    # all_features    = set(selected_t1).union(set(selected_t2), set(selected_slope))

    # # Create a boolean mask for all p features 
    # p = X_t1.shape[1]  

    # # Create a mask for tp2 features
    # final_mask_tp2= np.zeros(p, dtype=bool)
    # for f in selected_t2:
    #     final_mask_tp2[f] = True

    # # And for slope features
    # final_mask_slope = np.zeros(p, dtype=bool)
    # for f in selected_slope:
    #     final_mask_slope[f] = True

    # # Map feature indices back to ROI names 
    # roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling_grouped.csv"
    # roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    # # Print feature names for selected features
    # print("Features selected at Time 1 only (names):", [roi_names[int(f)] for f in selected_t1])
    # print("Features selected at Time 2 only (names):", [roi_names[int(f)] for f in selected_t2])
    # print("Features selected for slope (names):", [roi_names[int(f)] for f in selected_slope])
    # print("Common features (names):", [roi_names[int(f)] for f in sorted(common_features)])

    # # Gather group labels from memory_data["superager"] for each subject
    # group_labels = []
    # for sub in subjects:
    #     row = memory_data[memory_data["id"] == sub]
    #     if not row.empty:
    #         group_labels.append(row.iloc[0]["superager"])  # 0 or 1
    #     else:
    #         # If subject not found, append NaN or handle error
    #         group_labels.append(np.nan)

    # group_labels = np.array(group_labels)
    # group_labels = group_labels[valid]

    # # Subset Time 2 and slope data to final features only
    # X_t2_final = X_t2[:, final_mask_tp2]      # shape = [N, number_of_selected_features]
    # X_slope_final = X_slope[:, final_mask_slope]

    # # Corresponding memory outcomes
    # y_t2_final = y_t2
    # y_slope_final = y_slope

    # # Create masks for each group
    # groupA_mask = (group_labels == 0)
    # groupB_mask = (group_labels == 1)

    # # 1) Fit separate models on Time 2 data
    # print("\n--- Elastic Net on Time 2 data, separate by group ---")
    # X_A_t2 = X_t2_final[groupA_mask]
    # X_B_t2 = X_t2_final[groupB_mask]
    # y_A_t2 = y_t2_final[groupA_mask]
    # y_B_t2 = y_t2_final[groupB_mask]

    # enA_t2 = fit_and_report_en(X_A_t2, y_A_t2, group_name="A (Time 2)")
    # enB_t2 = fit_and_report_en(X_B_t2, y_B_t2, group_name="B (Time 2)")

    # # 2) Fit separate models on slope data
    # print("\n--- Elastic Net on Slope data, separate by group ---")
    # X_A_slope = X_slope_final[groupA_mask]
    # X_B_slope = X_slope_final[groupB_mask]
    # y_A_slope = y_slope_final[groupA_mask]
    # y_B_slope = y_slope_final[groupB_mask]

    # enA_slope = fit_and_report_en(X_A_slope, y_A_slope, group_name="A (Slope)")
    # enB_slope = fit_and_report_en(X_B_slope, y_B_slope, group_name="B (Slope)")

    # # Print coefficients for each selected feature set by group
    # final_feature_indices_slope = np.where(final_mask_slope)[0]
    # final_feature_indices_tp2 = np.where(final_mask_tp2)[0]

    # print("\n--- Coefficients for final selected features (Time 2) ---")
    # for idx, orig_idx in enumerate(final_feature_indices_tp2):
    #     print(f"{roi_names[orig_idx]}: Group A (non-SA) (Time 2): {enA_t2.coef_[idx]:.4f}, Group B (SA) (Time 2): {enB_t2.coef_[idx]:.4f}")


    # print("\n--- Coefficients for final selected features (Slope) ---")
    # for idx, orig_idx in enumerate(final_feature_indices_slope):
    #     print(f"{roi_names[orig_idx]}: Group A (non-SA) (Slope): {enA_slope.coef_[idx]:.4f}, Group B (SA) (Slope): {enB_slope.coef_[idx]:.4f}")

if __name__ == "__main__":
    main()