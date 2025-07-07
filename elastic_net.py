import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
import statsmodels.formula.api as smf
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

        # Extract subject ID 
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

def prep_data(subjects, root_path, fc_root_path, sc_root_path, memory_data, connectivity_type):
    """
    Prepares the data for analysis by extracting features and memory outcomes
    from the specified directories and memory data.

    Args:
        subjects (list): List of subject IDs to process.
        root_path (Path): Path to the root directory containing SFC data.
        fc_root_path (Path): Path to the root directory containing FC data.
        sc_root_path (Path): Path to the root directory containing SC data.
        memory_data (pd.DataFrame): DataFrame containing memory outcomes and demographics.
        connectivity_type (str): Type of connectivity data to process ("SFC", "FC", or "SC").
    """
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

    final_subjects = np.array(subjects)[valid]  # subjects in the same order as X_slope

    # Compute slopes
    X_slope = (X_t2 - X_t1) / age_diff[:, None]
    y_slope = (y_t2 - y_t1) / age_diff

    return X_t1, X_t2, y_t1, y_t2, X_slope, y_slope, age_diff, final_subjects

def bootstrap_stability_enet(
        X, y, roi_names,
        demographic_X=None,              
        stab_threshold=0.75,         
        n_boot=500,
        random_state=42,
        l1_grid=(0.1, 0.5, 0.9),
        alpha_grid=np.logspace(-2, 1, 10)):
    """
    Bootstraps subjects, refits ElasticNetCV, returns selection probability
    and prints extra diagnostics.

    Args:
        X (np.ndarray): Feature matrix of averaged feature values for SFC.
        y (np.ndarray): Target variable (e.g. memory scores).
        roi_names (list): List of ROI names corresponding to features in X.
        stab_threshold (float): Threshold for stability selection (default 0.75).
        n_boot (int): Number of bootstrap iterations (default 500).
        random_state (int): Random seed for reproducibility.
        l1_grid (tuple): Tuple of l1_ratio values to test in ElasticNetCV
            1 is pure Lasso, 0 is pure Ridge.
        alpha_grid (np.ndarray): Array of alpha values to test in ElasticNetCV.
            used to create lambda.
        demographic_X (np.ndarray): Compute 5-fold CV R² for:
            demographics only        (model_A)
            demographics + X         (model_B)
            and reports ΔR² = B – A
    """
    # Generates random number
    rng = np.random.default_rng(random_state) 
    p   = X.shape[1] # p is number of features
    counts  = np.zeros(p, dtype=int)
    coefs   = np.zeros((n_boot, p))

    for b in range(n_boot):
        idx = rng.integers(X.shape[0], size=X.shape[0])  # Subject bootstrap
        X_b, y_b = X[idx], y[idx] # X_b and y_b are bootstrapped samples

        en = ElasticNetCV(
            l1_ratio=l1_grid,
            alphas=alpha_grid,
            cv=5,
            random_state=rng.integers(1e9)
        ).fit(X_b, y_b)

        counts += (en.coef_ != 0)
        coefs[b] = en.coef_

    # Calculate stability
    stab = counts / n_boot
    stab_series = pd.Series(stab, index=roi_names).sort_values(ascending=False)

    # Add extra metrics 
    stable_mask = stab_series >= stab_threshold
    n_stable    = stable_mask.sum()
    med_abs_coef = (np.median(np.abs(coefs[:, stable_mask]).mean(axis=0))
                    if n_stable else np.nan)
    
    print(f"\nStability-selection summary (threshold ≥ {stab_threshold:.2f})")
    print(f"  • Stable ROIs      : {n_stable} / {p}")
    if n_stable:
        print(f"  • Median |coef|    : {med_abs_coef:.4f}")

    # Use this to estimate the role of just the demographics
    if demographic_X is not None:
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        r2_base, r2_full = [], []

        for train, test in kf.split(X):
            y_mean = y[train].mean()            # Trivial predictor
            r2_base.append(r2_score(y[test], np.full_like(y[test], y_mean)))

            # Demographics + ROIs via selected hyper-params from full data
            X_tr  = np.hstack([demographic_X[train], X[train]])
            X_te  = np.hstack([demographic_X[test],  X[test]])
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

def make_long_df(
    X_t1, X_t2,
    y_t1, y_t2,
    subjects,
    superager_vec,
    maintainer_vec,
    stable_rois,
    *,                          # keyword-only switches ↓
    scale_roi=True,
    sanitise_cols=True):
    """
    Build a long-format DataFrame and (optionally) standardise ROI columns
    and make column names Python safe (eg because some spaces in names are 
    creating weird formatting).

    Args:
        X_t1 (np.ndarray): Features at Time 1.
        X_t2 (np.ndarray): Features at Time 2.
        y_t1 (np.ndarray): Memory outcomes at Time 1.
        y_t2 (np.ndarray): Memory outcomes at Time 2.
        subjects (list): List of subject IDs.
        superager_vec (np.ndarray): Vector indicating superager status for each subject.
        maintainer_vec (np.ndarray): Vector indicating maintainer status for each subject.
        stable_rois (list): List of stable ROI names to include in the DataFrame.
    """
    # 1. Reshape into long form 
    rows = []
    for i, sid in enumerate(subjects):
        for tp, (Xrow, yval) in enumerate([(X_t1[i], y_t1[i]),
                                           (X_t2[i], y_t2[i])]):
            row = {
                "id":    sid,
                "time":  tp,                  # 0 = T1, 1 = T2
                "y":     yval,
                "Group_SA": superager_vec[i],    # 0 = control, 1 = superager
                "Group_maint": maintainer_vec[i], # 0 = control, 1 = maintainer
            }
            row.update({col: Xrow[j] for j, col in enumerate(stable_rois)})
            rows.append(row)

    df_long = pd.DataFrame(rows)

    # 2. Optionally scale ROIs 
    if scale_roi and stable_rois:
        sc = StandardScaler()
        df_long[stable_rois] = sc.fit_transform(df_long[stable_rois])

    # 3. Optionally sanitise names 
    rename_map = None
    if sanitise_cols:
        rename_map = {}
        for c in df_long.columns:
            safe = re.sub(r"[^0-9a-zA-Z_]", "_", c)   # only letters, digits, _
            if re.match(r"^[0-9]", safe):             # can’t start with digit
                safe = f"X_{safe}"
            rename_map[c] = safe
        df_long = df_long.rename(columns=rename_map)

    return df_long, rename_map

def build_wide_df(
    X_t1, X_t2, y_t1, y_t2, age_diff, subjects,
    roi_names, stable_rois,
    group_maint_vec, group_sa_vec,
    memory_data,
    demo_cols=("age_1", "sex", "YoE"),
    scale_demo=True,
):
    """
    Assemble a wide DataFrame ready for modelling
    memory-change (slope) as a function of ROI slopes, group status,
    and demographics.

    Args:
        X_t1, X_t2 (np.ndarray): Features at Time 1 and 2
        y_t1, y_t2 (np.ndarray): Memory outcomes at Time 1 and 2.
        age_diff (np.ndarray): Age difference between Time 1 and Time 2.
        subjects (list): List of subject IDs.
        roi_names (list): List of ROI names corresponding to features in X.
        stable_rois (list): List of stable ROI names to include in the DataFrame.
        group_maint_vec (np.ndarray): Vector indicating maintainer status for each subject.
        group_sa_vec (np.ndarray): Vector indicating superager status for each subject.
        memory_data (pd.DataFrame): DataFrame containing memory outcomes and demographics.
        demo_cols (tuple): Tuple of demographic columns to include in the DataFrame.
        scale_demo (bool): Whether to z-scale numeric demographic columns (default True).
    """
    # 1. Per-year slopes
    X_slope = (X_t2 - X_t1) / age_diff[:, None]
    y_slope = (y_t2 - y_t1) / age_diff

    # 2. Keep only stable ROIs and label columns <ROI>_slope
    idxs = [roi_names.index(r) for r in stable_rois]
    X_slope_stable = X_slope[:, idxs]
    roi_cols = [f"{r}_slope" for r in stable_rois]

    # 3. Base DataFrame with subject ids
    df_subj = (
        pd.DataFrame(X_slope_stable, columns=roi_cols, index=subjects)
        .reset_index()
        .rename(columns={"index": "id"})
    )
    df_subj["memory_slope"] = y_slope
    df_subj["Group_maint"] = group_maint_vec
    df_subj["Group_SA"] = group_sa_vec

    # 4. Demographics
    demo_df = (
        memory_data.set_index("id")
        .loc[subjects, demo_cols]        # keep same ordering
        .reset_index()
    )

    # convert sex to 0/1 if still strings
    if "sex" in demo_cols and demo_df["sex"].dtype == object:
        demo_df["sex"] = demo_df["sex"].map({"Female": 0, "Male": 1})

    # optional z-scaling for numeric demos (leave sex untouched)
    if scale_demo:
        num_cols = ["age_1", "YoE"]            # numeric columns only
        demo_df[num_cols] = StandardScaler().fit_transform(demo_df[num_cols])

    # safe column names ---------
    rename_map = {}
    for c in df_subj.columns:
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", c)
        if re.match(r"^[0-9]", safe):
            safe = f"X_{safe}"
        rename_map[c] = safe
    df_subj = df_subj.rename(columns=rename_map)

    # 5. Merge and return   (merge with the FULL demo_df, not a list)
    df_subj = df_subj.merge(demo_df, on="id")
    return df_subj

def main():
    connectivity_type = "SFC"  # Options: "SFC", "FC", "SC"
    sessions = ["ses-01", "ses-02"] 
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    memory_data = pd.read_csv("/home/rachel/Desktop/data/clean_data_all.csv")

    # Get the list of subjects to process
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
                

    # Prepare the data for analysis
    X_t1, X_t2, y_t1, y_t2, X_slope, y_slope, age_diff, final_subjects = prep_data(
        subjects, root_path, fc_root_path, sc_root_path, memory_data, connectivity_type)

    # Map feature indices back to ROI names 
    roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling_grouped.csv"
    roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    # Save the data as CSV
    out_file_1 = Path(f"/home/rachel/Desktop/data/grouped_{connectivity_type}_ses-01.csv")
    out_file_2 = Path(f"/home/rachel/Desktop/data/grouped_{connectivity_type}_ses-02.csv")

    df_t1 = pd.DataFrame(X_t1, columns=roi_names)
    df_t1.insert(0, "id", final_subjects)         # add subject IDs
    df_t1.to_csv(out_file_1, index=False)         # save with headers

    X_t2 = np.array(X_t2)
    df_t2 = pd.DataFrame(X_t2, columns=roi_names)
    df_t2.insert(0, "id", final_subjects)
    df_t2.to_csv(out_file_2, index=False)

    # Convert sex and cohort to numeric
    memory_data["sex"] = memory_data["sex"].map({"male": 0, "female": 1})
    
    print(f"Running bootstrap stability analysis for {connectivity_type}...")

    demographic_cols = memory_data.set_index("id").loc[subjects, ["age_1", "sex", "YoE"]].values

    # 1) Stability analysis for Time 1
    stab_t1 = bootstrap_stability_enet(
            X_t1, y_t1,
            roi_names=roi_names,
            demographic_X=demographic_cols,
            stab_threshold=0.75,
            n_boot=500,
            random_state=42)

    print("\nStable ROIs (≥ 75 %):")
    print(stab_t1[stab_t1 >= 0.75].apply(lambda x: f"{x:.2%}"))

    # Save the stable ROI names
    stable_rois_t1 = stab_t1[stab_t1 >= 0.75].index.tolist()

    # 2) Stability analysis for Time 2
    stab_t2 = bootstrap_stability_enet(
            X_t2, y_t2,
            roi_names=roi_names,
            demographic_X=demographic_cols,
            stab_threshold=0.75,
            n_boot=500,
            random_state=42)

    print("\nStable ROIs (≥ 75 %):")
    print(stab_t2[stab_t2 >= 0.75].apply(lambda x: f"{x:.2%}"))

    # Save the stable ROI names
    stable_rois_t2 = stab_t2[stab_t2 >= 0.75].index.tolist()

    # 3) Stability analysis for slope
    stab_long = bootstrap_stability_enet(
            X_slope, y_slope,
            roi_names=roi_names,
            demographic_X=demographic_cols,
            stab_threshold=0.75,
            n_boot=500,
            random_state=42)

    print("\nStable ROIs (≥ 75 %):")
    print(stab_long[stab_long >= 0.75].apply(lambda x: f"{x:.2%}"))

    # Save the stable ROI names
    stable_rois_long = stab_long[stab_long >= 0.75].index.tolist()

    # Generate a list of ROIs important at t1, t2, AND slope
    stable_rois = list(set(stable_rois_t1) & set(stable_rois_t2) & set(stable_rois_long))
    print(f"Stable ROIs across all analyses: {stable_rois}")

    # 2) Build the long DataFrame (make sure superager_vec is defined for each subject)
    superager_vec = memory_data.set_index("id").loc[final_subjects, "superager"].values
    maintainer_vec = memory_data.set_index("id").loc[final_subjects, "maintainer"].values

    df_subj = build_wide_df(
        X_t1, X_t2, y_t1, y_t2, age_diff,
        subjects, roi_names, stable_rois_long,
        group_maint_vec=maintainer_vec,             
        group_sa_vec=superager_vec,
        memory_data=memory_data
    )

    # Build the formula 
    roi_cols = [c for c in df_subj.columns if c.endswith("_slope") and c != "memory_slope"]
    roi_terms = " + ".join(roi_cols)
    int_terms  = " + ".join(f"Group_maint:{r}" for r in roi_cols)
    demo_vars = ["age_1", "sex", "YoE"] 
    demo_terms = " + ".join(demo_vars)

    formula = (
        f"memory_slope ~ Group_maint + {roi_terms} + {int_terms} + {demo_terms}"
    )
    print(formula)

    # Fit the mixed-effects model
    ols_fit = smf.ols(formula, data=df_subj).fit(cov_type="HC3")  # HC3 = robust SEs
    print(ols_fit.summary())

    # Now run an LME with ROIs important at tp1 and tp2
    overlap_rois = sorted(set(stable_rois_t1) & set(stable_rois_t2))
    print(f"ROIs stable at both waves ({len(overlap_rois)}):", overlap_rois)

    # Build + scale + sanitise 
    df_long, name_map = make_long_df(X_t1, X_t2, y_t1, y_t2, 
                                    subjects, superager_vec, maintainer_vec, overlap_rois)   

    # Build the model 
    roi_cols = [
        c for c in df_long.columns
        if c not in ["y", "id", "time", "Group_SA", "Group_maint"]
    ]
    if roi_cols:
        roi_terms = " + ".join(roi_cols)
        int_terms = " + ".join(f"Group_maint:{r}" for r in roi_cols)
        formula_lme = f"y ~ Group_maint * time + {roi_terms} + {int_terms}"
    else:
        formula_lme = "y ~ Group_maint * time"  # no ROI terms

    print("LME formula:\n", formula_lme)
    lme = smf.mixedlm(formula_lme, data=df_long, groups="id")
    fit = lme.fit(method="lbfgs")
    print(fit.summary())

if __name__ == "__main__":
    main()