import numpy as np
import pandas as pd
from merf import MERF
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score 
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

def flatten_connectivity_csv(matrix_csv, measure_col="pearson_rho"):
    """
    Reads an NxN connectivity CSV (214×214) and flattens it into a long DataFrame,
    returning columns: ROI_name, measure_col.

    Args:
        matrix_csv (str or Path): Path to the NxN CSV.
        measure_col (str): Name for the connectivity measure (e.g. "sc_value", "fc_value").

    Returns:
        pd.DataFrame with columns ["ROI_name", measure_col].
    """
    # Read the NxN matrix 
    df_matrix = pd.read_csv(matrix_csv, header=0, index_col=0)
    roi_list = df_matrix.index.tolist()  # ROI names

    # Compute the mean connectivity for each ROI (row-wise mean)
    avg_conn = df_matrix.mean(axis=1).values

    # Build the output DataFrame
    df_out = pd.DataFrame({
        "ROI_name": roi_list,
        measure_col: avg_conn
    })

    return df_out

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


def load_grouped_csv_as_series(csv_path: Path, roi_index):
    """
    Returns a one-row Series where index = roi names (≈70) and values = pearson_rho.
    Uses `roi_index` to enforce a consistent column order across subjects.
    """
    df = pd.read_csv(csv_path)

    # Ensure all expected ROIs present; if not, fill missing with NaN
    series = df.set_index("ROI_name")["pearson_rho"].reindex(roi_index)
    return series

def build_long_dataframe(root_path: Path,
                         fc_root_path: Path,
                         sc_root_path: Path,
                         sessions=("ses-01", "ses-02"),
                         subjects=None,
                         memory_df=None,
                         roi_index=None,
                         connectivity_type=None):
    """
    Returns df_long with columns:
        id, timepoint, y, age, sex, edu, cohort, superager, maintainer, roi_0 … roi_69
    """
    rows = []
    for sub in subjects:
        mem_row = memory_df.loc[memory_df["id"] == sub]
        if mem_row.empty:
            continue   # Skip subjects with no memory data

        for visit_idx, ses in enumerate(sessions):
            if connectivity_type == "FC":
                gcsv = (fc_root_path / ses / "individual_connectivity_matrices" / "grouped_rois" /
                            f"{sub}_{ses}_functional_connectivity_grouped.csv")
            elif connectivity_type == "SC":
                gcsv = (sc_root_path / ses / "individual_connectivity_matrices" / "grouped_rois" /
                            f"{sub}_{ses}_structural_connectivity_grouped.csv")
            else:
                gcsv = (root_path / ses / "individual_coupling_matrices" /
                            f"{sub}_{ses}_structure_function_coupling_grouped.csv")
            if not gcsv.is_file():
                print(f"Missing {gcsv}")
                continue

            roi_vals = load_grouped_csv_as_series(gcsv, roi_index)

            # Outcome & covariates
            y   = mem_row[f"memory_{visit_idx+1}"].values[0]
            age = mem_row[f"age_{visit_idx+1}"].values[0]

            row_dict = {
                "id": sub,
                "timepoint": visit_idx,   # 0=baseline,1=follow-up
                "y": y,
                "memory_1": mem_row["memory_1"].values[0],  # baseline memory
                "memory_2": mem_row["memory_2"].values[0],  # follow-up memory
                "age": age,
                "maintainer": mem_row["maintainer"].values[0],
                "sex": mem_row["sex"].values[0],
                "edu": mem_row["YoE"].values[0],
                "cohort": mem_row["cohort"].values[0],
                "superager": mem_row["superager"].values[0],
            }
            # Add ROI features with shortened names
            roi_cols = [r.replace("7Networks_", "roi_") for r in roi_index]   # shorten once
            row_dict.update({roi_cols[i]: v for i, v in enumerate(roi_vals)})

            rows.append(row_dict)

    df_long = pd.DataFrame(rows)
    return df_long

def fit_merf_once(df, feature_cols, n_trees=600, random_state=42):
    """
    Fits MERF with a random intercept and returns:
        merf       : fitted model object
        imp_series : pandas Series of permutation importances
    """
    # ----- Design matrices -----
    X = df[feature_cols].values                 # fixed-effect predictors
    Z = np.ones((len(df), 1))                   # random intercept
    y = df["y"].values
    df = df.reset_index(drop=True)
    clusters = df["id"]                         # subject IDs

    # ----- Base forest -----
    base_rf = RandomForestRegressor(
        n_estimators=n_trees,
        max_features="sqrt",
        min_samples_leaf=5,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1
    )

    merf = MERF(fixed_effects_model=base_rf)
    merf.fit(X, Z, clusters, y)

    # ----- Permutation importance on OOB preds -----
    y_hat = merf.predict(X, Z, clusters)          # includes random effects
    baseline_mse = np.mean((y - y_hat) ** 2)

    importances = []
    rng = np.random.default_rng(random_state)
    for j in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perm_pred = merf.predict(X_perm, Z, clusters)
        mse = np.mean((y - perm_pred)**2)
        importances.append(mse - baseline_mse)

    imp_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    # Calculate percent drop in R² 
    importances = np.array(importances)
    var_y = np.var(y, ddof=0)        # matches scikit-learn’s R² denominator
    delta_r2   = -importances / var_y            # signed change in R²

    return merf, imp_series, y, y_hat, delta_r2

def cv_merf_r2(df, feature_cols, n_splits=5, random_state=42):
    X = df[feature_cols].values
    Z = np.ones((len(df), 1))
    y = df["y"].values
    df = df.reset_index(drop=True)
    clusters = df["id"]

    base_rf = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )
    merf = MERF(fixed_effects_model=base_rf)

    # GroupKFold keeps both visits of each subject together
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []

    for train_idx, test_idx in gkf.split(X, y, groups=clusters):
        merf.fit(X[train_idx], Z[train_idx], clusters[train_idx], y[train_idx])
        preds = merf.predict(X[test_idx], Z[test_idx], clusters.iloc[test_idx])
        r2 = 1 - np.sum((y[test_idx] - preds) ** 2) / np.var(y[test_idx]) / len(test_idx)
        r2_scores.append(r2)

    return np.mean(r2_scores), np.std(r2_scores)

def main():
    sessions = ["ses-01", "ses-02"] 
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    memory_data = pd.read_csv("/home/rachel/Desktop/data/clean_data_all.csv")

    # Create a cohort column where bbhi is when id > 5000
    memory_data["id_num"] = memory_data["id"].str.replace("sub-", "", regex=False).astype(int)
    memory_data["cohort"] = memory_data["id_num"].apply(lambda x: "bbhi" if x > 5000 else "bbhi_senior")

    sample_csv = next((root_path / "ses-01" / "individual_coupling_matrices").glob("*_grouped.csv")) # e.g. a random subject's grouped CSV to get ROI names
    roi_index = pd.read_csv(sample_csv)["ROI_name"].tolist()

    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01")
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02")
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # # Make the flattened FC and SC CSVs for each subject
    # for sub in subjects:
    #     for ses in sessions:
    #         ses_path_fc = fc_root_path / ses / "individual_connectivity_matrices"
    #         ses_path_sc = sc_root_path / ses / "individual_connectivity_matrices"

    #         # ── Functional connectivity ──
    #         fc_csv = ses_path_fc / f"{sub}_{ses}_functional_connectivity_matrix.csv"
    #         if fc_csv.is_file():
    #             fc_flat = flatten_connectivity_csv(fc_csv, measure_col="pearson_rho")

    #             fc_output_dir = ses_path_fc / "grouped_rois"
    #             fc_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

    #             fc_output = fc_output_dir / f"{sub}_{ses}_functional_connectivity_flat.csv"
    #             fc_flat.to_csv(fc_output, index=False)  # (Optional) Save the flattened version

    #             # Group the flattened CSV by ROI
    #             grouped_output = fc_output_dir / f"{sub}_{ses}_functional_connectivity_grouped.csv"
    #             save_grouped_roi_averages(fc_output, grouped_output)

    #         # ── Structural connectivity ──
    #         sc_csv = ses_path_sc / f"{sub}_{ses}_structural_connectivity_matrix.csv"
    #         if sc_csv.is_file():
    #             sc_flat = flatten_connectivity_csv(sc_csv, measure_col="pearson_rho")

    #             sc_output_dir = ses_path_sc / "grouped_rois"
    #             sc_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

    #             sc_output = sc_output_dir / f"{sub}_{ses}_structural_connectivity_flat.csv"
    #             sc_flat.to_csv(sc_output, index=False)  # (Optional) Save the flattened version

    #             # Group the flattened CSV by ROI
    #             grouped_output = sc_output_dir / f"{sub}_{ses}_structural_connectivity_grouped.csv"
    #             save_grouped_roi_averages(sc_output, grouped_output)
    #         else:
    #             print(f"Missing SC CSV at path: {sc_csv}")

    # # Make the grouped averages for each subject's SFC CSV
    # for sub in subjects:
    #     for ses in sessions:
    #         ses_path = root_path / ses / "individual_coupling_matrices"
    #         csv_path = ses_path / f"{sub}_{ses}_structure_function_coupling.csv"
    #         output_path = ses_path / f"{sub}_{ses}_structure_function_coupling_grouped.csv"
    #         if csv_path.is_file():
    #             save_grouped_roi_averages(csv_path, output_path)

    df_long = build_long_dataframe(root_path=root_path,
                                fc_root_path=fc_root_path,
                                sc_root_path=sc_root_path,
                                sessions=("ses-01", "ses-02"),
                                subjects=subjects,
                                memory_df=memory_data,
                                roi_index=roi_index,
                                connectivity_type="SC") # Types: FC, SC, or SFC
    
    # Convert sex and cohort to numeric
    df_long["sex"] = df_long["sex"].map({"male": 0, "female": 1})
    df_long["cohort"] = df_long["cohort"].astype("category").cat.codes

    print(f"df_long shape: {df_long.shape}")   

    # Fiter the cohort 
    # df_long = df_long[df_long["superager"] == 1].copy()
    
    # feature_cols = ["age", "sex", "edu", "cohort", "memory_1"] + [c for c in df_long.columns if c.startswith("roi_")]
    feature_cols = ["age", "sex", 'edu', 'cohort', 'roi_LH_Cont_OFC', 'roi_RH_Limbic_OFC', 
                    'Subcortical 202: Left Amygdala', 'roi_LH_Limbic_TempPole', 'roi_RH_Cont_PFCv', 
                    'roi_RH_Default_PFCv', 'roi_LH_DorsAttn_PrCv', 'roi_LH_SalVentAttn_PFCl', 
                    'roi_RH_SalVentAttn_FrOperIns', 'roi_LH_Cont_pCun', 'roi_LH_Cont_Temp', 
                    'roi_RH_Default_Par', 'roi_LH_Cont_Cing', 'roi_LH_Limbic_OFC', 
                    'roi_LH_Default_PHC', 'roi_LH_Default_PFC', 'Subcortical 208: Right Hippocampus', 
                    'Subcortical 201: Left Hippocampus', 'Subcortical 210: Right Pallidum', 'roi_RH_Cont_Par', 
                    'roi_RH_Cont_pCun', 'Subcortical 213: Right Accumbens', 'roi_RH_Cont_PFCmp', 'roi_LH_DorsAttn_FEF', 
                    'roi_LH_SalVentAttn_ParOper', 'roi_RH_SalVentAttn_PrC', 'Subcortical 206: Left Accumbens',
                    'Subcortical 204: Left Putamen', 'Subcortical 207: Left Thalamus', 'roi_LH_Default_Par',
                    'Subcortical 209: Right Amygdala', 'roi_RH_SalVentAttn_Med', 'Subcortical 214: Right Thalamus', 'roi_RH_Cont_Temp']

    # One fit + importance
    merf_model, imp, y, y_hat, delta_r2 = fit_merf_once(df_long, feature_cols)
    print(f"Training R² (MERF full model): {r2_score(y, y_hat):.3f}")
    r2_series = pd.Series(delta_r2, index=feature_cols).sort_values(ascending=True)
    print("\nTop 10 ROI importance (change in R²):")
    print(r2_series.head(10).apply(lambda x: f"{x:.4f}"))
    print("\nTop 10 ROI importance (Δ MSE):")
    print(imp.head(10))

    # 5-fold group CV
    mean_r2, std_r2 = cv_merf_r2(df_long, feature_cols)
    print(f"\n5-fold Group-CV R²: {mean_r2:.3f} ± {std_r2:.3f}")     

if __name__ == "__main__":
    main()