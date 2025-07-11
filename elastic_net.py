import numpy as np
from sklearn.linear_model import ElasticNetCV
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, permutation_test_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from pathlib import Path
import os
import re

def get_subjects_to_process(output_folder, ses, id_csv_path):
    """
    Generate a list of subjects to process, ensuring each subject is also in the id column of the provided CSV.
    Args:
        output_folder (Path): Path to the directory coupling results
        ses (str): Session ID (format: ses-01).
        id_csv_path (Path or str): Path to CSV file with 'id' column.
    Returns:
        list: List of subject IDs to process.
    """
    # Load valid subject IDs from the CSV - this allows removing subs without sufficiently long follow-up
    valid_ids = set(pd.read_csv(id_csv_path)['id'].astype(str))

    subjects = []
    for fname in os.listdir(output_folder):
        if not fname.startswith("sub-") or not fname.endswith(f"{ses}_structure_function_coupling.csv"):
            continue

        # Extract subject ID 
        subject = fname.split("_")[0]

        sfc_path = output_folder / fname
        if sfc_path.exists() and subject in valid_ids:
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


def save_grouped_roi_averages(csv_path, output_path, group_level="ROI"):
    """
    Reads a subject's connectivity CSV (SFC/FC/SC),
    then groups & averages pearson_rho either:
      • by ROI prefix (strip trailing _<digits>), or
      • by network (for cortical: 3rd "_" field; for subcortical: region name).

    Args:
        csv_path (str or Path): Path to the input CSV file.
        output_path (str or Path): Path to save the grouped averages CSV.
        group_level (str): Grouping level, either "ROI" or "network".
                           "ROI" groups by ROI prefix (eg PFCv)
                           "network" groups by network name (eg DMN)
                           "streamline" groups by a predefined set of ROIs important in memory.
    """
    df = pd.read_csv(csv_path)
    df["ROI_name"] = df["ROI_name"].astype(str)

    # Group by ROIs to have 59 total
    if group_level.lower() == "roi":
        df["ROI_group"] = df["ROI_name"].str.replace(r'_\d+$', '', regex=True)
        grouped = (
            df
            .groupby("ROI_group", as_index=False)["pearson_rho"]
            .mean()
            .rename(columns={"ROI_group": "ROI_name"})
        )

    # Group by networks to have 7 cortical networks and 7 subcortical regions (classed as networks here)
    elif group_level.lower() == "network":
        def network_key(name):
            if name.startswith("Subcortical"):
                parts = name.split()
                # parts = ["Subcortical","208","Right","Hippocampus",...]
                region = " ".join(parts[3:]) # Combines the left and right hemispheres
                return region
            else:
                # Cortical case: "7Networks_RH_Cont_pCun"
                parts = name.split("_")
                if len(parts) >= 3:
                    return parts[2]
                # If the name does not match the expected format, return it as is
                return name

        df["network"] = df["ROI_name"].apply(network_key)

        # Group by network and average the connectivity values
        grouped = (
            df
            .groupby("network", as_index=False)["pearson_rho"]
            .mean()
            .rename(columns={"network": "ROI_name"})
        )

    elif group_level.lower() == "streamline":
        def streamline_key(name):
            if name.startswith("Subcortical"):
                parts = name.split(":")
                if len(parts) > 1:
                    sub_part = parts[1].strip()    # "Left Hippocampus"
                    tokens = sub_part.split()      # ["Left", "Hippocampus"]
                    return tokens[-1]              # "Hippocampus"
                return name  
            else:
                # Cortical case: "7Networks_RH_Cont_pCun"
                return re.sub(r"_\d+$", "", name)   

        df["roi_label"] = df["ROI_name"].apply(streamline_key)

        grouped = (
            df
            .groupby("roi_label", as_index=False)
            .agg({"pearson_rho": "mean"})
            .rename(columns={"roi_label": "ROI_name"})
        )
        
        # Exclude specified ROIs/networks/groups
        keep_rois = [
            "7Networks_LH_Limbic_OFC", # Wang et al., 2017
            "7Networks_RH_Cont_Cing", # For ACC / MCC
            "7Networks_LH_Cont_Cing",
            "7Networks_RH_Cont_PFCl",
            "7Networks_RH_Cont_PFCmp",
            "7Networks_RH_Cont_PFCv",
            "7Networks_LH_Cont_PFCl",
            "7Networks_LH_Cont_PFCmp",
            "7Networks_LH_Cont_PFCv",
            "7Networks_LH_Cont_Temp",
            "7Networks_RH_Cont_Temp",
            "7Networks_LH_Default_PFC",
            "7Networks_RH_Default_PFC",
            "7Networks_LH_Default_PHC",
            "7Networks_RH_Default_PHC",
            "7Networks_LH_Default_Temp",
            "7Networks_RH_Default_Temp",
            "7Networks_LH_Default_Par",
            "7Networks_RH_Default_Par",
            "7Networks_LH_Default_pCunPCC",
            "7Networks_RH_Default_pCunPCC",
            "Hippocampus",
            "Amygdala"
        ]

        if keep_rois:
            grouped = grouped[grouped["ROI_name"].isin(keep_rois)]
        
    else:
        raise ValueError("group_level must be 'ROI', 'network', or 'streamline'")

    grouped.to_csv(output_path, index=False)

def prep_data(subjects, root_path, fc_root_path, sc_root_path, memory_data, connectivity_type):
    """
    Prepares the data for analysis by extracting features, memory outcomes, and superager status
    from the specified directories and memory data.

    Args:
        subjects (list): List of subject IDs to process.
        root_path (Path): Path to the root directory containing SFC data.
        fc_root_path (Path): Path to the root directory containing FC data.
        sc_root_path (Path): Path to the root directory containing SC data.
        memory_data (pd.DataFrame): DataFrame containing memory outcomes and demographics.
        connectivity_type (str): Type of connectivity data to process ("SFC", "FC", or "SC").
    """
    # Prepare the data 
    def load_modality(sub, mod):
        """Helper function to load features for a given modality (eg SFC)."""
        if mod == "SFC":
            p1 = root_path / "ses-01" / "individual_coupling_matrices"
            p2 = root_path / "ses-02" / "individual_coupling_matrices"
            f1 = p1 / f"{sub}_ses-01_structure_function_coupling_grouped.csv"
            f2 = p2 / f"{sub}_ses-02_structure_function_coupling_grouped.csv"
            col = 'pearson_rho'
        elif mod == "FC":
            f1 = fc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_functional_connectivity_grouped.csv"
            f2 = fc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_functional_connectivity_grouped.csv"
            col = 'pearson_rho' 
        else:  # "SC"
            f1 = sc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_structural_connectivity_grouped.csv"
            f2 = sc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_structural_connectivity_grouped.csv"
            col = 'pearson_rho'  

        # Read both feature files if they exist (one from each tp), otherwise return None, None
        if f1.is_file() and f2.is_file():
            v1 = pd.to_numeric(pd.read_csv(f1)[col].values, errors='coerce')
            v2 = pd.to_numeric(pd.read_csv(f2)[col].values, errors='coerce')
            return v1, v2
        else:
            return None, None

    records = []
    for sub in subjects:
        # Load features
        if connectivity_type == "all":
            feats = []
            for mod in ("SFC","FC","SC"):
                v1, v2 = load_modality(sub, mod)
                if v1 is None:
                    feats = None
                    break
                feats.append((v1, v2))
            if feats is None:
                continue
            # Stack features from all modalities
            feat1 = np.hstack([v1 for v1, _ in feats])
            feat2 = np.hstack([v2 for _, v2 in feats])
        else:
            # Otherwise load a single modality
            feat1, feat2 = load_modality(sub, connectivity_type)
            if feat1 is None:
                continue

        # Make a cohort variable
        memory_data['numeric_id'] = memory_data['id'].str.replace("sub-", "").astype(int)
        memory_data['cohort'] = np.where(memory_data['numeric_id'] < 5000, "bbhi_senior", "bbhi")

        # Pull memory, ages, superager status from memory_data
        row = memory_data[memory_data['id'] == sub]
        if row.empty:
            continue
        mem1 = row.iloc[0]['memory_1']
        mem2 = row.iloc[0]['memory_2']
        age1 = row.iloc[0]['age_1']
        age2 = row.iloc[0]['age_2']
        superager = row.iloc[0]['superager']
        YoE = row.iloc[0]['YoE']
        sex = row.iloc[0]['sex']
        cohort = row.iloc[0]['cohort']

        records.append({
            'id'        : sub,
            'feat1'     : feat1,
            'feat2'     : feat2,
            'y1'        : mem1,
            'y2'        : mem2,
            'age1'      : age1,
            'age2'      : age2,
            'superager' : superager,
            'YoE'       : YoE,
            'sex'       : sex,
            'cohort'    : cohort
        })

    # Build DataFrame
    df = pd.DataFrame(records)

    # 1) Stack features
    X_t1 = np.stack(df['feat1'].values)
    X_t2 = np.stack(df['feat2'].values)
    age_diff = (df['age2'] - df['age1']).values

    # Elastic Net requires scaled features (not outputs)
    # Scale the features just before running the analysis
    scaler = StandardScaler()
    X_t1 = scaler.fit_transform(X_t1)
    X_t2 = scaler.fit_transform(X_t2)
    X_slope = (X_t2 - X_t1) / age_diff[:, None]
    X_slope = scaler.fit_transform(X_slope)

    # 3) Practice-effect residualization
    y_t1 = df['y1'].values
    y_t2 = df['y2'].values
    y_slope = (y_t2 - y_t1) / age_diff[:, None]
    y_slope_2 = (y_t2 - y_t1) / age_diff # Added to create the correct array shape for y_slope_adj
    valid = ~np.isnan(y_t1) & ~np.isnan(y_t2)
    slope, intercept, *_ = linregress(y_t1[valid], y_t2[valid])
    y_t2_adj = y_t2 - (intercept + slope * y_t1)
    y_slope_adj = (y_t2_adj - y_t1) / age_diff

    # 4) Build covariate matrices 
    #    Numeric: age, years of education
    cov_age1 = df['age1'].values.reshape(-1,1)
    cov_age2 = df['age2'].values.reshape(-1,1)
    cov_YoE  = df['YoE'].values.reshape(-1,1)
    #    Categorical: cohort, sex
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    cov_cat = ohe.fit_transform(df[['cohort','sex']])

    covs_t1   = np.hstack([cov_age1, cov_YoE, cov_cat])
    covs_t2   = np.hstack([cov_age2, cov_YoE, cov_cat])
    covs_slope= covs_t1.copy()

    # 5) Residualize each outcome on covariates
    lr1 = LinearRegression().fit(covs_t1, y_t1)
    lr2 = LinearRegression().fit(covs_t2, y_t2_adj)
    lrs = LinearRegression().fit(covs_slope, y_slope_adj)

    y_t1_resid        = y_t1         - lr1.predict(covs_t1)
    y_t2_adj_resid    = y_t2_adj     - lr2.predict(covs_t2)
    y_slope_adj_resid = y_slope_adj  - lrs.predict(covs_slope)
    y_t2_resid        = y_t2         - lr2.predict(covs_t2)
    y_slope_resid     = y_slope_2      - lrs.predict(covs_slope)

    # 6) Make a superager vector 
    superager_vec = df['superager'].values.astype(int)

    # 7) generate a stacked version of tp1 and tp2 features
    X_all = np.hstack([X_t1, X_t2, X_slope])   # shape: (N_subjects, 2 * len(roi_names))

    return X_t1, X_t2, y_t1, y_t2, X_slope, y_slope, age_diff, superager_vec, y_t1_resid, y_t2_adj_resid, y_slope_adj_resid, y_t2_resid, y_slope_resid, X_all

def bootstrap_stability_enet(
        X, y, roi_names, connectivity_type, timepoint, 
        max_iter=5000,
        stab_threshold=0.80,         
        n_boot=500,
        random_state=42,
        l1_grid=(0.1, 0.5, 0.9),
        alpha_grid=np.logspace(-2, 1, 10)):
    """
    Bootstraps subjects, refits ElasticNetCV, returns selection probability
    and prints extra diagnostics.

    Args:
        X (np.ndarray): Feature matrix of averaged feature values for connectivity.
        y (np.ndarray): Target variable (e.g. memory scores).
        roi_names (list): List of ROI names corresponding to features in X.
        connectivity_type (str): Type of connectivity data (e.g. "SFC", "FC", "SC").
        timepoint (str): Timepoint of the data being processed (e.g. "tp1", "tp2", "slope").
        stab_threshold (float): Threshold for stability selection (default 0.80).
        n_boot (int): Number of bootstrap iterations (default 500).
        random_state (int): Random seed for reproducibility.
        l1_grid (tuple): Tuple of l1_ratio values to test in ElasticNetCV
            1 is pure Lasso, 0 is pure Ridge.
        alpha_grid (np.ndarray): Array of alpha values to test in ElasticNetCV.
            used to create lambda.
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
            max_iter=max_iter,
            n_jobs      = 8, # Parallelization
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
    med_abs_coef = np.median(np.abs(coefs[:, stable_mask])
                    if n_stable else np.nan)
    
    print(f"\nStability-selection summary for {connectivity_type} for {timepoint} (threshold ≥ {stab_threshold:.2f})")
    print(f"  • Stable ROIs      : {n_stable} / {p}")

    rows = []
    for j, roi in enumerate(roi_names):
        sel_mask = coefs[:, j] != 0
        p_select = sel_mask.mean()           # same as stab[j]
        if p_select < stab_threshold:
            continue                         # skip unstable ROIs

        # Calculate statistics for selected ROIs
        med_all   = np.median(coefs[:, j])
        signs     = np.sign(coefs[sel_mask, j])
        sign_cons = (signs == 1).sum() - (signs == -1).sum()
        sign_cons /= (signs.size or 1)

        # Add confidence intervals for non-zero coefficients
        nonzero_coefs = coefs[sel_mask, j]
        med_nz   = np.median(nonzero_coefs)
        ci_low, ci_high = np.percentile(nonzero_coefs, [2.5, 97.5])

        # Determine direction now that CI no longer includes the zeros
        if ci_low > 0:
            direction = 'positive'
        elif ci_high < 0:
            direction = 'negative'
        else:
            direction = 'uncertain'

        rows.append({
            'ROI'            : roi,
            'stab'           : p_select,
            'sign_consist'   : sign_cons,
            'med_coef_all'   : med_all,
            'med_coef_sel'   : med_nz,
            'ci_low_sel'     : ci_low,
            'ci_high_sel'    : ci_high,
            'direction'      : direction
        })

    stable_df = pd.DataFrame(rows)
    if not stable_df.empty:
        stable_df = stable_df.sort_values('stab', ascending=False)
        print(f"\nStable ROIs (stab ≥ {stab_threshold}):")
        for _, r in stable_df.iterrows():
            print(f"  • {r.ROI:<25} "
                  f"stab={r.stab:.2f}   "
                  f"β̃_sel={r.med_coef_sel:+.3f}   "
                  f"CI_sel=[{r.ci_low_sel:+.3f},{r.ci_high_sel:+.3f}]   "
                  f"dir={r.direction}")
    else:
        print(f"\nStable ROIs (stab ≥ {stab_threshold}):")
        print("  none")

    # Now run a permutation test on the stable features
    stable_mask = stab_series.values >= stab_threshold
    stable_rois = stab_series.index[stable_mask].tolist()
    if stable_mask.sum() == 0:
        print("\nNo stable features—skipping reduced‐model permutation test.")
    else:
        # Subset X to just the stable columns
        X_stable = X[:, stable_mask]

        # Build the same pipeline
        full_pipe = ElasticNetCV(
            l1_ratio   = l1_grid,
            alphas     = alpha_grid,
            cv         = 5,
            max_iter   = max_iter,
            n_jobs     = 8,  # Parallelization
            random_state=random_state
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # Permutation test on the reduced feature set
        score_red, perm_scores_red, pval_red = permutation_test_score(
            full_pipe, X_stable, y,
            scoring       = 'r2',
            cv            = cv,
            n_permutations=1000,
            random_state  = random_state,
            n_jobs        = 4
        )

        print(f"\nReduced‐model permutation test (using {len(stable_rois)} stable features):")
        print(f"  • Features tested: {stable_rois}")
        print(f"  • CV R² (true labels)   : {score_red:.3f}")
        print(f"  • permutation p-value   : {pval_red:.3f}")

    return stab_series

def main():
    # Just a note that running all may lead to problems with colinearity
    connectivity_type = "FC"  # Options: "SFC", "FC", "SC", "all"
    sessions = ["ses-01", "ses-02"] 
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    csv_path = Path("/home/rachel/Desktop/data/clean_data_all.csv")
    memory_data = pd.read_csv(csv_path)

    # Get the list of subjects to process
    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01", csv_path)
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02", csv_path)
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # Make the flattened FC and SC CSVs for each subject
    for sub in subjects:
        for ses in sessions:
            ses_path_fc = fc_root_path / ses / "individual_connectivity_matrices"
            ses_path_sc = sc_root_path / ses / "individual_connectivity_matrices"

            # ── Functional connectivity ──
            fc_csv = ses_path_fc / f"{sub}_{ses}_functional_connectivity_matrix.csv"
            if fc_csv.is_file():
                fc_flat = flatten_connectivity_csv(fc_csv, measure_col="pearson_rho")

                fc_output_dir = ses_path_fc / "grouped_rois"
                fc_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

                fc_output = fc_output_dir / f"{sub}_{ses}_functional_connectivity_flat.csv"
                fc_flat.to_csv(fc_output, index=False)  # Save the flattened version

                # Group the flattened CSV by ROI
                grouped_output = fc_output_dir / f"{sub}_{ses}_functional_connectivity_grouped.csv"
                # Adding group_level="network" allows looking at all of the DMN for example rather than individual ROIs
                # Adding group_level="streamline" allows at a predefined subset of ROIs important in memory
                save_grouped_roi_averages(fc_output, grouped_output, group_level = "streamline") 

            # ── Structural connectivity ──
            sc_csv = ses_path_sc / f"{sub}_{ses}_structural_connectivity_matrix.csv"
            if sc_csv.is_file():
                sc_flat = flatten_connectivity_csv(sc_csv, measure_col="pearson_rho")

                sc_output_dir = ses_path_sc / "grouped_rois"
                sc_output_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

                sc_output = sc_output_dir / f"{sub}_{ses}_structural_connectivity_flat.csv"
                sc_flat.to_csv(sc_output, index=False)  # Save the flattened version

                # Group the flattened CSV by ROI
                grouped_output = sc_output_dir / f"{sub}_{ses}_structural_connectivity_grouped.csv"
                save_grouped_roi_averages(sc_output, grouped_output, group_level = "streamline") 

            else:
                print(f"Missing SC CSV at path: {sc_csv}")

    # Make the grouped averages for each subject's SFC CSV
    for sub in subjects:
        for ses in sessions:
            ses_path = root_path / ses / "individual_coupling_matrices"
            csv_path = ses_path / f"{sub}_{ses}_structure_function_coupling.csv"
            output_path = ses_path / f"{sub}_{ses}_structure_function_coupling_grouped.csv"
            if csv_path.is_file():
                save_grouped_roi_averages(csv_path, output_path, group_level = "streamline")
                
    # Prepare the data for analysis
    X_t1, X_t2, y_t1, y_t2, X_slope, y_slope, age_diff, superager_vec, y_t1_resid, y_t2_adj_resid, y_slope_adj_resid, y_t2_resid, y_slope_resid, X_all = prep_data(
        subjects, root_path, fc_root_path, sc_root_path, memory_data, connectivity_type)

    # Map feature indices back to ROI names 
    roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling_grouped.csv"
    roi_names_pre = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    if connectivity_type in ["SFC", "SC", "FC"]:
        roi_names = roi_names_pre
    else:
        # For "all", add modality prefixes to each ROI name
        modalities = ["sfc", "fc", "sc"]
        roi_names = [f"{mod}_{roi}" for mod in modalities for roi in roi_names_pre]

    # Get the ROI names with two timepoints + slope for X_all:
    roi_names_tp1 = [f"{r}_1" for r in roi_names]
    roi_names_tp2 = [f"{r}_2" for r in roi_names]
    roi_names_slope = [f"{r}_slope" for r in roi_names]
    all_roi_names = roi_names_tp1 + roi_names_tp2 + roi_names_slope
    
    print(f"Running bootstrap stability analysis for {connectivity_type}...")

    # 1) Stability analysis for Time 1
    bootstrap_stability_enet(
            X_t1, y_t1_resid, 
            roi_names=roi_names, 
            connectivity_type=connectivity_type,
            timepoint="tp1",
            max_iter=20000,
            stab_threshold=0.80,
            n_boot=500,
            random_state=42)

    # 2) Stability analysis for Time 2
    bootstrap_stability_enet(
            X_t2, y_t2_resid,
            roi_names=roi_names,
            connectivity_type=connectivity_type,
            timepoint="tp2",
            max_iter=20000,
            stab_threshold=0.80,
            n_boot=500,
            random_state=42)

    # 3) Stability analysis for slope
    bootstrap_stability_enet(
            X_slope, y_slope_resid,
            roi_names=roi_names,
            connectivity_type=connectivity_type,
            timepoint="slope",
            max_iter=20000,
            stab_threshold=0.80,
            n_boot=500,
            random_state=42)

if __name__ == "__main__":
    main()