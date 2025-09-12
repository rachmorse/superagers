import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, permutation_test_score
from sklearn.inspection import permutation_importance
import pandas as pd
from pathlib import Path
import os
import re
import nibabel as nib
from collections import Counter
from nilearn.datasets import fetch_atlas_schaefer_2018
from functools import lru_cache
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    balanced_accuracy_score
)
from sklearn.utils import check_random_state


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

_SCHAEFER_LABEL_TO_COUNT = None
def _get_schaefer_label_to_count():
    """
    Cached mapping of Schaefer 200 ROIs to their voxel counts
    to speed up weighted averaging.

    Returns:
        dict: Mapping of ROI label to voxel count.
    """
    global _SCHAEFER_LABEL_TO_COUNT
    if _SCHAEFER_LABEL_TO_COUNT is None:
        atl = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
        atlas = nib.load(atl.maps).get_fdata().astype(int)
        cortical_counts = Counter(atlas[atlas > 0].ravel())
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atl.labels]
        _SCHAEFER_LABEL_TO_COUNT = {labels[i]: cortical_counts[i]
                                    for i in cortical_counts.keys()
                                    if 0 <= i < len(labels)}
    return _SCHAEFER_LABEL_TO_COUNT

@lru_cache(maxsize=None) 
def _get_subcort_counts(subject: str, ses: str):
    """
    Load and cache the voxel counts for subcortical ROIs from the subject's aseg.mgz file.
    
    Args:
        subject (str): Subject ID (e.g. "sub-1234").
        ses (str): Session ID (e.g. "ses-01").
    
    Returns:
        Counter: Mapping of aseg label ID to voxel count.
    """
    cohort = "bbhi" if int(subject.split("-")[1]) > 5000 else "bbhi senior"
    if cohort == "bbhi":
        aseg_file = Path(f"/pool/guttmann/institut/BBHI/MRI/processed_data/freesurfer-reconall/{subject}_{ses}_run-01/mri/aseg.mgz")
    else:
        aseg_file = Path(f"/pool/guttmann/institut/UB/Superagers/MRI/freesurfer-reconall/{subject}_{ses}/mri/aseg.mgz")
    aseg = nib.load(aseg_file).get_fdata().astype(int)
    return Counter(aseg[aseg > 0].ravel())

def save_grouped_roi_averages(csv_path, output_path, group_level, subject, ses):
    """
    Reads a subject's connectivity CSV (SFC/FC/SC),
    then groups & averages the coefficients either:
      • by ROI prefix (strip trailing _<digits>), or
      • by network (for cortical: 3rd "_" field; for subcortical: region name).

    Args:
        csv_path (str or Path): Path to the input CSV file.
        output_path (str or Path): Path to save the grouped averages CSV.
        group_level (str): Grouping level, either "ROI" or "network".
                           "ROI" groups by ROI prefix (eg PFCv) 
                           "network" groups by network name (eg DMN)
        subject (str): Subject ID 
        ses (str): Session ID
    """
    df = pd.read_csv(csv_path)
    df["ROI_name"] = df["ROI_name"].astype(str)

    # Cached Schaefer voxel counts per ROI label 
    label_to_count = _get_schaefer_label_to_count()
    df["weight"] = df["ROI_name"].map(label_to_count).fillna(1)

    # Then get them for the subcortical ROIs
    subcort_counts = _get_subcort_counts(subject, ses)

    subcort_map = {
        "Subcortical 201: Left Hippocampus": 17, "Subcortical 202: Left Amygdala": 18, "Subcortical 203: Left Pallidum": 13,
        "Subcortical 204: Left Putamen": 12, "Subcortical 205: Left Caudate": 11, "Subcortical 206: Left Accumbens": 26,
        "Subcortical 207: Left Thalamus": 10, "Subcortical 208: Right Hippocampus": 53, "Subcortical 209: Right Amygdala": 54,
        "Subcortical 210: Right Pallidum": 52, "Subcortical 211: Right Putamen": 51, "Subcortical 212: Right Caudate": 50,
        "Subcortical 213: Right Accumbens": 58, "Subcortical 214: Right Thalamus": 49,
    }

    # Map subcortical voxel counts and override weight where applicable
    df["aseg_id"] = df["ROI_name"].map(subcort_map)
    df["weight_subcort"] = df["aseg_id"].map(subcort_counts)

    # Prefer subcortical voxel count when ROI_name is subcortical
    df.loc[df["ROI_name"].str.startswith("Subcortical"), "weight"] = df["weight_subcort"]
    df["weight"] = df["weight"].fillna(1).astype(float)

    # Clean up helper columns
    df.drop(columns=["aseg_id", "weight_subcort"], inplace=True)

    # Group by ROI prefix (strip trailing _<digits>)
    if group_level.lower() == "roi":
        df["ROI_group"] = df["ROI_name"].str.replace(r'_\d+$', '', regex=True)
        df["w_rho"] = df["pearson_rho"] * df["weight"]
        agg = df.groupby("ROI_group", as_index=False).agg(
            wsum=("w_rho", "sum"),
            w=("weight", "sum"),
        )
        grouped = (
            agg.assign(pearson_rho=agg["wsum"] / agg["w"])[["ROI_group", "pearson_rho"]]
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

        # Weighted average of pearson_rho by network using voxel-based weights
        tmp = df[["network", "pearson_rho", "weight"]].copy()
        tmp["w_rho"] = tmp["pearson_rho"] * tmp["weight"]
        agg = tmp.groupby("network", as_index=False).agg(
            wsum=("w_rho", "sum"),
            w=("weight", "sum"),
        )
        grouped = (
            agg.assign(pearson_rho=agg["wsum"] / agg["w"])[["network", "pearson_rho"]]
               .rename(columns={"network": "ROI_name"})
        )

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

        # Pull memory, ages, superager status from memory_data
        row = memory_data[memory_data['id'] == sub]
        if row.empty:
            continue
        age1 = row.iloc[0]['age_1']
        age2 = row.iloc[0]['age_2']
        superager = row.iloc[0]['superager']
        YoE = row.iloc[0]['YoE']
        sex = row.iloc[0]['sex']

        records.append({
            'id'        : sub,
            'feat1'     : feat1,
            'feat2'     : feat2,
            'age1'      : age1,
            'age2'      : age2,
            'superager' : superager,
            'YoE'       : YoE,
            'sex'       : sex
             })

    # Build DataFrame
    df = pd.DataFrame(records)

    # 1) Stack features
    X_t1 = np.stack(df['feat1'].values)
    X_t2 = np.stack(df['feat2'].values)
    age_diff = (df['age2'] - df['age1']).values
    X_slope = (X_t2 - X_t1) / age_diff[:, None]

    # 2) Make a superager vector 
    superager_vec = df['superager'].values.astype(int)

    # 3) generate a stacked version of tp1 and tp2 features
    X_all = np.hstack([X_t1, X_t2, X_slope])   # shape: (N_subjects, 2 * len(roi_names))

    return X_t1, X_t2, X_slope, age_diff, superager_vec, X_all


def run_elastic_net(
    X: np.ndarray,
    y,
    feature_names: list,
    n_permutations: int = 1000,
    n_repeats_importance: int = 1,
    class_weight=None,
    random_state: int = 42,
    verbose: int = 1,
):
    """
    Train/evaluate an Elastic-Net logistic classifier with:
      - Nested CV (inner tuning, outer evaluation)
      - Out-of-fold predictions and fold-level metrics
      - Model-level permutation test via label shuffling
      - Out-of-fold permutation importance aggregated across folds

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : array-like
        Binary labels (0=typical, 1=superager).
    feature_names : list[str]
        Names for features (len == n_features).
    n_permutations : int
        Number of label permutations to build the null distribution (model-level p-value).
    n_repeats_importance : int
        Repeats per feature for permutation importance (on each outer test fold).
    class_weight : {'balanced', dict, None}
        Class weighting for LogisticRegression.
    random_state : int
        RNG seed for splits and shuffles.
    verbose : int
        Verbosity; prints ~10% progress during permutations if >0.

    Returns
    -------
    results : dict
        {
          'observed': {'auc', 'pr_auc', 'brier', 'balanced_acc'},
          'cv_fold_metrics': pd.DataFrame,
          'oof': {'y_true', 'y_prob', 'test_index'},
          'best_params_per_fold': list[dict],
          'coef_mean': np.ndarray,
          'coef_std': np.ndarray,
          'perm_importance': pd.DataFrame,   # feature-wise mean/std across folds
          'permutation_test': {'aucs': np.ndarray, 'p_value': float},
        }

    Notes
    -----
    • If you duplicated subjects across rows (e.g., TP1 & TP2 stacked), replace
      StratifiedKFold with GroupKFold and pass subject IDs to both outer and inner CV
      so that no subject appears in both train and test within a fold.
    """
    # Create reproducible splits
    rng = check_random_state(random_state)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # Make sure inputs are valid
    assert set(np.unique(y)) <= {0, 1}, "y must be binary (0/1)."
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples."
    assert X.shape[1] == len(feature_names), "feature_names length must match n_features."

    # Fixed CV configuration for stability
    OUTER_SPLITS = 10
    INNER_SPLITS = 5

    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=random_state + 1)

    # Pipeline: scaling + Elastic-Net logistic 
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=20000,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1
        )),
    ])

    # Hyperparameter grid (log10 C from 1e-2 to 1e2; l1_ratio in (0,1])
    param_grid = {
        "clf__C": np.logspace(-2, 2, 9),
        "clf__l1_ratio": [0.2, 0.4, 0.6, 0.8, 1.0],
    }

    # Storage for outer-CV results
    y_prob_oof = np.zeros_like(y, dtype=float)
    y_true_oof = np.zeros_like(y, dtype=int)
    test_index_oof = np.zeros_like(y, dtype=bool)

    fold_metrics = []
    best_params_per_fold = []
    coefs = []
    perm_importance_accumulator = []

    # Precompute and cache outer splits 
    outer_splits = list(outer_cv.split(X, y))

    # Outer CV loop 
    for fold_id, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=14,
        )
        gs.fit(X_tr, y_tr)

        best = gs.best_estimator_
        best_params_per_fold.append(gs.best_params_)

        prob_te = best.predict_proba(X_te)[:, 1]
        y_prob_oof[te_idx] = prob_te
        y_true_oof[te_idx] = y_te
        test_index_oof[te_idx] = True

        # Fold metrics
        auc = roc_auc_score(y_te, prob_te)
        pr_auc = average_precision_score(y_te, prob_te)
        brier = brier_score_loss(y_te, prob_te)
        y_hat = (prob_te >= 0.5).astype(int)
        bal_acc = balanced_accuracy_score(y_te, y_hat)

        fold_metrics.append({
            "fold": fold_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "auc": auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "balanced_acc": bal_acc
        })

        # Coefficients (after scaling): for interpretability, store as-is
        coef = best.named_steps["clf"].coef_.ravel()
        # Align in case something odd happens:
        assert coef.shape[0] == X.shape[1]
        coefs.append(coef)

        # Permutation importance on the outer test set (no leakage)
        # Using ROC-AUC scorer; repeats kept small per your note
        pi = permutation_importance(
            best, X_te, y_te,
            scoring="roc_auc",
            n_repeats=n_repeats_importance,
            random_state=random_state + fold_id,
            n_jobs=14
        )
        perm_importance_accumulator.append(pi.importances_mean)

    # Aggregate observed performance
    observed_auc = roc_auc_score(y_true_oof, y_prob_oof)
    observed_pr_auc = average_precision_score(y_true_oof, y_prob_oof)
    observed_brier = brier_score_loss(y_true_oof, y_prob_oof)
    observed_bal_acc = balanced_accuracy_score(y_true_oof, (y_prob_oof >= 0.5).astype(int))

    # Aggregate coefficients and permutation importances
    coefs = np.vstack(coefs)
    coef_mean = coefs.mean(axis=0)
    coef_std = coefs.std(axis=0, ddof=1)

    perm_importance_accumulator = np.vstack(perm_importance_accumulator)
    pi_mean = perm_importance_accumulator.mean(axis=0)
    pi_std = perm_importance_accumulator.std(axis=0, ddof=1)

    perm_importance_df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": pi_mean,
        "perm_importance_std": pi_std,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
        "abs_coef_mean": np.abs(coef_mean)
    }).sort_values(["perm_importance_mean", "abs_coef_mean"], ascending=False).reset_index(drop=True)

    # Model-level permutation test (label shuffling) 
    # Train inside each outer split with permuted labels (train) and evaluate on true test labels.
    perm_aucs = np.empty(n_permutations, dtype=float)

    # Progress cadence (~10% updates)
    progress_every = max(1, n_permutations // 10)

    for p in range(n_permutations):
        # Shuffle labels globally once per permutation
        y_perm = rng.permutation(y)

        # Accumulate permuted out-of-fold predictions for the true test sets
        perm_oof = np.zeros_like(y, dtype=float)

        for (tr_idx, te_idx) in outer_splits:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr_perm = y_perm[tr_idx]       # permuted labels for training
            # Tune & fit on permuted labels
            gs_perm = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner_cv,
                refit=True,
                n_jobs=14,
            )
            gs_perm.fit(X_tr, y_tr_perm)
            best_perm = gs_perm.best_estimator_
            perm_oof[te_idx] = best_perm.predict_proba(X_te)[:, 1]

        perm_aucs[p] = roc_auc_score(y_true_oof, perm_oof)

        if verbose and ((p + 1) % progress_every == 0 or (p + 1) == n_permutations):
            print(f"[Permutation {p+1}/{n_permutations}] Null AUC (mean so far): {perm_aucs[:p+1].mean():.3f}")

    # One-sided p-value (>= observed AUC), add 1 to numerator/denominator for stability
    p_value = (np.sum(perm_aucs >= observed_auc) + 1.0) / (n_permutations + 1.0)

    results = {
        "observed": {
            "auc": observed_auc,
            "pr_auc": observed_pr_auc,
            "brier": observed_brier,
            "balanced_acc": observed_bal_acc
        },
        "cv_fold_metrics": pd.DataFrame(fold_metrics),
        "oof": {
            "y_true": y_true_oof,
            "y_prob": y_prob_oof,
            "test_index": test_index_oof
        },
        "best_params_per_fold": best_params_per_fold,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
        "perm_importance": perm_importance_df,
        "permutation_test": {
            "aucs": perm_aucs,
            "p_value": float(p_value)
        }
    }
    return results

def main():
    # Just a note that running all may lead to problems with colinearity
    connectivity_type = "SFC"  # Options: "SFC", "FC", "SC", "all"
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
                save_grouped_roi_averages(fc_output, grouped_output, group_level = "ROI", subject=sub, ses=ses) 

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
                save_grouped_roi_averages(sc_output, grouped_output, group_level = "ROI", subject=sub, ses=ses)

            else:
                print(f"Missing SC CSV at path: {sc_csv}")

    # Make the grouped averages for each subject's SFC CSV   
    for sub in subjects:
        for ses in sessions:
            ses_path = root_path / ses / "individual_coupling_matrices"
            csv_path = ses_path / f"{sub}_{ses}_structure_function_coupling.csv"
            output_path = ses_path / f"{sub}_{ses}_structure_function_coupling_grouped.csv"
            if csv_path.is_file():
                save_grouped_roi_averages(csv_path, output_path, group_level = "ROI", subject=sub, ses=ses)

    # Prepare the data for analysis
    X_t1, X_t2, X_slope, age_diff, superager_vec, X_all = prep_data(
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
    
    # --- Choose which feature matrix to classify on ---
    # Options: 't1', 't2', 'slope', or 'all' (stacked: t1 + t2 + slope)
    which_features = 'all'  # <- change here as needed

    if which_features == 't1':
        X_use = X_t1
        feat_names_use = roi_names           # e.g., 59 streamline ROIs
    elif which_features == 't2':
        X_use = X_t2
        feat_names_use = roi_names
    elif which_features == 'slope':
        X_use = X_slope
        feat_names_use = roi_names
    elif which_features == 'all':
        X_use = X_all
        feat_names_use = all_roi_names       # tp1 + tp2 + slope (with suffixes)
    else:
        raise ValueError("which_features must be one of: 't1', 't2', 'slope', 'all'")

    print(f"Training RF on {which_features} features "
          f"({X_use.shape[1]} predictors) for connectivity_type={connectivity_type}...")

    # import time

    # t0 = time.time()
    # results = run_elastic_net(
    #     X_use, superager_vec, feat_names_use,
    #     n_permutations=100,
    #     n_repeats_importance=1,
    #     class_weight=None,        
    #     random_state=7,
    #     verbose=1
    # )

    # print(results["observed"])
    # print("Model-level p-value:", results["permutation_test"]["p_value"])
    # results["perm_importance"].head(10)
    
    # dt = time.time() - t0
    
    # print(f"Mini-run took {dt:.2f}s")

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # from scipy.stats import mannwhitneyu
    # from statsmodels.stats.multitest import multipletests

    # # Build a tidy DataFrame
    # df = pd.DataFrame(X_use, columns=feat_names_use)
    # df["group"] = np.where(np.asarray(superager_vec).astype(int) == 1, "superager", "control")

    # # Mann–Whitney U test per feature
    # records = []
    # for feat in feat_names_use:
    #     a = df.loc[df["group"] == "control", feat].dropna().values
    #     b = df.loc[df["group"] == "superager", feat].dropna().values
    #     # Skip features with no variance in either group
    #     if (a.size < 2) or (b.size < 2) or (np.all(a == a[0])) or (np.all(b == b[0])):
    #         p = np.nan
    #         med_diff = np.nan
    #     else:
    #         stat = mannwhitneyu(a, b, alternative="two-sided")
    #         p = stat.pvalue
    #         med_diff = np.median(b) - np.median(a)  # direction (superager - control)
    #     records.append({"feature": feat, "p": p, "median_diff": med_diff})

    # stats_df = pd.DataFrame(records)
    # stats_df["significant"] = stats_df["p"] < 0.05

    # # Keep only significant features; sort by p then |median_diff|
    # sig = stats_df[stats_df["significant"]].copy()
    # sig = sig.sort_values(["p", "median_diff"], ascending=[True, False])

    # if sig.empty:
    #     print("No features are significantly different. No figure saved.")
    # else:
    #     # Plot in pages of 12 features per figure
    #     per_page = 12
    #     palettes = "Set2"
    #     sig_features = sig["feature"].tolist()

    #     for page_idx in range(0, len(sig_features), per_page):
    #         subset = sig_features[page_idx:page_idx + per_page]
    #         ncols = 3
    #         nrows = int(np.ceil(len(subset) / ncols))
    #         fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), constrained_layout=True)

    #         for i, feat in enumerate(subset):
    #             ax = axes.flat[i]
    #             sns.violinplot(
    #                 data=df, x="group", y=feat,
    #                 inner="box", cut=0, ax=ax, palette=palettes
    #             )
    #             # Title with p-value and direction (median diff)
    #             row = sig.loc[sig["feature"] == feat].iloc[0]
    #             p_text = f"p={row['p']:.3g}" if pd.notna(row["p"]) else "p=NA"
    #             md_text = f"Δ̃ (SA–Ctl)={row['median_diff']:.3g}"
    #             ax.set_title(f"{feat} • {p_text} • {md_text}", fontsize=10)
    #             ax.set_xlabel("")
    #             ax.set_ylabel("")

    #         # Hide any unused subplots
    #         total_axes = nrows * ncols
    #         for j in range(i + 1, total_axes):
    #             axes.flat[j].axis("off")

    #         plt.suptitle("Significant Feature Distributions (Superager vs Control, FDR q<0.05)", fontsize=14)
    #         out_file = f"feature_comparison_sig_p{page_idx // per_page + 1}.png"
    #         plt.savefig(out_file, dpi=300)
    #         plt.close()
    #         print(f"Saved {out_file} with {len(subset)} significant features.")


if __name__ == "__main__":
    main()