import numpy as np
import time
import pickle
from sklearn.inspection import permutation_importance
import pandas as pd
from pathlib import Path
import os, re
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from prep_data_for_en import get_subjects_to_process



def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, alpha=0.05, random_state=7):
    """Compute a bootstrap CI for AUC on out-of-fold predictions.
    So can think of this as how stable is the AUC across different resampling of
    the subjects in the outer folds (e.g., some subjects appear multiple times 
    and some not at all in each resample).
    
    Args:
        y_true (array-like): True binary labels (0/1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_boot (int): Number of bootstrap samples to draw.
        alpha (float): Significance level for the confidence interval (e.g., 0.05 for 95% CI).
        random_state (int): Seed for reproducibility.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = y_true.shape[0]
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return (np.nan, np.nan)
    lo = np.percentile(aucs, 100 * (alpha / 2))
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return (float(lo), float(hi))


def prep_data(subjects, root_path, fc_root_path, sc_root_path, demographic_data, connectivity_type):
    """Prepares the data for analysis by extracting features and superager status
    from the specified directories.

    Args:
        subjects (list): List of subject IDs to process.
        root_path (Path): Path to the root directory containing SFC data.
        fc_root_path (Path): Path to the root directory containing FC data.
        sc_root_path (Path): Path to the root directory containing SC data.
        demographic_data (pd.DataFrame): DataFrame containing demographic information.
        connectivity_type (str): Type of connectivity data to process ("SFC", "FC", or "SC").

    Returns:
        X_t1 (np.ndarray): Feature matrix for timepoint 1.
        X_slope (np.ndarray): Feature matrix representing annual change between timepoints.
        superager_vec_long (np.ndarray): Binary vector indicating longitudinal superager status.
        covariates (np.ndarray): Matrix of covariates (baseline age, years of education, sex).
    """
    # Prepare the data
    demographic_data = demographic_data.copy()
    demographic_data["id"] = demographic_data["id"].astype(str)
    if not demographic_data["id"].str.startswith("sub-").all():
        demographic_data["id"] = "sub-" + demographic_data["id"].str.replace("^sub-", "", regex=True)

    roi_refs = {}
    def load_modality(sub, mod):
        """Helper function to load per-ROI features for a given modality.

        Args:
            sub (str): Subject ID.
            mod (str): Modality ("SFC", "FC", or "SC").
        """
        # Use ungrouped per-ROI features (214 ROIs) for both timepoints
        if mod == "SFC":
            p1 = root_path / "ses-01" / "individual_coupling_matrices"
            p2 = root_path / "ses-02" / "individual_coupling_matrices"
            f1 = p1 / f"{sub}_ses-01_structure_function_coupling.csv"
            f2 = p2 / f"{sub}_ses-02_structure_function_coupling.csv"
        elif mod == "FC":
            f1 = fc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_functional_connectivity_flat.csv"
            f2 = fc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_functional_connectivity_flat.csv"
        else:  # SC
            f1 = sc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_structural_connectivity_flat.csv"
            f2 = sc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_structural_connectivity_flat.csv"
        col = 'pearson_rho'

        def read_one(path, ses_name):
            """Reads a single CSV file for one subject, modality, and session.
            Checks that the ROI order matches across subjects and sessions.

            Args:
                path (Path): Path to the CSV file to read.
                ses_name (str): Session name ("ses-01" or "ses-02") for ROI reference.
            """
            if not path.is_file():
                return None, None
            d = pd.read_csv(path)
            roi = d["ROI_name"].astype(str).tolist()
            key = f"{mod}_{ses_name}"
            if key not in roi_refs:
                roi_refs[key] = roi
            elif roi_refs[key] != roi:
                raise ValueError(f"ROI order mismatch across subjects for {mod} {ses_name}")
            vec = pd.to_numeric(d[col].values, errors='coerce')
            return vec, roi

        v1, roi1 = read_one(f1, "ses-01")
        v2, roi2 = read_one(f2, "ses-02")
        if v1 is None or v2 is None:
            return None, None
        if roi1 != roi2:
            raise ValueError(f"ROI order mismatch between timepoints for {sub} {mod}")
        return v1, v2

    records = []
    for sub in subjects:
        # Load features
        feat1, feat2 = load_modality(sub, connectivity_type)
        if feat1 is None:
            continue

        # Pull demographics and superager status from demographic_data
        row = demographic_data[demographic_data['id'] == sub]
        if row.empty:
            continue
        age1 = row.iloc[0]['age_1']
        age2 = row.iloc[0]['age_2']
        superager_long = row.iloc[0]['superager_long']
        YoE = row.iloc[0]['YoE']
        sex = row.iloc[0]['sex']

        # Convert sex to binary 0/1, NaN subjects are excluded later via valid_rows
        if pd.isna(sex):
            sex = np.nan
        elif sex == 'male':
            sex = 1
        elif sex == 'female':
            sex = 0
        else:
            sex = np.nan

        records.append({
            'id'             : sub,
            'feat1'          : feat1,
            'feat2'          : feat2,
            'age1'           : age1,
            'age2'           : age2,
            'superager_long' : superager_long,
            'YoE'            : YoE,
            'sex'            : sex,
        })

    # Build DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            "No valid subject records after feature/demographic matching. "
            "Check ID formatting and required columns in demographic_data."
        )

    # 1) Stack features and compute annual change
    X_t1 = np.stack(df['feat1'].values)
    X_t2 = np.stack(df['feat2'].values)
    age_diff = pd.to_numeric(df['age2'] - df['age1'], errors='coerce').values.astype(float)
    X_slope = np.full_like(X_t1, np.nan, dtype=float)
    valid_age_diff = np.isfinite(age_diff) & (age_diff != 0)
    X_slope[valid_age_diff] = (X_t2[valid_age_diff] - X_t1[valid_age_diff]) / age_diff[valid_age_diff, None]

    # 2) Covariates: baseline age, years of education, sex
    covariates = np.vstack([df['age1'].values, df['YoE'].values, df['sex'].values]).T

    # 3) Longitudinal superager label
    superager_vec_long = pd.to_numeric(df['superager_long'], errors='coerce').values

    return X_t1, X_slope, superager_vec_long, covariates



def _perm_importance_on_train_cv(pipe, best_params, X_tr, y_tr, inner_cv, n_repeats, random_state):
    """Compute permutation importance on training data via inner CV splits.
    This avoids using the outer test fold for importance estimation.

    Args:
        pipe: The sklearn Pipeline with the model to fit.
        best_params: The best hyperparameters found from GridSearchCV on the training fold.
        X_tr: The training feature matrix for the current outer fold.
        y_tr: The training labels for the current outer fold.
        inner_cv: The cross-validation splitter for the inner loop.
        n_repeats: The number of repeats for permutation importance.
        random_state: The random seed for reproducibility.

    Returns:
        np.ndarray: The mean permutation importance for each feature, averaged across inner CV folds.
    """
    per_fold_imps = []
    for fold_id, (itr, ival) in enumerate(inner_cv.split(X_tr, y_tr), start=1):
        est = clone(pipe).set_params(**best_params)
        est.fit(X_tr[itr], y_tr[itr])
        pi = permutation_importance(
            est, X_tr[ival], y_tr[ival],
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state + fold_id,
            n_jobs=1,
        )
        per_fold_imps.append(pi.importances_mean)
    return np.mean(np.vstack(per_fold_imps), axis=0)


def run_elastic_net(
    X: np.ndarray,
    y,
    feature_names: list,
    n_permutations: int = 1000,
    n_repeats_importance: int = 20,
    random_state: int = 42,
    verbose: int = 1,
    checkpoint_n=None,
    connectivity_type=None,
    group_level=None,
    which_features=None,
    label_type=None,
):
    """Train and evaluate an elastic net logistic classifier with nested cross-validation,
    out-of-fold predictions, permutation testing, and permutation-based feature importance.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (array-like): Binary labels (0 = non-superager, 1 = superager).
        feature_names (list[str]): Names of features (length = n_features).
        n_permutations (int): Number of label permutations for building the null distribution
            (used to compute model-level p-value).
        n_repeats_importance (int): Number of repeats per feature for permutation importance
            (evaluated on each outer training set across inner CV folds).
        random_state (int): Seed for random number generation in splits and shuffles.
        verbose (int): Verbosity level; prints ~10% progress during permutations if > 0.
        connectivity_type (str): Type of connectivity data ("SFC", "FC", "SC") for naming outputs.
        group_level (str): Grouping level for ROI averaging (always "ROI") for naming outputs.
        checkpoint_n (int): Save progress every `checkpoint_n` permutations to a pickle file.
        which_features (str): Which features are being used ('t1' or 't1_slope') for naming outputs.
        label_type (str): Outcome timepoint (always "long") for naming outputs.

    Returns:
        dict containing:
            - "observed": with metrics {"auc"}.
            - "cv_fold_metrics": with per-fold AUC.
            - "oof": with {"y_true", "y_prob", "test_index"} for out-of-fold predictions.
            - "best_params_per_fold": with best hyperparameters per fold.
            - "coef_mean": with mean coefficients across folds.
            - "coef_std": with coefficient standard deviations across folds.
            - "feat_importance": with feature-wise permutation importance, coefficients, and selection frequency.
            - "permutation_test": with {"aucs": np.ndarray, "p_value": float}.
    """
    # Create reproducible splits
    rng = check_random_state(random_state)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # Make sure inputs are valid
    assert set(np.unique(y)) <= {0, 1}, "y must be binary (0/1)."
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples."
    assert X.shape[1] == len(feature_names), "feature_names length must match n_features."
    if checkpoint_n is None or checkpoint_n <= 0:
        checkpoint_n = n_permutations

    # Fixed CV configuration for stability
    OUTER_SPLITS = 10
    INNER_SPLITS = 5

    # Stratified K-Folds keeps the same number of superagers/non-superagers in each fold
    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=random_state + 1)

    # Pipeline: scaling + Elastic-Net logistic 
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)), # EN requires scaling
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga", # only solver in scikit-learn that supports EN
            max_iter=20000, # increased because of convergence warnings at 5,000
            random_state=random_state,
            n_jobs=1
        )),
    ])

    # Hyperparameter grid 
    param_grid = {
        "clf__C": np.logspace(-4, 3, 10), # higher c = less regularization, may be overfit
        "clf__l1_ratio": [0.01, 0.1, 0.3, 0.5], # l1_ratio=1 is fully Lasso where some coeffs = 0
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

    # Outer CV: each fold holds out ~10% of subjects (X_te/y_te) as a test set 
    # This runs all 10 folds and every subject get exactly one OOF prediction
    for fold_id, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        X_tr, X_te = X[tr_idx], X[te_idx] # X_tr is the outer training set, X_te is the test set
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Inner CV: try all C/l1_ratio combos on X_tr via 5-fold CV, pick the best
        # refit=True trains a final model on all of X_tr with the winning params
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=20,
        )
        gs.fit(X_tr, y_tr)

        best = gs.best_estimator_  # model refitted on full X_tr with best hyperparameters
        best_params = gs.best_params_
        best_params_per_fold.append(best_params)

        # Use X_te and predict superager probability for held-out subjects
        prob_te = best.predict_proba(X_te)[:, 1]
        y_prob_oof[te_idx] = prob_te
        y_true_oof[te_idx] = y_te
        test_index_oof[te_idx] = True

        # AUC on this fold's test set — saved to check variance across folds
        auc = roc_auc_score(y_te, prob_te)
        fold_metrics.append({
            "fold": fold_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "auc": auc,
        })

        # Coefficients from the logistic regression step, averaged across folds after the loop
        coef = best.named_steps["clf"].coef_.ravel()
        assert coef.shape[0] == X.shape[1]
        coefs.append(coef)

        # Permutation importance on X_tr inner folds to keep the test set clean
        pi_mean = _perm_importance_on_train_cv(
            pipe, best_params, X_tr, y_tr, inner_cv, n_repeats_importance, random_state + fold_id
        )
        perm_importance_accumulator.append(pi_mean)

    # Metrics for how well the EN model did overall (on all out-of-fold predictions)
    observed_auc = roc_auc_score(y_true_oof, y_prob_oof)

    # Metrics for how important each feature was overall
    coefs = np.vstack(coefs)
    coef_mean = coefs.mean(axis=0)
    coef_std = coefs.std(axis=0, ddof=1)
    selection_freq = (coefs != 0).mean(axis=0)

    perm_importance_accumulator = np.vstack(perm_importance_accumulator)
    pi_mean = perm_importance_accumulator.mean(axis=0)

    feature_df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": pi_mean,
        "coef_mean": coef_mean,
        "coef_std": coef_std,
        "selected_freq": selection_freq,
        "abs_coef_mean": np.abs(coef_mean)
    }).sort_values(["abs_coef_mean"], ascending=False)

    # Model-level permutation test - now build the null distribution training the models on shuffled labels
    # Start by adding a check point for if the server crashes while this is running, not all progress is lost
    checkpoint_file = (
        f"results/{connectivity_type}_{group_level}_{which_features}_"
        f"{label_type}_include_perm_results.pkl"
    )
    start_p = 0
    completed = 0

    # Resume if checkpoint exists (i.e. from a previous crash or stoppage)
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            saved = pickle.load(f)

        completed = int(np.count_nonzero(~np.isnan(saved["perm_aucs"])))
        target_total = max(n_permutations, completed)

        if completed < target_total:
            # Expand arrays to the new total and copy old data
            perm_aucs = np.empty(target_total, dtype=float); perm_aucs[:] = np.nan
            valid_mask = ~np.isnan(saved["perm_aucs"])
            n_valid = valid_mask.sum()
            perm_aucs[:n_valid] = saved["perm_aucs"][valid_mask]
            print(f"Expanding from {completed} → {target_total} total permutations.")
        else:
            # Reuse existing arrays
            perm_aucs = saved["perm_aucs"]
            target_total = max(n_permutations, completed)
            print(f"Using existing {completed} total permutations.")

        start_p = completed
    else:
        print(f"Starting permutations from scratch, running {n_permutations} total.")
        target_total = n_permutations
        perm_aucs = np.empty(target_total, dtype=float); perm_aucs[:] = np.nan
        start_p = 0

    progress_every = max(1, target_total // 10) # report progress after every 10%
    
    if completed >= target_total:
        print("All requested permutations already done.")
    else:
        for p in range(start_p, target_total):
            # For each permutation, shuffle the labels randomly
            y_perm = rng.permutation(y)

            # Save the out-of-fold predictions for each permutation
            perm_oof = np.zeros_like(y, dtype=float)

            for (tr_idx, te_idx) in outer_splits:
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr_perm = y_perm[tr_idx]       # permuted labels for training
                # Train and run on permuted labels
                gs_perm = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=inner_cv, # runs inner CV search grid to make the comparisons fair (e.g. 5-fold)
                    refit=True,
                    n_jobs=20,
                )
                gs_perm.fit(X_tr, y_tr_perm)
                best_perm = gs_perm.best_estimator_
                perm_oof[te_idx] = best_perm.predict_proba(X_te)[:, 1]

            perm_aucs[p] = roc_auc_score(y, perm_oof)
        
            # Save results at each checkpoint_n permutations
            if (p + 1) % checkpoint_n == 0 or (p + 1) == target_total:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({
                        "last_p": p,
                        "perm_aucs": perm_aucs,
                        "feature": feature_names,   # safe mid-run
                    }, f)
                print(f"Checkpoint saved at permutation {p+1}")

            if verbose and ((p + 1) % progress_every == 0 or (p + 1) == target_total):
                print(f"[Permutation {p+1}/{target_total}] Null AUC (mean so far): {np.nanmean(perm_aucs):.3f}")

    # One-sided p-value (>= observed AUC), add 1 to numerator/denominator for stability
    # Needs a sufficient number of permutations to get a good estimate of the p-value
    p_value = (np.sum(perm_aucs >= observed_auc) + 1.0) / (target_total + 1.0)

    feature_df.sort_values("perm_importance_mean", ascending=False, inplace=True)

    # Save final state including p-value
    with open(f"{checkpoint_file}", "wb") as f:
        pickle.dump({
            "last_p": target_total - 1,
            "perm_aucs": perm_aucs,
            "feature": feature_names,
            "p_value": p_value,   # model-level p
        }, f)
         
    if start_p < target_total:  
        print(f"Final checkpoint saved at permutation {p+1}")

    results = {
        "observed": {
            "auc": observed_auc,
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
        "feat_importance": feature_df,
        "permutation_test": {
            "aucs": perm_aucs,
            "p_value": float(p_value)
        }
    }
    return results

def main():
    connectivity_type = "SFC"  # Options: "SFC", "FC", "SC"
    # These are the features used to predict superager status
    which_features = 't1'  # Options: 't1' (baseline only), 't1_slope' (baseline + annual change)
    root_path = Path("/home/rachel/Desktop/schaefer_analysis/structure_function_coupling")
    fc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/functional_connectivity/native_space")
    sc_root_path = Path("/home/rachel/Desktop/schaefer_analysis/structural_connectivity")
    csv_path = Path("/home/rachel/Desktop/data/superager.csv")
    demographic_data = pd.read_csv(csv_path)
    demographic_data.columns = [re.sub(r"^w(\d+)_(.*)", r"\2_\1", c) for c in demographic_data.columns]
    required_cols = {"id", "age_1", "age_2", "YoE", "sex", "superager_long"}
    missing_cols = sorted(required_cols - set(demographic_data.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")
    age_dir = Path("/home/rachel/Desktop/data")

    # Longitudinal superager definition requires both sessions
    subjects_tp1 = get_subjects_to_process(root_path / "ses-01" / "individual_coupling_matrices", "ses-01", age_dir)
    subjects_tp2 = get_subjects_to_process(root_path / "ses-02" / "individual_coupling_matrices", "ses-02", age_dir)
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # Prepare the data for analysis
    X_t1, X_slope, superager_vec_long, covariates = prep_data(
        subjects, root_path, fc_root_path, sc_root_path, demographic_data, connectivity_type)

    # Map feature indices back to ROI names 
    if connectivity_type == "SFC":
        roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling.csv"
    elif connectivity_type == "FC":
        roi_names_path = fc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{subjects[0]}_ses-01_functional_connectivity_flat.csv"
    else:  # SC
        roi_names_path = sc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{subjects[0]}_ses-01_structural_connectivity_flat.csv"
    roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    roi_names_tp1 = [f"{r}_1" for r in roi_names]
    roi_names_slope = [f"{r}_slope" for r in roi_names]
    X_t1_slope = np.hstack([X_t1, X_slope])

    # Select which features to use, matching them to ROI names
    match which_features:
        case 't1':
            X_use = X_t1
            feat_names_use = roi_names
        case 't1_slope':
            X_use = X_t1_slope
            feat_names_use = roi_names_tp1 + roi_names_slope
        case _:
            raise ValueError("which_features must be one of: 't1', 't1_slope'")

    y_use = superager_vec_long

    valid_rows = np.isfinite(y_use) & np.isfinite(X_use).all(axis=1)
    valid_rows = valid_rows & np.isfinite(covariates).all(axis=1)
    X_use = X_use[valid_rows]
    y_use = y_use[valid_rows].astype(int)
    covariates = covariates[valid_rows]

    # Include covariates as model features
    X_use = np.hstack([X_use, covariates.astype(float)])
    feat_names_use = feat_names_use + ["cov_age", "cov_YoE", "cov_sex"]

    print(
        f"Training EN on {which_features} features "
        f"({X_use.shape[1]} predictors) for "
        f"{connectivity_type=}..."
    )

    t0 = time.time()
    print("Starting time:", time.ctime(t0))
    results = run_elastic_net(
        X_use, y_use, feat_names_use,
        n_permutations=1000,
        n_repeats_importance=10,
        random_state=7,
        verbose=1,
        checkpoint_n=10,
        connectivity_type=connectivity_type,
        group_level="ROI",
        which_features=which_features,
        label_type="long",
    )

    feature_importance_csv = f"{connectivity_type}_ROI_{which_features}_long_include_feature_importance.csv"
    results["feat_importance"].to_csv(feature_importance_csv, index=False)
    print(f"Saved feature importance table: {feature_importance_csv}")

    print(results["observed"])
    fold_metrics = results["cv_fold_metrics"]
    print("Fold-wise AUCs:")
    for _, row in fold_metrics.iterrows():
        print(f"  fold {int(row['fold'])}: auc={row['auc']:.3f}")
    auc_ci = bootstrap_auc_ci(
        results["oof"]["y_true"],
        results["oof"]["y_prob"],
        n_boot=2000,
        alpha=0.05,
        random_state=7,
    )
    print(f"AUC 95% CI (bootstrap, oof): [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    print("Best params per fold (C, l1_ratio):")
    for i, p in enumerate(results["best_params_per_fold"], start=1):
        c_val = p.get("clf__C")
        l1_val = p.get("clf__l1_ratio")
        print(f"  fold {i}: C={c_val}, l1_ratio={l1_val}")
    print("Model-level p-value:", results["permutation_test"]["p_value"])

    feat = results["feat_importance"].copy()
    feat = feat[~feat["feature"].str.startswith("cov_", na=False)]
    compact = feat[["feature", "perm_importance_mean", "selected_freq"]].head(15)
    print("Top features by permutation importance:")
    with pd.option_context("display.max_columns", None):
        print(compact.to_string(index=False))
    dt = time.time() - t0
    print(f"Run took {dt:.2f}s")
    

if __name__ == "__main__":
    main()
