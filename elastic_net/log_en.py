import numpy as np
import time
import pickle
from sklearn.inspection import permutation_importance
import pandas as pd
from pathlib import Path
import os, re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests
from prep_data_for_en import get_subjects_to_process
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import seaborn as sns


def prep_data(subjects, root_path, fc_root_path, sc_root_path, memory_data, connectivity_type, group_level):
    """Prepares the data for analysis by extracting features, memory outcomes, and superager status
    from the specified directories and memory data.

    Args:
        subjects (list): List of subject IDs to process.
        root_path (Path): Path to the root directory containing SFC data.
        fc_root_path (Path): Path to the root directory containing FC data.
        sc_root_path (Path): Path to the root directory containing SC data.
        memory_data (pd.DataFrame): DataFrame containing memory outcomes and demographics.
        connectivity_type (str): Type of connectivity data to process ("SFC", "FC", or "SC").
        group_level (str): Grouping level for ROI averaging ("ROI" or "network").

    Returns:
        X_t1 (np.ndarray): Feature matrix for timepoint 1.
        X_t2 (np.ndarray): Feature matrix for timepoint 2.
        X_slope (np.ndarray): Feature matrix representing the slope between timepoints.
        X_t1_t2 (np.ndarray): Combined feature matrix for timepoints 1 and 2.
        superager_vec (np.ndarray): Binary vector indicating superager status.
        X_all (np.ndarray): Combined feature matrix for all features.
        age1 (np.ndarray): Ages at timepoint 1.
        YoE (np.ndarray): Years of education.
        sex (np.ndarray): Sex of the subjects.
    """
    # Prepare the data 
    def load_modality(sub, mod):
        """Helper function to load features for a given modality (eg SFC)."""
        if mod == "SFC":
            p1 = root_path / "ses-01" / "individual_coupling_matrices"
            p2 = root_path / "ses-02" / "individual_coupling_matrices"
            f1 = p1 / f"{sub}_ses-01_structure_function_coupling_grouped_by_{group_level}.csv"
            f2 = p2 / f"{sub}_ses-02_structure_function_coupling_grouped_by_{group_level}.csv"
            col = 'pearson_rho'
        elif mod == "FC":
            f1 = fc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_functional_connectivity_grouped_by_{group_level}.csv"
            f2 = fc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_functional_connectivity_grouped_by_{group_level}.csv"
            col = 'pearson_rho' 
        else:  # "SC"
            f1 = sc_root_path / f"ses-01/individual_connectivity_matrices/grouped_rois/{sub}_ses-01_structural_connectivity_grouped_by_{group_level}.csv"
            f2 = sc_root_path / f"ses-02/individual_connectivity_matrices/grouped_rois/{sub}_ses-02_structural_connectivity_grouped_by_{group_level}.csv"
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

        # Convert sex to binary 0/1
        sex = 1 if sex == 'male' else 0

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
    covariates = np.vstack([df['age1'].values, df['YoE'].values, df['sex'].values]).T   

    # 2) Make a superager vector 
    superager_vec = df['superager'].values.astype(int)

    # 3) Generate a stacked version of tp1 and tp2 features
    X_all = np.hstack([X_t1, X_t2, X_slope])   # shape: (N_subjects, 2 * len(roi_names))
    X_t1_t2 = np.hstack([X_t1, X_t2])  

    return X_t1, X_t2, X_slope, X_t1_t2, superager_vec, X_all, covariates


def take_residuals_covars(X, covariates):
    """
    Takes the residuals of ROI features using covariates (i.e., age, YoE, sex).
    For each feature column, fits a linear regression using the covariates,
    and the residuals (feature minus predicted component) returned. This
    ensures that downstream models operate only on variance unexplained by
    the covariates.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        covariates (np.ndarray): Matrix of covariates of shape (n_samples, n_covariates).

    Returns:
        np.ndarray: Residualized feature matrix of shape (n_samples, n_features),
                    where each column has been adjusted for the covariates.
    """
    n_samples, n_features = X.shape
    X_resid = np.zeros_like(X)

    for j in range(n_features):
        # Fit linear regression of feature j on covariates
        model = LinearRegression().fit(covariates, X[:, j])
        # Predicted values from covariates
        pred = model.predict(covariates)
        # Residuals = observed - predicted
        X_resid[:, j] = X[:, j] - pred

    return X_resid

def run_elastic_net(
    X: np.ndarray,
    y,
    feature_names: list,
    n_permutations: int = 1000,
    n_repeats_importance: int = 20,
    class_weight=None,
    random_state: int = 42,
    verbose: int = 1,
    checkpoint_n=None,
    connectivity_type=None,
    group_level=None,
    which_features=None,
    covariates: np.ndarray=None,
):
    """Train and evaluate an Elastic-Net logistic classifier with nested cross-validation,
    out-of-fold predictions, permutation testing, and permutation-based feature importance.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (array-like): Binary labels (0 = non-superager, 1 = superager).
        feature_names (list[str]): Names of features (length = n_features).
        n_permutations (int): Number of label permutations for building the null distribution
            (used to compute model-level p-value).
        n_repeats_importance (int): Number of repeats per feature for permutation importance
            (evaluated on each outer test fold).
        class_weight ({dict, None}): Class weighting for LogisticRegression.
        random_state (int): Seed for random number generation in splits and shuffles.
        verbose (int): Verbosity level; prints ~10% progress during permutations if > 0.
        connectivity_type (str): Type of connectivity data ("SFC", "FC", "SC", or "all") for naming outputs.
        group_level (str): Grouping level for ROI averaging ("ROI" or "network") for naming outputs.
        checkpoint_n (int): Save progress every `checkpoint_n` permutations to a pickle file.
        which_features (str): Which features are being used ('t1', 't2', 'slope', 't1_t2', 'all') for naming outputs.
        covariates (np.ndarray): Covariate matrix of shape (n_samples, n_covariates) for residualization.

    Returns:
        dict containing:
            - "observed": with metrics {"auc", "pr_auc", "brier", "balanced_acc"}.
            - "cv_fold_metrics": with metrics per cross-validation fold.
            - "oof": with {"y_true", "y_prob", "test_index"} for out-of-fold predictions.
            - "best_params_per_fold": with best hyperparameters per fold.
            - "coef_mean": with mean coefficients across folds.
            - "coef_std": with coefficient standard deviations across folds.
            - "feat_importance": with feature-wise mean/std permutation importance.
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

    # Fixed CV configuration for stability
    OUTER_SPLITS = 5
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
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=1
        )),
    ])

    # Hyperparameter grid 
    param_grid = {
        "clf__C": np.logspace(-2, 2, 9), # higher c = less regularization, may be overfit
        "clf__l1_ratio": [0.2, 0.4, 0.6, 0.8, 1.0], # l1_ratio=1 is fully Lasso where some coeffs = 0
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

    # Outer CV - for each outer split holds out X_te/y_te as the final test set
    for fold_id, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        X_tr, X_te = X[tr_idx], X[te_idx] # X_tr is the outer training set, X_te is the test set

        # Residualize ROI features against covariates inside the outer split
        cov_tr = covariates[tr_idx]   # age1, YoE, sex for training subjects
        cov_te = covariates[te_idx]   # covariates for test subjects
        X_tr = take_residuals_covars(X_tr, cov_tr)
        X_te = take_residuals_covars(X_te, cov_te)

        y_tr, y_te = y[tr_idx], y[te_idx]

        gs = GridSearchCV( # splits X_tr into inner train and validation sets
            estimator=pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv, # runs inner CV on X_tr split into inner train/val sets
            refit=True, # fits the best model on the whole X_tr after tuning
            n_jobs=9,
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
        pr_auc = average_precision_score(y_te, prob_te) # average precision = area under PR curve
        brier = brier_score_loss(y_te, prob_te) # measures calibration (lower is better) - how close are predicted probs to reality
        y_hat = (prob_te >= 0.5).astype(int) 
        bal_acc = balanced_accuracy_score(y_te, y_hat) # average of sensitivity and specificity

        # Metrics from the refit on the outer test set
        fold_metrics.append({
            "fold": fold_id,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "auc": auc,
            "pr_auc": pr_auc,
            "brier": brier,
            "balanced_acc": bal_acc
        })

        # Coefficients from the refit/best model 
        coef = best.named_steps["clf"].coef_.ravel()
        assert coef.shape[0] == X.shape[1]
        coefs.append(coef)

        # Now determine the importance of features
        # Permutation importance on the outer test set from the refit/best model
        # Idea remove one feature at a time and see how much the model performance drops
        pi = permutation_importance(
            best, X_te, y_te,
            scoring="roc_auc", # Considers sensitivity and specificity across all thresholds
            n_repeats=n_repeats_importance,
            random_state=random_state + fold_id,
            n_jobs=9
        )
        perm_importance_accumulator.append(pi.importances_mean)

    # Metrics for how well the EN model did overall (on all out-of-fold predictions)
    observed_auc = roc_auc_score(y_true_oof, y_prob_oof)
    observed_pr_auc = average_precision_score(y_true_oof, y_prob_oof)
    observed_brier = brier_score_loss(y_true_oof, y_prob_oof)
    observed_bal_acc = balanced_accuracy_score(y_true_oof, (y_prob_oof >= 0.5).astype(int))

    # Metrics for how important each feature was overall
    coefs = np.vstack(coefs)
    coef_mean = coefs.mean(axis=0)
    coef_std = coefs.std(axis=0, ddof=1)
    selection_freq = (coefs != 0).mean(axis=0)

    # Permutation importance for calculating feature-level p-values
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
    checkpoint_file = f"{connectivity_type}_{group_level}_{which_features}_perm_results.pkl"
    start_p = 0
    completed = 0

    # Resume if checkpoint exists (ie from a previous crash)
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            saved = pickle.load(f)

        completed = int(np.count_nonzero(~np.isnan(saved["perm_aucs"])))
        n_features = X.shape[1]
        target_total = max(n_permutations, completed)

        if completed < target_total:
            # Expand arrays to the new total and copy old data
            perm_aucs = np.empty(target_total, dtype=float); perm_aucs[:] = np.nan
            pi_null   = np.zeros((target_total, n_features), dtype=float)
            valid_mask = ~np.isnan(saved["perm_aucs"])
            n_valid = valid_mask.sum()
            perm_aucs[:n_valid] = saved["perm_aucs"][valid_mask]
            pi_null[:n_valid]   = saved["pi_null"][valid_mask]
            print(f"Expanding from {completed} → {target_total} total permutations.")
        else:
            # Reuse existing arrays
            perm_aucs = saved["perm_aucs"]
            pi_null   = saved["pi_null"]
            target_total = max(n_permutations, completed)
            print(f"Using existing {completed} total permutations.")

        start_p = completed
    else:
        print(f"Starting permutations from scratch, running {n_permutations} total.")
        target_total = n_permutations
        perm_aucs = np.empty(target_total, dtype=float); perm_aucs[:] = np.nan
        pi_null   = np.zeros((target_total, X.shape[1]), dtype=float)
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
            per_fold_imps = []

            for (tr_idx, te_idx) in outer_splits:
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr_perm = y_perm[tr_idx]       # permuted labels for training
                # Train and run on permuted labels
                gs_perm = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=inner_cv, # runs inner CV search grid to make the comparisons fair
                    refit=True,
                    n_jobs=9,
                )
                gs_perm.fit(X_tr, y_tr_perm)
                best_perm = gs_perm.best_estimator_
                perm_oof[te_idx] = best_perm.predict_proba(X_te)[:, 1]

                # Now calculate the importance of features in this permutation
                pi_perm = permutation_importance(
                    best_perm, X_te, y[te_idx],
                    scoring="roc_auc",
                    n_repeats=n_repeats_importance,
                    random_state=random_state + 10_000 + p,
                    n_jobs=9,
                )
                per_fold_imps.append(pi_perm.importances_mean)
        
            perm_aucs[p] = roc_auc_score(y, perm_oof)

            # Average feature importances across outer folds for this permutation
            if per_fold_imps:
                pi_null[p, :] = np.mean(np.vstack(per_fold_imps), axis=0)
            else:
                pi_null[p, :] = 0.0 

            if (p + 1) % checkpoint_n == 0 or (p + 1) == target_total:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump({
                        "last_p": p,
                        "perm_aucs": perm_aucs,
                        "pi_null": pi_null,
                        "feature": feature_names,   # safe mid-run
                    }, f)
                print(f"Checkpoint saved at permutation {p+1}")

            if verbose and ((p + 1) % progress_every == 0 or (p + 1) == target_total):
                print(f"[Permutation {p+1}/{target_total}] Null AUC (mean so far): {np.nanmean(perm_aucs):.3f}")

    # One-sided p-value (>= observed AUC), add 1 to numerator/denominator for stability
    # Must run a sufficient number of permutations to get a good estimate of the p-value
    p_value = (np.sum(perm_aucs >= observed_auc) + 1.0) / (target_total + 1.0)

    # Feature-level p-values corrected FDR
    completed = np.count_nonzero(~np.isnan(perm_aucs))
    pvals = ((pi_null[:completed] >= pi_mean).sum(axis=0) + 1.0) / (completed + 1.0)
    rej, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    feature_df["p_value"] = pvals
    feature_df["p_fdr"] = pvals_fdr
    feature_df.sort_values(["p_value", "perm_importance_mean"], ascending=[True, False], inplace=True)

    # Save results at each checkpoint_n permutations
    with open(f"{checkpoint_file}", "wb") as f:
        pickle.dump({
            "last_p": target_total - 1,
            "perm_aucs": perm_aucs,
            "pi_null": pi_null,
            "feature": feature_names,
            "p_value": p_value,   # model-level p
            "pvals": pvals,       # feature-wise raw p
            "p_fdr": pvals_fdr    # feature-wise FDR
        }, f)
         
    if start_p < target_total:  
        print(f"Final checkpoint saved at permutation {p+1}")

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
        "feat_importance": feature_df,
        "permutation_test": {
            "aucs": perm_aucs,
            "p_value": float(p_value)
        }
    }
    return results

def main():
    connectivity_type = "FC"  # Options: "SFC", "FC", "SC", "all"
    which_features = 't1_t2' # Options: 't1', 't2', 'slope', 't1_t2', 'all'
    group_level = "ROI" # Options: "ROI", "network"
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

    # Prepare the data for analysis 
    X_t1, X_t2, X_slope, X_t1_t2, superager_vec, X_all, covariates = prep_data(
        subjects, root_path, fc_root_path, sc_root_path, memory_data, connectivity_type, group_level)

    # Map feature indices back to ROI names 
    roi_names_path = root_path / "ses-01" / "individual_coupling_matrices" / f"{subjects[0]}_ses-01_structure_function_coupling_grouped_by_{group_level}.csv"
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
    
    # Select which features to use, matching them to ROI names
    match which_features:
        case 't1':
            X_use = X_t1
            feat_names_use = roi_names           # e.g., 59 streamline ROIs
        case 't2':
            X_use = X_t2
            feat_names_use = roi_names
        case 'slope':
            X_use = X_slope
            feat_names_use = roi_names
        case 'all':
            X_use = X_all
            feat_names_use = all_roi_names      # e.g., 59*3 = 177 features
        case 't1_t2':
            X_use = X_t1_t2
            feat_names_use = roi_names_tp1 + roi_names_tp2
        case _:
            raise ValueError("which_features must be one of: 't1', 't2', 'slope', 'all', 't1_t2'")

    print(f"Training EN on {which_features} features "
          f"({X_use.shape[1]} predictors) for {connectivity_type=}...")

    t0 = time.time()
    print("Starting time:", time.ctime(t0))
    results = run_elastic_net(
        X_use, superager_vec, feat_names_use,
        n_permutations=1,
        n_repeats_importance=20,
        class_weight=None,        
        random_state=7,
        verbose=1,
        checkpoint_n=10,
        connectivity_type=connectivity_type,
        group_level=group_level,
        which_features=which_features,
        covariates=covariates
    )

    print(results["observed"])
    print("Model-level p-value:", results["permutation_test"]["p_value"])
    with pd.option_context("display.max_columns", None):
        print(results["feat_importance"].head(50))
    dt = time.time() - t0
    print(f"Run took {dt:.2f}s")

    # Build dataframe of features to be able to look at correlation between timepoints for significant pairs
    df_roi_values = pd.DataFrame(X_use, columns=feat_names_use)
    df_roi_values["superager"] = superager_vec  # add the label

    sig_pairs = [
        "7Networks_RH_Cont_PFCl",
        "7Networks_RH_Default_PFCdPFCm",
        "Subcortical 213: Right Accumbens",
    ]

    for roi in sig_pairs:
        t1, t2 = f"{roi}_1", f"{roi}_2"
        if t1 in df_roi_values and t2 in df_roi_values:
            r = df_roi_values[t1].corr(df_roi_values[t2])
            print(f"{roi}: corr(t1 vs t2) = {r:.3f}")
        else:
            print(f"{roi}: missing one of t1/t2")

    # Then look whether they vary between superagers and non-superagers
    sig_pairs_by_tp = [
        "7Networks_RH_Cont_PFCl_2",
        "7Networks_RH_Default_PFCdPFCm_2",
        "Subcortical 213: Right Accumbens_1",
        "Subcortical 213: Right Accumbens_2",
    ]
    rois_to_plot = sig_pairs_by_tp  

    # Switch to long df
    plot_data = []
    for roi in rois_to_plot:
        col = f"{roi}"
        if col in df_roi_values:
            for _, row in df_roi_values.iterrows():
                plot_data.append({
                    "ROI": roi,
                    "Value": row[col],
                    "Group": "Superager" if row["superager"] == 1 else "Non-superager"
                })

    # Print superager mean and std for each ROI
    for roi in rois_to_plot:
        col = f"{roi}"
        if col in df_roi_values:
            superager_vals = df_roi_values[df_roi_values["superager"] == 1][col]
            mean_val = superager_vals.mean()
            std_val = superager_vals.std()
            print(f"{roi} (Superagers): mean={mean_val:.3f}, std={std_val:.3f}")
            nonsuperager_vals = df_roi_values[df_roi_values["superager"] == 0][col]
            mean_val_ns = nonsuperager_vals.mean()
            std_val_ns = nonsuperager_vals.std()
            print(f"{roi} (Non-Superagers): mean={mean_val_ns:.3f}, std={std_val_ns:.3f}")
    

if __name__ == "__main__":
    main()