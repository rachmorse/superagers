"""Classify superager status with nested-CV elastic net or linear SVM and a permutation test."""

import hashlib
import os
import pickle
import re
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prep_data_for_en import ATLAS, DATA_DIR, FC_ROOT, SC_ROOT, SFC_ROOT, get_subjects_to_process
from scipy.stats import norm
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state

MODEL: Literal["elastic_net", "linear_svm"] = "elastic_net"
CONNECTIVITY_TYPE: Literal["SFC", "FC", "SC"] = "FC"
# Features used to predict superager status: baseline only or baseline plus annual change
WHICH_FEATURES: Literal["t1", "t1_slope"] = "t1_slope"
N_JOBS = 20  # Parallel jobs for the main model's grid search and for the permutations
BC_H = 50 # Besag–Clifford sequential permutation test
# Atlas and SVM prefixes so outputs from different atlases or models never mix
MODEL_PREFIX = f"{ATLAS}_" + ("" if MODEL == "elastic_net" else "SVM_")
RESULTS_DIR = Path(__file__).resolve().parent / "results"



def delong_auc_ci(y_true, y_score, alpha=0.05):
    """Compute a DeLong CI for AUC on out-of-fold predictions.

    This is analytic, so it needs no resampling or seed.

    Args:
        y_true (array-like): True binary labels (0/1).
        y_score (array-like): Predicted probabilities or decision scores for the positive class.
        alpha (float): Significance level for the confidence interval (e.g., 0.05 for 95% CI).

    Returns:
        tuple: Lower and upper bounds of the AUC confidence interval.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]

    # Compare every positive with every negative. Ties count as half
    comparisons = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    auc = comparisons.mean()
    se = np.sqrt(
        comparisons.mean(axis=1).var(ddof=1) / len(pos)
        + comparisons.mean(axis=0).var(ddof=1) / len(neg)
    )
    z = norm.ppf(1 - alpha / 2)
    return (float(auc - z * se), float(auc + z * se))


def positive_scores(model, X):
    """Get the positive class scores from a fitted model.

    Elastic net returns probabilities and linear SVM returns decision scores. Both work for AUC.

    Args:
        model (Pipeline): Fitted scaling + classifier pipeline.
        X (np.ndarray): Feature matrix to score.

    Returns:
        np.ndarray: Score for the positive class for each row of X.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def prep_data(subjects, root_path, fc_root_path, sc_root_path, demographic_data, connectivity_type):
    """Extract features, superager status and covariates from the specified directories.

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
        subject_ids (np.ndarray): Subject IDs in the same row order as the arrays above.
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
        # Use ungrouped per-ROI features for both timepoints (ROI count depends on ATLAS)
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

    return X_t1, X_slope, superager_vec_long, covariates, df['id'].values



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


def _fit_outer_fold(pipe, param_grid, inner_cv, X_tr, y_tr, n_jobs):
    """Run the inner CV grid search on one outer training set and refit the best model.

    The main model and the permutations share this function so both use the same procedure.

    Args:
        pipe: The sklearn Pipeline with the model to fit.
        param_grid: The hyperparameter grid for GridSearchCV.
        inner_cv: The cross-validation splitter for the inner loop.
        X_tr: The outer training features.
        y_tr: The outer training labels.
        n_jobs: Number of parallel jobs for the grid search.

    Returns:
        GridSearchCV: The fitted search, with best_estimator_ refit on all of X_tr.
    """
    # Inner CV: try all C/l1_ratio combos on X_tr via 5-fold CV, pick the best
    # refit=True trains a final model on all of X_tr with the winning params
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=n_jobs,
    )
    return gs.fit(X_tr, y_tr)


def _permuted_auc(pipe, param_grid, inner_cv, outer_splits, X, y, y_perm):
    """Run the full nested CV on permuted labels and return the null AUC.

    Args:
        pipe: The sklearn Pipeline with the model to fit.
        param_grid: The hyperparameter grid for GridSearchCV.
        inner_cv: The cross-validation splitter for the inner loop.
        outer_splits: Cached (train, test) index pairs for the outer loop.
        X: The feature matrix.
        y: The original labels, used to score the out-of-fold predictions.
        y_perm: The permuted labels used for training.

    Returns:
        float: The out-of-fold AUC for this permutation.
    """
    # Save the out-of-fold predictions for each permutation
    perm_oof = np.zeros_like(y, dtype=float)
    for tr_idx, te_idx in outer_splits:  # permuted labels for training
        # Train and run on permuted labels
        gs_perm = _fit_outer_fold(
            pipe,
            param_grid,
            inner_cv,  # runs inner CV search grid to make the comparisons fair (e.g. 5-fold)
            X[tr_idx],
            y_perm[tr_idx],
            n_jobs=1,  # parallelism is across permutations
        )
        perm_oof[te_idx] = positive_scores(gs_perm.best_estimator_, X[te_idx])
    return roc_auc_score(y, perm_oof)


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
    """Train and evaluate the MODEL classifier (elastic net logistic or linear SVM).

    Uses nested cross-validation, out-of-fold predictions, permutation testing, and
    permutation-based feature importance.

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
            Also the parallel batch size; defaults to N_JOBS.
        which_features (str): Which features are being used ('t1' or 't1_slope') for naming outputs.
        label_type (str): Outcome timepoint (always "long") for naming outputs.

    Returns:
        dict containing:
            - "observed": with metrics {"auc"}.
            - "cv_fold_metrics": with per-fold AUC.
            - "oof": with {"y_true", "y_prob", "test_index"} for out-of-fold predictions (y_prob holds decision scores for linear SVM).
            - "best_params_per_fold": with best hyperparameters per fold.
            - "coef_mean": with mean coefficients across folds.
            - "coef_std": with coefficient standard deviations across folds.
            - "feat_importance": with feature-wise permutation importance, coefficients, and selection frequency (EN only).
            - "permutation_test": with {"aucs": np.ndarray, "p_value": float, "n_perm_used": int}.
    """
    # Create reproducible splits
    rng = check_random_state(random_state)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)

    # Make sure inputs are valid
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y must be binary (0/1).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if X.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match n_features.")
    # checkpoint_n is also the parallel batch size, so default to one permutation per worker
    if checkpoint_n is None or checkpoint_n <= 0:
        checkpoint_n = N_JOBS

    # Fixed CV configuration for stability
    OUTER_SPLITS = 10
    INNER_SPLITS = 5
    # Every outer test fold needs both classes to compute its AUC
    if np.bincount(y, minlength=2).min() < OUTER_SPLITS:
        raise ValueError(f"Need at least {OUTER_SPLITS} subjects per class for {OUTER_SPLITS}-fold CV.")

    # Stratified K-Folds keeps the same number of superagers/non-superagers in each fold
    outer_cv = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=random_state + 1)

    # Pipeline: scaling + Elastic-Net logistic or linear SVM
    if MODEL == "elastic_net":
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga", # only solver in scikit-learn that supports EN
            max_iter=20000, # increased because of convergence warnings at 5,000
            random_state=random_state,
            n_jobs=1
        )
    else:
        clf = LinearSVC(max_iter=200000, random_state=random_state)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)), # EN and SVM require scaling
        ("clf", clf),
    ])

    # Hyperparameter grid
    param_grid = {
        "clf__C": np.logspace(-4, 3, 10), # higher c = less regularization, may be overfit
    }
    if MODEL == "elastic_net":
        param_grid["clf__l1_ratio"] = [0.01, 0.1, 0.3, 0.5] # l1_ratio=1 is fully Lasso where some coeffs = 0

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

        gs = _fit_outer_fold(
            pipe,
            param_grid,
            inner_cv,
            X_tr,
            y_tr,
            n_jobs=N_JOBS,
        )

        best = gs.best_estimator_  # model refitted on full X_tr with best hyperparameters
        best_params = gs.best_params_
        best_params_per_fold.append(best_params)

        # Use X_te and predict superager probability (or SVM decision score) for held-out subjects
        prob_te = positive_scores(best, X_te)
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

        # Coefficients from the classifier step, averaged across folds after the loop
        coef = best.named_steps["clf"].coef_.ravel()
        assert coef.shape[0] == X.shape[1]
        coefs.append(coef)

        # Permutation importance on X_tr inner folds to keep the test set clean
        pi_mean = _perm_importance_on_train_cv(
            pipe, best_params, X_tr, y_tr, inner_cv, n_repeats_importance, random_state + fold_id
        )
        perm_importance_accumulator.append(pi_mean)

    # Metrics for how well the model did overall (on all out-of-fold predictions)
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
    if MODEL == "linear_svm":
        # L2 SVM never zeroes coefficients, so selection frequency is always 1
        feature_df = feature_df.drop(columns="selected_freq")

    # Model-level permutation test - now build the null distribution training the models on shuffled labels
    # Start by adding a check point for if the server crashes while this is running, not all progress is lost
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_file = RESULTS_DIR / (
        f"{MODEL_PREFIX}{connectivity_type}_{group_level}_{which_features}_"
        f"{label_type}_include_perm_results.pkl"
    )
    # Fingerprint data and settings so a checkpoint from a different run is never resumed
    run_info = {
        "data_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "model": MODEL,
        "random_state": random_state,
        "param_grid": repr(param_grid),
    }
    start_p = 0
    completed = 0

    # Resume if checkpoint exists (i.e. from a previous crash or stoppage)
    if os.path.exists(checkpoint_file):
        print(f"Checkpoint file found: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            saved = pickle.load(f)
        if saved.get("run_info") != run_info:
            raise ValueError(f"{checkpoint_file} is from different data or settings, delete it to start over.")

        completed = int(np.count_nonzero(~np.isnan(saved["perm_aucs"])))
        target_total = max(n_permutations, completed)

        if completed < target_total:
            # Expand arrays to the new total and copy old data
            perm_aucs = np.full(target_total, np.nan)
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
        perm_aucs = np.full(target_total, np.nan)
        start_p = 0

    progress_every = max(1, target_total // 10) # report progress after every 10%

    if completed >= target_total:
        print("All requested permutations already done.")
    elif np.sum(perm_aucs >= observed_auc) >= BC_H:
        print(f"Checkpoint already stopped early: {BC_H} null AUCs >= observed.")
    else:
        # Draw from permutation 0 so a resumed run continues the shuffle sequence instead of repeating it
        y_perms = [rng.permutation(y) for _ in range(target_total)][start_p:] # rng.permutation shuffles the labels randomly

        # Run each checkpoint batch in parallel with one permutation per worker
        for b_start in range(start_p, target_total, checkpoint_n):
            b_end = min(b_start + checkpoint_n, target_total)
            perm_aucs[b_start:b_end] = Parallel(n_jobs=N_JOBS)(
                delayed(_permuted_auc)(
                    pipe, param_grid, inner_cv, outer_splits, X, y, y_perms[p - start_p]
                )
                for p in range(b_start, b_end)
            )
            p = b_end - 1

            # Save results after each batch
            with open(checkpoint_file, "wb") as f:
                pickle.dump({
                    "last_p": p,
                    "perm_aucs": perm_aucs,
                    "feature": feature_names,   # safe mid-run
                    "run_info": run_info,
                }, f)
            print(f"Checkpoint saved at permutation {b_end}")

            if verbose and (b_end // progress_every > b_start // progress_every or b_end == target_total):
                print(f"[Permutation {b_end}/{target_total}] Null AUC (mean so far): {np.nanmean(perm_aucs):.3f}")

            # Besag-Clifford: stop once BC_H null AUCs reach the observed AUC
            if np.sum(perm_aucs[:b_end] >= observed_auc) >= BC_H:
                print(f"Stopping early: {BC_H} null AUCs >= observed by permutation {b_end}")
                break

    n_done = int(np.count_nonzero(~np.isnan(perm_aucs)))
    exceed_idx = np.flatnonzero(perm_aucs >= observed_auc)
    if n_done < target_total and len(exceed_idx) >= BC_H:
        # Stopped early
        n_perm_used = int(exceed_idx[BC_H - 1]) + 1
        p_value = BC_H / n_perm_used
    else:
        # All permutations ran: one-sided p-value (>= observed AUC), add 1 to numerator/denominator for stability
        n_perm_used = n_done
        p_value = (len(exceed_idx) + 1.0) / (n_done + 1.0)

    feature_df.sort_values("perm_importance_mean", ascending=False, inplace=True)

    # Save final state including p-value
    with open(f"{checkpoint_file}", "wb") as f:
        pickle.dump({
            "last_p": n_done - 1,
            "perm_aucs": perm_aucs,
            "feature": feature_names,
            "p_value": p_value,   # model-level p
            "n_perm_used": n_perm_used,
            "run_info": run_info,
        }, f)

    if start_p < target_total:
        print(f"Final checkpoint saved at permutation {n_done}")

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
            "p_value": float(p_value),
            "n_perm_used": n_perm_used,
        }
    }
    return results

def main():
    """Load the data, run the classifier, then save and print the results."""
    csv_path = DATA_DIR / "superager.csv"
    demographic_data = pd.read_csv(csv_path)
    demographic_data.columns = [re.sub(r"^w(\d+)_(.*)", r"\2_\1", c) for c in demographic_data.columns]
    required_cols = {"id", "age_1", "age_2", "YoE", "sex", "superager_long"}
    missing_cols = sorted(required_cols - set(demographic_data.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

    # Longitudinal superager definition requires both sessions
    subjects_tp1 = get_subjects_to_process(SFC_ROOT / "ses-01" / "individual_coupling_matrices", "ses-01", DATA_DIR)
    subjects_tp2 = get_subjects_to_process(SFC_ROOT / "ses-02" / "individual_coupling_matrices", "ses-02", DATA_DIR)
    subjects = sorted(set(subjects_tp1) & set(subjects_tp2))
    print(f"Subjects: {len(subjects)}")

    # Prepare the data for analysis
    X_t1, X_slope, superager_vec_long, covariates, subject_ids = prep_data(
        subjects, SFC_ROOT, FC_ROOT, SC_ROOT, demographic_data, CONNECTIVITY_TYPE)

    # Map feature indices back to ROI names, using a subject whose files loaded
    first_sub = subject_ids[0]
    if CONNECTIVITY_TYPE == "SFC":
        roi_names_path = SFC_ROOT / "ses-01" / "individual_coupling_matrices" / f"{first_sub}_ses-01_structure_function_coupling.csv"
    elif CONNECTIVITY_TYPE == "FC":
        roi_names_path = FC_ROOT / f"ses-01/individual_connectivity_matrices/grouped_rois/{first_sub}_ses-01_functional_connectivity_flat.csv"
    else:  # SC
        roi_names_path = SC_ROOT / f"ses-01/individual_connectivity_matrices/grouped_rois/{first_sub}_ses-01_structural_connectivity_flat.csv"
    roi_names = pd.read_csv(roi_names_path)['ROI_name'].tolist()

    roi_names_tp1 = [f"{r}_1" for r in roi_names]
    roi_names_slope = [f"{r}_slope" for r in roi_names]
    X_t1_slope = np.hstack([X_t1, X_slope])

    # Select which features to use, matching them to ROI names
    match WHICH_FEATURES:
        case 't1':
            X_use = X_t1
            feat_names_use = roi_names
        case 't1_slope':
            X_use = X_t1_slope
            feat_names_use = roi_names_tp1 + roi_names_slope
        case _:
            raise ValueError("WHICH_FEATURES must be one of: 't1', 't1_slope'")

    y_use = superager_vec_long

    valid_rows = np.isfinite(y_use) & np.isfinite(X_use).all(axis=1)
    valid_rows = valid_rows & np.isfinite(covariates).all(axis=1)
    X_use = X_use[valid_rows]
    y_use = y_use[valid_rows].astype(int)
    covariates = covariates[valid_rows]
    subject_ids = subject_ids[valid_rows]

    # Include covariates as model features
    X_use = np.hstack([X_use, covariates.astype(float)])
    feat_names_use = [*feat_names_use, "cov_age", "cov_YoE", "cov_sex"]

    # en_fdr.py finds EN models in the logs by "Training EN on"
    model_label = "EN" if MODEL == "elastic_net" else "SVM"
    print(
        f"Training {model_label} on {WHICH_FEATURES} features "
        f"({X_use.shape[1]} predictors) for "
        f"connectivity_type='{CONNECTIVITY_TYPE}'..."
    )

    t0 = time.time()
    print("Starting time:", time.ctime(t0))
    results = run_elastic_net(
        X_use, y_use, feat_names_use,
        n_permutations=1000,
        n_repeats_importance=10,
        random_state=7,
        verbose=1,
        checkpoint_n=20,
        connectivity_type=CONNECTIVITY_TYPE,
        group_level="ROI",
        which_features=WHICH_FEATURES,
        label_type="long",
    )

    feature_importance_csv = RESULTS_DIR / f"{MODEL_PREFIX}{CONNECTIVITY_TYPE}_ROI_{WHICH_FEATURES}_long_include_feature_importance.csv"
    results["feat_importance"].to_csv(feature_importance_csv, index=False)
    print(f"Saved feature importance table: {feature_importance_csv}")

    # Save out-of-fold predictions so AUC CIs and paired DeLong tests can be run without refitting
    oof_csv = RESULTS_DIR / f"{MODEL_PREFIX}{CONNECTIVITY_TYPE}_ROI_{WHICH_FEATURES}_long_include_oof_predictions.csv"
    pd.DataFrame({
        "subject": subject_ids,
        "y_true": results["oof"]["y_true"],
        "y_score": results["oof"]["y_prob"],
    }).to_csv(oof_csv, index=False)
    print(f"Saved out-of-fold predictions: {oof_csv}")

    print(results["observed"])
    fold_metrics = results["cv_fold_metrics"]
    print("Fold-wise AUCs:")
    for _, row in fold_metrics.iterrows():
        print(f"  fold {int(row['fold'])}: auc={row['auc']:.3f}")
    auc_ci = delong_auc_ci(
        results["oof"]["y_true"],
        results["oof"]["y_prob"],
        alpha=0.05,
    )
    print(f"AUC 95% CI (DeLong, oof): [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    print("Best params per fold:")
    for i, p in enumerate(results["best_params_per_fold"], start=1):
        params = ", ".join(f"{k.removeprefix('clf__')}={v}" for k, v in p.items())
        print(f"  fold {i}: {params}")
    print("Model-level p-value:", results["permutation_test"]["p_value"])
    print("Permutations used:", results["permutation_test"]["n_perm_used"])

    feat = results["feat_importance"].copy()
    feat = feat[~feat["feature"].str.startswith("cov_", na=False)]
    compact = feat[[c for c in ("feature", "perm_importance_mean", "selected_freq") if c in feat]].head(15)
    print("Top features by permutation importance:")
    with pd.option_context("display.max_columns", None):
        print(compact.to_string(index=False))
    dt = time.time() - t0
    print(f"Run took {dt:.2f}s")


if __name__ == "__main__":
    main()
