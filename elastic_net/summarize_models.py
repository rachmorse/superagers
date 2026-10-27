#!/usr/bin/env python3
"""Summarize the elastic net and linear SVM models and compute Benjamini-Hochberg FDR-adjusted p-values.

Reads the permutation results and out-of-fold predictions saved by run_classifier.py and
writes one row per model for results.Rmd.
"""

import pickle
from typing import Iterable, List

import pandas as pd
from run_classifier import RESULTS_DIR, delong_auc_ci
from prep_data_for_en import ATLAS, DATA_DIR
from sklearn.metrics import roc_auc_score

MODELS = {"elastic_net": "", "linear_svm": "SVM_"}  # File prefix after the atlas
CONNECTIVITY_TYPES = ["SFC", "FC", "SC"]
FEATURE_SETS = ["t1", "t1_slope"]
OUTPUT_CSV = DATA_DIR / f"en_model_results_{ATLAS}.csv"  # For results.Rmd


def bh_fdr(pvals: Iterable[float]) -> List[float]:
    """Return BH FDR adjusted p-values in original order."""
    pvals = list(pvals)
    m = len(pvals)
    if m == 0:
        return []

    for p in pvals:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"All p-values must be in [0, 1]. Found: {p}")

    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted_sorted = [0.0] * m

    for rank, (_, p) in enumerate(indexed, start=1):
        adjusted_sorted[rank - 1] = p * m / rank

    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted = [0.0] * m
    for i, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = min(adjusted_sorted[i], 1.0)

    return adjusted


def collect_results() -> pd.DataFrame:
    """Collect AUC, DeLong CI, p-value and permutations run for each model.

    Returns:
        pd.DataFrame: One row per model, connectivity type and feature set.
    """
    rows = []
    for model, file_prefix in MODELS.items():
        for conn_type in CONNECTIVITY_TYPES:
            for features in FEATURE_SETS:
                prefix = RESULTS_DIR / f"{ATLAS}_{file_prefix}{conn_type}_ROI_{features}_long_include"
                with open(f"{prefix}_perm_results.pkl", "rb") as f:
                    perm = pickle.load(f)
                oof = pd.read_csv(f"{prefix}_oof_predictions.csv")
                ci_low, ci_high = delong_auc_ci(oof["y_true"], oof["y_score"])

                rows.append({
                    "model": model,
                    "connectivity_type": conn_type,
                    "features": features,
                    "auc": roc_auc_score(oof["y_true"], oof["y_score"]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": perm["p_value"],
                    "n_perm": perm["n_perm_used"],  # Fewer than 1000 if stopped early
                })

    return pd.DataFrame(rows)


def main() -> None:
    """Save and print each model's results with FDR-adjusted p-values."""
    df = collect_results()
    # Correct within each model type (e.g. SVM or elastic net)
    df["p_fdr"] = df.groupby("model")["p_value"].transform(bh_fdr)
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"Saved model results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
