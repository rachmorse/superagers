#!/usr/bin/env python3
"""
Generate a supplementary table in Word format, extracting the top SFC features
from the baseline and slope elastic-net models, with their elastic net and SVM ranks.
"""

from pathlib import Path
import docx
import pandas as pd

# Import the logic and paths used in 'plot_feature_importance.py'
from plot_feature_importance import ATLAS, RESULTS_DIR, load_and_prepare

# Number of top features per connectivity type with display name
CONNECTIVITY_TYPES = {"SFC": (50, "structure-function coupling")}

OUTPUT_DOCX = Path(f"/home/rachel/Desktop/superagers/elastic_net/figures_tables/supplementary_table_{ATLAS}.docx")
OUTPUT_CSV = Path(f"/home/rachel/Desktop/data/en_features_table_{ATLAS}.csv")  # For results.Rmd

def process_top_n(df, is_slope_model, n):
    """Extract top n features by importance mean and tag annual change features.

    Args:
        df (pd.DataFrame): The feature importance DataFrame.
        is_slope_model (bool): Whether this is the slope model (for tagging).
        n (int): Number of features to keep.

    Returns:
        pd.DataFrame: Top n features sorted by descending importance.
    """
    top = df.sort_values("perm_importance_mean", ascending=False).head(n).copy()

    # Append (slope) label for longitudinal features
    if is_slope_model:
        top.loc[top["feature_type"] == "slope", "label"] = top["label"] + " (annual change)"

    return top


def importance_csv(conn_type, which_features, model_prefix=""):
    """Path to a feature importance CSV saved by log_en.py.

    Args:
        conn_type (str): Connectivity type ("SFC", "SC" or "FC").
        which_features (str): "t1" (baseline) or "t1_slope" (baseline + annual change).
        model_prefix (str): "" for elastic net or "SVM_" for linear SVM.

    Returns:
        Path: The CSV path.
    """
    return RESULTS_DIR / f"{ATLAS}_{model_prefix}{conn_type}_ROI_{which_features}_long_include_feature_importance.csv"


def add_ranks(df, svm_df):
    """Add each feature's elastic net and SVM importance rank across all features.

    Args:
        df (pd.DataFrame): Elastic net feature importance.
        svm_df (pd.DataFrame): Linear SVM feature importance for the same features.

    Returns:
        pd.DataFrame: df with en_rank and svm_rank columns (1 = most important).
    """
    df = df.copy()
    df["en_rank"] = df["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
    svm_rank = svm_df.set_index("feature")["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
    df["svm_rank"] = df["feature"].map(svm_rank)
    return df


def row_values(r):
    """Format one feature's importance and ranks for the table.

    Args:
        r (pd.Series): One row from process_top_n.

    Returns:
        list[str]: Importance, elastic net rank and SVM rank.
    """
    return [f"{r['perm_importance_mean']:.4f}", str(r["en_rank"]), str(r["svm_rank"])]


def main():
    # Baseline and baseline + annual change sections 
    sections = []
    for conn_type, (n, _) in CONNECTIVITY_TYPES.items():
        df_baseline = add_ranks(load_and_prepare(importance_csv(conn_type, "t1"), is_slope_model=False),
                                load_and_prepare(importance_csv(conn_type, "t1", "SVM_"), is_slope_model=False))
        df_slope = add_ranks(load_and_prepare(importance_csv(conn_type, "t1_slope"), is_slope_model=True),
                             load_and_prepare(importance_csv(conn_type, "t1_slope", "SVM_"), is_slope_model=True))
        sections += [
            ("Baseline model", process_top_n(df_baseline, is_slope_model=False, n=n)),
            ("Baseline + annual change model", process_top_n(df_slope, is_slope_model=True, n=n)),
        ]

    doc = docx.Document()
    
    headers = ["Feature", "Elastic net importance (ΔAUC)", "Elastic net rank", "SVM rank"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = 'Table Grid'

    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    def add_model_section(sub_header_text, df):
        row = table.add_row()
        merged_cell = row.cells[0].merge(row.cells[-1])
        merged_cell.text = sub_header_text
        merged_cell.paragraphs[0].runs[0].bold = True

        for _, r in df.iterrows():
            data_row = table.add_row()
            for i, value in enumerate([f"    {r['label']}", *row_values(r)]):
                data_row.cells[i].text = value

    # Fill the document
    for heading, df in sections:
        add_model_section(heading, df)

    doc.save(OUTPUT_DOCX)
    print(f"Saved supplementary table to: {OUTPUT_DOCX}")

    # Same table as CSV, with section headings as rows
    rows = []
    for heading, df in sections:
        rows.append(dict.fromkeys(headers, "") | {headers[0]: heading})
        rows += [dict(zip(headers, [r["label"], *row_values(r)])) for _, r in df.iterrows()]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved supplementary table to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()