#!/usr/bin/env python3
"""Build the supplementary table of top SFC features from the elastic net models.

Lists the top features from the baseline and baseline + annual change models with their
elastic net and SVM importance ranks. Saves them as a Word table and as a CSV for results.Rmd.
"""

import sys
from pathlib import Path
from typing import List

import docx
import pandas as pd

# Import the logic used in 'plot_feature_importance.py'
from plot_feature_importance import MODEL_TITLES, feature_importance_csv, load_and_prepare

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, ML_ATLAS

TOP_N = 50  # Top SFC features per model
OUTPUT_DOCX = Path(__file__).resolve().parents[1] / "analyses" / "figures" / f"supplementary_table_{ML_ATLAS}.docx"
OUTPUT_CSV = DATA_DIR / f"en_features_table_{ML_ATLAS}.csv"  # For results.Rmd


def process_top_n(df: pd.DataFrame, is_slope_model: bool, n: int) -> pd.DataFrame:
    """Extract top n features by importance mean and tag annual change features.

    Args:
        df: The feature importance DataFrame.
        is_slope_model: Whether this is the slope model (for tagging).
        n: Number of features to keep.

    Returns:
        Top n features sorted by descending importance.
    """
    top = df.sort_values("perm_importance_mean", ascending=False).head(n).copy()

    # Append (annual change) to the label of annual change features
    if is_slope_model:
        top.loc[top["feature_type"] == "slope", "label"] = top["label"] + " (annual change)"

    return top


def add_ranks(df: pd.DataFrame, svm_df: pd.DataFrame) -> pd.DataFrame:
    """Add each feature's elastic net and SVM importance rank across all features.

    Args:
        df: Elastic net feature importance.
        svm_df: Linear SVM feature importance for the same features.

    Returns:
        df with en_rank and svm_rank columns (1 = most important).
    """
    df = df.copy()
    df["en_rank"] = df["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
    svm_rank = svm_df.set_index("feature")["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
    df["svm_rank"] = df["feature"].map(svm_rank)
    return df


def row_values(r: pd.Series) -> List[str]:
    """Format one feature's importance and ranks for the table.

    Args:
        r: One row from process_top_n.

    Returns:
        Importance, elastic net rank, and SVM rank.
    """
    return [f"{r['perm_importance_mean']:.4f}", str(r["en_rank"]), str(r["svm_rank"])]


def main() -> None:
    """Build the top features table and save it as a Word table and a CSV."""
    # Baseline and baseline + annual change sections
    sections = []
    for features, heading in MODEL_TITLES.items():
        is_slope = features == "t1_slope"
        df = add_ranks(
            load_and_prepare(feature_importance_csv(features), is_slope_model=is_slope),
            load_and_prepare(feature_importance_csv(features, "linear_svm"), is_slope_model=is_slope),
        )
        sections.append((heading, process_top_n(df, is_slope_model=is_slope, n=TOP_N)))

    doc = docx.Document()

    headers = ["Feature", "Elastic net importance (ΔAUC)", "Elastic net rank", "SVM rank"]
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"

    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    def add_model_section(sub_header_text: str, df: pd.DataFrame) -> None:
        """Add a bold section heading row and one row per feature to the table.

        Args:
            sub_header_text: Section heading.
            df: Top features of one model from process_top_n.
        """
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

    # Same table as CSV with section headings as rows
    rows = []
    for heading, df in sections:
        rows.append(dict.fromkeys(headers, "") | {headers[0]: heading})
        rows += [dict(zip(headers, [r["label"], *row_values(r)])) for _, r in df.iterrows()]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved supplementary table to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
