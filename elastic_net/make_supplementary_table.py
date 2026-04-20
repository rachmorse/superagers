#!/usr/bin/env python3
"""
Generate a supplementary table in Word format, extracting the top 20 features 
from both the baseline and slope elastic-net models.
"""

import pandas as pd
from pathlib import Path
import docx

# Import the logic and paths used in 'plot_feature_importance.py'
from plot_feature_importance import (
    load_and_prepare,
    FEATURE_IMPORTANCE_T1_CSV,
    FEATURE_IMPORTANCE_SLOPE_CSV
)

OUTPUT_DOCX = Path("/home/rachel/Desktop/superagers/elastic_net/figures_tables/supplementary_table.docx")

def format_pval(val):
    """Format p-value to 4 decimal places and add significance asterisks.
    
    Args:
        val (float or NaN): The p-value to format.

    Returns:
        str: Formatted p-value with significance asterisks if applicable.
    """
    if pd.isna(val):
        return ""
    base = f"{val:.4f}"
    if val < 0.001:
        return base + "***"
    elif val < 0.01:
        return base + "**"
    elif val < 0.05:
        return base + "*"
    return base


def process_top_20(df, is_slope_model):
    """Extract top 20 features by importance mean and tag annual change features.

    Args:
        df (pd.DataFrame): The feature importance DataFrame.
        is_slope_model (bool): Whether this is the slope model (for tagging).

    Returns:
        pd.DataFrame: Top 20 features sorted by descending importance.
    """
    top20 = df.sort_values("perm_importance_mean", ascending=False).head(20).copy()

    # Append (slope) label for longitudinal features
    if is_slope_model:
        top20.loc[top20["feature_type"] == "slope", "label"] = top20["label"] + " (annual change)"
        
    return top20


def main():
    # Load and prep data using the imported function
    df_baseline = load_and_prepare(FEATURE_IMPORTANCE_T1_CSV, is_slope_model=False)
    df_slope = load_and_prepare(FEATURE_IMPORTANCE_SLOPE_CSV, is_slope_model=True)
    
    top20_baseline = process_top_20(df_baseline, is_slope_model=False)
    top20_slope = process_top_20(df_slope, is_slope_model=True)

    doc = docx.Document()
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    
    headers = ["Feature", "Importance Mean ΔAUC", "p-value", "pFDR"]
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True

    def add_model_section(sub_header_text, df):
        row = table.add_row()
        merged_cell = row.cells[0].merge(row.cells[3])
        merged_cell.text = sub_header_text
        merged_cell.paragraphs[0].runs[0].bold = True

        for _, r in df.iterrows():
            data_row = table.add_row()
            data_row.cells[0].text = f"    {r['label']}" # Small indent for readability
            data_row.cells[1].text = f"{r['perm_importance_mean']:.4f}"
            data_row.cells[2].text = format_pval(r['p_value'])
            data_row.cells[3].text = format_pval(r['p_fdr'])

    # Fill the document
    add_model_section("Baseline structure-function coupling model", top20_baseline)
    add_model_section("Baseline + annual change structure-function coupling model", top20_slope)

    doc.save(OUTPUT_DOCX)
    print(f"Saved supplementary table to: {OUTPUT_DOCX}")

if __name__ == "__main__":
    main()